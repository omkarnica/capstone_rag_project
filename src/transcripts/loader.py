"""
Bulk loads transcript records into PostgreSQL.

load_transcripts:
    Stages records into a temp table via COPY, upserts into `transcripts`
    with ON CONFLICT (accession_no) DO NOTHING, then queries back IDs for
    all passed accession numbers (both newly inserted and pre-existing).
    Returns {accession_no: transcript_id} — main.py uses this map to build
    FK-correct section records.

load_transcript_sections:
    COPY directly into `transcript_sections`. The FK to `transcripts`
    guarantees referential integrity; no unique conflict key is needed.

load_transcripts_data:
    Transaction wrapper — calls both loaders in sequence, commits on
    success, rolls back on any exception.
"""

from __future__ import annotations

from io import StringIO
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extensions import connection as PgConnection

from src.utils.exceptions import db_error_boundary
from src.utils.logger import get_logger
from src.xbrl.loader import get_connection

logger = get_logger(__name__)

_TRANSCRIPT_COLUMNS = [
    "cik", "accession_no", "company_name",
    "filed_date", "period_of_report", "form_type", "exhibit_url",
]


def load_transcripts(conn: PgConnection, records: list[dict]) -> dict[str, int]:
    """
    Bulk insert transcript metadata and return the accession_no → id mapping.

    Uses a temp staging table + COPY to avoid row-by-row inserts. Upserts
    with ON CONFLICT (accession_no) DO NOTHING so reruns are safe. After
    the upsert, queries transcript IDs for all passed accession numbers —
    this covers both newly inserted rows and rows that already existed.

    Args:
        conn:    Active psycopg2 connection. Caller manages lifecycle.
        records: List of dicts; each must contain all _TRANSCRIPT_COLUMNS keys.

    Returns:
        Dict mapping accession_no → transcript id for every record passed.

    Raises:
        DatabaseError: wraps any psycopg2.Error.
    """
    if not records:
        return {}

    df = pd.DataFrame(records)[_TRANSCRIPT_COLUMNS]

    with db_error_boundary("CREATE TEMP TABLE", table="transcripts_staging"):
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TEMP TABLE transcripts_staging (
                    cik              INTEGER,
                    accession_no     TEXT,
                    company_name     TEXT,
                    filed_date       DATE,
                    period_of_report DATE,
                    form_type        TEXT,
                    exhibit_url      TEXT
                ) ON COMMIT DROP
            """)

    buffer = StringIO()
    df.to_csv(buffer, sep="\t", header=False, index=False, na_rep="\\N")
    buffer.seek(0)

    columns_sql = ", ".join(_TRANSCRIPT_COLUMNS)
    copy_sql = (
        f"COPY transcripts_staging ({columns_sql}) "
        "FROM STDIN WITH (FORMAT TEXT, DELIMITER E'\\t', NULL '\\N')"
    )

    with db_error_boundary("COPY", table="transcripts_staging"):
        with conn.cursor() as cur:
            cur.copy_expert(copy_sql, buffer)

    with db_error_boundary("INSERT", table="transcripts"):
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO transcripts ({columns_sql})
                SELECT {columns_sql} FROM transcripts_staging
                ON CONFLICT (accession_no) DO NOTHING
            """)
            inserted = cur.rowcount

    accession_nos = [r["accession_no"] for r in records]
    with db_error_boundary("SELECT", table="transcripts"):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, accession_no FROM transcripts WHERE accession_no = ANY(%s)",
                (accession_nos,),
            )
            id_map: dict[str, int] = {row[1]: row[0] for row in cur.fetchall()}

    logger.info(
        "Loaded transcripts",
        extra={"inserted": inserted, "id_map_size": len(id_map)},
    )
    return id_map


def _escape_text_copy(value: str | None) -> str:
    """
    Escape a string for PostgreSQL COPY FORMAT TEXT.

    FORMAT TEXT does not support quoting — special characters must be
    backslash-escaped. pandas to_csv uses CSV quoting which FORMAT TEXT
    interprets as literal quote characters, corrupting the data.
    """
    if value is None:
        return "\\N"
    return (
        value
        .replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def load_transcript_sections(conn: PgConnection, sections: list[dict]) -> int:
    """
    Bulk insert transcript sections into `transcript_sections` via COPY.

    Builds the COPY buffer manually (not via pandas) to guarantee that
    transcript_id is serialized as a plain integer string and that
    section_text — which contains embedded newlines — is correctly escaped
    for PostgreSQL FORMAT TEXT rather than CSV-quoted.

    Args:
        conn:     Active psycopg2 connection. Caller manages lifecycle.
        sections: List of dicts with keys: transcript_id (int), section_item
                  (str), section_text (str).

    Returns:
        Number of rows loaded.

    Raises:
        DatabaseError: wraps any psycopg2.Error.
    """
    if not sections:
        return 0

    buffer = StringIO()
    for section in sections:
        transcript_id = str(section["transcript_id"])
        section_item  = _escape_text_copy(section.get("section_item"))
        section_text  = _escape_text_copy(section.get("section_text"))
        buffer.write(f"{transcript_id}\t{section_item}\t{section_text}\n")
    buffer.seek(0)

    copy_sql = (
        "COPY transcript_sections (transcript_id, section_item, section_text) "
        "FROM STDIN WITH (FORMAT TEXT, DELIMITER E'\\t', NULL '\\N')"
    )

    with db_error_boundary("COPY", table="transcript_sections"):
        with conn.cursor() as cur:
            cur.copy_expert(copy_sql, buffer)

    inserted = len(sections)
    logger.info(
        "Loaded transcript_sections",
        extra={"rows_inserted": inserted},
    )
    return inserted


def load_transcripts_data(
    transcript_records: list[dict],
    section_records: list[dict],
    conn: Optional[PgConnection] = None,
) -> tuple[int, int]:
    """
    Load transcripts and sections in a single transaction.

    Args:
        transcript_records: Passed to load_transcripts.
        section_records:    Passed to load_transcript_sections.
        conn:               Optional existing connection; creates one if None.

    Returns:
        (transcripts_id_map_size, sections_inserted)

    Raises:
        DatabaseError: rolls back and re-raises on any psycopg2 failure.
    """
    owns_connection = conn is None
    if conn is None:
        conn = get_connection()

    try:
        id_map = load_transcripts(conn, transcript_records)
        sections_inserted = load_transcript_sections(conn, section_records)
        conn.commit()
        return len(id_map), sections_inserted
    except Exception:
        conn.rollback()
        raise
    finally:
        if owns_connection:
            conn.close()
