"""
Bulk loads CourtListener litigation data into PostgreSQL.

load_cases:
    Stages litigation_cases into a temp table via COPY, upserts with
    ON CONFLICT (case_id) DO NOTHING.

load_opinions:
    COPY directly into litigation_opinions staging, upserts with
    ON CONFLICT (opinion_id) DO NOTHING.
    Requires cases to already exist (FK on case_id); call load_cases first.

load_litigation_data:
    Transaction wrapper — calls both loaders in sequence, commits on
    success, rolls back on any exception.
"""

from __future__ import annotations

from io import StringIO
from typing import Optional

from psycopg2.extensions import connection as PgConnection

from src.utils.exceptions import db_error_boundary
from src.utils.logger import get_logger
from src.xbrl.loader import get_connection

logger = get_logger(__name__)

_CASE_COLUMNS = [
    "case_id", "case_name", "court", "court_citation",
    "date_filed", "docket_number", "status", "company_name",
    "url", "cite_count",
]

_OPINION_COLUMNS = [
    "opinion_id", "case_id", "opinion_type", "plain_text", "snippet",
]


def _esc(value: str) -> str:
    """Escape a value for PostgreSQL COPY FORMAT TEXT."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def load_cases(conn: PgConnection, records: list[dict]) -> int:
    """Bulk insert litigation cases; skips existing case_ids."""
    if not records:
        return 0

    buffer = StringIO()
    for r in records:
        row = [
            _esc(r.get("case_id") or ""),
            _esc(r.get("case_name") or ""),
            _esc(r.get("court") or ""),
            _esc(r.get("court_citation") or ""),
            str(r["date_filed"]) if r.get("date_filed") else "\\N",
            _esc(r.get("docket_number") or ""),
            _esc(r.get("status") or ""),
            _esc(r.get("company_name") or ""),
            _esc(r.get("url") or ""),
            str(r.get("cite_count") or 0),
        ]
        buffer.write("\t".join(row) + "\n")
    buffer.seek(0)

    cols = ", ".join(_CASE_COLUMNS)

    with db_error_boundary("CREATE TEMP TABLE", table="litigation_cases_staging"):
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TEMP TABLE litigation_cases_staging (
                    case_id         TEXT,
                    case_name       TEXT,
                    court           TEXT,
                    court_citation  TEXT,
                    date_filed      DATE,
                    docket_number   TEXT,
                    status          TEXT,
                    company_name    TEXT,
                    url             TEXT,
                    cite_count      INTEGER
                ) ON COMMIT DROP
            """)

    with db_error_boundary("COPY", table="litigation_cases_staging"):
        with conn.cursor() as cur:
            cur.copy_expert(
                f"COPY litigation_cases_staging ({cols}) "
                "FROM STDIN WITH (FORMAT TEXT, DELIMITER E'\\t', NULL '\\N')",
                buffer,
            )

    with db_error_boundary("INSERT", table="litigation_cases"):
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO litigation_cases ({cols})
                SELECT {cols} FROM litigation_cases_staging
                ON CONFLICT (case_id) DO NOTHING
            """)
            inserted = cur.rowcount

    logger.info("Loaded litigation_cases", extra={"inserted": inserted, "total": len(records)})
    return inserted


def load_opinions(conn: PgConnection, records: list[dict]) -> int:
    """Bulk insert litigation opinions; skips existing opinion_ids."""
    if not records:
        return 0

    buffer = StringIO()
    for r in records:
        row = [
            _esc(r.get("opinion_id") or ""),
            _esc(r.get("case_id") or ""),
            _esc(r.get("opinion_type") or ""),
            _esc(r.get("plain_text") or ""),
            _esc(r.get("snippet") or ""),
        ]
        buffer.write("\t".join(row) + "\n")
    buffer.seek(0)

    cols = ", ".join(_OPINION_COLUMNS)

    with db_error_boundary("CREATE TEMP TABLE", table="litigation_opinions_staging"):
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TEMP TABLE litigation_opinions_staging (
                    opinion_id   TEXT,
                    case_id      TEXT,
                    opinion_type TEXT,
                    plain_text   TEXT,
                    snippet      TEXT
                ) ON COMMIT DROP
            """)

    with db_error_boundary("COPY", table="litigation_opinions_staging"):
        with conn.cursor() as cur:
            cur.copy_expert(
                f"COPY litigation_opinions_staging ({cols}) "
                "FROM STDIN WITH (FORMAT TEXT, DELIMITER E'\\t', NULL '\\N')",
                buffer,
            )

    with db_error_boundary("INSERT", table="litigation_opinions"):
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO litigation_opinions ({cols})
                SELECT {cols} FROM litigation_opinions_staging
                ON CONFLICT (opinion_id) DO NOTHING
            """)
            inserted = cur.rowcount

    logger.info("Loaded litigation_opinions", extra={"inserted": inserted, "total": len(records)})
    return inserted


def load_litigation_data(
    case_records: list[dict],
    opinion_records: list[dict],
    conn: Optional[PgConnection] = None,
) -> tuple[int, int]:
    """
    Load cases and opinions in a single transaction.

    Cases are inserted first to satisfy the FK constraint on litigation_opinions.

    Returns:
        (cases_inserted, opinions_inserted)
    """
    owns_connection = conn is None
    if conn is None:
        conn = get_connection()

    try:
        cases_inserted = load_cases(conn, case_records)
        opinions_inserted = load_opinions(conn, opinion_records)
        conn.commit()
        return cases_inserted, opinions_inserted
    except Exception:
        conn.rollback()
        raise
    finally:
        if owns_connection:
            conn.close()
