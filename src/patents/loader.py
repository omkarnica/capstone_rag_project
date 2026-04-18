"""
Bulk loads patent records into PostgreSQL.

load_patents:
    Stages records into a temp table via COPY, upserts into `patents`
    with ON CONFLICT (patent_id) DO NOTHING. The cpc_codes TEXT[] column
    is serialized to PostgreSQL array literal format before COPY.

load_patent_claims:
    COPY directly into `patent_claims`. Requires patent records to already
    exist (FK constraint on patent_id); always call load_patents first.

load_patents_data:
    Transaction wrapper — calls both loaders in sequence, commits on
    success, rolls back on any exception.
"""

from __future__ import annotations

import csv
import math
from io import StringIO
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extensions import connection as PgConnection

from src.utils.exceptions import db_error_boundary
from src.utils.logger import get_logger
from xbrl.loader import get_connection

logger = get_logger(__name__)

_PATENT_COLUMNS = [
    "patent_id",
    "patent_title",
    "grant_date",
    "assignee_organization",
    "cpc_codes",
    "citation_count",
]

_CLAIM_COLUMNS = [
    "patent_id",
    "claim_number",
    "claim_text",
    "is_independent",
]


def _format_pg_array(values: list[str]) -> str:
    """
    Serialize a Python list to a PostgreSQL TEXT[] literal for COPY.

    e.g. ["G06F", "H04L"] → '{"G06F","H04L"}'
    Empty list → '{}'
    """
    if not values:
        return "{}"
    return "{" + ",".join(values) + "}"


def load_patents(conn: PgConnection, records: list[dict]) -> int:
    """
    Bulk insert patent records and return the number of newly inserted rows.

    Stages records into a temp table via COPY then upserts with
    ON CONFLICT (patent_id) DO NOTHING so reruns are idempotent.
    The cpc_codes list is serialized to a PostgreSQL array literal
    before the COPY operation.

    Args:
        conn:    Active psycopg2 connection. Caller manages lifecycle.
        records: List of dicts; each must contain all _PATENT_COLUMNS keys.
                 cpc_codes must be a Python list[str].

    Returns:
        Number of newly inserted rows (pre-existing rows are skipped).

    Raises:
        DatabaseError: wraps any psycopg2.Error.
    """
    if not records:
        return 0

    # Serialize cpc_codes list → PostgreSQL array literal for COPY
    serialized = [
        {**r, "cpc_codes": _format_pg_array(r.get("cpc_codes") or [])}
        for r in records
    ]

    df = pd.DataFrame(serialized)[_PATENT_COLUMNS]

    with db_error_boundary("CREATE TEMP TABLE", table="patents_staging"):
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TEMP TABLE patents_staging (
                    patent_id             TEXT,
                    patent_title          TEXT,
                    grant_date            DATE,
                    assignee_organization TEXT,
                    cpc_codes             TEXT[],
                    citation_count        INTEGER
                ) ON COMMIT DROP
            """)

    buffer = StringIO()
    writer = csv.writer(buffer, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for row in df.itertuples(index=False, name=None):
        writer.writerow(["\\N" if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in row])
    buffer.seek(0)

    columns_sql = ", ".join(_PATENT_COLUMNS)
    copy_sql = (
        f"COPY patents_staging ({columns_sql}) "
        "FROM STDIN WITH (FORMAT TEXT, DELIMITER E'\\t', NULL '\\N')"
    )

    with db_error_boundary("COPY", table="patents_staging"):
        with conn.cursor() as cur:
            cur.copy_expert(copy_sql, buffer)

    with db_error_boundary("INSERT", table="patents"):
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO patents ({columns_sql})
                SELECT {columns_sql} FROM patents_staging
                ON CONFLICT (patent_id) DO NOTHING
            """)
            inserted = cur.rowcount

    logger.info(
        "Loaded patents",
        extra={"inserted": inserted, "total_records": len(records)},
    )
    return inserted


def load_patent_claims(conn: PgConnection, claims: list[dict]) -> int:
    """
    Bulk insert patent claims into `patent_claims` via COPY.

    Patents must already exist in the `patents` table before calling this
    function (foreign key constraint on patent_id).

    Args:
        conn:   Active psycopg2 connection. Caller manages lifecycle.
        claims: List of dicts with keys: patent_id, claim_number,
                claim_text, is_independent.

    Returns:
        Number of rows loaded.

    Raises:
        DatabaseError: wraps any psycopg2.Error.
    """
    if not claims:
        return 0

    df = pd.DataFrame(claims)[_CLAIM_COLUMNS]
    df["claim_text"] = df["claim_text"].str.replace("\n", " ", regex=False).str.replace("\r", " ", regex=False)

    buffer = StringIO()
    df.to_csv(buffer, sep="\t", header=False, index=False, na_rep="\\N")
    buffer.seek(0)

    columns_sql = ", ".join(_CLAIM_COLUMNS)
    copy_sql = (
        f"COPY patent_claims ({columns_sql}) "
        "FROM STDIN WITH (FORMAT TEXT, DELIMITER E'\\t', NULL '\\N')"
    )

    with db_error_boundary("COPY", table="patent_claims"):
        with conn.cursor() as cur:
            cur.copy_expert(copy_sql, buffer)

    inserted = len(df)
    logger.info(
        "Loaded patent_claims",
        extra={"rows_inserted": inserted},
    )
    return inserted


def load_patents_data(
    patent_records: list[dict],
    claim_records: list[dict],
    conn: Optional[PgConnection] = None,
) -> tuple[int, int]:
    """
    Load patents and claims in a single transaction.

    Patents are inserted first to satisfy the FK constraint on patent_claims.

    Args:
        patent_records: Passed to load_patents.
        claim_records:  Passed to load_patent_claims.
        conn:           Optional existing connection; creates one if None.

    Returns:
        (patents_inserted, claims_inserted)

    Raises:
        DatabaseError: rolls back and re-raises on any psycopg2 failure.
    """
    owns_connection = conn is None
    if conn is None:
        conn = get_connection()

    try:
        patents_inserted = load_patents(conn, patent_records)
        claims_inserted = load_patent_claims(conn, claim_records)
        conn.commit()
        return patents_inserted, claims_inserted
    except Exception:
        conn.rollback()
        raise
    finally:
        if owns_connection:
            conn.close()
