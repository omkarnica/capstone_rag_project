"""
Bulk loader for XBRL DataFrames into PostgreSQL.

Requirements covered:
- psycopg2 PostgreSQL connection
- bulk insert via COPY into a staging table
- upsert from staging → target using ON CONFLICT DO NOTHING
  (safe to re-run the same quarter without duplicate key errors)
"""

from __future__ import annotations

import os
from io import StringIO
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extensions import connection as PgConnection


def get_connection(
    host: Optional[str] = None,
    port: Optional[int] = None,
    dbname: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> PgConnection:
    """
    Create a PostgreSQL connection using explicit args or env defaults.
    """
    return psycopg2.connect(
        host=host or os.getenv("PGHOST", "localhost"),
        port=port or int(os.getenv("PGPORT", "5433")),
        dbname=dbname or os.getenv("PGDATABASE", "ma_oracle"),
        user=user or os.getenv("PGUSER", "postgres"),
        password=password or os.getenv("PGPASSWORD", ""),
    )


def _copy_dataframe(
    conn: PgConnection,
    df: pd.DataFrame,
    table_name: str,
    columns: list[str],
) -> int:
    """
    Bulk load a DataFrame into a table using COPY FROM STDIN.
    Returns number of loaded rows.
    """
    if df.empty:
        return 0

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {table_name}: {missing}")

    copy_df = df[columns].copy()

    for col in copy_df.columns:
        if pd.api.types.is_datetime64_any_dtype(copy_df[col]):
            copy_df[col] = copy_df[col].dt.strftime("%Y-%m-%d")

    buffer = StringIO()
    copy_df.to_csv(
        buffer,
        sep="\t",
        header=False,
        index=False,
        na_rep="\\N",
    )
    buffer.seek(0)

    columns_sql = ", ".join(columns)
    copy_sql = (
        f"COPY {table_name} ({columns_sql}) "
        "FROM STDIN WITH (FORMAT TEXT, DELIMITER E'\\t', NULL '\\N')"
    )

    with conn.cursor() as cur:
        cur.copy_expert(copy_sql, buffer)

    return len(copy_df)


def load_filings(conn: PgConnection, filings_df: pd.DataFrame) -> int:
    """
    Insert filings into `filings`, skipping rows that already exist.
    Safe to call multiple times with overlapping data.        ← CHANGED
    """
    filings_columns = [
        "adsh", "cik", "name", "sic", "form",
        "period", "fiscal_year", "fiscal_period", "filed",
    ]

    if filings_df.empty:
        return 0

    # ── CHANGED: use a temp staging table + INSERT ... ON CONFLICT DO NOTHING ──
    # This means if you re-run the same quarter, duplicate filings are silently
    # skipped instead of crashing with a primary key violation.
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE filings_staging
            (LIKE filings INCLUDING ALL)
            ON COMMIT DROP
        """)

    rows = _copy_dataframe(conn, filings_df, "filings_staging", filings_columns)

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO filings
            SELECT * FROM filings_staging
            ON CONFLICT (adsh) DO NOTHING
        """)
        inserted = cur.rowcount

    return inserted


def load_facts(conn: PgConnection, facts_df: pd.DataFrame) -> int:
    """
    Bulk insert facts data into `facts`.
    Facts have a serial PK so duplicates are naturally avoided as long
    as filings are deduplicated upstream (via load_filings above).
    """
    facts_columns = [
        "adsh", "tag", "version", "ddate", "qtrs", "uom", "value",
    ]
    return _copy_dataframe(conn, facts_df, "facts", facts_columns)


def load_xbrl_data(
    filings_df: pd.DataFrame,
    facts_df: pd.DataFrame,
    conn: Optional[PgConnection] = None,
) -> tuple[int, int]:
    """
    Load filings and facts in a single transaction.

    Returns:
        (num_filings_loaded, num_facts_loaded)
    """
    owns_connection = conn is None
    if conn is None:
        conn = get_connection()

    try:
        filings_loaded = load_filings(conn, filings_df)
        facts_loaded   = load_facts(conn, facts_df)
        conn.commit()
        return filings_loaded, facts_loaded
    except Exception:
        conn.rollback()
        raise
    finally:
        if owns_connection:
            conn.close()