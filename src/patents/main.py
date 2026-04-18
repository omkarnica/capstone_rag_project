"""
Orchestrator for the patents bulk TSV ingestion pipeline.

Ingestion flow:
1) read_all   — chunk-read four bulk TSV files, filter to assignee_id + date range
2) build_patent_records — merge DataFrames → patents schema dicts
3) build_claim_records  — filter to independent claims → patent_claims schema dicts
4) load_patents         — COPY into `patents` table, ON CONFLICT DO NOTHING
5) load_patent_claims   — COPY into `patent_claims` table

Patents must be loaded before claims to satisfy the FK constraint on
patent_claims.patent_id.

CLI:
    python -m src.patents.main
    python -m src.patents.main \\
        --assignee-id 38cff6e0-fc52-4faf-9fa9-b72912600dee \\
        --start 2020-01-01 --end 2025-12-31 \\
        --data-dir data/patents
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.patents import loader, parser, reader
from src.utils.logger import get_logger
from xbrl.loader import get_connection

logger = get_logger(__name__)

_DEFAULT_ASSIGNEE_ID = "38cff6e0-fc52-4faf-9fa9-b72912600dee"  # Apple Inc
_DEFAULT_DATA_DIR    = "data/patents"
_DEFAULT_START       = "2020-01-01"
_DEFAULT_END         = "2025-12-31"


def run_pipeline(
    assignee_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    data_dir: str | Path = _DEFAULT_DATA_DIR,
) -> tuple[int, int]:
    """
    Run the full patents ingestion for one assignee_id and date range.

    Reads four bulk TSV files using chunked I/O, merges and normalizes
    the records, then bulk loads into `patents` and `patent_claims`.

    Args:
        assignee_id: Assignee UUID from g_assignee_disambiguated.tsv.
        start_date:  Inclusive lower bound ISO date, or None.
        end_date:    Inclusive upper bound ISO date, or None.
        data_dir:    Path to the directory containing the bulk TSV files.

    Returns:
        (patents_inserted, claims_inserted)
    """
    logger.info(
        "Patents pipeline start",
        extra={
            "assignee_id": assignee_id,
            "start_date": start_date,
            "end_date": end_date,
            "data_dir": str(data_dir),
        },
    )

    # Steps 1-2: read + merge
    patents_df, assignee_df, cpc_df, claims_df = reader.read_all(
        data_dir, assignee_id, start_date, end_date
    )

    patent_records = parser.build_patent_records(patents_df, assignee_df, cpc_df)
    claim_records  = parser.build_claim_records(claims_df)

    if not patent_records:
        logger.info("No patent records to load", extra={"assignee_id": assignee_id})
        return 0, 0

    # Steps 3-4: bulk load
    conn = get_connection()
    try:
        patents_inserted = loader.load_patents(conn, patent_records)
        claims_inserted  = loader.load_patent_claims(conn, claim_records)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    logger.info(
        "Patents pipeline complete",
        extra={
            "assignee_id": assignee_id,
            "patents_inserted": patents_inserted,
            "claims_inserted": claims_inserted,
        },
    )
    return patents_inserted, claims_inserted


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Load USPTO patent data from bulk TSV files into PostgreSQL."
    )
    p.add_argument(
        "--assignee-id",
        default=_DEFAULT_ASSIGNEE_ID,
        help="Assignee UUID from g_assignee_disambiguated.tsv (default: Apple Inc)",
    )
    p.add_argument(
        "--start",
        default=_DEFAULT_START,
        help="Start grant date YYYY-MM-DD (default: 2020-01-01)",
    )
    p.add_argument(
        "--end",
        default=_DEFAULT_END,
        help="End grant date YYYY-MM-DD (default: 2025-12-31)",
    )
    p.add_argument(
        "--data-dir",
        default=_DEFAULT_DATA_DIR,
        help="Directory containing bulk TSV files (default: data/patents)",
    )
    return p


def main() -> None:
    """CLI entry point."""
    args = _build_arg_parser().parse_args()
    patents_inserted, claims_inserted = run_pipeline(
        assignee_id=args.assignee_id,
        start_date=args.start,
        end_date=args.end,
        data_dir=args.data_dir,
    )
    logger.info(
        "Run complete",
        extra={"patents_inserted": patents_inserted, "claims_inserted": claims_inserted},
    )


if __name__ == "__main__":
    main()
