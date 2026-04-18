"""
Orchestrator for the transcripts ingestion pipeline.

Ingestion flow for each Apple 8-K Item 2.02 filing:
1) fetch_8k_filings  — EDGAR submissions API → list of filing metadata
2) fetch_exhibit_url — filing index HTML → Exhibit 99.1 document URL
3) fetch_exhibit_text — Exhibit 99.1 HTML → raw HTML string
4) extract_exhibit_text — raw HTML → clean plain text
5) load_transcripts  — COPY into `transcripts`, returns accession→id map
6) load_transcript_sections — COPY into `transcript_sections` using FK ids

Individual filing failures (no exhibit found, empty text, DownloadError)
are logged and skipped — a single bad filing does not abort the run.

CLI:
    python -m src.transcripts.main
    python -m src.transcripts.main --cik 320193 --start 2020-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse

from src.transcripts import fetcher, loader, parser
from src.utils.exceptions import MAOracleError
from src.utils.logger import get_logger
from xbrl.loader import get_connection

logger = get_logger(__name__)

_DEFAULT_CIK = 320193
_DEFAULT_START = "2020-01-01"
_DEFAULT_END = "2025-12-31"


def run_pipeline(cik: int, start_date: str, end_date: str) -> tuple[int, int]:
    """
    Run the full transcripts ingestion for one CIK and date range.

    Fetches 8-K Item 2.02 filings from EDGAR, extracts Exhibit 99.1 text
    for each, and bulk loads results into `transcripts` and
    `transcript_sections`. Skips any filing that fails or has no exhibit.

    Args:
        cik:        SEC CIK, e.g. 320193 for Apple.
        start_date: Inclusive lower bound ISO date, e.g. "2020-01-01".
        end_date:   Inclusive upper bound ISO date, e.g. "2025-12-31".

    Returns:
        (transcripts_loaded, sections_loaded)
    """
    logger.info(
        "Transcripts pipeline start",
        extra={"cik": cik, "start_date": start_date, "end_date": end_date},
    )

    filings = fetcher.fetch_8k_filings(cik, start_date, end_date)

    transcript_records: list[dict] = []
    raw_sections: list[tuple[str, str]] = []  # (accession_no, section_text)

    for filing in filings:
        accession_no = filing["accession_no"]

        try:
            # Step 2: get exhibit URL from filing index
            exhibit_url = fetcher.fetch_exhibit_url(cik, accession_no)
            if exhibit_url is None:
                logger.warning(
                    "No Exhibit 99.1 found — skipping filing",
                    extra={"accession_no": accession_no},
                )
                continue

            # Step 3: fetch exhibit HTML
            html = fetcher.fetch_exhibit_text(exhibit_url)

            # Step 4: extract clean text
            text = parser.extract_exhibit_text(html, accession_no)
            if text is None:
                logger.warning(
                    "Exhibit text empty after cleaning — skipping filing",
                    extra={"accession_no": accession_no},
                )
                continue

        except MAOracleError as exc:
            logger.warning(
                "Error fetching filing — skipping",
                extra={"accession_no": accession_no, "error": str(exc)},
            )
            continue

        transcript_records.append({
            "cik": filing["cik"],
            "accession_no": filing["accession_no"],
            "company_name": filing["company_name"],
            "filed_date": filing["filed_date"],
            "period_of_report": filing["period_of_report"],
            "form_type": filing["form_type"],
            "exhibit_url": exhibit_url,
        })
        raw_sections.append((accession_no, text))

    if not transcript_records:
        logger.info("No transcript records to load", extra={"cik": cik})
        return 0, 0

    conn = get_connection()
    try:
        id_map = loader.load_transcripts(conn, transcript_records)

        print("id_map keys sample:", list(id_map.keys())[:2])
        print("raw_sections keys sample:", [acc for acc, _ in raw_sections[:2]])

        # Normalize both sides to guard against whitespace differences
        id_map = {k.strip(): v for k, v in id_map.items()}

        section_records = [
            {
                "transcript_id": id_map[acc_no.strip()],
                "section_item": "EX-99.1",
                "section_text": text,
            }
            for acc_no, text in raw_sections
            if acc_no.strip() in id_map
        ]

        sections_loaded = loader.load_transcript_sections(conn, section_records)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    logger.info(
        "Transcripts pipeline complete",
        extra={
            "cik": cik,
            "transcripts_loaded": len(id_map),
            "sections_loaded": sections_loaded,
        },
    )
    return len(id_map), sections_loaded


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Load SEC 8-K Exhibit 99.1 earnings transcripts into PostgreSQL."
    )
    p.add_argument(
        "--cik", type=int, default=_DEFAULT_CIK,
        help="SEC CIK (default: 320193 — Apple Inc.)",
    )
    p.add_argument(
        "--start", default=_DEFAULT_START,
        help="Start date YYYY-MM-DD (default: 2020-01-01)",
    )
    p.add_argument(
        "--end", default=_DEFAULT_END,
        help="End date YYYY-MM-DD (default: 2025-12-31)",
    )
    return p


def main() -> None:
    """CLI entry point."""
    args = _build_arg_parser().parse_args()
    transcripts_loaded, sections_loaded = run_pipeline(args.cik, args.start, args.end)
    logger.info(
        "Run complete",
        extra={"transcripts_loaded": transcripts_loaded, "sections_loaded": sections_loaded},
    )


if __name__ == "__main__":
    main()
