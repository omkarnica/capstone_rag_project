"""
Main entrypoint for the XBRL ingestion pipeline.

Ingestion flow:
1) Download — fetch the SEC XBRL quarterly zip from EDGAR
   (https://www.sec.gov/files/dera/data/financial-statement-data-sets/<quarter>.zip)
2) Extract — unzip to obtain sub.txt (filing metadata) and num.txt (numeric facts)
3) Parse — read sub.txt and num.txt into Pandas DataFrames; filter to target CIKs
4) Normalize — map raw XBRL tags to canonical names via TAG_MAP
5) Bulk load — COPY DataFrames into PostgreSQL using staging tables and
   ON CONFLICT DO NOTHING to make reruns safe
"""

from __future__ import annotations

import argparse
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from . import loader, normalizer, parser
from src.utils.exceptions import MAOracleError, ParseError, download_error_boundary
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Pipeline result ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Structured outcome for a single quarter's pipeline run."""

    quarter: str
    success: bool
    filings_loaded: int = 0
    facts_loaded: int = 0
    error: str | None = None


# ── Target companies ──────────────────────────────────────────────────────────
TARGET_CIKS = {
    789019,   # Microsoft
}

# ── Quarters to load (all missing quarters) ───────────────────────────────────
# You already have 2025q4. Load everything from 2021q1 through 2025q3.
QUARTERS = [
    "2021q1", "2021q2", "2021q3", "2021q4",
    "2022q1", "2022q2", "2022q3", "2022q4",
    "2023q1", "2023q2", "2023q3", "2023q4",
    "2024q1", "2024q2", "2024q3", "2024q4",
    "2025q1", "2025q2", "2025q3", "2025q4"
]

SEC_BASE_URL = "https://www.sec.gov/Archives/edgar/full-index"


# ── Download helper ───────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _download_zip(url: str) -> bytes:
    """
    Fetch a ZIP archive from SEC EDGAR. Retried up to 3 times with
    exponential backoff (2s → 4s → 8s). Re-raises the original
    requests exception on exhaustion so the caller's error boundary
    can translate it.
    """
    response = requests.get(
        url,
        headers={"User-Agent": "karnicajain.ds@gmail.com"},
        timeout=120,
    )
    response.raise_for_status()
    return response.content


def download_and_extract(quarter: str, extract_dir: Path) -> tuple[Path, Path]:
    """
    Download a SEC quarterly zip file and extract sub.txt + num.txt.

    Args:
        quarter:     e.g. "2024q1"
        extract_dir: local folder to extract into (created if missing)

    Returns:
        (sub_path, num_path)

    Raises:
        DownloadError: if the HTTP fetch fails after 3 attempts.
        ParseError:    if sub.txt or num.txt are missing from the archive.
    """
    xbrl_url = (
        f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
    )

    logger.info("Downloading quarter archive", extra={"quarter": quarter, "url": xbrl_url})

    # Boundary sits outside _download_zip: retries exhaust first, then the
    # surviving requests exception is translated to DownloadError exactly once.
    with download_error_boundary(xbrl_url):
        content = _download_zip(xbrl_url)

    extract_dir = extract_dir / quarter
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        zf.extractall(extract_dir)

    sub_path = extract_dir / "sub.txt"
    num_path = extract_dir / "num.txt"

    for expected_path in (sub_path, num_path):
        if not expected_path.exists():
            raise ParseError(
                f"Expected file not found after extracting {quarter}.zip",
                path=str(expected_path),
            )

    return sub_path, num_path


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(sub_path: str | Path, num_path: str | Path) -> tuple[int, int]:
    """
    Execute the end-to-end parse -> filter -> normalize -> load workflow.

    Returns:
        (filings_loaded, facts_loaded)
    """
    logger.info(
        "Pipeline start",
        extra={"sub_path": str(sub_path), "num_path": str(num_path)},
    )
    # 1. Parse
    filings_df, facts_df = parser.parse_xbrl(sub_path=sub_path, num_path=num_path)

    # 2. Filter to target companies only ← NEW
    filings_df = filings_df[filings_df["cik"].isin(TARGET_CIKS)]
    valid_adsh = set(filings_df["adsh"])
    facts_df   = facts_df[facts_df["adsh"].isin(valid_adsh)]

    logger.info(
        "Applied CIK filter",
        extra={"filtered_filings": len(filings_df), "filtered_facts": len(facts_df)},
    )

    # 3. Normalize tags
    normalized_facts_df = normalizer.normalize_facts_tags(facts_df)

    # 4. Load
    conn = loader.get_connection()
    try:
        filings_loaded, facts_loaded = loader.load_xbrl_data(
            filings_df=filings_df,
            facts_df=normalized_facts_df,
            conn=conn,
        )
    finally:
        conn.close()

    logger.info(
        "Pipeline end",
        extra={"filings_loaded": filings_loaded, "facts_loaded": facts_loaded},
    )
    return filings_loaded, facts_loaded


# ── Multi-quarter loop ────────────────────────────────────────────────────────

def run_all_quarters(
    extract_dir: str | Path = Path("data/raw"),
) -> list[PipelineResult]:
    """
    Download and load all quarters defined in QUARTERS.

    Args:
        extract_dir: local folder where zip files are extracted

    Returns:
        One PipelineResult per quarter — callers can inspect failures
        without parsing log output.
    """
    extract_dir = Path(extract_dir)
    results: list[PipelineResult] = []

    for quarter in QUARTERS:
        logger.info("Processing quarter", extra={"quarter": quarter})
        try:
            sub_path, num_path = download_and_extract(quarter, extract_dir)
            filings_loaded, facts_loaded = run_pipeline(sub_path, num_path)
            logger.info(
                "Quarter processed",
                extra={
                    "quarter": quarter,
                    "filings_loaded": filings_loaded,
                    "facts_loaded": facts_loaded,
                },
            )
            results.append(
                PipelineResult(
                    quarter=quarter,
                    success=True,
                    filings_loaded=filings_loaded,
                    facts_loaded=facts_loaded,
                )
            )
        except MAOracleError as e:
            logger.warning(
                "Skipped quarter due to error",
                extra={"quarter": quarter, "error": str(e)},
            )
            results.append(
                PipelineResult(quarter=quarter, success=False, error=str(e))
            )

    total_filings = sum(r.filings_loaded for r in results)
    total_facts   = sum(r.facts_loaded   for r in results)
    failed        = [r.quarter for r in results if not r.success]

    logger.info(
        "All quarters completed",
        extra={
            "total_filings_loaded": total_filings,
            "total_facts_loaded": total_facts,
            "failed_quarters": failed,
        },
    )
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(description="Load SEC XBRL data into PostgreSQL.")

    subparsers = arg_parser.add_subparsers(dest="command")

    # Single quarter from local files (original behaviour)
    single = subparsers.add_parser("single", help="Load one quarter from local sub.txt + num.txt")
    single.add_argument("--sub-path", required=True, help="Path to sub.txt")
    single.add_argument("--num-path", required=True, help="Path to num.txt")

    # All quarters — download + load automatically
    subparsers.add_parser("all", help="Download and load all missing quarters from SEC EDGAR")

    return arg_parser


def main() -> None:
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()

    if args.command == "single":
        filings_loaded, facts_loaded = run_pipeline(args.sub_path, args.num_path)
        logger.info(
            "Single run completed",
            extra={"filings_loaded": filings_loaded, "facts_loaded": facts_loaded},
        )

    elif args.command == "all":
        run_all_quarters()

    else:
        arg_parser.print_help()


if __name__ == "__main__":
    main()