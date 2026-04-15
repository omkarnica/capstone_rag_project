"""
Main entrypoint for the XBRL loader pipeline.

Flow:
1) Download and extract SEC quarterly zip (sub.txt + num.txt)
2) Parse SEC files
3) Filter to target companies only
4) Normalize facts tags to canonical tags
5) Bulk load filings and facts into PostgreSQL
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import requests

from . import loader, normalizer, parser
from src.utils.logger import get_logger

logger = get_logger(__name__)

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

def download_and_extract(quarter: str, extract_dir: Path) -> tuple[Path, Path]:
    """
    Download a SEC quarterly zip file and extract sub.txt + num.txt.

    Args:
        quarter:     e.g. "2024q1"
        extract_dir: local folder to extract into (created if missing)

    Returns:
        (sub_path, num_path)
    """
    year = quarter[:4]       # "2024"
    q    = quarter[4:]       # "q1"

    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/{q.upper()}/company.zip"

    # The XBRL bulk data lives at a different URL pattern
    xbrl_url = f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"

    logger.info("Downloading quarter archive", extra={"quarter": quarter, "url": xbrl_url})
    response = requests.get(xbrl_url, headers={"User-Agent": "karnicajain.ds@gmail.com"}, timeout=120)
    response.raise_for_status()

    extract_dir = extract_dir / quarter
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(extract_dir)

    sub_path = extract_dir / "sub.txt"
    num_path = extract_dir / "num.txt"

    if not sub_path.exists() or not num_path.exists():
        raise FileNotFoundError(
            f"Expected sub.txt and num.txt inside {quarter}.zip but did not find them. "
            f"Files found: {list(extract_dir.iterdir())}"
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

def run_all_quarters(extract_dir: str | Path = Path("data/raw")) -> None:
    """
    Download and load all quarters defined in QUARTERS.

    Args:
        extract_dir: local folder where zip files are extracted
    """
    extract_dir = Path(extract_dir)
    total_filings = 0
    total_facts   = 0

    for quarter in QUARTERS:
        logger.info("Processing quarter", extra={"quarter": quarter})
        try:
            sub_path, num_path = download_and_extract(quarter, extract_dir)
            filings_loaded, facts_loaded = run_pipeline(sub_path, num_path)
            total_filings += filings_loaded
            total_facts   += facts_loaded
            logger.info(
                "Quarter processed",
                extra={
                    "quarter": quarter,
                    "filings_loaded": filings_loaded,
                    "facts_loaded": facts_loaded,
                },
            )
        except Exception as e:
            # Log the error but keep going — don't let one bad quarter stop the rest
            logger.warning(
                "Skipped quarter due to error",
                extra={"quarter": quarter, "error": str(e)},
            )

    logger.info(
        "All quarters completed",
        extra={"total_filings_loaded": total_filings, "total_facts_loaded": total_facts},
    )


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