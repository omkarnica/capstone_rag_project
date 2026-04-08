"""
Main entrypoint for the XBRL loader pipeline.

Flow:
1) Parse SEC files (`sub.txt`, `num.txt`)
2) Normalize facts tags to canonical tags
3) Connect to PostgreSQL
4) Bulk load filings and facts
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import loader, normalizer, parser


def run_pipeline(sub_path: str | Path, num_path: str | Path) -> tuple[int, int]:
    """
    Execute the end-to-end parse -> normalize -> load workflow.

    Returns:
        (filings_loaded, facts_loaded)
    """
    filings_df, facts_df = parser.parse_xbrl(sub_path=sub_path, num_path=num_path)
    normalized_facts_df = normalizer.normalize_facts_tags(facts_df)

    conn = loader.get_connection()
    try:
        filings_loaded, facts_loaded = loader.load_xbrl_data(
            filings_df=filings_df,
            facts_df=normalized_facts_df,
            conn=conn,
        )
    finally:
        conn.close()

    return filings_loaded, facts_loaded


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI arguments for running the loader from terminal.
    """
    arg_parser = argparse.ArgumentParser(description="Load SEC XBRL data into PostgreSQL.")
    arg_parser.add_argument("--sub-path", required=True, help="Path to sub.txt")
    arg_parser.add_argument("--num-path", required=True, help="Path to num.txt")
    return arg_parser


def main() -> None:
    """
    CLI entrypoint.
    """
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()

    filings_loaded, facts_loaded = run_pipeline(args.sub_path, args.num_path)
    print(f"Loaded filings: {filings_loaded}")
    print(f"Loaded facts: {facts_loaded}")


if __name__ == "__main__":
    main()

