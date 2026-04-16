"""
XBRL parser for SEC bulk files.

Reads `sub.txt` (filings metadata) and `num.txt` (numeric facts) into
Pandas DataFrames with sensible dtypes and preserves the `adsh` join key.

This module does not touch the database; it is purely for file → DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from src.utils.exceptions import parse_error_boundary
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _read_tsv(path: Path, dtype: dict | None = None, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """
    Internal helper to read the SEC tab-separated text files.

    The SEC bulk files are actually TSVs despite the .txt extension.
    """
    with parse_error_boundary(str(path)):
        return pd.read_csv(
            path,
            sep="\t",
            dtype=dtype,
            parse_dates=parse_dates or [],
            low_memory=False,
        )


def parse_filings(sub_path: str | Path) -> pd.DataFrame:
    """
    Parse `sub.txt` into a DataFrame representing the `filings` table.

    Expected key columns from SEC `sub.txt`:
    - adsh: Accession number / filing ID (join key)
    - cik: Central Index Key (integer, but keep as string-safe)
    - name: Company name
    - sic: Standard Industrial Classification
    - form: Filing form type (e.g., 10-K, 10-Q)
    - period: Period end date
    - fy: Fiscal year
    - fp: Fiscal period (e.g., FY, Q1)
    - filed: Filing date
    """
    sub_path = Path(sub_path)
    logger.info("Starting filings parsing", extra={"file_path": str(sub_path)})

    dtype = {
        "adsh": "string",
        "cik": "Int64",
        "name": "string",
        "sic": "Int64",
        "form": "string",
        "fy": "Int64",
        "fp": "string",
    }

    df = _read_tsv(
        sub_path,
        dtype=dtype,
        parse_dates=["period", "filed"]
    )

    # Align column names with the `filings` table schema
    column_mapping = {
        "fy": "fiscal_year",
        "fp": "fiscal_period",
    }
    df = df.rename(columns=column_mapping)

    # Keep only the columns needed for the target schema (others can be added later if needed)
    wanted_columns = [
        "adsh",
        "cik",
        "name",
        "sic",
        "form",
        "period",
        "fiscal_year",
        "fiscal_period",
        "filed",
    ]
    existing_columns = [col for col in wanted_columns if col in df.columns]
    parsed_df = df[existing_columns].copy()
    logger.info(
        "Completed filings parsing",
        extra={"file_path": str(sub_path), "row_count": len(parsed_df)},
    )
    return parsed_df


def parse_facts(num_path: str | Path) -> pd.DataFrame:
    """
    Parse `num.txt` into a DataFrame representing the `facts` table.

    Expected key columns from SEC `num.txt`:
    - adsh: Filing ID (join key)
    - tag: XBRL tag
    - version: Taxonomy version
    - ddate: Data date
    - qtrs: Number of quarters represented
    - uom: Unit of measure
    - value: Numeric value
    """
    num_path = Path(num_path)
    logger.info("Starting facts parsing", extra={"file_path": str(num_path)})

    dtype = {
        "adsh": "string",
        "tag": "string",
        "version": "string",
        "qtrs": "Int64",
        "uom": "string",
        "value": "float64"
    }

    df = _read_tsv(
        num_path,
        dtype=dtype,
        parse_dates=["ddate"],
    )

    # Keep only the columns needed for the target schema
    wanted_columns = [
        "adsh",
        "tag",
        "version",
        "ddate",
        "qtrs",
        "uom",
        "value",
    ]
    existing_columns = [col for col in wanted_columns if col in df.columns]
    parsed_df = df[existing_columns].copy()
    logger.info(
        "Completed facts parsing",
        extra={"file_path": str(num_path), "row_count": len(parsed_df)},
    )
    return parsed_df


def parse_xbrl(sub_path: str | Path, num_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to parse both filings (`sub.txt`) and facts (`num.txt`).

    Returns:
        (filings_df, facts_df)
    """
    filings_df = parse_filings(sub_path)
    facts_df = parse_facts(num_path)
    return filings_df, facts_df

