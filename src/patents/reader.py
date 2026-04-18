"""
Reads and filters USPTO bulk TSV files for a given assignee.

All four bulk files (g_patent.tsv, g_assignee_disambiguated.tsv,
g_cpc_current.tsv, g_claims_2025.tsv) are 1-3 GB each. This module
reads them in chunks of 100,000 rows so only matching rows accumulate
in memory. The filtered result (a single assignee's patents) is small
enough to hold in memory for downstream merging.

Fetch flow:
    Step 1 — read_assignee_patents:
        Filter g_assignee_disambiguated.tsv to the given assignee_id.
        Collect matched patent_ids and the disambig_assignee_organization name.

    Step 2 — read_patents:
        Filter g_patent.tsv to the patent_ids from Step 1.
        Optionally apply a grant_date range filter.

    Step 3 — read_cpc:
        Filter g_cpc_current.tsv to those patent_ids.
        One row per (patent_id, cpc_group) — parser.py groups into lists.

    Step 4 — read_claims:
        Filter g_claims_2025.tsv to those patent_ids.

    read_all:
        Orchestrates all four steps and returns
        (patents_df, assignee_df, cpc_df, claims_df).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_CHUNK_SIZE = 100_000

# Columns actually needed from each file — reduces memory per chunk
_ASSIGNEE_COLS = ["patent_id", "assignee_id", "disambig_assignee_organization"]
_PATENT_COLS   = ["patent_id", "patent_date", "patent_title"]
_CPC_COLS      = ["patent_id", "cpc_group"]
_CLAIM_COLS    = ["patent_id", "claim_number", "claim_text", "dependent"]


def _tsv_path(data_dir: Path, filename: str) -> Path:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected TSV file not found: {path}")
    return path


def read_assignee_patents(
    data_dir: Path,
    assignee_id: str,
) -> pd.DataFrame:
    """
    Read g_assignee_disambiguated.tsv in chunks and return rows for assignee_id.

    Args:
        data_dir:    Directory containing the bulk TSV files.
        assignee_id: UUID identifying the target assignee,
                     e.g. "38cff6e0-fc52-4faf-9fa9-b72912600dee" for Apple Inc.

    Returns:
        DataFrame with columns: patent_id, assignee_id,
        disambig_assignee_organization. One row per matching patent.
    """
    path = _tsv_path(data_dir, "g_assignee_disambiguated.tsv")
    chunks: list[pd.DataFrame] = []
    total_rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=_ASSIGNEE_COLS,
        dtype=str,
        chunksize=_CHUNK_SIZE,
    ):
        total_rows += len(chunk)
        matched = chunk[chunk["assignee_id"] == assignee_id]
        if not matched.empty:
            chunks.append(matched)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=_ASSIGNEE_COLS)

    logger.info(
        "Read assignee file",
        extra={
            "assignee_id": assignee_id,
            "total_rows_scanned": total_rows,
            "matched_patents": len(df),
        },
    )
    return df


def read_patents(
    data_dir: Path,
    patent_ids: set[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Read g_patent.tsv in chunks and return rows for the given patent_ids.

    Optionally filters to a grant_date range (patent_date column).

    Args:
        data_dir:   Directory containing the bulk TSV files.
        patent_ids: Set of patent_id strings to keep.
        start_date: Inclusive lower bound ISO date "YYYY-MM-DD", or None.
        end_date:   Inclusive upper bound ISO date "YYYY-MM-DD", or None.

    Returns:
        DataFrame with columns: patent_id, patent_date, patent_title.
    """
    path = _tsv_path(data_dir, "g_patent.tsv")
    chunks: list[pd.DataFrame] = []
    total_rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=_PATENT_COLS,
        dtype={"patent_id": str, "patent_title": str},
        parse_dates=["patent_date"],
        chunksize=_CHUNK_SIZE,
    ):
        total_rows += len(chunk)
        matched = chunk[chunk["patent_id"].isin(patent_ids)].copy()

        if matched.empty:
            continue

        if start_date:
            matched = matched[matched["patent_date"] >= pd.Timestamp(start_date)]
        if end_date:
            matched = matched[matched["patent_date"] <= pd.Timestamp(end_date)]

        if not matched.empty:
            chunks.append(matched)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=_PATENT_COLS)

    logger.info(
        "Read patents file",
        extra={
            "total_rows_scanned": total_rows,
            "matched_patents": len(df),
            "start_date": start_date,
            "end_date": end_date,
        },
    )
    return df


def read_cpc(
    data_dir: Path,
    patent_ids: set[str],
) -> pd.DataFrame:
    """
    Read g_cpc_current.tsv in chunks and return CPC rows for the given patent_ids.

    Returns one row per (patent_id, cpc_group) combination. parser.py
    groups these into a list per patent.

    Args:
        data_dir:   Directory containing the bulk TSV files.
        patent_ids: Set of patent_id strings to keep.

    Returns:
        DataFrame with columns: patent_id, cpc_group.
    """
    path = _tsv_path(data_dir, "g_cpc_current.tsv")
    chunks: list[pd.DataFrame] = []
    total_rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=_CPC_COLS,
        dtype=str,
        chunksize=_CHUNK_SIZE,
    ):
        total_rows += len(chunk)
        matched = chunk[chunk["patent_id"].isin(patent_ids)]
        if not matched.empty:
            chunks.append(matched)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=_CPC_COLS)

    logger.info(
        "Read CPC file",
        extra={
            "total_rows_scanned": total_rows,
            "matched_cpc_rows": len(df),
        },
    )
    return df


def read_claims(
    data_dir: Path,
    patent_ids: set[str],
) -> pd.DataFrame:
    """
    Read g_claims_2025.tsv in chunks and return claim rows for the given patent_ids.

    Returns all claims (independent and dependent). parser.py filters to
    independent claims only (dependent column is null/empty).

    Args:
        data_dir:   Directory containing the bulk TSV files.
        patent_ids: Set of patent_id strings to keep.

    Returns:
        DataFrame with columns: patent_id, claim_number, claim_text, dependent.
    """
    path = _tsv_path(data_dir, "g_claims_2025.tsv")
    chunks: list[pd.DataFrame] = []
    total_rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=_CLAIM_COLS,
        dtype={"patent_id": str, "claim_text": str},
        chunksize=_CHUNK_SIZE,
    ):
        total_rows += len(chunk)
        matched = chunk[chunk["patent_id"].isin(patent_ids)]
        if not matched.empty:
            chunks.append(matched)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=_CLAIM_COLS)

    logger.info(
        "Read claims file",
        extra={
            "total_rows_scanned": total_rows,
            "matched_claim_rows": len(df),
        },
    )
    return df


def read_all(
    data_dir: str | Path,
    assignee_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate all four TSV reads and return filtered DataFrames.

    Reads in dependency order: assignee first (to get patent_ids), then
    patents (with optional date filter), then CPC and claims (using the
    date-filtered patent_ids so downstream tables stay consistent).

    Args:
        data_dir:    Path to the directory containing bulk TSV files.
        assignee_id: Assignee UUID, e.g. "38cff6e0-fc52-4faf-9fa9-b72912600dee".
        start_date:  Inclusive lower bound ISO date, or None.
        end_date:    Inclusive upper bound ISO date, or None.

    Returns:
        (patents_df, assignee_df, cpc_df, claims_df)
        All DataFrames share the same patent_id universe (date-filtered).
    """
    data_dir = Path(data_dir)

    logger.info(
        "Starting bulk TSV reads",
        extra={"assignee_id": assignee_id, "start_date": start_date, "end_date": end_date},
    )

    # Step 1: find all patent_ids for this assignee
    assignee_df = read_assignee_patents(data_dir, assignee_id)
    all_patent_ids: set[str] = set(assignee_df["patent_id"].dropna().unique())

    # Step 2: filter patents by those ids + optional date range
    patents_df = read_patents(data_dir, all_patent_ids, start_date, end_date)

    # Use the date-filtered patent_ids for CPC and claims
    filtered_patent_ids: set[str] = set(patents_df["patent_id"].dropna().unique())

    # Step 3: CPC codes for date-filtered patents only
    cpc_df = read_cpc(data_dir, filtered_patent_ids)

    # Step 4: claims for date-filtered patents only
    claims_df = read_claims(data_dir, filtered_patent_ids)

    # Filter assignee_df to match date-filtered patents
    assignee_df = assignee_df[assignee_df["patent_id"].isin(filtered_patent_ids)]

    logger.info(
        "Completed bulk TSV reads",
        extra={
            "patents": len(patents_df),
            "cpc_rows": len(cpc_df),
            "claim_rows": len(claims_df),
        },
    )
    return patents_df, assignee_df, cpc_df, claims_df
