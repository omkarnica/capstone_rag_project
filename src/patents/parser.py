"""
Merges and normalizes bulk TSV DataFrames into schema-ready records.

Actual TSV column names (verified from file headers):
    g_patent.tsv:                   patent_id, patent_date, patent_title
    g_assignee_disambiguated.tsv:   patent_id, disambig_assignee_organization
    g_cpc_current.tsv:              patent_id, cpc_group
    g_claims_2025.tsv:              patent_id, claim_number, claim_text, dependent

build_patent_records:
    Merges patents_df + assignee_df + cpc_df into flat dicts matching the
    `patents` table schema. CPC codes are grouped into a sorted, deduplicated
    Python list per patent. citation_count defaults to 0 (not in bulk files).

build_claim_records:
    Converts claims_df into `patent_claims` schema dicts, keeping only
    independent claims (where the `dependent` column is null or empty).
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_patent_records(
    patents_df: pd.DataFrame,
    assignee_df: pd.DataFrame,
    cpc_df: pd.DataFrame,
) -> list[dict]:
    """
    Merge three DataFrames into `patents`-table-ready dicts.

    Merge order:
    1. LEFT JOIN patents_df ← assignee_df on patent_id
       (keeps all patents even if the assignee row is missing)
    2. For each patent, collect cpc_group values from cpc_df into a
       sorted, deduplicated list.

    citation_count is set to 0 — the bulk TSV files do not include
    forward citation counts.

    Args:
        patents_df:  From reader.read_patents — patent_id, patent_date,
                     patent_title.
        assignee_df: From reader.read_assignee_patents — patent_id,
                     disambig_assignee_organization.
        cpc_df:      From reader.read_cpc — patent_id, cpc_group.

    Returns:
        List of dicts with keys: patent_id, patent_title, grant_date,
        assignee_organization, cpc_codes (list[str]), citation_count (int).
    """
    if patents_df.empty:
        logger.info("No patent rows to parse", extra={})
        return []

    # Deduplicate assignee rows — keep one org name per patent
    assignee_dedup = (
        assignee_df[["patent_id", "disambig_assignee_organization"]]
        .drop_duplicates(subset="patent_id")
    )

    merged = patents_df.merge(assignee_dedup, on="patent_id", how="left")

    # Group CPC codes into a list per patent
    if not cpc_df.empty:
        cpc_grouped = (
            cpc_df.dropna(subset=["cpc_group"])
            .groupby("patent_id")["cpc_group"]
            .apply(lambda codes: sorted(set(codes)))
            .reset_index()
            .rename(columns={"cpc_group": "cpc_codes"})
        )
        merged = merged.merge(cpc_grouped, on="patent_id", how="left")
    else:
        merged["cpc_codes"] = None

    records: list[dict] = []
    for _, row in merged.iterrows():
        cpc_codes = row.get("cpc_codes")
        records.append({
            "patent_id": str(row["patent_id"]),
            "patent_title": row.get("patent_title") or None,
            "grant_date": (
                row["patent_date"].strftime("%Y-%m-%d")
                if pd.notna(row.get("patent_date"))
                else None
            ),
            "assignee_organization": row.get("disambig_assignee_organization") or None,
            "cpc_codes": list(cpc_codes) if isinstance(cpc_codes, (list, pd.Series)) else [],
            "citation_count": 0,
        })

    logger.info(
        "Built patent records",
        extra={"count": len(records)},
    )
    return records


def build_claim_records(claims_df: pd.DataFrame) -> list[dict]:
    """
    Convert claims_df into `patent_claims`-table-ready dicts.

    A claim is independent when the `dependent` column is NaN, None,
    empty string, or 0. The bulk TSV uses NaN for independent claims
    and an integer claim_sequence for dependent ones.

    Only independent claims are returned. Rows with empty claim_text
    after stripping are also skipped.

    Args:
        claims_df: From reader.read_claims — patent_id, claim_number,
                   claim_text, dependent.

    Returns:
        List of dicts with keys: patent_id, claim_number, claim_text,
        is_independent (always True — dependents are excluded).
    """
    if claims_df.empty:
        logger.info("No claim rows to parse", extra={})
        return []

    # Independent = dependent is NaN (float) or falsy
    independent_mask = claims_df["dependent"].isna() | (claims_df["dependent"] == 0)
    independent_df = claims_df[independent_mask].copy()

    # Drop rows with empty text
    independent_df["claim_text"] = independent_df["claim_text"].astype(str).str.strip()
    independent_df = independent_df[independent_df["claim_text"] != ""]

    records: list[dict] = [
        {
            "patent_id": str(row["patent_id"]),
            "claim_number": int(row["claim_number"]) if pd.notna(row["claim_number"]) else 0,
            "claim_text": row["claim_text"],
            "is_independent": True,
        }
        for _, row in independent_df.iterrows()
    ]

    logger.info(
        "Built claim records",
        extra={
            "total_claims": len(claims_df),
            "independent_claims": len(records),
        },
    )
    return records
