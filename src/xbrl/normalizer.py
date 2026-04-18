"""
Normalize raw XBRL tags in a facts DataFrame to canonical names.

Replaces raw XBRL tag strings in facts_df['tag'] with canonical names
defined in TAG_MAP (xbrl/tag_map.py). Tags not present in TAG_MAP are
passed through unchanged.

Expected facts_df columns:
    tag   (string)   — raw XBRL tag, e.g. "us-gaap/Revenues"
    value (float64)  — numeric fact value
    adsh  (string)   — accession number; join key to the filings table
    ddate (date)     — data date for the reported value
"""

from __future__ import annotations

import pandas as pd

from .tag_map import TAG_MAP
from src.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_facts_tags(facts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw tags in `facts_df['tag']` to canonical tags using TAG_MAP.

    - Replaces the `tag` column with canonical values.
    - Preserves all other columns.
    - Returns a new DataFrame (does not mutate input in place).
    """
    if "tag" not in facts_df.columns:
        raise ValueError("facts_df must contain a 'tag' column.")

    normalized_count = int(facts_df["tag"].isin(TAG_MAP.keys()).sum())
    unknown_count = int((~facts_df["tag"].isin(TAG_MAP.keys())).sum())
    normalized_df = facts_df.copy()
    normalized_df["tag"] = normalized_df["tag"].replace(TAG_MAP)
    logger.info(
        "Normalized facts tags",
        extra={
            "normalized_tags": normalized_count,
            "unknown_tags": unknown_count,
            "total_rows": len(facts_df),
        },
    )
    return normalized_df

