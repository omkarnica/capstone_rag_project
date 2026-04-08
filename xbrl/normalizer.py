"""
Normalize raw XBRL tags in a facts DataFrame to canonical tags.
"""

from __future__ import annotations

import pandas as pd

from .tag_map import TAG_MAP


def normalize_facts_tags(facts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw tags in `facts_df['tag']` to canonical tags using TAG_MAP.

    - Replaces the `tag` column with canonical values.
    - Preserves all other columns.
    - Returns a new DataFrame (does not mutate input in place).
    """
    if "tag" not in facts_df.columns:
        raise ValueError("facts_df must contain a 'tag' column.")

    normalized_df = facts_df.copy()
    normalized_df["tag"] = normalized_df["tag"].replace(TAG_MAP)
    return normalized_df

