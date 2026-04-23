from __future__ import annotations

from typing import Literal

RouteName = Literal["llm_direct", "vectorstore", "web_search"]
TierName = Literal["tier_0", "tier_1", "tier_2"]

TIER_LABELS: dict[TierName, str] = {
    "tier_0": "Tier 0 - llm_direct",
    "tier_1": "Tier 1 - single-route RAG",
    "tier_2": "Tier 2 - multi-hop RAG",
}


def tier_label(tier: TierName) -> str:
    return TIER_LABELS[tier]


def tier_for_single_route(route: str | None) -> TierName:
    if route == "llm_direct":
        return "tier_0"
    return "tier_1"


def tier_for_plan(plan_type: str, route: str | None = None) -> TierName:
    if plan_type == "multi":
        return "tier_2"
    return tier_for_single_route(route)