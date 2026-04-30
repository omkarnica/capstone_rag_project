from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from src.model_config import get_planner_llm
from src.nodes.graph_topics import is_graph_topic
from src.tiering import tier_for_plan

RouteHint = Literal[
    "sql", "filings", "transcripts", "patents",
    "litigation", "graph", "contradiction", "llm_direct",
]
TierHint = Literal["tier_0", "tier_1", "tier_2"]

RECENCY_MARKERS = {
    "current", "currently", "latest", "recent", "recently",
    "today", "now", "q1", "q2", "q3", "q4", "fiscal year",
}

FINANCIAL_MARKERS = {
    "revenue", "net income", "gross margin", "operating income", "ebitda",
    "cash flow", "eps", "earnings per share", "balance sheet", "debt",
    "assets", "liabilities", "equity", "ratio", "margin", "profit",
}

FILING_MARKERS = {
    "10-k", "10-q", "risk factor", "item 1a", "md&a", "management discussion",
    "material weakness", "footnote", "disclosure", "sec filing",
}

TRANSCRIPT_MARKERS = {
    "earnings call", "management said", "ceo said", "cfo said",
    "guidance", "investor", "transcript", "press release", "8-k",
}

PATENT_MARKERS = {
    "patent", "ip portfolio", "cpc code", "invention", "patent claim",
    "intellectual property", "patent filing", "grant date",
}

LITIGATION_MARKERS = {
    "lawsuit", "litigation", "court", "settlement", "legal exposure",
    "case", "plaintiff", "defendant", "judgment",
}

SOURCE_TOPIC_SPECS: dict[str, dict] = {
    "financial_metrics": {
        "phrases": ("revenue", "net income", "gross margin", "operating income", "cash", "ebitda"),
        "summary_question": "What are the key financial metrics?",
        "route_hint": "sql",
    },
    "risk_factors": {
        "phrases": ("risk factor", "10-k", "10-q", "item 1a", "md&a", "material weakness"),
        "summary_question": "What are the key risk disclosures in the SEC filings?",
        "route_hint": "filings",
    },
    "management_statements": {
        "phrases": ("earnings call", "management said", "ceo", "guidance", "transcript"),
        "summary_question": "What did management say on the earnings call?",
        "route_hint": "transcripts",
    },
    "patents": {
        "phrases": ("patent", "ip portfolio", "cpc", "invention", "claim"),
        "summary_question": "What is the patent portfolio?",
        "route_hint": "patents",
    },
    "litigation": {
        "phrases": ("lawsuit", "litigation", "court", "settlement", "legal"),
        "summary_question": "What is the litigation exposure?",
        "route_hint": "litigation",
    },
}

COMPARISON_PATTERNS = [
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bdiscrepanc", re.IGNORECASE),
    re.compile(r"\bcontradict", re.IGNORECASE),
    re.compile(r"\bdifference between\b", re.IGNORECASE),
    re.compile(r"\bwhat management said .+ vs .+filing", re.IGNORECASE),
    re.compile(r"\bfilings? vs\.? transcript", re.IGNORECASE),
    re.compile(r"\btranscript vs\.? filing", re.IGNORECASE),
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "its", "of",
    "on", "or", "say", "that", "the", "this", "to", "what",
    "when", "which", "why",
}


class PlannedSubquestion(BaseModel):
    question: str = Field(description="A standalone sub-question.")
    route_hint: RouteHint | None = Field(
        default=None,
        description="Optional preferred route for the downstream graph.",
    )


class QueryPlan(BaseModel):
    plan_type: Literal["single", "multi"] = Field(
        description="Whether the question should stay whole or be decomposed."
    )
    reason: str = Field(description="Short reason for the planning decision.")
    tier_hint: TierHint = Field(description="Expected execution tier for the plan.")
    subquestions: list[PlannedSubquestion] = Field(
        description="One or more sub-questions to run."
    )


def _clean_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip("?!.")


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS
    }


def _overlap_ratio(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _extract_source_topics(question: str) -> list[str]:
    lowered = question.lower()
    return [
        topic
        for topic, spec in SOURCE_TOPIC_SPECS.items()
        if any(phrase in lowered for phrase in spec["phrases"])
    ]


def _guess_route_hint(question: str) -> RouteHint:
    lowered = question.lower()

    # Contradiction: explicit comparison or discrepancy across sources
    if any(p.search(question) for p in COMPARISON_PATTERNS):
        if any(m in lowered for m in TRANSCRIPT_MARKERS) or any(m in lowered for m in FILING_MARKERS):
            return "contradiction"

    if is_graph_topic(question):
        return "graph"
    if any(m in lowered for m in FINANCIAL_MARKERS):
        return "sql"
    if any(m in lowered for m in FILING_MARKERS):
        return "filings"
    if any(m in lowered for m in TRANSCRIPT_MARKERS):
        return "transcripts"
    if any(m in lowered for m in PATENT_MARKERS):
        return "patents"
    if any(m in lowered for m in LITIGATION_MARKERS):
        return "litigation"
    return "llm_direct"


def _canonical_tier_hint(plan_type: str, subquestions: list[dict]) -> TierHint:
    route = subquestions[0].get("route_hint") if subquestions else None
    return tier_for_plan(plan_type, route)


def _single_plan(question: str, reason: str) -> dict:
    route_hint = _guess_route_hint(question)
    subquestions = [{"question": question, "route_hint": route_hint}]
    return {
        "plan_type": "single",
        "reason": reason,
        "tier_hint": _canonical_tier_hint("single", subquestions),
        "subquestions": subquestions,
    }


def _multi_plan(subquestions: list[dict], reason: str) -> dict:
    return {
        "plan_type": "multi",
        "reason": reason,
        "tier_hint": "tier_2",
        "subquestions": subquestions,
    }


def _normalize_subquestions(subquestions: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    seen: set[str] = set()
    for sq in subquestions:
        cleaned = _clean_question(sq["question"])
        if not cleaned:
            continue
        canonical = cleaned.lower()
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append({
            "question": cleaned,
            "route_hint": sq.get("route_hint") or _guess_route_hint(cleaned),
        })
    return normalized


def _extract_explicit_question_parts(question: str) -> list[str]:
    return [_clean_question(p) for p in re.split(r"\?+", question) if _clean_question(p)]


def _has_mixed_retrieval_intents(parts: list[str]) -> bool:
    return len({_guess_route_hint(p) for p in parts}) > 1


def _requires_multi_source_synthesis(question: str) -> bool:
    source_topics = _extract_source_topics(question)
    if any(p.search(question) for p in COMPARISON_PATTERNS):
        return len(source_topics) >= 2 or (
            any(m in question.lower() for m in TRANSCRIPT_MARKERS)
            and any(m in question.lower() for m in FILING_MARKERS)
        )
    return False


def _build_synthesis_subquestions(question: str) -> list[dict]:
    source_topics = _extract_source_topics(question)
    subquestions = []
    for topic in source_topics:
        spec = SOURCE_TOPIC_SPECS[topic]
        subquestions.append({
            "question": str(spec["summary_question"]),
            "route_hint": str(spec["route_hint"]),
        })
    # Always add contradiction if comparing transcripts vs filings
    lowered = question.lower()
    if (
        any(m in lowered for m in TRANSCRIPT_MARKERS)
        and any(m in lowered for m in FILING_MARKERS | FINANCIAL_MARKERS)
        and any(p.search(question) for p in COMPARISON_PATTERNS)
    ):
        subquestions.append({
            "question": f"Are there contradictions between management statements and SEC filings for this company?",
            "route_hint": "contradiction",
        })
    return _normalize_subquestions(subquestions)


def _extract_candidate_subquestions(question: str) -> list[str]:
    explicit_parts = _extract_explicit_question_parts(question)
    if len(explicit_parts) > 1:
        return explicit_parts
    return [_clean_question(question)]


def _subquestions_overlap_too_much(subquestions: list[dict]) -> bool:
    if len(subquestions) < 2:
        return False
    for i, left in enumerate(subquestions):
        for right in subquestions[i + 1:]:
            if _overlap_ratio(left["question"], right["question"]) >= 0.6:
                return True
    return False


def should_decompose(question: str) -> bool:
    clean_question = _clean_question(question)
    explicit_parts = _extract_candidate_subquestions(clean_question)
    if len(explicit_parts) > 1:
        if _has_mixed_retrieval_intents(explicit_parts):
            return True
        if any(_overlap_ratio(explicit_parts[0], p) < 0.55 for p in explicit_parts[1:]):
            return True
    return _requires_multi_source_synthesis(clean_question)


def _fallback_plan_query(question: str) -> dict:
    if not should_decompose(question):
        return _single_plan(question, "The question can be answered in one pass.")
    explicit_parts = _extract_candidate_subquestions(question)
    if len(explicit_parts) > 1:
        subquestions = _normalize_subquestions(
            [{"question": p, "route_hint": _guess_route_hint(p)} for p in explicit_parts]
        )
    else:
        subquestions = _build_synthesis_subquestions(question)
    if len(subquestions) < 2:
        return _single_plan(question, "The question can be answered in one pass.")
    return _multi_plan(subquestions, "The question requires combining multiple M&A evidence pools.")


def plan_query(question: str) -> dict:
    """Plan whether a user M&A query should stay whole or be decomposed into sub-queries."""
    clean_question = _clean_question(question)
    heuristic_multi = should_decompose(clean_question)

    if not heuristic_multi:
        return _single_plan(clean_question, "The question is a single intent and should stay as one pass.")

    try:
        planner = get_planner_llm().with_structured_output(QueryPlan)
        system_prompt = """
You are the planning step for an M&A due diligence RAG pipeline.

Be conservative — keep the query as "single" for:
- questions about one financial metric
- questions about one company's risk factors
- questions about what management said
- questions about a specific patent or lawsuit

Use "multi" when the question clearly requires:
- comparing management statements vs. SEC filings (transcripts + sql/filings)
- comprehensive due diligence across multiple data sources
- sequential sub-questions across different evidence pools
- contradiction detection alongside financial or filing analysis

Route hints:
- sql: financial metrics, XBRL numbers
- filings: 10-K/10-Q narratives
- transcripts: earnings call statements
- patents: IP portfolio
- litigation: lawsuits, legal exposure
- contradiction: compare filings vs transcripts
- llm_direct: general concepts

Constraints:
- Do not split on "and" or "also" by default.
- If two sub-questions overlap heavily, collapse back to single.
- tier_hint must be one of: tier_0, tier_1, tier_2.
""".strip()

        result = planner.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": clean_question},
        ])

        subquestions = _normalize_subquestions([
            {"question": sub.question, "route_hint": sub.route_hint}
            for sub in result.subquestions
        ])

        if len(subquestions) < 2:
            return _fallback_plan_query(clean_question)

        if _subquestions_overlap_too_much(subquestions):
            return _fallback_plan_query(clean_question)

        if not heuristic_multi:
            return _single_plan(clean_question, "The question is a single intent and should stay as one pass.")

        return _multi_plan(subquestions, result.reason)

    except Exception:
        return _fallback_plan_query(clean_question)
