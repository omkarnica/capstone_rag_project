from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from src.model_config import get_planner_llm
from src.tiering import tier_for_plan

RouteHint = Literal["llm_direct", "vectorstore", "web_search"]
TierHint = Literal["tier_0", "tier_1", "tier_2"]

RECENCY_MARKERS = {
    "current",
    "currently",
    "latest",
    "recent",
    "recently",
    "new",
    "newly",
    "today",
    "now",
    "launched",
    "launch",
    "announced",
    "updated",
    "listed",
}

DOC_MARKERS = {
    "anthropic",
    "claude",
    "exam",
    "guide",
    "syllabus",
    "prepare",
    "preparation",
    "docs",
    "documentation",
    "course",
    "courses",
    "mcp",
    "model context protocol",
}

CONCEPTUAL_META_TERMS = {
    "time-sensitive",
    "semantic cache",
    "single-hop",
    "multi-hop",
    "hallucination",
    "prompt engineering",
    "context management",
    "adaptive rag",
    "web search",
    "routing",
}

SOURCE_TOPIC_SPECS: dict[str, dict[str, str | tuple[str, ...]]] = {
    "exam_guide": {
        "phrases": (
            "certification exam guide",
            "claude certification exam",
            "claude certified architect",
            "exam guide",
        ),
        "summary_question": "What does the certification exam guide cover?",
        "route_hint": "vectorstore",
    },
    "claude_101": {
        "phrases": ("claude 101",),
        "summary_question": "What does Claude 101 cover?",
        "route_hint": "vectorstore",
    },
    "claude_code_in_action": {
        "phrases": ("claude code in action",),
        "summary_question": "What does Claude Code in Action cover?",
        "route_hint": "vectorstore",
    },
    "building_api": {
        "phrases": ("building with the claude api", "claude api"),
        "summary_question": "What does Building with the Claude API cover?",
        "route_hint": "vectorstore",
    },
    "mcp_course": {
        "phrases": (
            "introduction to model context protocol",
            "model context protocol",
        ),
        "summary_question": "What does Introduction to Model Context Protocol cover?",
        "route_hint": "vectorstore",
    },
}

COMPARISON_PATTERNS = [
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bhow do .+ and .+ differ\b", re.IGNORECASE),
    re.compile(r"\bdifference between\b", re.IGNORECASE),
    re.compile(r"\bwhat themes appear in both\b", re.IGNORECASE),
    re.compile(r"\bboth .+ and .+\b", re.IGNORECASE),
]

ALIGNMENT_PATTERNS = [
    re.compile(r"\bbased on .+ and .+, which\b", re.IGNORECASE),
    re.compile(r"\busing the local course summaries and the current anthropic courses listings\b", re.IGNORECASE),
    re.compile(r"\bwhich currently listed anthropic course\b", re.IGNORECASE),
    re.compile(r"\bwhich currently listed course\b", re.IGNORECASE),
    re.compile(r"\bwhich listed course\b", re.IGNORECASE),
    re.compile(r"\balign(?:s|ed|ment)?\b", re.IGNORECASE),
    re.compile(r"\bclosest in spirit\b", re.IGNORECASE),
    re.compile(r"\bbest matches that need\b", re.IGNORECASE),
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "say",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "which",
    "why",
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
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned.rstrip("?!.")


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
    if not union:
        return 0.0

    return len(left_tokens & right_tokens) / len(union)


def _is_conceptual_meta_question(question: str) -> bool:
    lowered = question.lower()
    asks_conceptual = bool(
        re.match(r"^(what is|what are|why|how)\b", lowered)
    )
    return asks_conceptual and any(term in lowered for term in CONCEPTUAL_META_TERMS)


def _extract_source_topics(question: str) -> list[str]:
    lowered = question.lower()
    found = []

    for topic, spec in SOURCE_TOPIC_SPECS.items():
        phrases = spec["phrases"]
        if any(phrase in lowered for phrase in phrases):
            found.append(topic)

    return found


def _current_listing_requested(question: str) -> bool:
    lowered = question.lower()
    return "anthropic courses" in lowered and any(
        marker in lowered for marker in ("currently listed", "current listings", "listed course")
    )


def _is_same_entity_listing_overview_question(question: str) -> bool:
    lowered = question.lower()
    return (
        lowered.startswith("is ")
        and "currently listed on anthropic courses" in lowered
        and "course page say it covers" in lowered
    )


def _guess_route_hint(question: str) -> RouteHint:
    lowered = question.lower()

    if _is_conceptual_meta_question(question):
        return "llm_direct"

    if _is_same_entity_listing_overview_question(question):
        return "web_search"

    if any(marker in lowered for marker in RECENCY_MARKERS) and not _is_conceptual_meta_question(question):
        return "web_search"

    if _extract_source_topics(question):
        return "vectorstore"

    if any(marker in lowered for marker in DOC_MARKERS):
        return "vectorstore"

    return "llm_direct"


def _canonical_tier_hint(
    plan_type: str,
    subquestions: list[dict[str, str | None]],
) -> TierHint:
    route = subquestions[0]["route_hint"] if subquestions else None
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


def _multi_plan(subquestions: list[dict[str, str | None]], reason: str) -> dict:
    return {
        "plan_type": "multi",
        "reason": reason,
        "tier_hint": "tier_2",
        "subquestions": subquestions,
    }


def _normalize_subquestions(subquestions: list[dict[str, str | None]]) -> list[dict[str, str | None]]:
    normalized: list[dict[str, str | None]] = []
    seen_questions: set[str] = set()

    for subquestion in subquestions:
        cleaned = _clean_question(subquestion["question"])
        if not cleaned:
            continue

        canonical = cleaned.lower()
        if canonical in seen_questions:
            continue

        seen_questions.add(canonical)
        normalized.append(
            {
                "question": cleaned,
                "route_hint": subquestion.get("route_hint") or _guess_route_hint(cleaned),
            }
        )

    return normalized


def _expand_contextual_reference(part: str, full_question: str) -> str:
    lowered_part = part.lower()
    lowered_full = full_question.lower()

    if "this exam" in lowered_part and "claude" in lowered_full and "certification" in lowered_full:
        return re.sub(
            r"\bthis exam\b",
            "the Claude certification exam",
            part,
            flags=re.IGNORECASE,
        )

    if "the exam" in lowered_part and "claude" in lowered_full and "certification" in lowered_full:
        return re.sub(
            r"\bthe exam\b",
            "the Claude certification exam",
            part,
            flags=re.IGNORECASE,
        )

    return part


def _extract_explicit_question_parts(question: str) -> list[str]:
    return [
        _clean_question(part)
        for part in re.split(r"\?+", question)
        if _clean_question(part)
    ]


def _has_mixed_retrieval_intents(parts: list[str]) -> bool:
    route_hints = {_guess_route_hint(part) for part in parts if part}
    return len(route_hints) > 1


def _requires_multi_source_synthesis(question: str) -> bool:
    lowered = question.lower()
    source_topics = _extract_source_topics(question)

    if any(pattern.search(question) for pattern in COMPARISON_PATTERNS):
        return len(source_topics) >= 2

    if any(pattern.search(question) for pattern in ALIGNMENT_PATTERNS):
        return True

    if _current_listing_requested(question) and len(source_topics) >= 1 and lowered.startswith("what does"):
        return True

    return False


def _build_synthesis_subquestions(question: str) -> list[dict[str, str | None]]:
    source_topics = _extract_source_topics(question)
    subquestions: list[dict[str, str | None]] = []

    for topic in source_topics:
        spec = SOURCE_TOPIC_SPECS[topic]
        subquestions.append(
            {
                "question": str(spec["summary_question"]),
                "route_hint": str(spec["route_hint"]),
            }
        )

    if _current_listing_requested(question):
        specific_courses = [topic for topic in source_topics if topic != "exam_guide"]
        if specific_courses and _is_same_entity_listing_overview_question(question):
            specific = specific_courses[0]
            spec = SOURCE_TOPIC_SPECS[specific]
            title_question = str(spec["summary_question"]).removeprefix("What does ").removesuffix(" cover?")
            subquestions.append(
                {
                    "question": f"Is {title_question} currently listed on Anthropic Courses?",
                    "route_hint": "web_search",
                }
            )
        else:
            subquestions.append(
                {
                    "question": "What courses are currently listed on Anthropic Courses?",
                    "route_hint": "web_search",
                }
            )

    if not subquestions and any(pattern.search(question) for pattern in ALIGNMENT_PATTERNS):
        subquestions.extend(
            [
                {
                    "question": "What courses are currently listed on Anthropic Courses?",
                    "route_hint": "web_search",
                },
                {
                    "question": "What do the locally indexed Anthropic course and exam-guide summaries cover?",
                    "route_hint": "vectorstore",
                },
            ]
        )

    return _normalize_subquestions(subquestions)


def _extract_candidate_subquestions(question: str) -> list[str]:
    explicit_parts = _extract_explicit_question_parts(question)
    if len(explicit_parts) > 1:
        return [_expand_contextual_reference(part, question) for part in explicit_parts]

    return [_clean_question(question)]


def should_decompose(question: str) -> bool:
    """
    Conservative but synthesis-aware multi-hop heuristic.

    Decompose when a question clearly spans multiple evidence pools, explicit
    multi-part asks, or multi-source synthesis/comparison/alignment patterns.
    """
    clean_question = _clean_question(question)

    if _is_same_entity_listing_overview_question(clean_question):
        return False

    explicit_parts = _extract_candidate_subquestions(clean_question)
    if len(explicit_parts) > 1:
        if _has_mixed_retrieval_intents(explicit_parts):
            return True
        if any(_overlap_ratio(explicit_parts[0], part) < 0.55 for part in explicit_parts[1:]):
            return True

    return _requires_multi_source_synthesis(clean_question)


def _subquestions_overlap_too_much(subquestions: list[dict[str, str | None]]) -> bool:
    if len(subquestions) < 2:
        return False

    for index, left in enumerate(subquestions):
        for right in subquestions[index + 1 :]:
            if _overlap_ratio(left["question"], right["question"]) >= 0.6:
                return True

    return False


def _fallback_plan_query(question: str) -> dict:
    if not should_decompose(question):
        return _single_plan(question, "The question can be answered in one pass.")

    explicit_parts = _extract_candidate_subquestions(question)
    if len(explicit_parts) > 1:
        subquestions = _normalize_subquestions(
            [{"question": part, "route_hint": _guess_route_hint(part)} for part in explicit_parts]
        )
    else:
        subquestions = _build_synthesis_subquestions(question)

    if len(subquestions) < 2:
        return _single_plan(question, "The question can be answered in one pass.")

    return _multi_plan(
        subquestions,
        "The question requires combining multiple evidence pools and should be answered as multi-hop.",
    )


def plan_query(question: str) -> dict:
    """
    Plan whether a user query should stay whole or be decomposed.

    The planner stays conservative for ordinary single-source questions, but it
    explicitly recognizes comparison, overlap, alignment, and recommendation
    patterns that require multi-source synthesis.
    """
    clean_question = _clean_question(question)
    heuristic_multi = should_decompose(clean_question)

    if not heuristic_multi:
        return _single_plan(clean_question, "The question is a single intent and should stay as one pass.")

    try:
        planner = get_planner_llm().with_structured_output(QueryPlan)
        system_prompt = """
You are the planning step for a 3-tier adaptive RAG workflow.

Be conservative, but do not miss true multi-source synthesis questions.

Keep the query as "single" for:
- one-topic explanations
- one-document summaries
- straightforward detail questions from one source
- conceptual questions about current/time-sensitive/routing terms
- same-entity course-listing questions that can be answered from one current source

Use "multi" when the question clearly requires:
- comparison between two sources or topics
- overlap/synthesis across two sources
- alignment or recommendation based on local summaries and current listings
- combining exam-guide evidence with current course listings
- multiple distinct evidence pools or sequential sub-questions

Important constraints:
- Do not split on "and" or "also" by default.
- Do not decompose only because a question contains recency words.
- route_hint must not be used as the reason to decompose.
- If two returned sub-questions overlap heavily, collapse back to single.
- tier_hint must be one of: tier_0, tier_1, tier_2.
""".strip()

        result = planner.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": clean_question},
            ]
        )

        subquestions = _normalize_subquestions(
            [
                {
                    "question": sub.question,
                    "route_hint": sub.route_hint,
                }
                for sub in result.subquestions
            ]
        )

        if len(subquestions) < 2:
            return _fallback_plan_query(clean_question)

        if _subquestions_overlap_too_much(subquestions):
            return _fallback_plan_query(clean_question)

        if not heuristic_multi:
            return _single_plan(clean_question, "The question is a single intent and should stay as one pass.")

        return _multi_plan(subquestions, result.reason)
    except Exception:
        return _fallback_plan_query(clean_question)