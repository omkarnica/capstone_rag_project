from __future__ import annotations

import re

from src.graph import build_graph
from src.nodes.planner import plan_query
from src.nodes.merge import merge_answers
from src.cache.semantic_cache import (
    SemanticCache,
    compute_corpus_version,
    is_time_sensitive_question,
)
from src.tiering import tier_for_plan, tier_for_single_route, tier_label

_GRAPH = None
_CACHE = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def get_cache():
    global _CACHE
    if _CACHE is None:
        _CACHE = SemanticCache()
    return _CACHE


def _doc_contexts(docs: list[dict]) -> list[str]:
    contexts = []

    for i, doc in enumerate(docs, start=1):
        metadata = doc.get("metadata", {})
        title = (
            metadata.get("title")
            or metadata.get("file_name")
            or metadata.get("source")
            or "unknown"
        )
        page = metadata.get("page_label") or metadata.get("page")
        prefix = f"[Document {i}] {title}"
        if page is not None:
            prefix += f" (page {page})"

        content = doc.get("content", "").strip()
        contexts.append(f"{prefix}\n{content}")

    return contexts


def _web_contexts(results: list[dict]) -> list[str]:
    contexts = []

    for i, item in enumerate(results, start=1):
        title = item.get("title", "unknown")
        url = item.get("url", "unknown")
        content = item.get("content", "").strip()
        contexts.append(f"[Web Result {i}] {title}\nURL: {url}\n{content}")

    return contexts


def _extract_retrieved_contexts(result: dict) -> list[str]:
    docs = result.get("filtered_docs") or result.get("retrieved_docs") or []
    web_results = result.get("web_results") or []

    if docs:
        return _doc_contexts(docs)

    if web_results:
        return _web_contexts(web_results)

    return []


def _context_identity_key(context: str) -> str:
    lines = [line.strip() for line in context.splitlines() if line.strip()]

    if not lines:
        return ""

    first_line = re.sub(r"^\[(?:Document|Web Result)\s+\d+\]\s*", "", lines[0])

    for line in lines[1:3]:
        if line.startswith("URL: "):
            return f"url::{line.removeprefix('URL: ').strip()}"

    body_preview = " ".join(lines[1:])[:180]
    return f"{first_line}::{body_preview}"


def _dedupe_contexts(contexts: list[str]) -> list[str]:
    seen = set()
    deduped = []

    for context in contexts:
        key = _context_identity_key(context) or context
        if key not in seen:
            seen.add(key)
            deduped.append(context)

    return deduped


def _aggregate_sub_result_contexts(sub_results: list[dict]) -> list[str]:
    return _dedupe_contexts(
        [
            context
            for sub_result in sub_results
            for context in sub_result.get("retrieved_contexts", [])
        ]
    )


def _finalize_result_metadata(
    result: dict,
    *,
    chunking_strategy: str,
    cache_allowed: bool,
    cache_enabled: bool,
) -> dict:
    if "plan_type" not in result:
        result["plan_type"] = "single"

    if result["plan_type"] == "single" and not result.get("final_route"):
        sub_results = result.get("sub_results") or []
        if sub_results:
            first_sub = sub_results[0]
            result["initial_route"] = result.get("initial_route") or first_sub.get("initial_route")
            result["final_route"] = first_sub.get("final_route")
            result["retrieved_contexts"] = result.get("retrieved_contexts") or first_sub.get(
                "retrieved_contexts", []
            )

    if result["plan_type"] == "multi":
        result["retrieved_contexts"] = _aggregate_sub_result_contexts(result.get("sub_results", []))

    tier = tier_for_plan(result["plan_type"], result.get("final_route"))
    result["tier"] = tier
    result["tier_label"] = tier_label(tier)
    result["chunking_strategy"] = result.get("chunking_strategy", chunking_strategy)
    result["cache_allowed"] = cache_allowed
    result["cache_enabled"] = cache_enabled
    result["retrieved_contexts"] = _dedupe_contexts(result.get("retrieved_contexts", []))
    result["retrieved_context_count"] = len(result["retrieved_contexts"])

    return result


def run_single_question(
    question: str,
    route_hint: str | None = None,
    force_route: bool = False,
    chunking_strategy: str = "hierarchical",
) -> dict:
    """
    Run one question through the existing LangGraph pipeline.
    """
    graph = get_graph()

    state = {
        "question": question,
        "max_iterations": 3,
        "max_retrieval_attempts": 1,
        "chunking_strategy": chunking_strategy,
    }

    if route_hint:
        state["route_hint"] = route_hint
    if force_route:
        state["force_route"] = True

    result = graph.invoke(state)
    single_tier = tier_for_single_route(result.get("route"))
    retrieved_contexts = _extract_retrieved_contexts(result)

    return {
        "question": question,
        "planned_route_hint": route_hint,
        "tier": single_tier,
        "tier_label": tier_label(single_tier),
        "initial_route": result.get("initial_route"),
        "final_route": result.get("route"),
        "route_reason": result.get("route_reason"),
        "retrieval_query": result.get("retrieval_query"),
        "rewritten_question": result.get("rewritten_question"),
        "relevant_doc_count": result.get("relevant_doc_count"),
        "retrieval_attempt": result.get("retrieval_attempt"),
        "answer": result.get("answer"),
        "citations": result.get("citations", []),
        "hallucination_grade": result.get("hallucination_grade"),
        "answer_quality_grade": result.get("answer_quality_grade"),
        "retrieved_contexts": retrieved_contexts,
        "retrieved_context_count": len(retrieved_contexts),
        "chunking_strategy": chunking_strategy,
    }


def run_adaptive_query(
    question: str,
    chunking_strategy: str = "hierarchical",
    similarity_threshold: float = 0.92,
    use_cache: bool = True,
) -> dict:
    """
    Adaptive wrapper:
    1. semantic cache lookup
    2. plan query
    3. single => one graph run
    4. multi => run graph per sub-question, then merge
    5. cache stable result
    """
    corpus_version = compute_corpus_version(
        chunking_strategy=chunking_strategy
    )

    cache_allowed = use_cache and not is_time_sensitive_question(question)

    if cache_allowed:
        cache = get_cache()
        cached = cache.get_similar(
            question=question,
            corpus_version=corpus_version,
            chunking_strategy=chunking_strategy,
            similarity_threshold=similarity_threshold,
        )
        if cached is not None:
            cached["cache_hit"] = True
            cached["cache_similarity"] = cached.get("cache_similarity")
            cached["cache_source_question"] = cached.get("cache_source_question")
            return _finalize_result_metadata(
                cached,
                chunking_strategy=chunking_strategy,
                cache_allowed=True,
                cache_enabled=use_cache,
            )

    plan = plan_query(question)

    if plan["plan_type"] == "single":
        single = run_single_question(
            question=plan["subquestions"][0]["question"],
            route_hint=plan["subquestions"][0].get("route_hint"),
            force_route=False,
            chunking_strategy=chunking_strategy,
        )

        final_result = {
            "plan_type": "single",
            "plan_reason": plan["reason"],
            "plan_tier_hint": plan.get("tier_hint"),
            "subquestions": plan["subquestions"],
            "initial_route": single["initial_route"],
            "final_route": single["final_route"],
            "final_answer": single["answer"],
            "citations": single["citations"],
            "retrieved_contexts": single["retrieved_contexts"],
            "sub_results": [single],
            "cache_hit": False,
            "cache_similarity": None,
            "cache_source_question": None,
            "chunking_strategy": chunking_strategy,
        }

    else:
        sub_results = []
        for subq in plan["subquestions"]:
            result = run_single_question(
                question=subq["question"],
                route_hint=subq.get("route_hint"),
                force_route=True,
                chunking_strategy=chunking_strategy,
            )
            sub_results.append(result)

        merged = merge_answers(question, sub_results)
        retrieved_contexts = _aggregate_sub_result_contexts(sub_results)

        final_result = {
            "plan_type": "multi",
            "plan_reason": plan["reason"],
            "plan_tier_hint": plan.get("tier_hint"),
            "subquestions": plan["subquestions"],
            "initial_route": "multi_hop",
            "final_route": "multi_hop",
            "final_answer": merged["answer"],
            "citations": merged["citations"],
            "sub_results": merged["sub_results"],
            "retrieved_contexts": retrieved_contexts,
            "cache_hit": False,
            "cache_similarity": None,
            "cache_source_question": None,
            "chunking_strategy": chunking_strategy,
        }

    # Store only stable queries in semantic cache
    if cache_allowed:
        cache = get_cache()
        cache.store(
            question=question,
            result=final_result,
            corpus_version=corpus_version,
            chunking_strategy=chunking_strategy,
        )

    return _finalize_result_metadata(
        final_result,
        chunking_strategy=chunking_strategy,
        cache_allowed=cache_allowed,
        cache_enabled=use_cache,
    )