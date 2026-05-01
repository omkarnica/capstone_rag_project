from __future__ import annotations

import re

from src.graph import build_graph
from src.nodes.planner import plan_query
from src.nodes.merge import merge_answers
from src.observability import (
    build_langsmith_metadata,
    build_langsmith_runnable_config,
    langsmith_trace_scope,
)
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
    company: str | None = None,
    period: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
    trace_source: str | None = None,
) -> dict:
    """
    Run one question through the M&A Oracle LangGraph pipeline.
    """
    graph = get_graph()

    state = {
        "question": question,
        "max_iterations": 3,
        "max_retrieval_attempts": 3,
        "chunking_strategy": chunking_strategy,
    }

    if route_hint:
        state["route_hint"] = route_hint
    if force_route:
        state["force_route"] = True
    if company:
        state["company"] = company
    if period:
        state["period"] = period

    metadata = build_langsmith_metadata(
        question=question,
        company=company,
        period=period,
        route=route_hint,
        user_id=user_id,
        plan_type="single",
        trace_source=trace_source,
    )
    tags = [tag for tag in [trace_source or "direct", "single", route_hint or "auto"] if tag]
    config = build_langsmith_runnable_config(
        metadata=metadata,
        tags=tags,
        run_name="single_question",
    )

    with langsmith_trace_scope(metadata=metadata, tags=tags):
        result = graph.invoke(state, config=config)
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
        "contradiction_report": result.get("contradiction_report", []),
        "company": result.get("company"),
        "period": result.get("period"),
    }


def _cache_scoped_question(
    question: str,
    *,
    company: str | None = None,
    period: str | None = None,
) -> str:
    scope_parts = []
    if company:
        scope_parts.append(f"company={company.lower()}")
    if period:
        scope_parts.append(f"period={period.lower()}")
    if not scope_parts:
        return question
    return f"{question}\nContext: {'; '.join(scope_parts)}"


def _cache_scope(
    chunking_strategy: str,
    *,
    company: str | None = None,
    period: str | None = None,
) -> str:
    scope_parts = [chunking_strategy]
    if company:
        scope_parts.append(f"company={company.lower()}")
    if period:
        scope_parts.append(f"period={period.lower()}")
    return "|".join(scope_parts)


def run_adaptive_query(
    question: str,
    chunking_strategy: str = "hierarchical",
    similarity_threshold: float = 0.92,
    use_cache: bool = True,
    company: str | None = None,
    period: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
    trace_source: str | None = None,
) -> dict:
    """
    Adaptive wrapper:
    1. semantic cache lookup
    2. plan query
    3. single => one graph run
    4. multi => run graph per sub-question, then merge
    5. cache stable result
    """
    metadata = build_langsmith_metadata(
        question=question,
        company=company,
        period=period,
        user_id=user_id,
        trace_source=trace_source,
    )
    tags = [tag for tag in [trace_source or "direct", "adaptive"] if tag]

    with langsmith_trace_scope(metadata=metadata, tags=tags):
        corpus_version = compute_corpus_version(
            chunking_strategy=chunking_strategy
        )
        cache_question = _cache_scoped_question(question, company=company, period=period)
        cache_scope = _cache_scope(chunking_strategy, company=company, period=period)

        cache_allowed = use_cache and not is_time_sensitive_question(question)

        if cache_allowed:
            cache = get_cache()

            # O(1) exact match — same question text, same scope
            exact = cache.get_exact(cache_question, corpus_version, cache_scope)
            if exact is not None:
                exact["cache_hit"] = True
                exact["cache_similarity"] = 1.0
                exact["cache_source_question"] = cache_question
                return _finalize_result_metadata(
                    exact,
                    chunking_strategy=chunking_strategy,
                    cache_allowed=True,
                    cache_enabled=use_cache,
                )

            # O(n) semantic scan — similar but not identical question
            cached = cache.get_similar(
                question=cache_question,
                corpus_version=corpus_version,
                chunking_strategy=cache_scope,
                similarity_threshold=similarity_threshold,
            )
            if cached is not None:
                # Promote to exact cache so the next identical query is O(1)
                cache.set_exact(cache_question, cached, corpus_version, cache_scope)
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
                company=company,
                period=period,
                tenant_id=tenant_id,
                user_id=user_id,
                trace_source=trace_source,
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
                "company": company,
                "period": period,
            }

        else:
            sub_results = []
            for subq in plan["subquestions"]:
                result = run_single_question(
                    question=subq["question"],
                    route_hint=subq.get("route_hint"),
                    force_route=True,
                    chunking_strategy=chunking_strategy,
                    company=company,
                    period=period,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    trace_source=trace_source,
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
                "company": company,
                "period": period,
            }

        _dynamic_routes = {"contradiction", "graph", "multi_hop"}
        cache_store_allowed = cache_allowed and final_result.get("final_route") not in _dynamic_routes
        if cache_store_allowed:
            cache = get_cache()
            cache.store(
                question=cache_question,
                result=final_result,
                corpus_version=corpus_version,
                chunking_strategy=cache_scope,
            )
            cache.set_exact(cache_question, final_result, corpus_version, cache_scope)

        return _finalize_result_metadata(
            final_result,
            chunking_strategy=chunking_strategy,
            cache_allowed=cache_allowed,
            cache_enabled=use_cache,
        )
