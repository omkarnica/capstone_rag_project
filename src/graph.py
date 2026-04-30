from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from src.state import GraphState
from src.nodes.router import route_question
from src.nodes.retriever import retrieve_docs
from src.nodes.grader import (
    grade_documents,
    grade_hallucination,
    grade_answer_quality,
)
from src.nodes.rewriter import rewrite_query
from src.nodes.generator import generate_answer, generate_direct_answer
from src.nodes.fallback import web_search_fallback

_RETRIEVAL_ROUTES = {"sql", "filings", "transcripts", "patents", "litigation", "graph"}


# Not used by the compiled graph (build_graph uses an inner closure that also injects eval_config).
# Retained for external callers that may reference it directly.
def initialize_state(state: GraphState) -> GraphState:
    return {
        **state,
        "iteration": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", 3),
        "retrieval_attempt": state.get("retrieval_attempt", 0),
        "max_retrieval_attempts": state.get("max_retrieval_attempts", 3),
    }


def increment_iteration(state: GraphState) -> GraphState:
    return {**state, "iteration": state.get("iteration", 0) + 1}


def increment_retrieval_attempt(state: GraphState) -> GraphState:
    return {**state, "retrieval_attempt": state.get("retrieval_attempt", 0) + 1}


def run_contradiction_check(state: GraphState) -> GraphState:
    from src.contradictions.detector import run_due_diligence
    company = state.get("company", "")
    period = state.get("period", "")
    fiscal_year = 2024
    if period and period[-4:].isdigit():
        fiscal_year = int(period[-4:])
    quarter = "Q4"
    if period:
        for token in period.upper().split():
            if token in {"Q1", "Q2", "Q3", "Q4", "FY"}:
                quarter = token
                break
    findings = run_due_diligence(
        company=company,
        transcript_company=company,
        fiscal_year=fiscal_year,
        quarter=quarter,
    )
    summary_lines = []
    for f in findings:
        severity = f.get("severity", "unknown")
        metric = f.get("metric", "")
        period_label = f.get("period", "")
        score = f.get("contradiction_score", 0)
        summary_lines.append(
            f"[{severity.upper()}] {metric} ({period_label}): contradiction score {score:.2f}"
        )
    summary = "\n".join(summary_lines) if summary_lines else "No contradictions detected."
    citations = [f"{f.get('metric', '')} ({f.get('period', '')})" for f in findings]
    return {
        **state,
        "contradiction_report": findings,
        "answer": summary,
        "citations": citations,
    }


def route_after_router(state: GraphState) -> str:
    route = state["route"]
    if route == "contradiction":
        return "contradiction"
    if route == "llm_direct":
        return "llm_direct"
    return "retrieve"


def route_after_doc_grading(state: GraphState) -> str:
    if state.get("relevant_doc_count", 0) > 0:
        return "generate"
    if (
        state.get("route") == "filings"
        and not state.get("graph_fallback_attempted", False)
    ):
        return "graph_fallback"
    if state.get("retrieval_attempt", 0) < state.get("max_retrieval_attempts", 1):
        return "rewrite"
    return "fallback"


def use_graph_fallback(state: GraphState) -> GraphState:
    return {
        **state,
        "route": "graph",
        "graph_fallback_attempted": True,
        "retrieved_docs": [],
        "filtered_docs": [],
        "doc_relevance": [],
        "relevant_doc_count": 0,
    }


def route_after_hallucination(state: GraphState) -> str:
    if state.get("web_search_error"):
        return "end"
    if state.get("hallucination_grade") == "yes":
        return "quality"
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return "end"
    return "retry"


def route_after_quality(state: GraphState) -> str:
    if state.get("web_search_error"):
        return "end"
    if state.get("answer_quality_grade") == "yes":
        return "end"
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return "end"
    return "retry"


def retry_route(state: GraphState) -> str:
    if state.get("route") == "graph":
        return "end"
    if state.get("route") in _RETRIEVAL_ROUTES:
        return "web_search"
    return "end"


def build_graph(eval_config: dict | None = None):
    graph = StateGraph(GraphState)

    _eval_config = eval_config or {}

    def _initialize_state(state: GraphState) -> GraphState:
        return {
            **state,
            "iteration": state.get("iteration", 0),
            "max_iterations": state.get("max_iterations", 3),
            "retrieval_attempt": state.get("retrieval_attempt", 0),
            "max_retrieval_attempts": state.get("max_retrieval_attempts", 3),
            "graph_fallback_attempted": state.get("graph_fallback_attempted", False),
            "eval_config": _eval_config,
        }

    graph.add_node("initialize", _initialize_state)
    graph.add_node("router", route_question)

    # M&A contradiction node (agentic — bypasses retrieve/grade/generate)
    graph.add_node("contradiction_check", run_contradiction_check)

    # Corrective RAG nodes
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("grade_docs", grade_documents)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("increment_retrieval_attempt", increment_retrieval_attempt)
    graph.add_node("graph_fallback", use_graph_fallback)
    graph.add_node("web_search", web_search_fallback)

    # Generation + Self-RAG nodes
    graph.add_node("generate", generate_answer)
    graph.add_node("llm_direct_generate", generate_direct_answer)
    graph.add_node("grade_hallucination", grade_hallucination)
    graph.add_node("grade_quality", grade_answer_quality)
    graph.add_node("increment_iteration", increment_iteration)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "router")

    # Adaptive RAG — routes to contradiction, llm_direct, or retrieval
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "contradiction": "contradiction_check",
            "llm_direct": "llm_direct_generate",
            "retrieve": "retrieve",
        },
    )

    # Contradiction check goes straight to END
    graph.add_edge("contradiction_check", END)

    # Corrective RAG
    graph.add_edge("retrieve", "grade_docs")

    graph.add_conditional_edges(
        "grade_docs",
        route_after_doc_grading,
        {
            "generate": "generate",
            "graph_fallback": "graph_fallback",
            "rewrite": "rewrite",
            "fallback": "web_search",
        },
    )

    graph.add_edge("graph_fallback", "retrieve")
    graph.add_edge("rewrite", "increment_retrieval_attempt")
    graph.add_edge("increment_retrieval_attempt", "retrieve")

    # Fallback web search
    graph.add_edge("web_search", "generate")

    # Self-RAG
    graph.add_edge("generate", "grade_hallucination")
    graph.add_edge("llm_direct_generate", "grade_quality")

    graph.add_conditional_edges(
        "grade_hallucination",
        route_after_hallucination,
        {
            "quality": "grade_quality",
            "retry": "increment_iteration",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "grade_quality",
        route_after_quality,
        {
            "retry": "increment_iteration",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "increment_iteration",
        retry_route,
        {
            "web_search": "web_search",
            "end": END,
        },
    )

    return graph.compile()
