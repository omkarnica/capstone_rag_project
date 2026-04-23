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


def initialize_state(state: GraphState) -> GraphState:
    return {
        **state,
        "iteration": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", 3),
        "retrieval_attempt": state.get("retrieval_attempt", 0),
        "max_retrieval_attempts": state.get("max_retrieval_attempts", 1),
    }


def increment_iteration(state: GraphState) -> GraphState:
    return {
        **state,
        "iteration": state.get("iteration", 0) + 1,
    }


def increment_retrieval_attempt(state: GraphState) -> GraphState:
    return {
        **state,
        "retrieval_attempt": state.get("retrieval_attempt", 0) + 1,
    }


def route_after_router(state: GraphState) -> str:
    return state["route"]


def route_after_doc_grading(state: GraphState) -> str:
    """
    Corrective RAG controller:
    - if relevant docs exist -> generate
    - else if retry budget remains -> rewrite
    - else -> web fallback
    """
    if state.get("relevant_doc_count", 0) > 0:
        return "generate"

    if state.get("retrieval_attempt", 0) < state.get("max_retrieval_attempts", 1):
        return "rewrite"

    return "fallback"


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
    """
    Self-RAG retry control:
    - if original path was vectorstore/web, use fallback web on retry
    - if direct answer path, just end for now or move to web if you want later
    """
    if state.get("route") in {"vectorstore", "web_search"}:
        return "web_search"

    return "end"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("initialize", initialize_state)
    graph.add_node("router", route_question)

    # Corrective RAG nodes
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("grade_docs", grade_documents)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("increment_retrieval_attempt", increment_retrieval_attempt)
    graph.add_node("web_search", web_search_fallback)

    # Generation + Self-RAG nodes
    graph.add_node("generate", generate_answer)
    graph.add_node("llm_direct_generate", generate_direct_answer)
    graph.add_node("grade_hallucination", grade_hallucination)
    graph.add_node("grade_quality", grade_answer_quality)
    graph.add_node("increment_iteration", increment_iteration)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "router")

    # Adaptive RAG
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "llm_direct": "llm_direct_generate",
            "vectorstore": "retrieve",
            "web_search": "web_search",
        },
    )

    # Corrective RAG
    graph.add_edge("retrieve", "grade_docs")

    graph.add_conditional_edges(
        "grade_docs",
        route_after_doc_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fallback": "web_search",
        },
    )

    graph.add_edge("rewrite", "increment_retrieval_attempt")
    graph.add_edge("increment_retrieval_attempt", "retrieve")

    # Direct web route and fallback route both go here
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