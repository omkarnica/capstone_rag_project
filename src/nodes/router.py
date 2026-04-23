from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.model_config import get_router_llm
from src.state import GraphState


class RouteDecision(BaseModel):
    route: Literal["llm_direct", "vectorstore", "web_search"] = Field(
        description="Routing decision for the question."
    )
    reason: str = Field(description="Short reason for the routing decision.")

def route_question(state: GraphState) -> GraphState:
    """
    Adaptive RAG router.
    Honors force_route for planned sub-questions.
    """
    question = state["question"]
    route_hint = state.get("route_hint")
    force_route = state.get("force_route", False)

    # Hard route for decomposed sub-questions
    if force_route and route_hint in {"llm_direct", "vectorstore", "web_search"}:
        return {
            **state,
            "route": route_hint,
            "initial_route": route_hint,
            "route_reason": f"Forced by planner route_hint: {route_hint}",
        }

    system_prompt = f"""
You are the router for an agentic RAG system.

Choose exactly one route:
- llm_direct: general conceptual knowledge that does not need documents
- vectorstore: answer should come from the local Anthropic document corpus
- web_search: answer depends on recent, current, or newly announced information

Routing rules:
- Questions about Claude certification details, exam guide content, syllabus, topics covered, preparation, eligibility, courses, Academy resources, or Claude Code docs should go to vectorstore.
- Questions asking for recent launches, latest announcements, or current status should go to web_search.
- Questions that define or explain concepts such as time-sensitive queries, semantic cache behavior, single-hop vs multi-hop retrieval, prompt engineering, hallucination, or context management should go to llm_direct even if they contain words like current, latest, or time-sensitive.
- Questions about MCP or Introduction to Model Context Protocol course overviews should go to vectorstore.
- Only use llm_direct for broad conceptual questions.

Planner route hint:
{route_hint if route_hint else "none"}

If a planner route hint is present, follow it unless it is clearly wrong.

Return structured output with:
- route
- reason
""".strip()

    llm = get_router_llm().with_structured_output(RouteDecision)

    decision = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    )

    return {
        **state,
        "route": decision.route,
        "initial_route": decision.route,
        "route_reason": decision.reason,
    }