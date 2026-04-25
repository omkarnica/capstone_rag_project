from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.model_config import get_router_llm
from src.state import GraphState

_VALID_ROUTES = {
    "sql", "filings", "transcripts", "patents",
    "litigation", "graph", "contradiction", "llm_direct",
}


class RouteDecision(BaseModel):
    route: Literal[
        "sql", "filings", "transcripts", "patents",
        "litigation", "graph", "contradiction", "llm_direct",
    ] = Field(description="Routing decision for the M&A query.")
    reason: str = Field(description="Short reason for the routing decision.")


_SYSTEM_PROMPT = """
You are the router for an M&A due diligence RAG system.

Choose exactly one route based on what the query needs:

- sql          — financial metrics, revenue, profit, margins, operating income, EBITDA,
                 cash flow, EPS, ratios, XBRL numbers, balance sheet, structured financial data
- filings      — 10-K/10-Q narrative content: risk factors, MD&A, management discussion,
                 footnotes, disclosures, business description, Item 1A, material weaknesses
- transcripts  — what management said on earnings calls, CEO/CFO quotes, guidance,
                 investor Q&A, earnings press releases
- patents      — IP portfolio, patent claims, CPC codes, invention descriptions,
                 patent citations, patent filing/grant dates
- litigation   — court cases, lawsuits, settlements, legal exposure, litigation risk,
                 CourtListener data
- graph        — entity relationships, subsidiaries, board connections, ownership structure,
                 corporate hierarchy (NOTE: stub — may return limited results)
- contradiction — compare management statements vs. filed disclosures, detect discrepancies
                 between what was said on earnings calls and what SEC filings show,
                 due diligence contradiction analysis
- llm_direct   — general M&A concepts, finance theory, valuation methods, definitions,
                 questions that do not require specific company data

Routing rules:
- If a question asks for specific numbers, use sql.
- If a question asks about risk factors or 10-K language, use filings.
- If a question asks what management "said" or "mentioned", use transcripts.
- If a question asks about patents, IP, or CPC codes, use patents.
- If a question asks about lawsuits, courts, or legal, use litigation.
- If a question explicitly asks to compare filings vs. transcripts or detect contradictions, use contradiction.
- For general concepts (what is DCF, how does M&A work), use llm_direct.

Planner route hint: {route_hint}
If a planner route hint is present and valid, follow it.

Return structured output with route and reason.
""".strip()


def route_question(state: GraphState) -> GraphState:
    question = state["question"]
    route_hint = state.get("route_hint", "none") or "none"
    force_route = state.get("force_route", False)

    if force_route and route_hint in _VALID_ROUTES:
        return {
            **state,
            "route": route_hint,
            "initial_route": route_hint,
            "route_reason": f"Forced by planner route_hint: {route_hint}",
        }

    llm = get_router_llm().with_structured_output(RouteDecision)
    decision = llm.invoke([
        {"role": "system", "content": _SYSTEM_PROMPT.format(route_hint=route_hint)},
        {"role": "user", "content": question},
    ])

    return {
        **state,
        "route": decision.route,
        "initial_route": decision.route,
        "route_reason": decision.reason,
    }
