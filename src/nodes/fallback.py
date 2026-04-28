from __future__ import annotations

from google.genai import types

from src.model_config import get_genai_client, get_model_name
from src.state import GraphState


def _rewrite_for_web_search(question: str, route: str | None = None) -> str:
    route = route or ""
    q = question.lower()

    if route in {"sql", "filings"} or any(term in q for term in ("10-k", "10-q", "sec", "filing")):
        return f"{question} SEC filing 10-K 10-Q"
    if route == "transcripts" or any(term in q for term in ("earnings call", "transcript", "management said")):
        return f"{question} earnings call transcript investor relations SEC 8-K"
    if route == "patents" or any(term in q for term in ("patent", "ip", "invention")):
        return f"{question} patent USPTO Google Patents"
    if route == "litigation" or any(term in q for term in ("lawsuit", "litigation", "court", "case")):
        return f"{question} litigation court case CourtListener"
    return f"{question} M&A due diligence official source"


def _search_with_gemini_google_tool(query: str) -> list[dict]:
    client = get_genai_client()
    response = client.models.generate_content(
        model=get_model_name(),
        contents=(
            "Search the web for authoritative M&A due diligence evidence. "
            "Prefer SEC, USPTO, CourtListener, regulator, and company investor "
            f"relations sources.\n\nQuery: {query}"
        ),
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    text = (response.text or "").strip()
    if not text:
        return []
    return [
        {
            "title": "Gemini Google Search grounded result",
            "url": "google_search://grounded",
            "content": text,
        }
    ]


def web_search_fallback(state: GraphState) -> GraphState:
    original_query = state.get("rewritten_question") or state["question"]
    query = _rewrite_for_web_search(original_query, state.get("route"))

    try:
        results = _search_with_gemini_google_tool(query)
    except Exception as exc:
        message = str(exc).strip() or type(exc).__name__
        return {
            **state,
            "route": "web_search",
            "retrieval_query": query,
            "web_results": [],
            "web_search_error": f"Web search is currently unavailable: {message}",
        }

    return {
        **state,
        "route": "web_search",
        "retrieval_query": query,
        "web_results": results,
        "web_search_error": "",
    }
