from __future__ import annotations

import json
import os
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.state import GraphState
from src.utils.secrets import get_secret

load_dotenv()

_GCP_PROJECT = "codelab-2-485215"
_GCP_LOCATION = "us-central1"
_GEMINI_MODEL = "gemini-2.5-flash"

TRUSTED_MA_DOMAINS = [
    "sec.gov",
    "www.sec.gov",
    "uspto.gov",
    "www.uspto.gov",
    "patents.google.com",
    "courtlistener.com",
    "www.courtlistener.com",
    "justice.gov",
    "www.justice.gov",
    "ftc.gov",
    "www.ftc.gov",
]

LOW_TRUST_DOMAINS = {
    "linkedin.com",
    "reddit.com",
    "medium.com",
    "youtube.com",
    "facebook.com",
    "x.com",
    "twitter.com",
}


def _get_google_api_key() -> str:
    return get_secret("GOOGLE_SEARCH_API_KEY")


def _get_google_search_engine_id() -> str:
    try:
        return get_secret("GOOGLE_SEARCH_ENGINE_ID")
    except ValueError:
        return get_secret("GOOGLE_CSE_ID")


def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _is_trusted_domain(url: str) -> bool:
    domain = _get_domain(url)
    return any(domain == d or domain.endswith("." + d) for d in TRUSTED_MA_DOMAINS)


def _is_low_trust_domain(url: str) -> bool:
    domain = _get_domain(url)
    return any(domain == d or domain.endswith("." + d) for d in LOW_TRUST_DOMAINS)


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


def _score_result(item: dict) -> int:
    title = (item.get("title") or "").lower()
    url = item.get("url") or ""
    content = (item.get("content") or "").lower()

    score = 0

    if _is_trusted_domain(url):
        score += 100

    if _is_low_trust_domain(url):
        score -= 60

    trusted_terms = (
        "10-k", "10-q", "8-k", "annual report", "quarterly report",
        "sec", "edgar", "investor relations", "patent", "uspto",
        "court", "opinion", "litigation", "complaint", "settlement",
    )
    for term in trusted_terms:
        if term in title:
            score += 8
        if term in content:
            score += 3

    if not item.get("content"):
        score -= 10

    return score


def _normalize_results(results: list[dict]) -> list[dict]:
    normalized = []

    for item in results:
        url = item.get("url", "")
        normalized.append(
            {
                "title": item.get("title", ""),
                "url": url,
                "content": item.get("content", ""),
                "is_trusted": _is_trusted_domain(url),
                "trust_score": _score_result(item),
            }
        )

    normalized.sort(
        key=lambda x: (x["is_trusted"], x["trust_score"]),
        reverse=True,
    )
    return normalized


def _normalize_google_result(item: dict) -> dict:
    return {
        "title": item.get("title", ""),
        "url": item.get("link", ""),
        "content": item.get("snippet", ""),
    }


def _google_request(query: str, count: int = 5) -> dict:
    params = urlencode(
        {
            "key": _get_google_api_key(),
            "cx": _get_google_search_engine_id(),
            "q": query,
            "num": min(max(count, 1), 10),
            "safe": "active",
        }
    )
    request = Request(
        url=f"https://www.googleapis.com/customsearch/v1?{params}",
        headers={"Accept": "application/json"},
    )

    with urlopen(request, timeout=20) as response:
        payload = response.read().decode("utf-8")

    return json.loads(payload)


def _dedupe_results(results: list[dict]) -> list[dict]:
    seen_urls = set()
    deduped = []

    for item in results:
        url = item.get("url") or item.get("link") or ""
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(item)

    return deduped


def _search_google(query: str, include_domains: list[str] | None = None) -> list[dict]:
    if include_domains:
        all_results = []
        for domain in include_domains:
            scoped_query = f"site:{domain} {query}"
            response = _google_request(scoped_query, count=5)
            all_results.extend(response.get("items", []))

        return [
            _normalize_google_result(item)
            for item in _dedupe_results(all_results)
        ]

    response = _google_request(query, count=5)
    return [
        _normalize_google_result(item)
        for item in response.get("items", [])
    ]


def _search_with_gemini_google_tool(query: str) -> list[dict]:
    client = genai.Client(
        vertexai=True,
        project=_GCP_PROJECT,
        location=_GCP_LOCATION,
    )
    response = client.models.generate_content(
        model=_GEMINI_MODEL,
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


def _web_search_error_message(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    return f"Web search is currently unavailable: {message}"


def web_search_fallback(state: GraphState) -> GraphState:
    """
    Web fallback strategy:
    1. Search trusted M&A source domains first via Google Custom Search.
    2. If nothing useful comes back, broaden the search.
    3. Re-rank all results so trusted sources appear first.
    """
    original_query = state.get("rewritten_question") or state["question"]
    query = _rewrite_for_web_search(original_query, state.get("route"))

    try:
        final_results = _search_with_gemini_google_tool(query)
        if final_results:
            return {
                **state,
                "route": "web_search",
                "retrieval_query": query,
                "web_results": final_results,
                "web_search_error": "",
            }

        # Pass 1: trusted M&A domains only
        results = _search_google(query=query, include_domains=TRUSTED_MA_DOMAINS)
        ranked_results = _normalize_results(results)

        # Keep trusted results only if we found any
        trusted_results = [r for r in ranked_results if r["is_trusted"]]

        # Pass 2: broaden only if trusted search returned nothing useful
        if not trusted_results:
            broad_results = _search_google(query=query, include_domains=None)
            ranked_results = _normalize_results(broad_results)
            final_results = ranked_results[:3]
        else:
            final_results = trusted_results[:3]
    except Exception as exc:
        return {
            **state,
            "route": "web_search",
            "retrieval_query": query,
            "web_results": [],
            "web_search_error": _web_search_error_message(exc),
        }

    return {
        **state,
        "route": "web_search",
        "retrieval_query": query,
        "web_results": final_results,
        "web_search_error": "",
    }
