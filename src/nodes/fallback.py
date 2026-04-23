from __future__ import annotations

import json
import os
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from src.state import GraphState

load_dotenv()

OFFICIAL_DOMAINS = [
    "anthropic.com",
    "docs.anthropic.com",
    "anthropic.skilljar.com"
]

LOW_TRUST_DOMAINS = {
    "linkedin.com",
    "coursera.org",
    "mashable.com",
    "reddit.com",
    "medium.com",
    "youtube.com",
}


def _get_brave_api_key() -> str:
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise ValueError("BRAVE_SEARCH_API_KEY is missing. Put it in your .env file.")
    return api_key


def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _is_official_domain(url: str) -> bool:
    domain = _get_domain(url)
    return any(domain == d or domain.endswith("." + d) for d in OFFICIAL_DOMAINS)


def _is_low_trust_domain(url: str) -> bool:
    domain = _get_domain(url)
    return any(domain == d or domain.endswith("." + d) for d in LOW_TRUST_DOMAINS)


def _rewrite_for_web_search(question: str) -> str:
    q = question.lower()

    if any(word in q for word in ["certification", "certified", "exam guide", "exam"]):
        return f"{question} official Anthropic Claude certification"
    if "claude code" in q:
        return f"{question} official Anthropic Claude Code docs"
    if any(word in q for word in ["recent", "latest", "new", "launched", "announcement", "current"]):
        return f"{question} official Anthropic announcement"
    return f"{question} official Anthropic"


def _score_result(item: dict) -> int:
    title = (item.get("title") or "").lower()
    url = item.get("url") or ""
    content = (item.get("content") or "").lower()

    score = 0

    if _is_official_domain(url):
        score += 100

    if _is_low_trust_domain(url):
        score -= 60

    if "anthropic" in title:
        score += 15
    if "claude" in title:
        score += 10
    if "certification" in title or "certified" in title:
        score += 12
    if "announcement" in title or "news" in title:
        score += 8
    if "exam guide" in title:
        score += 15

    if "anthropic" in content:
        score += 5
    if "claude" in content:
        score += 5

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
                "is_official": _is_official_domain(url),
                "trust_score": _score_result(item),
            }
        )

    normalized.sort(
        key=lambda x: (x["is_official"], x["trust_score"]),
        reverse=True,
    )
    return normalized


def _join_brave_snippets(item: dict) -> str:
    parts = []

    description = (item.get("description") or "").strip()
    if description:
        parts.append(description)

    for snippet in item.get("extra_snippets") or []:
        text = str(snippet).strip()
        if text and text not in parts:
            parts.append(text)

    return "\n\n".join(parts)


def _normalize_brave_result(item: dict) -> dict:
    return {
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "content": _join_brave_snippets(item),
    }


def _brave_request(query: str, count: int = 5) -> dict:
    params = urlencode(
        {
            "q": query,
            "count": count,
            "extra_snippets": "true",
            "safesearch": "moderate",
            "search_lang": "en",
            "ui_lang": "en-US",
            "country": "US",
        }
    )
    request = Request(
        url=f"https://api.search.brave.com/res/v1/web/search?{params}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": _get_brave_api_key(),
        },
    )

    with urlopen(request, timeout=20) as response:
        payload = response.read().decode("utf-8")

    return json.loads(payload)


def _dedupe_results(results: list[dict]) -> list[dict]:
    seen_urls = set()
    deduped = []

    for item in results:
        url = item.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(item)

    return deduped


def _search_brave(query: str, include_domains: list[str] | None = None) -> list[dict]:
    if include_domains:
        all_results = []
        for domain in include_domains:
            scoped_query = f"site:{domain} {query}"
            response = _brave_request(scoped_query, count=5)
            all_results.extend(response.get("web", {}).get("results", []))

        return [
            _normalize_brave_result(item)
            for item in _dedupe_results(all_results)
        ]

    response = _brave_request(query, count=5)
    return [
        _normalize_brave_result(item)
        for item in response.get("web", {}).get("results", [])
    ]


def _web_search_error_message(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    return f"Web search is currently unavailable: {message}"


def web_search_fallback(state: GraphState) -> GraphState:
    """
    Web fallback strategy:
    1. Search only official Anthropic domains first.
    2. If nothing useful comes back, broaden the search.
    3. Re-rank all results so official/trusted sources appear first.
    """
    original_query = state.get("rewritten_question") or state["question"]
    query = _rewrite_for_web_search(original_query)

    try:
        # Pass 1: official domains only
        results = _search_brave(query=query, include_domains=OFFICIAL_DOMAINS)
        ranked_results = _normalize_results(results)

        # Keep official results only if we found any
        official_results = [r for r in ranked_results if r["is_official"]]

        # Pass 2: broaden only if official search returned nothing useful
        if not official_results:
            broad_results = _search_brave(query=query, include_domains=None)
            ranked_results = _normalize_results(broad_results)
            final_results = ranked_results[:3]
        else:
            final_results = official_results[:3]
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