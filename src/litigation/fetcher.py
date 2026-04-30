"""
CourtListener opinion fetcher for M&A Oracle litigation pipeline.

fetch_cases:
    Searches CourtListener /api/rest/v4/search/ for opinions mentioning a
    company. Paginates up to max_results. Returns case + opinion metadata.

fetch_opinion_text:
    Fetches full plain_text for one opinion from /api/rest/v4/opinions/{id}/.
    Falls back to empty string if plain_text is unavailable.

API:  https://www.courtlistener.com/api/rest/v4/
Auth: Optional token stored as COURTLISTNER_API_KEY in GCP Secret Manager / .env.
      Unauthenticated requests are allowed but rate-limited to ~5k/day.
"""

from __future__ import annotations

import time

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.exceptions import download_error_boundary
from src.utils.logger import get_logger
from src.utils.secrets import get_secret

logger = get_logger(__name__)

_BASE       = "https://www.courtlistener.com/api/rest/v4"
_SEARCH_URL = f"{_BASE}/search/"
_OPINION_URL = f"{_BASE}/opinions/{{opinion_id}}/"
_PAGE_SIZE  = 20
_POLITE_DELAY = 0.25  # seconds between paginated requests


def _headers() -> dict:
    """Return auth headers if COURTLISTNER_API_KEY is available."""
    try:
        token = get_secret("COURTLISTNER_API_KEY")
        return {"Authorization": f"Token {token}"}
    except Exception:
        return {}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.15, min=0.15, max=4))
def _get(url: str, params: dict) -> dict:
    with download_error_boundary(url):
        resp = requests.get(url, params=params, headers=_headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()


def fetch_cases(
    company_name: str,
    filed_after: str | None = None,
    filed_before: str | None = None,
    max_results: int = 100,
) -> list[dict]:
    """
    Search CourtListener for court opinions involving a company.

    Args:
        company_name: Company name to search for (e.g. "Apple Inc").
        filed_after:  ISO date lower bound on date_filed, e.g. "2018-01-01".
        filed_before: ISO date upper bound on date_filed, e.g. "2024-12-31".
        max_results:  Maximum number of cases to return.

    Returns:
        List of case dicts with keys: case_id, case_name, court,
        court_citation, date_filed, docket_number, status, company_name,
        url, cite_count, opinions (list of {id, type, snippet}).
    """
    cases: list[dict] = []
    params: dict = {
        "q":        company_name,
        "type":     "o",
        "order_by": "score desc",
    }
    if filed_after:
        params["filed_after"] = filed_after
    if filed_before:
        params["filed_before"] = filed_before

    page = 1
    while len(cases) < max_results:
        params["page"] = page
        data = _get(_SEARCH_URL, params)
        results = data.get("results", [])
        if not results:
            break

        for item in results:
            cases.append({
                "case_id":        str(item.get("cluster_id", "")),
                "case_name":      item.get("caseName", ""),
                "court":          item.get("court", ""),
                "court_citation": item.get("court_citation_string", ""),
                "date_filed":     item.get("dateFiled"),
                "docket_number":  item.get("docketNumber", ""),
                "status":         item.get("status", ""),
                "company_name":   company_name,
                "url":            f"https://www.courtlistener.com{item.get('absolute_url', '')}",
                "cite_count":     item.get("citeCount", 0),
                "opinions":       [
                    {
                        "id":      op.get("id"),
                        "type":    op.get("type", ""),
                        "snippet": op.get("snippet", ""),
                    }
                    for op in item.get("opinions", [])
                ],
            })

        logger.info(
            "Fetched CourtListener page",
            extra={"company": company_name, "page": page, "page_count": len(results)},
        )

        if not data.get("next"):
            break
        page += 1
        time.sleep(_POLITE_DELAY)

    logger.info(
        "CourtListener fetch complete",
        extra={"company": company_name, "total_cases": len(cases[:max_results])},
    )
    return cases[:max_results]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.15, min=0.15, max=4))
def fetch_opinion_text(opinion_id: int | str) -> str:
    """
    Fetch the full plain text of a single CourtListener opinion.

    Args:
        opinion_id: Integer or string opinion ID from the search results.

    Returns:
        Plain text string, or empty string if unavailable.
    """
    url = _OPINION_URL.format(opinion_id=opinion_id)
    with download_error_boundary(url):
        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
    return (data.get("plain_text") or "").strip()
