"""
Fetches SEC EDGAR 8-K earnings filings for a given CIK via a 3-step flow.

Step 1 — fetch_8k_filings:
    Calls the EDGAR submissions API and returns metadata for 8-K filings
    that contain Item 2.02 within the requested date range.

Step 2 — fetch_exhibit_url:
    Fetches the filing index page for a single accession number and parses
    the document table to find the Exhibit 99.1 URL. Apple's Item 2.02
    section contains only "see Exhibit 99.1" — the actual earnings text
    lives in that exhibit.

Step 3 — fetch_exhibit_text:
    Fetches and returns the raw HTML of the Exhibit 99.1 document.
    parser.py converts it to clean text.

All HTTP calls go through _get, a private helper with tenacity retry
(3 attempts, exponential backoff 2s→8s). A 0.15 s sleep before each
request keeps traffic within SEC's 10 req/s limit.
download_error_boundary translates exhausted requests exceptions to
DownloadError.
"""

from __future__ import annotations

import time
from datetime import date

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.exceptions import download_error_boundary
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
_SEC_BASE = "https://www.sec.gov"
_HEADERS = {"User-Agent": "karnicajain.ai@gmail.com"}
_REQUEST_DELAY_S = 0.15  # SEC rate limit: 10 req/s


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _get(url: str) -> requests.Response:
    """
    HTTP GET with tenacity retry (3 attempts, exponential backoff 2s→8s).

    Sleeps 0.15 s before each attempt to stay within SEC's rate limit.
    Re-raises the original requests exception on exhaustion so the
    caller's download_error_boundary can translate it to DownloadError.
    """
    time.sleep(_REQUEST_DELAY_S)
    response = requests.get(url, headers=_HEADERS, timeout=30)
    response.raise_for_status()
    return response


def fetch_8k_filings(cik: int, start_date: str, end_date: str) -> list[dict]:
    """
    Return metadata for 8-K Item 2.02 filings from the EDGAR submissions API.

    Reads the parallel arrays in filings.recent and filters to rows where
    form == "8-K", "2.02" appears in the items field, and filingDate falls
    within the requested range.

    Args:
        cik:        SEC Central Index Key, e.g. 320193 for Apple Inc.
        start_date: Inclusive lower bound, ISO format "YYYY-MM-DD".
        end_date:   Inclusive upper bound, ISO format "YYYY-MM-DD".

    Returns:
        List of dicts with keys: cik, accession_no, company_name,
        filed_date, period_of_report, form_type.
        Note: exhibit_url is NOT included — main.py adds it after Step 2.

    Raises:
        DownloadError: if the submissions API call fails after 3 retries.
    """
    url = f"{_SUBMISSIONS_BASE}/CIK{cik:010d}.json"

    with download_error_boundary(url):
        response = _get(url)

    data = response.json()
    company_name: str = data.get("name", "")
    recent: dict = data["filings"]["recent"]

    forms = recent.get("form", [])
    items_list = recent.get("items", [""] * len(forms))
    accessions = recent.get("accessionNumber", [])
    filed_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    filings: list[dict] = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        if "2.02" not in items_list[i]:
            continue
        filed = date.fromisoformat(filed_dates[i])
        if not (start <= filed <= end):
            continue

        filings.append({
            "cik": cik,
            "accession_no": accessions[i],
            "company_name": company_name,
            "filed_date": filed_dates[i],
            "period_of_report": report_dates[i] or None,
            "form_type": form,
        })

    logger.info(
        "Fetched 8-K Item 2.02 filings",
        extra={"cik": cik, "count": len(filings)},
    )
    return filings


def fetch_exhibit_url(cik: int, accession_no: str) -> str | None:
    """
    Fetch the filing index page and return the URL of the Exhibit 99.1 document.

    Parses the EDGAR filing index HTML table and matches rows where the
    Type column equals "EX-99.1". Returns None if no such row exists.

    Args:
        cik:          SEC CIK.
        accession_no: Accession number with dashes, e.g. "0000320193-24-000006".

    Returns:
        Full HTTPS URL of the Exhibit 99.1 document, or None if not found.

    Raises:
        DownloadError: if the index page fetch fails after 3 retries.
    """
    accession_clean = accession_no.replace("-", "")
    index_url = f"{_ARCHIVE_BASE}/{cik}/{accession_clean}/{accession_no}-index.htm"

    with download_error_boundary(index_url):
        response = _get(index_url)

    soup = BeautifulSoup(response.text, "lxml")

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        type_text = cells[3].get_text(strip=True)
        if type_text != "EX-99.1":
            continue

        link = cells[2].find("a")
        if not link or not link.get("href"):
            continue

        href = link["href"]
        full_url = f"{_SEC_BASE}{href}" if href.startswith("/") else href

        logger.info(
            "Found Exhibit 99.1",
            extra={"accession_no": accession_no, "exhibit_url": full_url},
        )
        return full_url

    logger.info(
        "Exhibit 99.1 not found in filing index",
        extra={"accession_no": accession_no},
    )
    return None


def fetch_exhibit_text(exhibit_url: str) -> str:
    """
    Fetch the raw HTML of an Exhibit 99.1 document.

    Args:
        exhibit_url: Full HTTPS URL returned by fetch_exhibit_url.

    Returns:
        Raw HTML string. parser.extract_exhibit_text converts this to
        clean plain text.

    Raises:
        DownloadError: if the fetch fails after 3 retries.
    """
    with download_error_boundary(exhibit_url):
        response = _get(exhibit_url)
    return response.text
