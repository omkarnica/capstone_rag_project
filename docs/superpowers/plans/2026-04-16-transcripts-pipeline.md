# Transcripts Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fetch Apple's SEC 8-K earnings press releases (CIK 320193) via the 3-step EDGAR flow, extract Exhibit 99.1 text, and bulk load into `transcripts` + `transcript_sections`.

**Architecture:** `fetcher.py` owns all HTTP logic via three public functions matching the 3-step flow (submissions API → filing index → exhibit document). `parser.py` extracts clean text from Exhibit 99.1 HTML. `loader.py` mirrors the xbrl/loader.py COPY pattern and returns an `accession_no → id` map so `main.py` can build FK-correct section records. `main.py` wires them together and skips individual filing failures without aborting the run.

**Tech Stack:** `requests` + `tenacity` (HTTP retry), `beautifulsoup4` + `lxml` (HTML parsing for both index and exhibit), `psycopg2` (PostgreSQL COPY), `src/utils/exceptions.py`, `src/utils/logger.py`

---

## Example URLs — Apple CIK 320193, Q1 FY2024 8-K

| Step | URL |
|------|-----|
| Step 1 — Submissions API | `https://data.sec.gov/submissions/CIK0000320193.json` |
| Step 2 — Filing index | `https://www.sec.gov/Archives/edgar/data/320193/000032019324000006/` |
| Step 3 — Exhibit 99.1 | `https://www.sec.gov/Archives/edgar/data/320193/000032019324000006/aapl-20240201ex991.htm` |

Accession `0000320193-24-000006` → clean `000032019324000006` (strip dashes).

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/transcripts/schema.sql` | Create | DDL for `transcripts` + `transcript_sections` |
| `src/transcripts/__init__.py` | Create | Empty — makes package importable |
| `src/transcripts/fetcher.py` | Create | `fetch_8k_filings`, `fetch_exhibit_url`, `fetch_exhibit_text` + private `_get` |
| `src/transcripts/parser.py` | Create | `extract_exhibit_text` — BS4 clean text from Exhibit 99.1 HTML |
| `src/transcripts/loader.py` | Create | `load_transcripts` → id map, `load_transcript_sections`, `load_transcripts_data` |
| `src/transcripts/main.py` | Create | `run_pipeline` + CLI (`--cik`, `--start`, `--end`) |
| `tests/test_transcripts.py` | Create | All tests for this feature |

---

## Task 1: Schema + Apply to Database

**Files:**
- Create: `src/transcripts/schema.sql`

- [ ] **Step 1: Write the schema file**

```sql
-- src/transcripts/schema.sql

CREATE TABLE IF NOT EXISTS transcripts (
    id               SERIAL PRIMARY KEY,
    cik              INTEGER NOT NULL,
    accession_no     TEXT    NOT NULL,
    company_name     TEXT,
    filed_date       DATE,
    period_of_report DATE,
    form_type        TEXT    DEFAULT '8-K',
    exhibit_url      TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT transcripts_accession_no_key UNIQUE (accession_no)
);

CREATE INDEX IF NOT EXISTS idx_transcripts_cik
    ON transcripts (cik);
CREATE INDEX IF NOT EXISTS idx_transcripts_filed_date
    ON transcripts (filed_date);

CREATE TABLE IF NOT EXISTS transcript_sections (
    id            SERIAL  PRIMARY KEY,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(id),
    section_item  TEXT    NOT NULL,
    section_text  TEXT    NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transcript_sections_transcript_id
    ON transcript_sections (transcript_id);
```

- [ ] **Step 2: Apply schema to the database**

```bash
psql -h localhost -p 5433 -U postgres -d ma_oracle -f src/transcripts/schema.sql
```

Expected output: six lines, each `CREATE TABLE` or `CREATE INDEX`, no errors.

- [ ] **Step 3: Commit**

```bash
git add src/transcripts/schema.sql
git commit -m "feat: add transcripts and transcript_sections schema"
```

---

## Task 2: Install Dependencies + Package Init

**Files:**
- Create: `src/transcripts/__init__.py`

- [ ] **Step 1: Install new dependencies**

```bash
uv add beautifulsoup4 lxml
```

- [ ] **Step 2: Create the empty package init**

```python
# src/transcripts/__init__.py
```

- [ ] **Step 3: Commit**

```bash
git add src/transcripts/__init__.py
git commit -m "chore: add transcripts package init and beautifulsoup4/lxml deps"
```

---

## Task 3: fetcher.py — Three-Step EDGAR Fetch

**Files:**
- Create: `src/transcripts/fetcher.py`
- Create: `tests/test_transcripts.py`

### Design

`_get(url)` — private `@retry(stop=3, wait=exponential(2s→8s), reraise=True)` helper; sleeps 0.15 s before each request for SEC rate limit compliance.

`fetch_8k_filings(cik, start_date, end_date)` — calls Step 1 URL, filters parallel arrays by `form=="8-K"`, `"2.02" in items`, and date range. Returns list of metadata dicts (no `exhibit_url` yet — that's added by `main.py` after Step 2).

`fetch_exhibit_url(cik, accession_no)` — Step 2: fetches the filing index HTML, parses its `<table>` to find the row whose Type cell equals `"EX-99.1"`, returns the full exhibit URL. Returns `None` if no such row exists.

`fetch_exhibit_text(exhibit_url)` — Step 3: fetches and returns raw HTML of the exhibit.

- [ ] **Step 1: Write the failing fetcher tests**

```python
# tests/test_transcripts.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Shared fixtures ───────────────────────────────────────────────────────────

SAMPLE_SUBMISSIONS = {
    "cik": "320193",
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "accessionNumber": [
                "0000320193-24-000006",
                "0000320193-24-000099",
                "0000320193-23-000088",
            ],
            "filingDate": ["2024-02-01", "2024-03-15", "2023-11-02"],
            "reportDate": ["2024-01-27", "2024-03-28", "2023-09-30"],
            "form":       ["8-K", "10-Q", "8-K"],
            "items":      ["2.02,9.01", "", "2.02,9.01"],
        }
    },
}

# Minimal EDGAR filing index HTML with one EX-99.1 row
INDEX_HTML = """
<html><body>
<table class="tableFile">
  <tr>
    <th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th>
  </tr>
  <tr>
    <td>1</td><td>8-K</td>
    <td><a href="/Archives/edgar/data/320193/000032019324000006/aapl-20240201.htm">
        aapl-20240201.htm</a></td>
    <td>8-K</td><td>10 KB</td>
  </tr>
  <tr>
    <td>2</td><td>Exhibit 99.1</td>
    <td><a href="/Archives/edgar/data/320193/000032019324000006/aapl-20240201ex991.htm">
        aapl-20240201ex991.htm</a></td>
    <td>EX-99.1</td><td>180 KB</td>
  </tr>
</table>
</body></html>
"""

INDEX_HTML_NO_EXHIBIT = """
<html><body>
<table class="tableFile">
  <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
  <tr>
    <td>1</td><td>8-K</td>
    <td><a href="/Archives/edgar/data/320193/000032019324000006/aapl-20240201.htm">
        aapl-20240201.htm</a></td>
    <td>8-K</td><td>10 KB</td>
  </tr>
</table>
</body></html>
"""

EXHIBIT_HTML = "<html><body><p>Apple Q1 FY2024 revenue was $119.6 billion.</p></body></html>"


# ── Fetcher: fetch_8k_filings ─────────────────────────────────────────────────

def test_fetch_8k_filings_excludes_non_8k_and_non_202():
    """10-Q filings and 8-Ks without Item 2.02 must be excluded."""
    from src.transcripts.fetcher import fetch_8k_filings

    mock_resp = MagicMock()
    mock_resp.json.return_value = SAMPLE_SUBMISSIONS

    with patch("src.transcripts.fetcher._get", return_value=mock_resp):
        result = fetch_8k_filings(320193, "2020-01-01", "2024-12-31")

    assert len(result) == 2
    assert all(r["form_type"] == "8-K" for r in result)


def test_fetch_8k_filings_respects_date_range():
    """Filings before start_date or after end_date must be excluded."""
    from src.transcripts.fetcher import fetch_8k_filings

    mock_resp = MagicMock()
    mock_resp.json.return_value = SAMPLE_SUBMISSIONS

    with patch("src.transcripts.fetcher._get", return_value=mock_resp):
        result = fetch_8k_filings(320193, "2024-01-01", "2024-12-31")

    assert len(result) == 1
    assert result[0]["accession_no"] == "0000320193-24-000006"


def test_fetch_8k_filings_record_has_required_keys():
    """Each dict must contain the keys main.py will read."""
    from src.transcripts.fetcher import fetch_8k_filings

    mock_resp = MagicMock()
    mock_resp.json.return_value = SAMPLE_SUBMISSIONS

    with patch("src.transcripts.fetcher._get", return_value=mock_resp):
        result = fetch_8k_filings(320193, "2020-01-01", "2024-12-31")

    required = {"cik", "accession_no", "company_name", "filed_date", "period_of_report", "form_type"}
    for r in result:
        assert required.issubset(r.keys())


# ── Fetcher: fetch_exhibit_url ────────────────────────────────────────────────

def test_fetch_exhibit_url_finds_ex991_by_type():
    """Must return the full URL of the EX-99.1 document."""
    from src.transcripts.fetcher import fetch_exhibit_url

    mock_resp = MagicMock()
    mock_resp.text = INDEX_HTML

    with patch("src.transcripts.fetcher._get", return_value=mock_resp):
        result = fetch_exhibit_url(320193, "0000320193-24-000006")

    assert result == (
        "https://www.sec.gov/Archives/edgar/data/320193"
        "/000032019324000006/aapl-20240201ex991.htm"
    )


def test_fetch_exhibit_url_returns_none_when_not_found():
    """Must return None when no EX-99.1 row is in the index."""
    from src.transcripts.fetcher import fetch_exhibit_url

    mock_resp = MagicMock()
    mock_resp.text = INDEX_HTML_NO_EXHIBIT

    with patch("src.transcripts.fetcher._get", return_value=mock_resp):
        result = fetch_exhibit_url(320193, "0000320193-24-000006")

    assert result is None


# ── Fetcher: fetch_exhibit_text ───────────────────────────────────────────────

def test_fetch_exhibit_text_returns_raw_html():
    """Must return the raw HTML string of the exhibit document."""
    from src.transcripts.fetcher import fetch_exhibit_text

    mock_resp = MagicMock()
    mock_resp.text = EXHIBIT_HTML

    with patch("src.transcripts.fetcher._get", return_value=mock_resp):
        result = fetch_exhibit_text(
            "https://www.sec.gov/Archives/edgar/data/320193"
            "/000032019324000006/aapl-20240201ex991.htm"
        )

    assert result == EXHIBIT_HTML
```

- [ ] **Step 2: Run — verify ImportError**

```bash
pytest tests/test_transcripts.py -v -k "fetch"
```

Expected: `ImportError: No module named 'src.transcripts.fetcher'`

- [ ] **Step 3: Write fetcher.py**

```python
# src/transcripts/fetcher.py
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
    index_url = f"{_ARCHIVE_BASE}/{cik}/{accession_clean}/"

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
```

- [ ] **Step 4: Run fetcher tests — verify they pass**

```bash
pytest tests/test_transcripts.py -v -k "fetch"
```

Expected: 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/transcripts/fetcher.py tests/test_transcripts.py
git commit -m "feat: add transcripts fetcher with 3-step EDGAR flow"
```

---

## Task 4: parser.py — Exhibit Text Extraction

**Files:**
- Create: `src/transcripts/parser.py`
- Modify: `tests/test_transcripts.py` (append)

### Design

`extract_exhibit_text(html_content, accession_no)`:
1. Parse HTML with BeautifulSoup + lxml
2. Remove all `<script>` and `<style>` tags in-place
3. Call `soup.get_text(separator="\n")`
4. Strip each line; drop empty lines; collapse runs of 3+ blank lines to 1
5. Return joined text, or `None` if nothing remains after cleaning

- [ ] **Step 1: Append parser tests**

```python
# Append to tests/test_transcripts.py

# ── Parser: extract_exhibit_text ──────────────────────────────────────────────

_EXHIBIT_WITH_CONTENT = """
<html><head>
  <script>alert('remove me');</script>
  <style>.x { color: red; }</style>
</head><body>
  <p>Apple Reports First Quarter Results</p>
  <p>Revenue was $119.6 billion, up 2% year over year.</p>
  <p>   </p>
  <p></p>
  <p>Tim Cook said: "Strong results."</p>
</body></html>
"""

_EXHIBIT_EMPTY = "<html><body>   \n   \n   </body></html>"


def test_extract_exhibit_text_returns_cleaned_text():
    """Must return non-None text containing the meaningful content."""
    from src.transcripts.parser import extract_exhibit_text

    result = extract_exhibit_text(_EXHIBIT_WITH_CONTENT, "0000320193-24-000006")

    assert result is not None
    assert "Apple Reports First Quarter Results" in result
    assert "119.6 billion" in result
    assert "Tim Cook" in result


def test_extract_exhibit_text_removes_script_and_style():
    """Script and style tag contents must not appear in the output."""
    from src.transcripts.parser import extract_exhibit_text

    result = extract_exhibit_text(_EXHIBIT_WITH_CONTENT, "0000320193-24-000006")

    assert result is not None
    assert "alert(" not in result
    assert "color: red" not in result


def test_extract_exhibit_text_returns_none_when_empty():
    """Returns None when the document has no meaningful text after cleaning."""
    from src.transcripts.parser import extract_exhibit_text

    result = extract_exhibit_text(_EXHIBIT_EMPTY, "0000320193-24-000006")

    assert result is None
```

- [ ] **Step 2: Run — verify ImportError**

```bash
pytest tests/test_transcripts.py -v -k "extract_exhibit"
```

Expected: `ImportError: No module named 'src.transcripts.parser'`

- [ ] **Step 3: Write parser.py**

```python
# src/transcripts/parser.py
"""
Extracts clean plain text from SEC Exhibit 99.1 HTML documents.

Apple's 8-K Exhibit 99.1 is the earnings press release. It is an HTML
document containing financial highlights, management quotes, and detailed
segment results. This module strips structural HTML noise (scripts, styles,
excessive blank lines) and returns clean plain text ready for storage and
downstream embedding.

Returns None if the document is empty or contains only whitespace after
cleaning (e.g. placeholder or redirect pages).
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from src.utils.logger import get_logger

logger = get_logger(__name__)

_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def extract_exhibit_text(html_content: str, accession_no: str) -> str | None:
    """
    Extract clean plain text from an Exhibit 99.1 HTML document.

    Processing steps:
    1. Parse HTML with BeautifulSoup + lxml.
    2. Remove all <script> and <style> tags in-place.
    3. Extract plain text with newline separators.
    4. Strip each line; discard empty lines; collapse runs of 3+
       consecutive blank lines to a single blank line.
    5. Return the joined text, or None if nothing remains.

    Args:
        html_content: Raw HTML string of the Exhibit 99.1 document.
        accession_no: Used in log messages for traceability.

    Returns:
        Cleaned plain text string, or None if the document is empty
        after cleaning.
    """
    soup = BeautifulSoup(html_content, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    raw_text = soup.get_text(separator="\n")
    lines = [line.strip() for line in raw_text.splitlines()]
    text = "\n".join(lines)
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text).strip()

    if not text:
        logger.info(
            "Exhibit text is empty after cleaning",
            extra={"accession_no": accession_no},
        )
        return None

    logger.info(
        "Extracted exhibit text",
        extra={"accession_no": accession_no, "char_count": len(text)},
    )
    return text
```

- [ ] **Step 4: Run parser tests — verify they pass**

```bash
pytest tests/test_transcripts.py -v -k "extract_exhibit"
```

Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/transcripts/parser.py tests/test_transcripts.py
git commit -m "feat: add transcripts parser for Exhibit 99.1 text extraction"
```

---

## Task 5: loader.py — Bulk Load

**Files:**
- Create: `src/transcripts/loader.py`
- Modify: `tests/test_transcripts.py` (append)

### Design

Mirrors `xbrl/loader.py`. `load_transcripts` stages via COPY into a temp table, upserts with `ON CONFLICT (accession_no) DO NOTHING`, then queries back IDs for all passed accession numbers — returning `{accession_no: id}` regardless of whether rows were newly inserted or already existed. `main.py` uses this map to build FK-correct section records.

- [ ] **Step 1: Append loader tests**

```python
# Append to tests/test_transcripts.py

import psycopg2

from src.utils.exceptions import DatabaseError

# ── Loader ────────────────────────────────────────────────────────────────────

_TRANSCRIPT_RECORDS = [
    {
        "cik": 320193,
        "accession_no": "0000320193-24-000006",
        "company_name": "Apple Inc.",
        "filed_date": "2024-02-01",
        "period_of_report": "2024-01-27",
        "form_type": "8-K",
        "exhibit_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000006/aapl-20240201ex991.htm",
    }
]

_SECTION_RECORDS = [
    {
        "transcript_id": 1,
        "section_item": "EX-99.1",
        "section_text": "Apple Q1 FY2024 revenue was $119.6 billion.",
    }
]


def test_load_transcripts_returns_empty_dict_for_empty_input():
    """No DB calls should be made when records is empty."""
    from src.transcripts.loader import load_transcripts

    mock_conn = MagicMock()
    result = load_transcripts(mock_conn, [])

    assert result == {}
    mock_conn.cursor.assert_not_called()


def test_load_transcripts_raises_database_error_on_psycopg2_failure():
    """A psycopg2.Error during the CREATE TEMP TABLE must raise DatabaseError."""
    from src.transcripts.loader import load_transcripts

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.execute.side_effect = psycopg2.OperationalError("connection lost")
    mock_conn.cursor.return_value = mock_cursor

    with pytest.raises(DatabaseError):
        load_transcripts(mock_conn, _TRANSCRIPT_RECORDS)


def test_load_transcript_sections_returns_zero_for_empty_input():
    """No DB calls should be made when sections is empty."""
    from src.transcripts.loader import load_transcript_sections

    mock_conn = MagicMock()
    result = load_transcript_sections(mock_conn, [])

    assert result == 0
    mock_conn.cursor.assert_not_called()


def test_load_transcripts_data_rolls_back_on_error():
    """A psycopg2 failure at any point must trigger rollback."""
    from src.transcripts.loader import load_transcripts_data

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.execute.side_effect = psycopg2.ProgrammingError("relation does not exist")
    mock_conn.cursor.return_value = mock_cursor

    with pytest.raises(DatabaseError):
        load_transcripts_data(
            _TRANSCRIPT_RECORDS,
            _SECTION_RECORDS,
            conn=mock_conn,
        )

    mock_conn.rollback.assert_called_once()
```

- [ ] **Step 2: Run — verify ImportError**

```bash
pytest tests/test_transcripts.py -v -k "load_transcript"
```

Expected: `ImportError: No module named 'src.transcripts.loader'`

- [ ] **Step 3: Write loader.py**

```python
# src/transcripts/loader.py
"""
Bulk loads transcript records into PostgreSQL.

load_transcripts:
    Stages records into a temp table via COPY, upserts into `transcripts`
    with ON CONFLICT (accession_no) DO NOTHING, then queries back IDs for
    all passed accession numbers (both newly inserted and pre-existing).
    Returns {accession_no: transcript_id} — main.py uses this map to build
    FK-correct section records.

load_transcript_sections:
    COPY directly into `transcript_sections`. The FK to `transcripts`
    guarantees referential integrity; no unique conflict key is needed.

load_transcripts_data:
    Transaction wrapper — calls both loaders in sequence, commits on
    success, rolls back on any exception.
"""

from __future__ import annotations

from io import StringIO
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extensions import connection as PgConnection

from src.utils.exceptions import db_error_boundary
from src.utils.logger import get_logger
from xbrl.loader import get_connection

logger = get_logger(__name__)

_TRANSCRIPT_COLUMNS = [
    "cik", "accession_no", "company_name",
    "filed_date", "period_of_report", "form_type", "exhibit_url",
]


def load_transcripts(conn: PgConnection, records: list[dict]) -> dict[str, int]:
    """
    Bulk insert transcript metadata and return the accession_no → id mapping.

    Uses a temp staging table + COPY to avoid row-by-row inserts. Upserts
    with ON CONFLICT (accession_no) DO NOTHING so reruns are safe. After
    the upsert, queries transcript IDs for all passed accession numbers —
    this covers both newly inserted rows and rows that already existed.

    Args:
        conn:    Active psycopg2 connection. Caller manages lifecycle.
        records: List of dicts; each must contain all _TRANSCRIPT_COLUMNS keys.

    Returns:
        Dict mapping accession_no → transcript id for every record passed.

    Raises:
        DatabaseError: wraps any psycopg2.Error.
    """
    if not records:
        return {}

    df = pd.DataFrame(records)[_TRANSCRIPT_COLUMNS]

    with db_error_boundary("CREATE TEMP TABLE", table="transcripts_staging"):
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TEMP TABLE transcripts_staging (
                    cik              INTEGER,
                    accession_no     TEXT,
                    company_name     TEXT,
                    filed_date       DATE,
                    period_of_report DATE,
                    form_type        TEXT,
                    exhibit_url      TEXT
                ) ON COMMIT DROP
            """)

    buffer = StringIO()
    df.to_csv(buffer, sep="\t", header=False, index=False, na_rep="\\N")
    buffer.seek(0)

    columns_sql = ", ".join(_TRANSCRIPT_COLUMNS)
    copy_sql = (
        f"COPY transcripts_staging ({columns_sql}) "
        "FROM STDIN WITH (FORMAT TEXT, DELIMITER E'\\t', NULL '\\N')"
    )

    with db_error_boundary("COPY", table="transcripts_staging"):
        with conn.cursor() as cur:
            cur.copy_expert(copy_sql, buffer)

    with db_error_boundary("INSERT", table="transcripts"):
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO transcripts ({columns_sql})
                SELECT {columns_sql} FROM transcripts_staging
                ON CONFLICT (accession_no) DO NOTHING
            """)
            inserted = cur.rowcount

    accession_nos = [r["accession_no"] for r in records]
    with db_error_boundary("SELECT", table="transcripts"):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, accession_no FROM transcripts WHERE accession_no = ANY(%s)",
                (accession_nos,),
            )
            id_map: dict[str, int] = {row[1]: row[0] for row in cur.fetchall()}

    logger.info(
        "Loaded transcripts",
        extra={"inserted": inserted, "id_map_size": len(id_map)},
    )
    return id_map


def load_transcript_sections(conn: PgConnection, sections: list[dict]) -> int:
    """
    Bulk insert transcript sections into `transcript_sections` via COPY.

    Args:
        conn:     Active psycopg2 connection. Caller manages lifecycle.
        sections: List of dicts with keys: transcript_id, section_item,
                  section_text.

    Returns:
        Number of rows loaded.

    Raises:
        DatabaseError: wraps any psycopg2.Error.
    """
    if not sections:
        return 0

    columns = ["transcript_id", "section_item", "section_text"]
    df = pd.DataFrame(sections)[columns]

    buffer = StringIO()
    df.to_csv(buffer, sep="\t", header=False, index=False, na_rep="\\N")
    buffer.seek(0)

    columns_sql = ", ".join(columns)
    copy_sql = (
        f"COPY transcript_sections ({columns_sql}) "
        "FROM STDIN WITH (FORMAT TEXT, DELIMITER E'\\t', NULL '\\N')"
    )

    with db_error_boundary("COPY", table="transcript_sections"):
        with conn.cursor() as cur:
            cur.copy_expert(copy_sql, buffer)

    inserted = len(df)
    logger.info(
        "Loaded transcript_sections",
        extra={"rows_inserted": inserted},
    )
    return inserted


def load_transcripts_data(
    transcript_records: list[dict],
    section_records: list[dict],
    conn: Optional[PgConnection] = None,
) -> tuple[int, int]:
    """
    Load transcripts and sections in a single transaction.

    Args:
        transcript_records: Passed to load_transcripts.
        section_records:    Passed to load_transcript_sections.
        conn:               Optional existing connection; creates one if None.

    Returns:
        (transcripts_id_map_size, sections_inserted)

    Raises:
        DatabaseError: rolls back and re-raises on any psycopg2 failure.
    """
    owns_connection = conn is None
    if conn is None:
        conn = get_connection()

    try:
        id_map = load_transcripts(conn, transcript_records)
        sections_inserted = load_transcript_sections(conn, section_records)
        conn.commit()
        return len(id_map), sections_inserted
    except Exception:
        conn.rollback()
        raise
    finally:
        if owns_connection:
            conn.close()
```

- [ ] **Step 4: Run loader tests — verify they pass**

```bash
pytest tests/test_transcripts.py -v -k "load_transcript"
```

Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/transcripts/loader.py tests/test_transcripts.py
git commit -m "feat: add transcripts loader with COPY bulk load and accession→id map"
```

---

## Task 6: main.py — Orchestrator + CLI

**Files:**
- Create: `src/transcripts/main.py`
- Modify: `tests/test_transcripts.py` (append)

### Design

`run_pipeline` iterates over filings, calling `fetch_exhibit_url` then `fetch_exhibit_text` then `extract_exhibit_text` for each. Any `MAOracleError` or `None` return (no exhibit found, empty text) skips that filing with a warning and continues. After the loop, `load_transcripts` returns the `id_map`, which is used to build section records before `load_transcript_sections`.

- [ ] **Step 1: Append main tests**

```python
# Append to tests/test_transcripts.py

from src.utils.exceptions import DownloadError

# ── Main orchestrator ─────────────────────────────────────────────────────────

_FILING_META = {
    "cik": 320193,
    "accession_no": "0000320193-24-000006",
    "company_name": "Apple Inc.",
    "filed_date": "2024-02-01",
    "period_of_report": "2024-01-27",
    "form_type": "8-K",
}

_EXHIBIT_URL = (
    "https://www.sec.gov/Archives/edgar/data/320193"
    "/000032019324000006/aapl-20240201ex991.htm"
)


def test_run_pipeline_skips_filing_when_no_exhibit_url():
    """When fetch_exhibit_url returns None, the filing must be skipped."""
    from src.transcripts.main import run_pipeline

    with (
        patch("src.transcripts.main.fetcher.fetch_8k_filings", return_value=[_FILING_META]),
        patch("src.transcripts.main.fetcher.fetch_exhibit_url", return_value=None),
        patch("src.transcripts.main.loader.load_transcripts", return_value={}),
        patch("src.transcripts.main.loader.load_transcript_sections", return_value=0),
        patch("src.transcripts.main.get_connection") as mock_gc,
    ):
        mock_gc.return_value = MagicMock()
        t, s = run_pipeline(320193, "2024-01-01", "2024-12-31")

    assert t == 0
    assert s == 0


def test_run_pipeline_skips_filing_on_download_error():
    """A DownloadError on fetch_exhibit_text must skip the filing, not abort."""
    from src.transcripts.main import run_pipeline

    with (
        patch("src.transcripts.main.fetcher.fetch_8k_filings", return_value=[_FILING_META]),
        patch("src.transcripts.main.fetcher.fetch_exhibit_url", return_value=_EXHIBIT_URL),
        patch(
            "src.transcripts.main.fetcher.fetch_exhibit_text",
            side_effect=DownloadError("timeout", url=_EXHIBIT_URL),
        ),
        patch("src.transcripts.main.loader.load_transcripts", return_value={}),
        patch("src.transcripts.main.loader.load_transcript_sections", return_value=0),
        patch("src.transcripts.main.get_connection") as mock_gc,
    ):
        mock_gc.return_value = MagicMock()
        t, s = run_pipeline(320193, "2024-01-01", "2024-12-31")

    assert t == 0
    assert s == 0


def test_run_pipeline_builds_section_records_with_correct_transcript_id():
    """Section records must use the transcript_id from the id_map."""
    from src.transcripts.main import run_pipeline

    id_map = {"0000320193-24-000006": 42}
    captured: list[dict] = []

    def fake_load_sections(conn, sections):
        captured.extend(sections)
        return len(sections)

    with (
        patch("src.transcripts.main.fetcher.fetch_8k_filings", return_value=[_FILING_META]),
        patch("src.transcripts.main.fetcher.fetch_exhibit_url", return_value=_EXHIBIT_URL),
        patch("src.transcripts.main.fetcher.fetch_exhibit_text", return_value="<html/>"),
        patch("src.transcripts.main.parser.extract_exhibit_text", return_value="Earnings text."),
        patch("src.transcripts.main.loader.load_transcripts", return_value=id_map),
        patch("src.transcripts.main.loader.load_transcript_sections", side_effect=fake_load_sections),
        patch("src.transcripts.main.get_connection") as mock_gc,
    ):
        mock_gc.return_value = MagicMock()
        run_pipeline(320193, "2024-01-01", "2024-12-31")

    assert len(captured) == 1
    assert captured[0]["transcript_id"] == 42
    assert captured[0]["section_item"] == "EX-99.1"
    assert captured[0]["section_text"] == "Earnings text."


def test_run_pipeline_omits_section_when_extract_returns_none():
    """When extract_exhibit_text returns None, no section record is created."""
    from src.transcripts.main import run_pipeline

    id_map = {"0000320193-24-000006": 42}
    captured: list[dict] = []

    def fake_load_sections(conn, sections):
        captured.extend(sections)
        return len(sections)

    with (
        patch("src.transcripts.main.fetcher.fetch_8k_filings", return_value=[_FILING_META]),
        patch("src.transcripts.main.fetcher.fetch_exhibit_url", return_value=_EXHIBIT_URL),
        patch("src.transcripts.main.fetcher.fetch_exhibit_text", return_value="<html/>"),
        patch("src.transcripts.main.parser.extract_exhibit_text", return_value=None),
        patch("src.transcripts.main.loader.load_transcripts", return_value=id_map),
        patch("src.transcripts.main.loader.load_transcript_sections", side_effect=fake_load_sections),
        patch("src.transcripts.main.get_connection") as mock_gc,
    ):
        mock_gc.return_value = MagicMock()
        run_pipeline(320193, "2024-01-01", "2024-12-31")

    assert captured == []
```

- [ ] **Step 2: Run — verify ImportError**

```bash
pytest tests/test_transcripts.py -v -k "pipeline"
```

Expected: `ImportError: No module named 'src.transcripts.main'`

- [ ] **Step 3: Write main.py**

```python
# src/transcripts/main.py
"""
Orchestrator for the transcripts ingestion pipeline.

Ingestion flow for each Apple 8-K Item 2.02 filing:
1) fetch_8k_filings  — EDGAR submissions API → list of filing metadata
2) fetch_exhibit_url — filing index HTML → Exhibit 99.1 document URL
3) fetch_exhibit_text — Exhibit 99.1 HTML → raw HTML string
4) extract_exhibit_text — raw HTML → clean plain text
5) load_transcripts  — COPY into `transcripts`, returns accession→id map
6) load_transcript_sections — COPY into `transcript_sections` using FK ids

Individual filing failures (no exhibit found, empty text, DownloadError)
are logged and skipped — a single bad filing does not abort the run.

CLI:
    python -m src.transcripts.main
    python -m src.transcripts.main --cik 320193 --start 2020-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse

from src.transcripts import fetcher, loader, parser
from src.utils.exceptions import MAOracleError
from src.utils.logger import get_logger
from xbrl.loader import get_connection

logger = get_logger(__name__)

_DEFAULT_CIK = 320193
_DEFAULT_START = "2020-01-01"
_DEFAULT_END = "2025-12-31"


def run_pipeline(cik: int, start_date: str, end_date: str) -> tuple[int, int]:
    """
    Run the full transcripts ingestion for one CIK and date range.

    Fetches 8-K Item 2.02 filings from EDGAR, extracts Exhibit 99.1 text
    for each, and bulk loads results into `transcripts` and
    `transcript_sections`. Skips any filing that fails or has no exhibit.

    Args:
        cik:        SEC CIK, e.g. 320193 for Apple.
        start_date: Inclusive lower bound ISO date, e.g. "2020-01-01".
        end_date:   Inclusive upper bound ISO date, e.g. "2025-12-31".

    Returns:
        (transcripts_loaded, sections_loaded)
    """
    logger.info(
        "Transcripts pipeline start",
        extra={"cik": cik, "start_date": start_date, "end_date": end_date},
    )

    filings = fetcher.fetch_8k_filings(cik, start_date, end_date)

    transcript_records: list[dict] = []
    raw_sections: list[tuple[str, str]] = []  # (accession_no, section_text)

    for filing in filings:
        accession_no = filing["accession_no"]

        try:
            # Step 2: get exhibit URL from filing index
            exhibit_url = fetcher.fetch_exhibit_url(cik, accession_no)
            if exhibit_url is None:
                logger.warning(
                    "No Exhibit 99.1 found — skipping filing",
                    extra={"accession_no": accession_no},
                )
                continue

            # Step 3: fetch exhibit HTML
            html = fetcher.fetch_exhibit_text(exhibit_url)

            # Step 4: extract clean text
            text = parser.extract_exhibit_text(html, accession_no)
            if text is None:
                logger.warning(
                    "Exhibit text empty after cleaning — skipping filing",
                    extra={"accession_no": accession_no},
                )
                continue

        except MAOracleError as exc:
            logger.warning(
                "Error fetching filing — skipping",
                extra={"accession_no": accession_no, "error": str(exc)},
            )
            continue

        transcript_records.append({
            "cik": filing["cik"],
            "accession_no": filing["accession_no"],
            "company_name": filing["company_name"],
            "filed_date": filing["filed_date"],
            "period_of_report": filing["period_of_report"],
            "form_type": filing["form_type"],
            "exhibit_url": exhibit_url,
        })
        raw_sections.append((accession_no, text))

    if not transcript_records:
        logger.info("No transcript records to load", extra={"cik": cik})
        return 0, 0

    conn = get_connection()
    try:
        id_map = loader.load_transcripts(conn, transcript_records)

        section_records = [
            {
                "transcript_id": id_map[acc_no],
                "section_item": "EX-99.1",
                "section_text": text,
            }
            for acc_no, text in raw_sections
            if acc_no in id_map
        ]

        sections_loaded = loader.load_transcript_sections(conn, section_records)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    logger.info(
        "Transcripts pipeline complete",
        extra={
            "cik": cik,
            "transcripts_loaded": len(id_map),
            "sections_loaded": sections_loaded,
        },
    )
    return len(id_map), sections_loaded


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Load SEC 8-K Exhibit 99.1 earnings transcripts into PostgreSQL."
    )
    p.add_argument(
        "--cik", type=int, default=_DEFAULT_CIK,
        help="SEC CIK (default: 320193 — Apple Inc.)",
    )
    p.add_argument(
        "--start", default=_DEFAULT_START,
        help="Start date YYYY-MM-DD (default: 2020-01-01)",
    )
    p.add_argument(
        "--end", default=_DEFAULT_END,
        help="End date YYYY-MM-DD (default: 2025-12-31)",
    )
    return p


def main() -> None:
    """CLI entry point."""
    args = _build_arg_parser().parse_args()
    transcripts_loaded, sections_loaded = run_pipeline(args.cik, args.start, args.end)
    logger.info(
        "Run complete",
        extra={"transcripts_loaded": transcripts_loaded, "sections_loaded": sections_loaded},
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run full test suite — verify all pass**

```bash
pytest tests/test_transcripts.py -v
```

Expected: 18 tests PASSED.

- [ ] **Step 5: Run regression check**

```bash
pytest tests/ -v
```

Expected: all pre-existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/transcripts/main.py tests/test_transcripts.py
git commit -m "feat: add transcripts orchestrator and CLI"
```

---

## Self-Review

**Spec coverage:**
- ✅ 3-step fetch flow: submissions API → filing index (fetch_exhibit_url) → Exhibit 99.1
- ✅ Apple 8-K Item 2.02 only says "see Exhibit 99.1" — content fetched from exhibit, not main doc
- ✅ `exhibit_url` column (not `raw_url`)
- ✅ `section_item = "EX-99.1"` (not "99.1")
- ✅ New tables have `created_at TIMESTAMPTZ DEFAULT NOW()`
- ✅ All HTTP: tenacity 3 attempts, exponential backoff 2s→8s, 0.15s sleep
- ✅ All HTTP wrapped in `download_error_boundary`
- ✅ All DB ops wrapped in `db_error_boundary`
- ✅ `ON CONFLICT (accession_no) DO NOTHING` — reruns safe
- ✅ Bulk COPY not row-by-row INSERT
- ✅ `get_connection()` from `xbrl.loader`
- ✅ No `print()` — all `get_logger(__name__)`
- ✅ Docstrings on every module and function

**Type consistency:**
- `fetch_8k_filings` returns dicts without `exhibit_url` ← `main.py` adds it after `fetch_exhibit_url` ✅
- `load_transcripts` returns `dict[str, int]` ← consumed as `id_map` in `main.py` ✅
- `run_pipeline` returns `tuple[int, int]` ✅
- `section_item` is `"EX-99.1"` in both test assertions and `main.py` ✅
