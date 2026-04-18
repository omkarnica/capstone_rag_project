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


# ── Loader ────────────────────────────────────────────────────────────────────

import psycopg2

from src.utils.exceptions import DatabaseError

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


# ── Main orchestrator ─────────────────────────────────────────────────────────

from src.utils.exceptions import DownloadError

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
