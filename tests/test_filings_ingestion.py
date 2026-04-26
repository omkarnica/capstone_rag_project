from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


INDEX_JSON_WITH_EX21 = {
    "directory": {
        "item": [
            {"name": "aapl-20241031x10k.htm", "type": "10-K"},
            {"name": "aapl-ex21_1.htm", "type": "EX-21.1"},
        ]
    }
}


INDEX_JSON_WITHOUT_EX21 = {
    "directory": {
        "item": [
            {"name": "aapl-20241031x10k.htm", "type": "10-K"},
            {"name": "aapl-ex23.htm", "type": "EX-23"},
        ]
    }
}


EXHIBIT_21_HTML = """
<html>
  <body>
    <h1>Exhibit 21.1</h1>
    <p>List of Subsidiaries</p>
    <table>
      <tr><th>Name of Subsidiary</th><th>Jurisdiction</th></tr>
      <tr><td>Apple Operations International Limited</td><td>Ireland</td></tr>
      <tr><td>Apple Sales International Limited</td><td>Ireland</td></tr>
      <tr><td>Apple India Private Limited</td><td>India</td></tr>
      <tr><td>Apple India Private Limited</td><td>India</td></tr>
    </table>
  </body>
</html>
"""


DEF14A_HTML = """
<html>
  <body>
    <table>
      <tr><th>Director</th><th>Position</th></tr>
      <tr><td>Satya Nadella</td><td>Chairman & CEO</td></tr>
      <tr><td>Amy Hood</td><td>Director</td></tr>
      <tr><td>Satya Nadella</td><td>Chairman & CEO</td></tr>
    </table>
  </body>
</html>
"""


DEF14A_NOISY_HTML = """
<html>
  <body>
    <table>
      <tr><td>2025 Proxy Statement</td><td>10</td></tr>
      <tr><td>Audit and Finance Committee Report</td><td>94</td></tr>
    </table>
    <h2>Nominees for Election</h2>
    <table>
      <tr><th>Name</th><th>Occupation</th></tr>
      <tr><td>Art Levinson Chair</td><td>Founder and CEO, Calico</td></tr>
      <tr><td>Tim Cook</td><td>CEO, Apple</td></tr>
      <tr><td>Andrea Jung</td><td>President and CEO, Grameen America</td></tr>
    </table>
    <table>
      <tr><th>Name</th><th>Fees Earned or Paid in Cash ($)</th></tr>
      <tr><td>Tim Cook</td><td>100,000</td></tr>
    </table>
  </body>
</html>
"""


SUBMISSIONS_WITH_CROSS_YEAR_FILING = {
    "filings": {
        "recent": {
            "form": ["10-K"],
            "filingDate": ["2025-11-01"],
            "reportDate": ["2024-09-28"],
            "accessionNumber": ["0000320193-25-000123"],
            "primaryDocument": ["aapl-20250928x10k.htm"],
        }
    }
}


def test_find_exhibit_21_filename_returns_first_match() -> None:
    from src.filings.ingestion import _find_exhibit_21_filename

    result = _find_exhibit_21_filename(INDEX_JSON_WITH_EX21)

    assert result == "aapl-ex21_1.htm"


def test_filter_def14a_filings_keeps_only_eligible_proxy_years() -> None:
    from src.filings.ingestion import _filter_def14a_filings

    filings = [
        {"form": "DEF 14A", "year": 2025},
        {"form": "DEF 14A", "year": 2020},
        {"form": "10-K", "year": 2025},
        {"form": "DEF 14A", "year": None},
        {"form": "DEF 14A", "year": 2023},
    ]

    result = _filter_def14a_filings(filings)

    assert result == [
        {"form": "DEF 14A", "year": 2025},
        {"form": "DEF 14A", "year": 2023},
    ]


def test_get_filings_uses_report_date_year_for_saved_metadata() -> None:
    from src.filings.ingestion import get_filings

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SUBMISSIONS_WITH_CROSS_YEAR_FILING

    with patch("src.filings.ingestion.requests.get", return_value=mock_resp):
        result = get_filings("0000320193")

    assert result[0]["year"] == 2024
    assert result[0]["filing_date"] == "2025-11-01"
    assert result[0]["report_date"] == "2024-09-28"


def test_find_exhibit_21_filename_returns_none_when_missing() -> None:
    from src.filings.ingestion import _find_exhibit_21_filename

    result = _find_exhibit_21_filename(INDEX_JSON_WITHOUT_EX21)

    assert result is None


def test_extract_subsidiaries_from_exhibit_html_returns_unique_names() -> None:
    from src.filings.ingestion import _extract_subsidiaries_from_exhibit_html

    result = _extract_subsidiaries_from_exhibit_html(EXHIBIT_21_HTML)

    assert result == [
        "Apple Operations International Limited",
        "Apple Sales International Limited",
        "Apple India Private Limited",
    ]


def test_extract_board_members_from_def14a_html_returns_unique_name_title_pairs() -> None:
    from src.filings.ingestion import _extract_board_members_from_def14a_html

    result = _extract_board_members_from_def14a_html(DEF14A_HTML)

    assert result == [
        {"name": "Satya Nadella", "title": "Chairman & CEO"},
        {"name": "Amy Hood", "title": "Director"},
    ]


def test_extract_board_members_from_def14a_html_skips_toc_and_fee_tables() -> None:
    from src.filings.ingestion import _extract_board_members_from_def14a_html

    result = _extract_board_members_from_def14a_html(DEF14A_NOISY_HTML)

    assert result == [
        {"name": "Art Levinson Chair", "title": "Founder and CEO, Calico"},
        {"name": "Tim Cook", "title": "CEO, Apple"},
        {"name": "Andrea Jung", "title": "President and CEO, Grameen America"},
    ]


def test_collect_subsidiaries_by_year_writes_company_json(tmp_path: Path) -> None:
    from src.filings.ingestion import collect_subsidiaries_by_year

    filings = [
        {"year": 2025, "accession": "000032019325000073"},
        {"year": 2024, "accession": "000032019324000123"},
    ]

    with (
        patch("src.filings.ingestion._lookup_exhibit_21_filename", side_effect=["aapl-ex21-2025.htm", "aapl-ex21-2024.htm"]),
        patch("src.filings.ingestion._fetch_exhibit_21_html", return_value=EXHIBIT_21_HTML),
    ):
        output_path = collect_subsidiaries_by_year(
            cik="0000320193",
            ticker="AAPL",
            company_title="Apple Inc.",
            form_type="10-K",
            filings=filings,
            data_dir=tmp_path,
        )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path == tmp_path / "aapl_10-k_subsidiaries.json"
    assert saved["ticker"] == "AAPL"
    assert saved["form_type"] == "10-K"
    assert [entry["year"] for entry in saved["subsidiaries_by_year"]] == [2025, 2024]
    assert saved["subsidiaries_by_year"][0]["subsidiaries"][0] == "Apple Operations International Limited"


def test_collect_board_members_by_year_writes_company_json(tmp_path: Path) -> None:
    from src.filings.ingestion import collect_board_members_by_year

    filings = [
        {"year": 2025, "filing_url": "https://sec.gov/2025-def14a"},
        {"year": 2024, "filing_url": "https://sec.gov/2024-def14a"},
    ]

    with patch("src.filings.ingestion._fetch_proxy_filing_html", return_value=DEF14A_HTML):
        output_path = collect_board_members_by_year(
            ticker="MSFT",
            company_title="Microsoft Corporation",
            filings=filings,
            data_dir=tmp_path,
        )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path == tmp_path / "msft_def14a_board_members.json"
    assert saved["ticker"] == "MSFT"
    assert saved["form_type"] == "DEF 14A"
    assert saved["board_members"] == [
        {"name": "Amy Hood", "title": "Director", "years_present": [2024, 2025]},
        {"name": "Satya Nadella", "title": "Chairman & CEO", "years_present": [2024, 2025]},
    ]
    assert saved["board_members_by_year"] == [
        {
            "year": 2025,
            "members": [
                {"name": "Satya Nadella", "title": "Chairman & CEO"},
                {"name": "Amy Hood", "title": "Director"},
            ],
        },
        {
            "year": 2024,
            "members": [
                {"name": "Satya Nadella", "title": "Chairman & CEO"},
                {"name": "Amy Hood", "title": "Director"},
            ],
        },
    ]


def test_collect_subsidiaries_by_year_skips_missing_exhibit(tmp_path: Path) -> None:
    from src.filings.ingestion import collect_subsidiaries_by_year

    filings = [
        {"year": 2025, "accession": "000032019325000073"},
        {"year": 2024, "accession": "000032019324000123"},
    ]

    with (
        patch("src.filings.ingestion._lookup_exhibit_21_filename", side_effect=[None, "aapl-ex21-2024.htm"]),
        patch("src.filings.ingestion._fetch_exhibit_21_html", return_value=EXHIBIT_21_HTML),
    ):
        output_path = collect_subsidiaries_by_year(
            cik="0000320193",
            ticker="AAPL",
            company_title="Apple Inc.",
            form_type="10-K",
            filings=filings,
            data_dir=tmp_path,
        )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert [entry["year"] for entry in saved["subsidiaries_by_year"]] == [2024]


def test_ingestion_filing_returns_subsidiaries_path_without_disrupting_existing_outputs() -> None:
    from src.filings.ingestion import ingestion_filing

    with (
        patch("src.filings.ingestion.ingestion_pipeline", return_value=("data/aapl_10-k", "AAPL", "0000320193", [{"year": 2025, "accession": "000032019325000073"}])),
        patch("src.filings.ingestion.get_filings", return_value=[{"form": "DEF 14A", "year": 2025, "filing_url": "https://sec.gov/def14a"}]),
        patch("src.filings.ingestion.extract_xbrl_from_filings") as mock_xbrl,
        patch("src.filings.ingestion.extract_docling_from_filings") as mock_docling,
        patch("src.filings.ingestion.verify_docling_output"),
        patch("src.filings.ingestion.validate_docling_content"),
        patch("src.filings.ingestion.validate_xbrl_output"),
        patch("src.filings.ingestion.collect_subsidiaries_by_year", return_value=Path("data/aapl_10-k_subsidiaries.json")),
        patch("src.filings.ingestion.collect_board_members_by_year", return_value=Path("data/aapl_def14a_board_members.json")),
    ):
        result = ingestion_filing("Apple Inc.", form_type="10-K")

    mock_xbrl.assert_called_once_with("data/aapl_10-k")
    mock_docling.assert_called_once()
    assert result["folder"] == "data/aapl_10-k"
    assert result["ticker"] == "AAPL"
    assert result["docling_json_path"] == "data/aapl_10-k_docling.json"
    assert result["xbrl_json_path"] == "data/aapl_10-k.json"
    assert result["subsidiaries_json_path"] == "data/aapl_10-k_subsidiaries.json"
    assert result["board_members_json_path"] == "data/aapl_def14a_board_members.json"
