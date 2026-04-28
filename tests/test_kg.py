from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import builtins
import csv
import logging
import pytest
import shutil
from unittest.mock import patch


def load_kg_module(module_name: str = "kg_module"):
    kg_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "Knowledge graph"
        / "kg.py"
    )
    spec = spec_from_file_location(module_name, kg_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_test_dir(name: str) -> Path:
    path = Path(__file__).resolve().parent / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir()
    return path


def test_import_does_not_configure_root_logging():
    with patch.object(logging, "basicConfig", side_effect=AssertionError("basicConfig called")):
        load_kg_module("kg_module_logging")


def test_import_succeeds_without_neo4j_dependency():
    original_import = builtins.__import__

    def import_with_neo4j_error(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "neo4j":
            raise ImportError("neo4j not installed")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=import_with_neo4j_error):
        kg = load_kg_module("kg_module_no_neo4j")

    assert kg.GraphDatabase is None


def test_resolve_neo4j_settings_prefers_env_then_config():
    kg = load_kg_module("kg_module_neo4j_settings")

    with patch.dict(
        "os.environ",
        {
            "NEO4J_URI": "bolt://env-host:9999",
            "NEO4J_USER": "env-user",
            "NEO4J_PASSWORD": "env-pass",
        },
        clear=False,
    ):
        assert kg._resolve_neo4j_settings(
            {
                "NEO4J_URI": "bolt://config-host:7687",
                "NEO4J_USER": "config-user",
                "NEO4J_PASSWORD": "config-pass",
            }
        ) == ("bolt://env-host:9999", "env-user", "env-pass")

    with patch.dict(
        "os.environ",
        {
            "NEO4J_URI": "",
            "NEO4J_USER": "",
            "NEO4J_PASSWORD": "",
        },
        clear=False,
    ):
        assert kg._resolve_neo4j_settings(
            {
                "NEO4J_URI": "bolt://config-host:7687",
                "NEO4J_USER": "config-user",
                "NEO4J_PASSWORD": "config-pass",
            }
        ) == ("bolt://config-host:7687", "config-user", "config-pass")


def test_year_from_report_date_preferred():
    kg = load_kg_module()
    entry = {
        "report_date": "2025-12-31",
        "filing_date": "2026-02-05",
    }

    assert kg.extract_year(entry) == 2025


def test_build_filing_id_uses_report_year():
    kg = load_kg_module()

    assert kg.build_filing_id("AAPL", "10-K", 2025) == "AAPL_10-K_2025"


def test_is_page_stamp_detects_docling_footer_pattern():
    kg = load_kg_module("kg_module_page_stamp")

    assert kg._is_page_stamp("Apple Inc. | 2021 Form 10-K | 26") is True


def test_is_title_block_accepts_short_heading():
    kg = load_kg_module("kg_module_title_true")

    assert kg._is_title_block("Foreign Currency Risk") is True


def test_is_title_block_accepts_item_heading_pattern():
    kg = load_kg_module("kg_module_title_item")

    assert kg._is_title_block("Item 7. Management's Discussion and Analysis") is True


def test_is_title_block_rejects_short_non_title_cased_block():
    kg = load_kg_module("kg_module_title_short_false")

    assert kg._is_title_block("foreign currency risk") is False


def test_is_title_block_rejects_numeric_only_short_block():
    kg = load_kg_module("kg_module_title_numeric_false")

    assert kg._is_title_block("2024") is False


def test_is_title_block_rejects_long_paragraph():
    kg = load_kg_module("kg_module_title_false")
    long_paragraph = (
        "We are exposed to market risks arising from changes in foreign currency "
        "exchange rates, interest rates, and equity prices, and we actively "
        "monitor those exposures as part of our broader financial risk "
        "management program."
    )

    assert kg._is_title_block(long_paragraph) is False


def test_extract_sections_from_docling_entry_skips_noise_and_attaches_pending_title():
    kg = load_kg_module("kg_module_extract_sections")
    body_text = (
        "We are exposed to market risks arising from changes in foreign currency "
        "exchange rates in the normal course of business, and we use a range of "
        "operational and financial controls to monitor and manage those risks."
    )
    entry = {
        "ticker": "AAPL",
        "form_type": "10-K",
        "report_date": "2021-09-25",
        "docling": {
            "texts": [
                {"text": "Table of Contents", "content_layer": "body", "hyperlink": "#toc"},
                {"text": "Risk Factors", "content_layer": "body"},
                {"text": "Apple Inc. | 2021 Form 10-K | 26", "content_layer": "body"},
                {"text": "Foreign Currency Risk", "content_layer": "body"},
                {"text": body_text, "content_layer": "body", "self_ref": "#/texts/12"},
            ]
        },
    }

    assert kg.extract_sections_from_docling_entry(entry, "aapl_2021_10k.json") == [
        {
            "section_id": "AAPL_10-K_2021_section_12",
            "filing_id": "AAPL_10-K_2021",
            "ticker": "AAPL",
            "form_type": "10-K",
            "year": 2021,
            "ordinal": 12,
            "title": "Foreign Currency Risk",
            "text": body_text,
            "source_file": "aapl_2021_10k.json",
        }
    ]


def test_extract_sections_from_docling_entry_avoids_duplicate_section_ids_with_missing_ordinals():
    kg = load_kg_module("kg_module_extract_sections_ordinals")
    first_body_text = (
        "This section has no self reference, so the extractor needs to assign a "
        "deterministic fallback ordinal that stays unique within the filing."
    )
    second_body_text = (
        "This later section has an explicit self reference that would collide "
        "with a naive fallback strategy based only on the current section count."
    )
    entry = {
        "ticker": "AAPL",
        "form_type": "10-K",
        "report_date": "2021-09-25",
        "docling": {
            "texts": [
                {"text": first_body_text, "content_layer": "body"},
                {"text": second_body_text, "content_layer": "body", "self_ref": "#/texts/1"},
            ]
        },
    }

    assert kg.extract_sections_from_docling_entry(entry, "aapl_2021_10k.json") == [
        {
            "section_id": "AAPL_10-K_2021_section_1",
            "filing_id": "AAPL_10-K_2021",
            "ticker": "AAPL",
            "form_type": "10-K",
            "year": 2021,
            "ordinal": 1,
            "title": None,
            "text": first_body_text,
            "source_file": "aapl_2021_10k.json",
        },
        {
            "section_id": "AAPL_10-K_2021_section_2",
            "filing_id": "AAPL_10-K_2021",
            "ticker": "AAPL",
            "form_type": "10-K",
            "year": 2021,
            "ordinal": 2,
            "title": None,
            "text": second_body_text,
            "source_file": "aapl_2021_10k.json",
        },
    ]


def test_normalize_board_member_name_strips_role_suffix_at_end():
    kg = load_kg_module("kg_module_normalize_board_member_suffix")

    assert kg.normalize_board_member_name("Art Levinson Board Chair") == "Art Levinson"


def test_normalize_board_member_name_trims_and_collapses_spaces_before_suffix_handling():
    kg = load_kg_module("kg_module_normalize_board_member_spacing")

    assert kg.normalize_board_member_name("  Art   Levinson   Board Chair  ") == "Art Levinson"


def test_normalize_board_member_name_leaves_plain_name_unchanged():
    kg = load_kg_module("kg_module_normalize_board_member_plain")

    assert kg.normalize_board_member_name("Tim Cook") == "Tim Cook"


def test_normalize_board_member_name_preserves_role_phrase_when_not_trailing():
    kg = load_kg_module("kg_module_normalize_board_member_non_trailing_suffix")

    assert kg.normalize_board_member_name("Board Chair Art Levinson") == "Board Chair Art Levinson"


def test_normalize_board_member_name_rejects_role_only_string():
    kg = load_kg_module("kg_module_normalize_board_member_role_only")

    assert kg.normalize_board_member_name("Board Chair") == ""


def test_aggregate_board_members_merges_multi_year_records_and_uses_latest_title():
    kg = load_kg_module("kg_module_aggregate_board_members")
    records = [
        {"year": 2023, "members": [{"name": "Art Levinson Chair", "title": "Founder"}]},
        {"year": 2025, "members": [{"name": "Art Levinson Board Chair", "title": "Board Chair"}]},
        {"year": 2024, "members": [{"name": "Tim Cook", "title": "CEO"}]},
        {"year": 2025, "members": [{"name": "Tim Cook", "title": "Chief Executive Officer"}]},
    ]

    assert kg._aggregate_board_members(records, current_year=2025) == [
        {
            "id": "AAPL_art_levinson",
            "name": "Art Levinson",
            "ticker": "AAPL",
            "title": "Board Chair",
            "years_present": [2023, 2025],
            "is_current": True,
        },
        {
            "id": "AAPL_tim_cook",
            "name": "Tim Cook",
            "ticker": "AAPL",
            "title": "Chief Executive Officer",
            "years_present": [2024, 2025],
            "is_current": True,
        },
    ]


def test_aggregate_board_members_infers_current_year_from_latest_record():
    kg = load_kg_module("kg_module_aggregate_board_members_inferred_year")
    records = [
        {"year": 2022, "members": [{"name": "Al Gore Lead Independent Director", "title": "Lead Independent Director"}]},
        {"year": 2024, "members": [{"name": "Al Gore", "title": "Director"}]},
        {"year": 2023, "members": [{"name": "Sue Wagner", "title": "Director"}]},
    ]

    assert kg._aggregate_board_members(records, ticker="msft") == [
        {
            "id": "MSFT_al_gore",
            "name": "Al Gore",
            "ticker": "MSFT",
            "title": "Director",
            "years_present": [2022, 2024],
            "is_current": True,
        },
        {
            "id": "MSFT_sue_wagner",
            "name": "Sue Wagner",
            "ticker": "MSFT",
            "title": "Director",
            "years_present": [2023],
            "is_current": False,
        },
    ]


def test_aggregate_board_members_skips_role_only_name_entries():
    kg = load_kg_module("kg_module_aggregate_board_members_role_only")
    records = [
        {"year": 2025, "members": [{"name": "Director", "title": "Director"}]},
        {"year": 2025, "members": [{"name": "Tim Cook", "title": "Chief Executive Officer"}]},
    ]

    assert kg._aggregate_board_members(records, current_year=2025) == [
        {
            "id": "AAPL_tim_cook",
            "name": "Tim Cook",
            "ticker": "AAPL",
            "title": "Chief Executive Officer",
            "years_present": [2025],
            "is_current": True,
        }
    ]


def test_aggregate_board_members_uses_deterministic_title_for_same_year_duplicates():
    kg = load_kg_module("kg_module_aggregate_board_members_same_year_duplicates")
    records = [
        {"year": 2025, "members": [{"name": "Tim Cook", "title": "CEO"}]},
        {"year": 2025, "members": [{"name": "Tim Cook", "title": "Chief Executive Officer"}]},
        {"year": 2025, "members": [{"name": "Tim Cook", "title": "CEO"}]},
    ]

    assert kg._aggregate_board_members(records, current_year=2025) == [
        {
            "id": "AAPL_tim_cook",
            "name": "Tim Cook",
            "ticker": "AAPL",
            "title": "Chief Executive Officer",
            "years_present": [2025],
            "is_current": True,
        }
    ]


def test_normalize_subsidiary_id_slugifies_name_and_prefixes_ticker():
    kg = load_kg_module("kg_module_normalize_subsidiary_id")

    assert (
        kg.normalize_subsidiary_id("AAPL", "Apple Operations International Limited")
        == "AAPL_apple_operations_international_limited"
    )


def test_build_subsidiary_relationship_payload_emits_one_relationship_per_subsidiary_year():
    kg = load_kg_module("kg_module_build_subsidiary_relationship_payload")
    payload = {
        "form_type": "10-K",
        "subsidiaries_by_year": [
            {
                "year": 2023,
                "subsidiaries": [
                    "Apple Operations International Limited",
                    "Apple Sales International",
                ],
            },
            {
                "year": 2024,
                "subsidiaries": [
                    "Apple Operations International Limited",
                ],
            },
        ]
    }

    assert kg.build_subsidiary_relationship_payload(payload, "AAPL") == [
        {
            "subsidiary_id": "AAPL_apple_operations_international_limited",
            "name": "Apple Operations International Limited",
            "ticker": "AAPL",
            "year": 2023,
            "source_form_type": "10-K",
        },
        {
            "subsidiary_id": "AAPL_apple_sales_international",
            "name": "Apple Sales International",
            "ticker": "AAPL",
            "year": 2023,
            "source_form_type": "10-K",
        },
        {
            "subsidiary_id": "AAPL_apple_operations_international_limited",
            "name": "Apple Operations International Limited",
            "ticker": "AAPL",
            "year": 2024,
            "source_form_type": "10-K",
        },
    ]


def test_parse_cpc_codes_parses_postgres_array_string():
    kg = load_kg_module("kg_module_parse_cpc_codes")

    assert kg.parse_cpc_codes("{A61B5/02,G01C22/006,H04L9/32}") == [
        "A61B5/02",
        "G01C22/006",
        "H04L9/32",
    ]


def test_parse_cpc_codes_strips_quotes_from_postgres_array_elements():
    kg = load_kg_module("kg_module_parse_cpc_codes_quotes")

    assert kg.parse_cpc_codes('{"H04W72/00","G06F3/01"}') == [
        "H04W72/00",
        "G06F3/01",
    ]


def test_extract_cpc_sections_returns_deduplicated_three_char_prefixes():
    kg = load_kg_module("kg_module_extract_cpc_sections")

    assert kg.extract_cpc_sections(
        ["A61B5/02", "H04W72/00", "H04L9/32", "G06F3/01", "Y02E10/10"]
    ) == {"A61", "H04", "G06"}


def test_filter_patents_csv_excludes_2020_and_preserves_raw_input():
    kg = load_kg_module("kg_module_filter_patents_csv")
    data_dir = make_test_dir("_tmp_filter_patents_csv")
    raw_path = data_dir / "patents.csv"
    filtered_path = data_dir / "patents_2021_2025.csv"
    raw_csv = "\n".join(
        [
            "patent_id,patent_title,grant_date,assignee_organization,cpc_codes,citation_count,created_at",
            '1,Older patent,2020-01-07,Apple Inc.,"{A61B5/02}",0,2026-04-18',
            '2,Window patent,2021-01-05,Apple Inc.,"{H04W72/00}",0,2026-04-18',
            '3,Later patent,2025-12-30,Apple Inc.,"{G06F3/01}",0,2026-04-18',
            '4,Too new patent,2026-01-02,Apple Inc.,"{G06F3/01}",0,2026-04-18',
            '5,Other company patent,2024-03-01,Other Co.,"{H04L9/32}",0,2026-04-18',
        ]
    )
    try:
        raw_path.write_text(raw_csv, encoding="utf-8")

        output_path = kg.filter_patents_csv(str(raw_path), str(filtered_path))

        assert Path(output_path) == filtered_path
        assert raw_path.read_text(encoding="utf-8") == raw_csv
        with filtered_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["patent_id"] for row in rows] == ["2", "3"]
        assert all(row["grant_date"][:4] in {"2021", "2025"} for row in rows)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_ingest_patents_creates_one_domain_edge_per_unique_cpc_prefix():
    kg = load_kg_module("kg_module_ingest_patents")
    data_dir = make_test_dir("_tmp_ingest_patents")
    patents_path = data_dir / "patents_2021_2025.csv"
    patents_path.write_text(
        "\n".join(
            [
                "patent_id,patent_title,grant_date,assignee_organization,cpc_codes,citation_count,created_at",
                '123,Wireless security,2024-03-15,Apple Inc.,"{H04W72/00,H04L9/32,G06F3/01,Y02E10/10}",0,2026-04-18',
            ]
        ),
        encoding="utf-8",
    )

    calls = []

    class RecordingSession:
        def run(self, query, **params):
            if "RETURN c.ticker AS ticker" in query:
                return DummyResult({"ticker": params["ticker"]})
            calls.append((query, params))
            return DummyResult()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyResult:
        def __init__(self, row=None):
            self.row = row

        def single(self):
            return self.row

    class RecordingDriver:
        def session(self):
            return RecordingSession()

    try:
        kg.ingest_patents(RecordingDriver(), patents_path, "aapl")

        assert len(calls) == 3
        patent_query, patent_params = calls[0]
        assert "MERGE (c)-[:HAS_PATENT]->(p)" in patent_query
        assert "MERGE (p:Patent {patent_id: $patent_id})" in patent_query
        assert patent_params == {
            "patent_id": "123",
            "title": "Wireless security",
            "grant_date": "2024-03-15",
            "grant_year": 2024,
            "cpc_codes": ["H04W72/00", "H04L9/32", "G06F3/01", "Y02E10/10"],
            "ticker": "AAPL",
        }

        domain_prefixes = {params["prefix"] for _, params in calls[1:]}
        assert domain_prefixes == {"H04", "G06"}
        assert {params["section_letter"] for _, params in calls[1:]} == {"H", "G"}
        assert {params["label"] for _, params in calls[1:]} == {
            "Telecommunications & Signal Processing",
            "Computing & Data Processing",
        }
        assert all(
            "MERGE (d:TechnologyDomain {cpc_prefix: $prefix})" in query
            and "MERGE (p)-[:BELONGS_TO_DOMAIN {cpc_prefix: $prefix}]->(d)" in query
            for query, _ in calls[1:]
        )
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_ingest_patents_skips_malformed_grant_date_rows():
    kg = load_kg_module("kg_module_ingest_patents_bad_grant_date")
    data_dir = make_test_dir("_tmp_ingest_patents_bad_grant_date")
    patents_path = data_dir / "patents_2021_2025.csv"
    patents_path.write_text(
        "\n".join(
            [
                "patent_id,patent_title,grant_date,assignee_organization,cpc_codes,citation_count,created_at",
                '123,Malformed patent,bad-date,Apple Inc.,"{H04W72/00}",0,2026-04-18',
                '124,Valid patent,2024-03-15,Apple Inc.,"{G06F3/01}",0,2026-04-18',
            ]
        ),
        encoding="utf-8",
    )

    calls = []

    class RecordingSession:
        def run(self, query, **params):
            if "RETURN c.ticker AS ticker" in query:
                return DummyResult({"ticker": params["ticker"]})
            calls.append((query, params))
            return DummyResult()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyResult:
        def __init__(self, row=None):
            self.row = row

        def single(self):
            return self.row

    class RecordingDriver:
        def session(self):
            return RecordingSession()

    try:
        kg.ingest_patents(RecordingDriver(), patents_path, "AAPL")

        assert len(calls) == 2
        assert calls[0][1]["patent_id"] == "124"
        assert calls[1][1]["prefix"] == "G06"
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_ingest_patents_requires_existing_company_node():
    kg = load_kg_module("kg_module_ingest_patents_missing_company")
    data_dir = make_test_dir("_tmp_ingest_patents_missing_company")
    patents_path = data_dir / "patents_2021_2025.csv"
    patents_path.write_text(
        "\n".join(
            [
                "patent_id,patent_title,grant_date,assignee_organization,cpc_codes,citation_count,created_at",
                '123,Wireless security,2024-03-15,Apple Inc.,"{H04W72/00}",0,2026-04-18',
            ]
        ),
        encoding="utf-8",
    )

    class RecordingSession:
        def run(self, query, **params):
            if "RETURN c.ticker AS ticker" in query:
                return DummyResult(None)
            raise AssertionError("Patent writes should not run when the company node is missing")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyResult:
        def __init__(self, row=None):
            self.row = row

        def single(self):
            return self.row

    class RecordingDriver:
        def session(self):
            return RecordingSession()

    try:
        with pytest.raises(ValueError, match="Company node for ticker AAPL must exist"):
            kg.ingest_patents(RecordingDriver(), patents_path, "AAPL")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_create_schema_includes_patent_constraints():
    kg = load_kg_module("kg_module_create_schema_patents")
    statements = []

    class RecordingSession:
        def run(self, statement, **params):
            statements.append(statement)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class RecordingDriver:
        def session(self):
            return RecordingSession()

    kg.create_schema(RecordingDriver())

    assert (
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patent) REQUIRE p.patent_id IS UNIQUE"
        in statements
    )
    assert (
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:TechnologyDomain) REQUIRE d.cpc_prefix IS UNIQUE"
        in statements
    )


def test_run_filings_kg_build_wires_standalone_sources_with_injected_driver():
    kg = load_kg_module("kg_module_run_builder")
    data_dir = Path(__file__).resolve().parent / "_tmp_run_filings_kg_build"
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir()

    expected_files = {
        "aapl_10-k_docling.json",
        "aapl_10-q_docling.json",
        "aapl_8-k_docling.json",
        "aapl_10-k_subsidiaries.json",
        "aapl_def14a_board_members.json",
        "patents.csv",
    }
    for file_name in expected_files:
        content = "[]" if file_name != "patents.csv" else (
            "patent_id,patent_title,grant_date,assignee_organization,cpc_codes,citation_count,created_at\n"
        )
        (data_dir / file_name).write_text(content, encoding="utf-8")

    calls = []

    class DummyDriver:
        def close(self):
            raise AssertionError("Injected driver should not be closed")

    driver = DummyDriver()

    def record(name):
        def _recorder(*args, **kwargs):
            calls.append((name, args, kwargs))

        return _recorder

    try:
        with (
            patch.object(kg, "create_schema", side_effect=record("create_schema")),
            patch.object(kg, "upsert_company", side_effect=record("upsert_company")),
            patch.object(
                kg,
                "ingest_docling_filings",
                side_effect=record("ingest_docling_filings"),
            ),
            patch.object(
                kg,
                "ingest_subsidiaries",
                side_effect=record("ingest_subsidiaries"),
            ),
            patch.object(
                kg,
                "ingest_board_members",
                side_effect=record("ingest_board_members"),
            ),
            patch.object(
                kg,
                "filter_patents_csv",
                side_effect=record("filter_patents_csv"),
            ),
            patch.object(
                kg,
                "ingest_patents",
                side_effect=record("ingest_patents"),
            ),
            patch.object(
                kg,
                "log_graph_summary",
                side_effect=record("log_graph_summary"),
            ),
        ):
            summary = kg.run_filings_kg_build("aapl", str(data_dir), driver=driver)

        assert summary == {
            "ticker": "AAPL",
            "data_dir": str(data_dir),
            "docling_files": 3,
        }

        assert [name for name, _, _ in calls] == [
            "create_schema",
            "upsert_company",
            "ingest_docling_filings",
            "ingest_docling_filings",
            "ingest_docling_filings",
            "ingest_subsidiaries",
            "ingest_board_members",
            "filter_patents_csv",
            "ingest_patents",
            "log_graph_summary",
        ]
        assert calls[0][1] == (driver,)
        assert calls[1][1] == (driver, "AAPL", "Apple Inc.")
        assert [
            Path(args[1]).name
            for name, args, _ in calls
            if name == "ingest_docling_filings"
        ] == [
            "aapl_10-k_docling.json",
            "aapl_10-q_docling.json",
            "aapl_8-k_docling.json",
        ]
        assert calls[5][1][0] is driver
        assert Path(calls[5][1][1]) == data_dir / "aapl_10-k_subsidiaries.json"
        assert calls[5][1][2] == "AAPL"
        assert calls[6][1][0] is driver
        assert Path(calls[6][1][1]) == data_dir / "aapl_def14a_board_members.json"
        assert calls[6][1][2] == "AAPL"
        assert Path(calls[7][1][0]) == data_dir / "patents.csv"
        assert Path(calls[7][1][1]) == data_dir / "patents_2021_2025.csv"
        assert calls[8][1][0] is driver
        assert Path(calls[8][1][1]) == data_dir / "patents_2021_2025.csv"
        assert calls[8][1][2] == "AAPL"
        assert calls[9][1] == (driver, "AAPL")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_run_filings_kg_build_rejects_non_aapl_ticker_before_any_ingestion():
    kg = load_kg_module("kg_module_run_builder_non_aapl")

    class DummyDriver:
        def close(self):
            raise AssertionError("Injected driver should not be closed")

    driver = DummyDriver()

    with pytest.raises(
        ValueError,
        match="This standalone filings KG builder currently supports only AAPL.",
    ):
        kg.run_filings_kg_build("MSFT", "unused-data-dir", driver=driver)
