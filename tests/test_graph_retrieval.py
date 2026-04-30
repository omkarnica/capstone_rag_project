from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def load_graph_retrieval_module(module_name: str = "graph_retrieval_module"):
    module_path = Path(__file__).resolve().parents[1] / "src" / "graph_retrieval.py"
    spec = spec_from_file_location(module_name, module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_graph_retrieval_import_loads_dotenv() -> None:
    with patch("dotenv.load_dotenv") as mock_load_dotenv:
        load_graph_retrieval_module("graph_retrieval_module_dotenv")

    mock_load_dotenv.assert_called_once()


def test_resolve_neo4j_settings_prefers_env_values() -> None:
    from src import graph_retrieval

    with patch.dict(
        "os.environ",
        {
            "NEO4J_URI": "neo4j+s://db.example.io",
            "NEO4J_USER": "neo4j-user",
            "NEO4J_PASSWORD": "secret",
        },
        clear=False,
    ):
        assert graph_retrieval._resolve_neo4j_settings() == (
            "neo4j+s://db.example.io",
            "neo4j-user",
            "secret",
        )


def test_resolve_neo4j_settings_uses_secret_manager_ids_from_config_when_env_missing() -> None:
    from src import graph_retrieval

    with patch.dict("os.environ", {}, clear=True):
        with patch.object(
            graph_retrieval,
            "_CONFIG",
            {
                "NEO4J_URI_SECRET": "NEO4J_URI",
                "NEO4J_USER_SECRET": "NEO4J_USER",
                "NEO4J_PASSWORD_SECRET": "NEO4J_PASSWORD",
            },
        ):
            with patch(
                "src.graph_retrieval.get_secret",
                side_effect=["neo4j+s://secret.db.io", "secret-user", "secret-password"],
            ) as mock_get_secret:
                assert graph_retrieval._resolve_neo4j_settings() == (
                    "neo4j+s://secret.db.io",
                    "secret-user",
                    "secret-password",
                )

    requested_secret_ids = [call.args[0] for call in mock_get_secret.call_args_list]
    assert requested_secret_ids == ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]


def test_strip_cypher_fences_removes_markdown_wrapper() -> None:
    from src.graph_retrieval import _strip_cypher_fences

    fenced = "```cypher\nMATCH (c:Company {ticker: 'AAPL'}) RETURN c\n```"

    assert _strip_cypher_fences(fenced) == "MATCH (c:Company {ticker: 'AAPL'}) RETURN c"


def test_validate_read_only_cypher_rejects_write_keywords() -> None:
    from src.graph_retrieval import validate_read_only_cypher

    with pytest.raises(ValueError, match="read-only"):
        validate_read_only_cypher("MATCH (c:Company {ticker: 'AAPL'}) SET c.name = 'Apple' RETURN c")


def test_validate_read_only_cypher_requires_match_and_return() -> None:
    from src.graph_retrieval import validate_read_only_cypher

    with pytest.raises(ValueError, match="MATCH"):
        validate_read_only_cypher("RETURN 1")


def test_graph_prompt_instructs_semantic_aliases_and_year_fields() -> None:
    from src.graph_retrieval import _graph_prompt

    prompt = _graph_prompt("Who are the board members of Apple in 2024?", company="AAPL")

    assert "return semantic aliases" in prompt.lower()
    assert "include company and year fields when available" in prompt.lower()
    assert "prefer the exact stored company name" in prompt.lower()
    assert "for board-member questions" in prompt.lower()
    assert "treat years_present as a list/array field" in prompt.lower()
    assert "year in node.years_present" in prompt.lower()
    assert 'contains "2024"' in prompt.lower()
    assert "for subsidiary questions" in prompt.lower()
    assert "for filing questions" in prompt.lower()
    assert "for patent questions" in prompt.lower()


def test_graph_prompt_surfaces_exact_company_name_hint_from_question() -> None:
    from src.graph_retrieval import _graph_prompt

    prompt = _graph_prompt("Who are the board members of Apple Inc. in 2024?", company="AAPL")

    assert "preferred company full name: apple inc." in prompt.lower()
    assert "preserve the full company name exactly in the cypher match" in prompt.lower()


def test_generic_row_to_doc_falls_back_to_key_value_lines() -> None:
    from src.graph_retrieval import _row_to_doc

    doc = _row_to_doc({"custom_field": "value", "count": 3}, cypher="MATCH ... RETURN ...")

    assert "custom_field: value" in doc["content"]
    assert "count: 3" in doc["content"]
    assert doc["metadata"]["cypher"] == "MATCH ... RETURN ..."


def test_list_values_are_preserved_in_fallback_formatting() -> None:
    from src.graph_retrieval import _row_to_doc

    doc = _row_to_doc({"years_present": [2024, 2025]}, cypher="MATCH ... RETURN ...")

    assert "years_present: [2024, 2025]" in doc["content"]


def test_board_member_row_formats_as_explicit_evidence() -> None:
    from src.graph_retrieval import _row_to_doc

    row = {
        "company": "Apple",
        "year": 2024,
        "board_member": "Art Levinson",
        "title": "Founder and CEO, Calico",
    }

    doc = _row_to_doc(row, cypher="MATCH ... RETURN ...")

    assert "Apple board member in 2024: Art Levinson" in doc["content"]
    assert "Title: Founder and CEO, Calico" in doc["content"]


def test_board_member_row_formats_capitalized_aliases_with_year_and_company() -> None:
    from src.graph_retrieval import _row_to_doc

    row = {
        "CompanyName": "Apple Inc.",
        "Year": 2023,
        "BoardMemberName": "Tim Cook",
        "BoardMemberTitle": "CEO, Apple",
        "YearsPresent": [2023, 2024],
    }

    doc = _row_to_doc(row, cypher="MATCH ... RETURN ...")

    assert "Apple Inc. board member in 2023: Tim Cook" in doc["content"]
    assert "Title: CEO, Apple" in doc["content"]
    assert "Years present: [2023, 2024]" in doc["content"]


def test_subsidiary_row_formats_as_explicit_evidence() -> None:
    from src.graph_retrieval import _row_to_doc

    row = {
        "company": "Apple",
        "subsidiary": "Apple Operations International Limited",
        "year": 2024,
        "source_form_type": "10-K",
    }

    doc = _row_to_doc(row, cypher="MATCH ... RETURN ...")

    assert "Apple subsidiary in 2024: Apple Operations International Limited" in doc["content"]
    assert "Source form: 10-K" in doc["content"]


def test_subsidiary_row_formats_langsmith_style_aliases() -> None:
    from src.graph_retrieval import _row_to_doc

    row = {
        "Company": "Apple Inc.",
        "Subsidiary_Name": "Apple Operations International Limited",
    }

    doc = _row_to_doc(row, cypher="MATCH ... RETURN ...")

    assert "Apple Inc. subsidiary: Apple Operations International Limited" in doc["content"]


def test_filing_row_formats_as_explicit_evidence() -> None:
    from src.graph_retrieval import _row_to_doc

    row = {
        "company": "Apple",
        "year": 2024,
        "form_type": "10-K",
        "filing_id": "AAPL_10-K_2024",
        "source_file": "aapl_10-k_docling.json",
    }

    doc = _row_to_doc(row, cypher="MATCH ... RETURN ...")

    assert "Apple filing: 2024 10-K" in doc["content"]
    assert "Filing ID: AAPL_10-K_2024" in doc["content"]
    assert "Source file: aapl_10-k_docling.json" in doc["content"]


def test_section_row_formats_as_explicit_evidence() -> None:
    from src.graph_retrieval import _row_to_doc

    row = {
        "company": "Apple",
        "year": 2024,
        "form_type": "10-K",
        "section_title": "Risk Factors",
        "section_id": "AAPL_10-K_2024_section_12",
        "text": "Supply chain constraints may materially affect operations.",
    }

    doc = _row_to_doc(row, cypher="MATCH ... RETURN ...")

    assert "Apple 2024 10-K section: Risk Factors" in doc["content"]
    assert "Section ID: AAPL_10-K_2024_section_12" in doc["content"]
    assert "Supply chain constraints may materially affect operations." in doc["content"]


def test_patent_row_formats_as_explicit_evidence() -> None:
    from src.graph_retrieval import _row_to_doc

    row = {
        "company": "Apple",
        "patent_id": "US1234567",
        "patent_title": "Wireless security system",
        "grant_date": "2024-03-15",
        "domain": "G06",
    }

    doc = _row_to_doc(row, cypher="MATCH ... RETURN ...")

    assert "Apple patent: US1234567" in doc["content"]
    assert "Title: Wireless security system" in doc["content"]
    assert "Grant date: 2024-03-15" in doc["content"]
    assert "Domain: G06" in doc["content"]


def test_domain_row_formats_as_explicit_evidence() -> None:
    from src.graph_retrieval import _row_to_doc

    row = {
        "company": "Apple",
        "cpc_prefix": "G06",
        "label": "Computing & Data Processing",
    }

    doc = _row_to_doc(row, cypher="MATCH ... RETURN ...")

    assert "Apple technology domain: G06 | Computing & Data Processing" in doc["content"]


def test_retrieve_graph_docs_returns_semantic_subsidiary_evidence() -> None:
    from src.graph_retrieval import retrieve_graph_docs

    llm = MagicMock()
    llm.invoke.return_value.content = (
        "MATCH (c:Company {ticker: 'AAPL'})-[:HAS_SUBSIDIARY]->(s:Subsidiary) "
        "RETURN 'Apple' AS company, s.name AS subsidiary, 2024 AS year LIMIT 1"
    )

    result_cursor = MagicMock()
    result_cursor.data.return_value = [
        {"company": "Apple", "subsidiary": "Apple Operations International Limited", "year": 2024}
    ]

    session = MagicMock()
    session.__enter__.return_value = session
    session.run.return_value = result_cursor

    driver = MagicMock()
    driver.session.return_value = session

    with (
        patch("src.graph_retrieval.get_graph_llm", return_value=llm),
        patch("src.graph_retrieval.get_neo4j_driver", return_value=driver),
    ):
        docs = retrieve_graph_docs("List Apple's subsidiaries", company="AAPL")

    assert docs[0]["content"].startswith("Apple subsidiary in 2024: Apple Operations International Limited")
    assert docs[0]["metadata"]["cypher"].startswith("MATCH")


def test_retrieve_graph_docs_formats_rows_as_docs() -> None:
    from src.graph_retrieval import retrieve_graph_docs

    llm = MagicMock()
    llm.invoke.return_value.content = "MATCH (c:Company {ticker: 'AAPL'})-[:HAS_SUBSIDIARY]->(s:Subsidiary) RETURN s.name AS subsidiary LIMIT 2"

    record_one = {"subsidiary": "Apple Operations International Limited"}
    record_two = {"subsidiary": "Beats Electronics LLC"}
    result_cursor = MagicMock()
    result_cursor.data.return_value = [record_one, record_two]

    session = MagicMock()
    session.__enter__.return_value = session
    session.run.return_value = result_cursor

    driver = MagicMock()
    driver.session.return_value = session

    with (
        patch("src.graph_retrieval.get_graph_llm", return_value=llm),
        patch("src.graph_retrieval.get_neo4j_driver", return_value=driver),
    ):
        docs = retrieve_graph_docs("List Apple's subsidiaries", company="AAPL")

    assert len(docs) == 2
    assert docs[0]["metadata"]["source"] == "Knowledge Graph"
    assert "Apple Operations International Limited" in docs[0]["content"]
    assert docs[0]["metadata"]["cypher"].startswith("MATCH")


def test_retrieve_docs_dispatches_graph_route() -> None:
    from src.nodes.retriever import retrieve_docs

    graph_docs = [
        {
            "content": "subsidiary: Apple Operations International Limited",
            "metadata": {"source": "Knowledge Graph"},
        }
    ]

    with patch("src.nodes.retriever.retrieve_graph_docs", return_value=graph_docs) as mock_retrieve:
        result = retrieve_docs(
            {
                "route": "graph",
                "question": "List Apple's subsidiaries",
                "company": "AAPL",
            }
        )

    mock_retrieve.assert_called_once_with("List Apple's subsidiaries", company="AAPL")
    assert result["retrieved_docs"] == graph_docs
    assert result["retrieval_query"] == "List Apple's subsidiaries"
