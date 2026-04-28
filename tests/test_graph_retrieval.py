from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


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
