from __future__ import annotations


def test_derive_query_metadata_filter_adds_form_type_and_year() -> None:
    from src.filings.raptor_retrieval import _derive_query_metadata_filter

    result = _derive_query_metadata_filter(
        "What did Apple report in its 2024 10-Q about Services revenue?"
    )

    assert result == {
        "form_type": {"$eq": "10-Q"},
        "year": {"$eq": 2024},
    }


def test_derive_query_metadata_filter_adds_only_year_when_form_missing() -> None:
    from src.filings.raptor_retrieval import _derive_query_metadata_filter

    result = _derive_query_metadata_filter(
        "What did Apple report in 2023 about foreign exchange risk?"
    )

    assert result == {"year": {"$eq": 2023}}


def test_derive_query_metadata_filter_adds_only_form_type_when_year_missing() -> None:
    from src.filings.raptor_retrieval import _derive_query_metadata_filter

    result = _derive_query_metadata_filter(
        "Summarize Apple's 10-K discussion of net sales."
    )

    assert result == {"form_type": {"$eq": "10-K"}}


def test_derive_query_metadata_filter_returns_none_without_detectable_filters() -> None:
    from src.filings.raptor_retrieval import _derive_query_metadata_filter

    result = _derive_query_metadata_filter(
        "What risks did Apple describe around supply chain disruption?"
    )

    assert result is None


def test_derive_query_metadata_filter_merges_with_explicit_filter() -> None:
    from src.filings.raptor_retrieval import _derive_query_metadata_filter

    result = _derive_query_metadata_filter(
        "What did Apple report in its 2024 10-Q about Services revenue?",
        metadata_filter={"ticker": {"$eq": "AAPL"}},
    )

    assert result == {
        "ticker": {"$eq": "AAPL"},
        "form_type": {"$eq": "10-Q"},
        "year": {"$eq": 2024},
    }


def test_raptor_retrieve_passes_derived_metadata_filter_to_query() -> None:
    from src.filings.raptor_retrieval import raptor_retrieve

    captured = {}

    def fake_query_pinecone(**kwargs):
        captured["metadata_filter"] = kwargs["metadata_filter"]
        return []

    result = None
    from unittest.mock import patch

    with (
        patch("src.filings.raptor_retrieval.embed_query", return_value=[0.1, 0.2]),
        patch("src.filings.raptor_retrieval.query_pinecone", side_effect=fake_query_pinecone),
        patch("src.filings.raptor_retrieval.hybrid_rrf_rank", return_value=[]),
    ):
        result = raptor_retrieve(
            query="What did Apple report in its 2024 10-Q about Services revenue?",
            namespace="filings",
        )

    assert captured["metadata_filter"] == {
        "form_type": {"$eq": "10-Q"},
        "year": {"$eq": 2024},
    }
    assert result["final_context_count"] == 0
