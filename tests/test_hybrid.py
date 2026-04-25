from __future__ import annotations

from src.utils.hybrid import bm25_rank, hybrid_rrf_rank, rrf_fuse, tokenize_for_bm25


def test_tokenize_for_bm25_normalizes_terms() -> None:
    assert tokenize_for_bm25("Revenue, AI-growth!") == ["revenue", "ai", "growth"]


def test_bm25_rank_prioritizes_keyword_match() -> None:
    docs = [
        {"id": "a", "text": "cloud revenue growth"},
        {"id": "b", "text": "patent litigation exposure"},
        {"id": "c", "text": "supply chain inventory risk"},
    ]

    ranked = bm25_rank("litigation risk", docs, lambda d: d["text"])

    assert ranked[0]["id"] == "b"


def test_rrf_fuse_combines_ranked_lists() -> None:
    dense = [{"id": "a"}, {"id": "b"}]
    keyword = [{"id": "b"}, {"id": "a"}]

    fused = rrf_fuse([dense, keyword], key=lambda d: d["id"])

    assert {item["id"] for item in fused} == {"a", "b"}


def test_hybrid_rrf_rank_returns_top_k() -> None:
    dense = [
        {"id": "a", "text": "generic financial statement"},
        {"id": "b", "text": "patent infringement litigation court opinion"},
        {"id": "c", "text": "supply chain risk factor"},
    ]

    ranked = hybrid_rrf_rank(
        "patent litigation",
        dense,
        text_getter=lambda d: d["text"],
        key=lambda d: d["id"],
        top_k=2,
    )

    assert len(ranked) == 2
    assert ranked[0]["id"] in {"a", "b"}
