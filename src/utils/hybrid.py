from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TypeVar

from rank_bm25 import BM25Okapi

T = TypeVar("T")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize_for_bm25(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def bm25_rank(
    query: str,
    items: Sequence[T],
    text_getter: Callable[[T], str],
) -> list[T]:
    if not items:
        return []

    corpus = [tokenize_for_bm25(text_getter(item)) for item in items]
    query_tokens = tokenize_for_bm25(query)

    if not query_tokens or not any(corpus):
        return list(items)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(
        zip(items, scores, strict=False),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )
    return [item for item, _score in ranked]


def rrf_fuse(
    ranked_lists: Sequence[Sequence[T]],
    *,
    key: Callable[[T], str],
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[T]:
    scores: dict[str, float] = {}
    items_by_key: dict[str, T] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            item_key = key(item)
            if not item_key:
                continue
            items_by_key.setdefault(item_key, item)
            scores[item_key] = scores.get(item_key, 0.0) + 1.0 / (rrf_k + rank)

    fused_keys = sorted(scores, key=scores.get, reverse=True)
    fused = [items_by_key[item_key] for item_key in fused_keys]
    return fused[:top_k] if top_k is not None else fused


def hybrid_rrf_rank(
    query: str,
    dense_ranked_items: Sequence[T],
    text_getter: Callable[[T], str],
    key: Callable[[T], str],
    *,
    top_k: int | None = None,
    rrf_k: int = 60,
) -> list[T]:
    dense_ranked = list(dense_ranked_items)
    bm25_ranked = bm25_rank(query, dense_ranked, text_getter)
    return rrf_fuse(
        [dense_ranked, bm25_ranked],
        key=key,
        rrf_k=rrf_k,
        top_k=top_k,
    )
