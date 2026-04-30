"""
Adapter exposing the interface api.py expects:
  SemanticCache, compute_corpus_version, is_time_sensitive_question
"""

import json
from typing import Optional

from src.cache.redis_backend import RedisCacheBackend
from src.cache.embedding_cache import embed_query

_RECENCY_TERMS = {"current", "latest", "recent", "today", "now", "live"}

_backend_instance: Optional[RedisCacheBackend] = None


def _get_backend() -> Optional[RedisCacheBackend]:
    global _backend_instance
    if _backend_instance is None:
        try:
            _backend_instance = RedisCacheBackend()
        except Exception:
            return None
    return _backend_instance


def get_cache_backend() -> Optional[RedisCacheBackend]:
    return _get_backend()


def is_time_sensitive_question(question: str) -> bool:
    return any(t in question.lower() for t in _RECENCY_TERMS)


def compute_corpus_version(chunking_strategy: str = "hierarchical") -> int:
    backend = _get_backend()
    if backend is None:
        return 0
    try:
        return backend.get_doc_version()
    except Exception:
        return 0


class SemanticCache:

    def get_similar(
        self,
        question: str,
        corpus_version: int,
        chunking_strategy: str,
        similarity_threshold: float,
    ) -> Optional[dict]:
        backend = _get_backend()
        if backend is None:
            return None
        try:
            embedding = embed_query(question)
            raw = backend.get_semantic(
                embedding,
                threshold=similarity_threshold,
                source_filter=chunking_strategy,
            )
        except Exception:
            return None
        if raw is None:
            return None
        # answer field holds the full serialized result dict written by store()
        try:
            restored = json.loads(raw["answer"])
        except (json.JSONDecodeError, KeyError, TypeError):
            restored = {
                "final_answer": raw.get("answer", ""),
                "citations": json.loads(raw.get("sources_json", "[]")),
            }
        restored["cache_similarity"] = raw.get("similarity")
        restored["cache_source_question"] = raw.get("question")
        return restored

    def store(
        self,
        question: str,
        result: dict,
        corpus_version: int,
        chunking_strategy: str,
    ) -> None:
        backend = _get_backend()
        if backend is None:
            return
        try:
            embedding = embed_query(question)
            backend.set_semantic(
                question=question,
                embedding=embedding,
                answer=json.dumps(result),
                sources_json=json.dumps(result.get("citations", [])),
                doc_version=corpus_version,
                ttl_seconds=3600,
                source_filter=chunking_strategy,
            )
        except Exception:
            pass
