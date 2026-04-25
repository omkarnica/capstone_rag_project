"""
Cache package — 3-tier caching for the RAG pipeline.
"""

import os

from src.cache.base import CacheBackend


def get_cache_backend() -> CacheBackend:
    backend = (os.getenv("CACHE_BACKEND", "redis")).strip().lower()
    if backend == "redis":
        from src.cache.redis_backend import RedisCacheBackend
        return RedisCacheBackend()
    raise ValueError(f"Unsupported CACHE_BACKEND '{backend}'. Supported: redis")
