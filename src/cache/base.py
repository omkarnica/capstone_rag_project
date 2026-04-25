"""
Abstract base class for cache backends.
"""

from abc import ABC, abstractmethod
from typing import Optional


class CacheBackend(ABC):
    """Interface that all cache backends must implement."""

    @abstractmethod
    def get_exact(self, query_hash: str) -> Optional[dict]:
        ...

    @abstractmethod
    def set_exact(
        self,
        query_hash: str,
        question: str,
        answer: str,
        sources_json: str,
        doc_version: int,
        ttl_seconds: int,
    ) -> None:
        ...

    @abstractmethod
    def get_semantic(
        self, embedding: list[float], threshold: float, source_filter: str = ""
    ) -> Optional[dict]:
        ...

    @abstractmethod
    def set_semantic(
        self,
        question: str,
        embedding: list[float],
        answer: str,
        sources_json: str,
        doc_version: int,
        ttl_seconds: int,
        source_filter: str = "",
    ) -> None:
        ...

    @abstractmethod
    def get_retrieval(
        self, embedding: list[float], threshold: float, source_filter: str = ""
    ) -> Optional[dict]:
        ...

    @abstractmethod
    def set_retrieval(
        self,
        question: str,
        embedding: list[float],
        chunks_json: str,
        doc_version: int,
        ttl_seconds: int,
        source_filter: str = "",
    ) -> None:
        ...

    @abstractmethod
    def get_doc_version(self) -> int:
        ...

    @abstractmethod
    def bump_doc_version(self) -> int:
        ...

    @abstractmethod
    def clear_all(self) -> dict:
        ...

    @abstractmethod
    def get_stats(self) -> dict:
        ...

    @abstractmethod
    def cleanup_expired(self) -> int:
        ...

    @abstractmethod
    def get_document_hash(self, file_hash: str) -> Optional[dict]:
        ...

    @abstractmethod
    def set_document_hash(self, file_hash: str, metadata: dict) -> None:
        ...

    @abstractmethod
    def remove_document_hash_by_name(self, file_name: str) -> bool:
        ...

    @abstractmethod
    def clear_document_hashes(self) -> int:
        ...
