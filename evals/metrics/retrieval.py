from __future__ import annotations

import numpy as np
from sklearn.metrics import ndcg_score
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class MRRMetric(BaseMetric):
    """Mean Reciprocal Rank — position of first relevant source in retrieved list."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.score = 0.0

    @property
    def __name__(self) -> str:
        return "MRR"

    def compute(self, retrieved_sources: list[str], expected_sources: list[str]) -> float:
        expected_set = {s.lower() for s in expected_sources}
        for rank, source in enumerate(retrieved_sources, start=1):
            if source.lower() in expected_set:
                return 1.0 / rank
        return 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        retrieved = _extract_sources(test_case.retrieval_context or [])
        expected = getattr(test_case, "expected_sources", [])
        self.score = self.compute(retrieved, expected)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


class NDCGMetric(BaseMetric):
    """Normalized Discounted Cumulative Gain @ k using binary relevance."""

    def __init__(self, k: int = 5, threshold: float = 0.5):
        self.k = k
        self.threshold = threshold
        self.score = 0.0

    @property
    def __name__(self) -> str:
        return f"NDCG@{self.k}"

    def compute(self, retrieved_sources: list[str], expected_sources: list[str]) -> float:
        if not expected_sources:
            return 0.0
        expected_set = {s.lower() for s in expected_sources}
        relevance = [1 if s.lower() in expected_set else 0 for s in retrieved_sources]
        if sum(relevance) == 0:
            return 0.0
        ideal = sorted(relevance, reverse=True)
        score = ndcg_score(
            y_true=np.array([ideal]),
            y_score=np.array([relevance]),
            k=self.k,
        )
        return float(score)

    def measure(self, test_case: LLMTestCase) -> float:
        retrieved = _extract_sources(test_case.retrieval_context or [])
        expected = getattr(test_case, "expected_sources", [])
        self.score = self.compute(retrieved, expected)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


def _extract_sources(retrieval_context: list[str]) -> list[str]:
    """Source labels are passed directly as retrieval_context strings by the runner."""
    return retrieval_context
