from __future__ import annotations

import json
import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class NumericalAccuracyMetric(BaseMetric):
    """Fraction of expected_numbers found in the answer within 1% tolerance."""

    def __init__(self, tolerance: float = 0.01, threshold: float = 0.8):
        self.tolerance = tolerance
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.async_mode = False

    @property
    def __name__(self) -> str:
        return "NumericalAccuracy"

    def compute(self, answer: str, expected_numbers: list[float]) -> float:
        if not expected_numbers:
            return 1.0
        extracted = _extract_numbers(answer)
        hits = 0
        for expected in expected_numbers:
            for found in extracted:
                if expected == 0:
                    if found == 0:
                        hits += 1
                        break
                elif abs(found - expected) / abs(expected) <= self.tolerance:
                    hits += 1
                    break
        return hits / len(expected_numbers)

    def measure(self, test_case: LLMTestCase) -> float:
        expected_numbers = getattr(test_case, "expected_numbers", [])
        self.score = self.compute(test_case.actual_output or "", expected_numbers)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


class EntityMatchPrecisionMetric(BaseMetric):
    """Fraction of expected_entities present (case-insensitive) in the answer."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.async_mode = False

    @property
    def __name__(self) -> str:
        return "EntityMatchPrecision"

    def compute(self, answer: str, expected_entities: list[str]) -> float:
        if not expected_entities:
            return 1.0
        answer_lower = answer.lower()
        hits = sum(1 for e in expected_entities if e.lower() in answer_lower)
        return hits / len(expected_entities)

    def measure(self, test_case: LLMTestCase) -> float:
        expected_entities = getattr(test_case, "expected_entities", [])
        self.score = self.compute(test_case.actual_output or "", expected_entities)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


class CitationAccuracyMetric(BaseMetric):
    """Overlap between cited sources in the answer and expected_sources."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.async_mode = False

    @property
    def __name__(self) -> str:
        return "CitationAccuracy"

    def compute(self, cited_sources: list[str], expected_sources: list[str]) -> float:
        if not expected_sources:
            return 1.0
        cited_lower = {s.lower() for s in cited_sources}
        expected_lower = {s.lower() for s in expected_sources}
        overlap = cited_lower & expected_lower
        return len(overlap) / len(expected_lower)

    def measure(self, test_case: LLMTestCase) -> float:
        cited = getattr(test_case, "cited_sources", [])
        expected = getattr(test_case, "expected_sources", [])
        self.score = self.compute(cited, expected)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text, stripping $ , B T suffixes.

    Excludes bare integers in the calendar year range (1900–2100) to prevent
    false positives from dates like FY2024.
    """
    pattern = r"\$?\s*(\d[\d,]*\.?\d*)\s*(?:billion|trillion|million|B|T|M)?"
    matches = re.findall(pattern, text, re.IGNORECASE)
    results = []
    for m in matches:
        try:
            val = float(m.replace(",", ""))
            # Skip bare integers in calendar year range
            if val == int(val) and 1900 <= val <= 2100:
                continue
            results.append(val)
        except ValueError:
            pass
    return results


class CompletenessMetric(BaseMetric):
    """LLM judge: does the answer cover all required points from expected_output?"""

    def __init__(self, model=None, threshold: float = 0.7):
        self.model = model
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.async_mode = False

    @property
    def __name__(self) -> str:
        return "Completeness"

    def _get_model(self):
        if self.model is not None:
            return self.model
        from evals.metrics.gemini_judge import GeminiJudge
        return GeminiJudge()

    def compute(self, actual_output: str, expected_output: str) -> float:
        prompt = (
            "You are evaluating whether an AI answer is complete relative to a reference answer.\n\n"
            f"Reference answer:\n{expected_output}\n\n"
            f"AI answer:\n{actual_output}\n\n"
            "Return JSON with keys 'score' (float 0-1) and 'reason' (string).\n"
            "Score 1.0 = covers all key points. Score 0.0 = misses most key points."
        )
        raw = self._get_model().generate(prompt)
        try:
            data = json.loads(raw.strip().strip("```json").strip("```"))
            return float(data.get("score", 0.0))
        except (json.JSONDecodeError, ValueError):
            return 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        self.score = self.compute(
            actual_output=test_case.actual_output or "",
            expected_output=test_case.expected_output or "",
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


class ContradictionDetectionRate(BaseMetric):
    """Tier-4 only: did the system surface the expected contradictions?"""

    def __init__(self, model=None, threshold: float = 0.6):
        self.model = model
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.async_mode = False

    @property
    def __name__(self) -> str:
        return "ContradictionDetectionRate"

    def _get_model(self):
        if self.model is not None:
            return self.model
        from evals.metrics.gemini_judge import GeminiJudge
        return GeminiJudge()

    def compute(self, actual_output: str, expected_contradictions: list[str]) -> float:
        if not expected_contradictions:
            return 1.0
        prompt = (
            "You are evaluating whether an AI response identified specific contradictions "
            "in M&A due diligence analysis.\n\n"
            "Expected contradictions to find:\n" +
            "\n".join(f"- {c}" for c in expected_contradictions) +
            f"\n\nAI response:\n{actual_output}\n\n"
            "Return JSON with 'score' (float 0-1) and 'reason' (string).\n"
            "Score = fraction of expected contradictions surfaced in the response."
        )
        raw = self._get_model().generate(prompt)
        try:
            data = json.loads(raw.strip().strip("```json").strip("```"))
            return float(data.get("score", 0.0))
        except (json.JSONDecodeError, ValueError):
            return 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        expected_contradictions = getattr(test_case, "expected_contradictions", [])
        self.score = self.compute(
            actual_output=test_case.actual_output or "",
            expected_contradictions=expected_contradictions,
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


class DueDiligenceConfidenceScore(BaseMetric):
    """Composite weighted score: faithfulness × completeness × numerical accuracy × citation."""

    WEIGHTS = {
        "faithfulness": 0.35,
        "completeness": 0.30,
        "numerical_accuracy": 0.25,
        "citation_accuracy": 0.10,
    }

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.async_mode = False

    @property
    def __name__(self) -> str:
        return "DueDiligenceConfidence"

    @staticmethod
    def weighted_score(
        faithfulness: float,
        completeness: float,
        numerical_accuracy: float,
        citation_accuracy: float,
    ) -> float:
        w = DueDiligenceConfidenceScore.WEIGHTS
        return (
            faithfulness * w["faithfulness"]
            + completeness * w["completeness"]
            + numerical_accuracy * w["numerical_accuracy"]
            + citation_accuracy * w["citation_accuracy"]
        )

    def measure(self, test_case: LLMTestCase) -> float:
        scores = getattr(test_case, "component_scores", {})
        self.score = self.weighted_score(
            faithfulness=scores.get("faithfulness", 0.0),
            completeness=scores.get("completeness", 0.0),
            numerical_accuracy=scores.get("numerical_accuracy", 0.0),
            citation_accuracy=scores.get("citation_accuracy", 0.0),
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success
