# Deep Evals — Evaluation Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a DeepEval-based evaluation framework that measures retrieval quality, generation quality, and domain-specific correctness across 4 ablation configurations, with results served via FastAPI endpoints.

**Architecture:** DeepEval metrics (Gemini 2.5 Flash as judge via Vertex AI) + custom `BaseMetric` subclasses for MRR, NDCG, and domain metrics. A Python runner invokes the LangGraph with 4 ablation configs (naive → +router → +reranker → full), writes timestamped JSON results. FastAPI exposes those results for the future React dashboard.

**Tech Stack:** `deepeval>=1.0.0`, `scikit-learn` (NDCG via `ndcg_score`), `google-genai` (Vertex AI judge), LangGraph, FastAPI, Pinecone, PostgreSQL.

**Spec:** `docs/superpowers/specs/2026-04-25-deep-evals-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `evals/__init__.py` | package marker |
| Create | `evals/dataset/__init__.py` | package marker |
| Create | `evals/dataset/golden_queries.json` | 30 ground truth Q&A pairs (AAPL/MSFT/NVDA) |
| Create | `evals/configs/__init__.py` | package marker |
| Create | `evals/configs/ablation_configs.py` | 4 ablation config dicts |
| Create | `evals/metrics/__init__.py` | package marker |
| Create | `evals/metrics/gemini_judge.py` | `DeepEvalBaseLLM` wrapper for Gemini 2.5 Flash |
| Create | `evals/metrics/retrieval.py` | `MRRMetric`, `NDCGMetric` |
| Create | `evals/metrics/due_diligence.py` | 6 custom domain metrics |
| Create | `evals/runner.py` | Eval orchestrator |
| Create | `evals/results/.gitkeep` | keep results dir in git |
| Create | `src/eval_api.py` | FastAPI router at `/eval` |
| Create | `tests/test_evals.py` | Unit tests for all custom metrics + runner + API |
| Modify | `pyproject.toml` | Add `deepeval>=1.0.0` |
| Modify | `src/state.py` | Add `eval_config: dict` field |
| Modify | `src/graph.py` | Add `eval_config` param to `build_graph()` |
| Modify | `src/nodes/router.py` | Short-circuit when `eval_config["router"] is False` |
| Modify | `src/nodes/retriever.py` | Skip reranker when `eval_config["reranker"] is False` |
| Modify | `src/nodes/grader.py` | Short-circuit when corrective/self_rag flags are False |
| Modify | `src/filings/raptor_retrieval.py` | Add `use_reranker: bool = True` param |
| Modify | `src/app.py` | Mount `eval_router` at `/eval` |

---

## Task 1: Add `deepeval` dependency and scaffold directories

**Files:**
- Modify: `pyproject.toml`
- Create: `evals/__init__.py`, `evals/dataset/__init__.py`, `evals/configs/__init__.py`, `evals/metrics/__init__.py`, `evals/results/.gitkeep`

- [ ] **Step 1: Add deepeval to pyproject.toml**

In `pyproject.toml`, add `"deepeval>=1.0.0",` to the `dependencies` list, after the existing `langgraph` entry:

```toml
    "langgraph>=0.2.0",
    "langchain-google-vertexai>=2.0.0",
    "redis>=5.0.0",
    "deepeval>=1.0.0",
```

- [ ] **Step 2: Install the dependency**

```bash
uv sync
```

Expected: resolves without conflicts. DeepEval pulls in `openai` as a transitive dep — that is fine, we will override the LLM judge with Gemini.

- [ ] **Step 3: Create package markers and results dir**

```bash
mkdir -p evals/dataset evals/configs evals/metrics evals/results
touch evals/__init__.py evals/dataset/__init__.py evals/configs/__init__.py evals/metrics/__init__.py evals/results/.gitkeep
```

- [ ] **Step 4: Verify import works**

```bash
uv run python -c "import deepeval; print(deepeval.__version__)"
```

Expected: prints a version string like `1.x.x`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock evals/
git commit -m "chore(evals): add deepeval dependency and scaffold evals/ directory"
```

---

## Task 2: Gemini judge wrapper

**Files:**
- Create: `evals/metrics/gemini_judge.py`

DeepEval's LLM-judge metrics (FaithfulnessMetric, etc.) call an LLM internally. We must subclass `DeepEvalBaseLLM` to plug in Gemini 2.5 Flash on Vertex AI instead of OpenAI.

- [ ] **Step 1: Write the failing test**

In `tests/test_evals.py` (create file):

```python
from unittest.mock import MagicMock, patch


def test_gemini_judge_get_model_name():
    from evals.metrics.gemini_judge import GeminiJudge
    judge = GeminiJudge()
    assert judge.get_model_name() == "gemini-2.5-flash"


def test_gemini_judge_generate_calls_genai(monkeypatch):
    mock_response = MagicMock()
    mock_response.text = "yes"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("evals.metrics.gemini_judge.get_genai_client", return_value=mock_client):
        from evals.metrics.gemini_judge import GeminiJudge
        judge = GeminiJudge()
        result = judge.generate("Is this faithful?")

    assert result == "yes"
    mock_client.models.generate_content.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_evals.py::test_gemini_judge_get_model_name tests/test_evals.py::test_gemini_judge_generate_calls_genai -v
```

Expected: `ModuleNotFoundError: No module named 'evals.metrics.gemini_judge'`

- [ ] **Step 3: Implement `evals/metrics/gemini_judge.py`**

```python
from __future__ import annotations

from deepeval.models.base_model import DeepEvalBaseLLM

from src.model_config import get_genai_client, get_model_name


class GeminiJudge(DeepEvalBaseLLM):
    def get_model_name(self) -> str:
        return get_model_name()

    def load_model(self):
        return get_genai_client()

    def generate(self, prompt: str) -> str:
        client = get_genai_client()
        response = client.models.generate_content(
            model=get_model_name(),
            contents=prompt,
        )
        return response.text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_evals.py::test_gemini_judge_get_model_name tests/test_evals.py::test_gemini_judge_generate_calls_genai -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/metrics/gemini_judge.py tests/test_evals.py
git commit -m "feat(evals): add GeminiJudge wrapper for DeepEval LLM judge"
```

---

## Task 3: Custom retrieval ranking metrics (MRR, NDCG)

**Files:**
- Create: `evals/metrics/retrieval.py`
- Modify: `tests/test_evals.py`

These metrics are pure math — no LLM calls. They compare `retrieved_sources` (list of source strings the system returned) against `expected_sources` (ground truth). The test case's `retrieval_context` list provides the retrieved chunks; we extract a source label from each chunk's first line or metadata tag.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evals.py`:

```python
def test_mrr_perfect_match():
    from evals.metrics.retrieval import MRRMetric
    metric = MRRMetric()
    # First retrieved source matches — reciprocal rank = 1.0
    score = metric.compute(
        retrieved_sources=["AAPL 10-K 2024", "MSFT 10-K 2024"],
        expected_sources=["AAPL 10-K 2024"],
    )
    assert score == 1.0


def test_mrr_second_position():
    from evals.metrics.retrieval import MRRMetric
    metric = MRRMetric()
    # First match is at position 2 — reciprocal rank = 0.5
    score = metric.compute(
        retrieved_sources=["MSFT 10-K 2024", "AAPL 10-K 2024"],
        expected_sources=["AAPL 10-K 2024"],
    )
    assert score == 0.5


def test_mrr_no_match():
    from evals.metrics.retrieval import MRRMetric
    metric = MRRMetric()
    score = metric.compute(
        retrieved_sources=["MSFT 10-K 2024"],
        expected_sources=["AAPL 10-K 2024"],
    )
    assert score == 0.0


def test_ndcg_perfect():
    from evals.metrics.retrieval import NDCGMetric
    metric = NDCGMetric(k=3)
    score = metric.compute(
        retrieved_sources=["AAPL 10-K 2024", "AAPL XBRL FY2024", "MSFT 10-K 2024"],
        expected_sources=["AAPL 10-K 2024", "AAPL XBRL FY2024"],
    )
    assert score == 1.0


def test_ndcg_none_relevant():
    from evals.metrics.retrieval import NDCGMetric
    metric = NDCGMetric(k=3)
    score = metric.compute(
        retrieved_sources=["NVDA 10-K 2024"],
        expected_sources=["AAPL 10-K 2024"],
    )
    assert score == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_evals.py -k "mrr or ndcg" -v
```

Expected: `ModuleNotFoundError: No module named 'evals.metrics.retrieval'`

- [ ] **Step 3: Implement `evals/metrics/retrieval.py`**

```python
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
        # ndcg_score expects shape (n_samples, n_labels)
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
    """Extract source label from retrieval context strings.

    Retrieval context entries are plain text chunks. We tag them with their
    source during test case construction in the runner, so by the time they
    reach here they are already source labels (e.g. 'AAPL 10-K 2024').
    """
    return retrieval_context
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_evals.py -k "mrr or ndcg" -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/metrics/retrieval.py tests/test_evals.py
git commit -m "feat(evals): add MRRMetric and NDCGMetric custom retrieval ranking metrics"
```

---

## Task 4: Custom domain metrics — no LLM

**Files:**
- Create: `evals/metrics/due_diligence.py`
- Modify: `tests/test_evals.py`

Three metrics that use pure computation: `NumericalAccuracyMetric`, `EntityMatchPrecisionMetric`, `CitationAccuracyMetric`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evals.py`:

```python
def test_numerical_accuracy_exact():
    from evals.metrics.due_diligence import NumericalAccuracyMetric
    m = NumericalAccuracyMetric()
    # Answer contains 391.0 — within 1% of expected 391.0
    score = m.compute(
        answer="Apple's FY2024 revenue was $391.0 billion.",
        expected_numbers=[391.0],
    )
    assert score == 1.0


def test_numerical_accuracy_within_tolerance():
    from evals.metrics.due_diligence import NumericalAccuracyMetric
    m = NumericalAccuracyMetric()
    # 392.0 is within 1% of 391.0 (diff = 0.256%)
    score = m.compute(
        answer="Revenue was approximately $392.0 billion.",
        expected_numbers=[391.0],
    )
    assert score == 1.0


def test_numerical_accuracy_wrong():
    from evals.metrics.due_diligence import NumericalAccuracyMetric
    m = NumericalAccuracyMetric()
    score = m.compute(
        answer="Revenue was $300 billion.",
        expected_numbers=[391.0],
    )
    assert score == 0.0


def test_entity_match_all_present():
    from evals.metrics.due_diligence import EntityMatchPrecisionMetric
    m = EntityMatchPrecisionMetric()
    score = m.compute(
        answer="Apple reported FY2024 revenue of $391.0 billion.",
        expected_entities=["Apple", "FY2024", "$391.0 billion"],
    )
    assert score == 1.0


def test_entity_match_partial():
    from evals.metrics.due_diligence import EntityMatchPrecisionMetric
    m = EntityMatchPrecisionMetric()
    score = m.compute(
        answer="Apple reported revenue.",
        expected_entities=["Apple", "FY2024", "$391.0 billion"],
    )
    assert abs(score - 1/3) < 0.01


def test_citation_accuracy_full_overlap():
    from evals.metrics.due_diligence import CitationAccuracyMetric
    m = CitationAccuracyMetric()
    score = m.compute(
        cited_sources=["AAPL 10-K 2024", "AAPL XBRL FY2024"],
        expected_sources=["AAPL 10-K 2024", "AAPL XBRL FY2024"],
    )
    assert score == 1.0


def test_citation_accuracy_no_overlap():
    from evals.metrics.due_diligence import CitationAccuracyMetric
    m = CitationAccuracyMetric()
    score = m.compute(
        cited_sources=["MSFT 10-K 2024"],
        expected_sources=["AAPL 10-K 2024"],
    )
    assert score == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_evals.py -k "numerical or entity_match or citation" -v
```

Expected: `ModuleNotFoundError: No module named 'evals.metrics.due_diligence'`

- [ ] **Step 3: Implement the three no-LLM metrics in `evals/metrics/due_diligence.py`**

```python
from __future__ import annotations

import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class NumericalAccuracyMetric(BaseMetric):
    """Fraction of expected_numbers found in the answer within 1% tolerance."""

    def __init__(self, tolerance: float = 0.01, threshold: float = 0.8):
        self.tolerance = tolerance
        self.threshold = threshold
        self.score = 0.0

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
    """Extract all numeric values from text, stripping $ , B T suffixes."""
    pattern = r"\$?\s*(\d[\d,]*\.?\d*)\s*(?:billion|trillion|million|B|T|M)?"
    matches = re.findall(pattern, text, re.IGNORECASE)
    results = []
    for m in matches:
        try:
            val = float(m.replace(",", ""))
            results.append(val)
        except ValueError:
            pass
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_evals.py -k "numerical or entity_match or citation" -v
```

Expected: all 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/metrics/due_diligence.py tests/test_evals.py
git commit -m "feat(evals): add NumericalAccuracyMetric, EntityMatchPrecisionMetric, CitationAccuracyMetric"
```

---

## Task 5: Custom domain metrics — LLM judge

**Files:**
- Modify: `evals/metrics/due_diligence.py`
- Modify: `tests/test_evals.py`

Three metrics that use the Gemini judge: `CompletenessMetric`, `ContradictionDetectionRate`, `DueDiligenceConfidenceScore`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evals.py`:

```python
def test_completeness_metric_mocked():
    from unittest.mock import MagicMock, patch
    from evals.metrics.due_diligence import CompletenessMetric

    mock_judge = MagicMock()
    mock_judge.generate.return_value = '{"score": 0.9, "reason": "covers all key points"}'

    m = CompletenessMetric(model=mock_judge)
    score = m.compute(
        actual_output="Apple FY2024 revenue was $391.0B driven by iPhone sales.",
        expected_output="Apple's total revenue for FY2024 was $391.0 billion.",
    )
    assert score == 0.9


def test_due_diligence_confidence_weighted():
    from evals.metrics.due_diligence import DueDiligenceConfidenceScore
    score = DueDiligenceConfidenceScore.weighted_score(
        faithfulness=1.0,
        completeness=1.0,
        numerical_accuracy=1.0,
        citation_accuracy=1.0,
    )
    assert abs(score - 1.0) < 0.001


def test_due_diligence_confidence_partial():
    from evals.metrics.due_diligence import DueDiligenceConfidenceScore
    # faithfulness=0, completeness=0, numerical=0, citation=0 → 0.0
    score = DueDiligenceConfidenceScore.weighted_score(
        faithfulness=0.0,
        completeness=0.0,
        numerical_accuracy=0.0,
        citation_accuracy=0.0,
    )
    assert score == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_evals.py -k "completeness or due_diligence_confidence" -v
```

Expected: `ImportError` or `AttributeError` — `CompletenessMetric` not yet defined.

- [ ] **Step 3: Append the three LLM-judge metrics to `evals/metrics/due_diligence.py`**

Add after the `_extract_numbers` function:

```python
import json


class CompletenessMetric(BaseMetric):
    """LLM judge: does the answer cover all required points from expected_output?"""

    def __init__(self, model=None, threshold: float = 0.7):
        self.model = model
        self.threshold = threshold
        self.score = 0.0

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
            f"Expected contradictions to find:\n" +
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_evals.py -k "completeness or due_diligence_confidence" -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Run the full test suite so far**

```bash
uv run pytest tests/test_evals.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add evals/metrics/due_diligence.py tests/test_evals.py
git commit -m "feat(evals): add CompletenessMetric, ContradictionDetectionRate, DueDiligenceConfidenceScore"
```

---

## Task 6: Ablation configurations

**Files:**
- Create: `evals/configs/ablation_configs.py`

- [ ] **Step 1: Create `evals/configs/ablation_configs.py`**

```python
from __future__ import annotations

ABLATION_CONFIGS: dict[str, dict] = {
    "naive_rag": {
        "router": False,
        "reranker": False,
        "corrective": False,
        "self_rag": False,
    },
    "plus_router": {
        "router": True,
        "reranker": False,
        "corrective": False,
        "self_rag": False,
    },
    "plus_reranker": {
        "router": True,
        "reranker": True,
        "corrective": False,
        "self_rag": False,
    },
    "full_system": {
        "router": True,
        "reranker": True,
        "corrective": True,
        "self_rag": True,
    },
    # Plug in when teammate's KG branch merges:
    # "plus_kg": {
    #     "router": True, "reranker": True, "corrective": True,
    #     "self_rag": True, "kg": True,
    # },
}
```

- [ ] **Step 2: Verify import**

```bash
uv run python -c "from evals.configs.ablation_configs import ABLATION_CONFIGS; print(list(ABLATION_CONFIGS))"
```

Expected: `['naive_rag', 'plus_router', 'plus_reranker', 'full_system']`

- [ ] **Step 3: Commit**

```bash
git add evals/configs/ablation_configs.py
git commit -m "feat(evals): add ablation configuration dicts"
```

---

## Task 7: Graph eval config flags

**Files:**
- Modify: `src/state.py` — add `eval_config` field
- Modify: `src/graph.py` — pass `eval_config` into initial state
- Modify: `src/nodes/router.py` — short-circuit when `router=False`
- Modify: `src/nodes/retriever.py` — skip reranker when `reranker=False`
- Modify: `src/nodes/grader.py` — short-circuit corrective/self_rag flags
- Modify: `src/filings/raptor_retrieval.py` — add `use_reranker` param

- [ ] **Step 1: Add `eval_config` to `src/state.py`**

In `src/state.py`, add `eval_config` as an optional field. The full file after the change:

```python
from typing import Any, Dict, List, Literal, TypedDict


class GraphState(TypedDict, total=False):
    chunking_strategy: str
    question: str
    rewritten_question: str
    retrieval_query: str
    web_search_error: str

    route: Literal[
        "sql",
        "filings",
        "transcripts",
        "patents",
        "litigation",
        "graph",
        "contradiction",
        "llm_direct",
    ]
    route_reason: str
    route_hint: str
    initial_route: str
    force_route: bool

    # M&A context
    company: str | None
    period: str | None
    source_type: str | None
    data_source_result: dict
    contradiction_report: list

    retrieved_docs: List[dict]
    filtered_docs: List[dict]
    web_results: List[dict]

    answer: str
    citations: List[str]

    doc_relevance: List[str]
    relevant_doc_count: int

    hallucination_grade: str
    answer_quality_grade: str

    iteration: int
    max_iterations: int

    retrieval_attempt: int
    max_retrieval_attempts: int

    # Evaluation ablation config — None in production, dict during eval runs
    eval_config: Dict[str, Any]
```

- [ ] **Step 2: Update `build_graph()` in `src/graph.py` to accept `eval_config`**

Change the `build_graph` function signature and the `initialize_state` node to inject the config:

```python
def build_graph(eval_config: dict | None = None):
    graph = StateGraph(GraphState)

    _eval_config = eval_config or {}

    def _initialize_state(state: GraphState) -> GraphState:
        return {
            **state,
            "iteration": state.get("iteration", 0),
            "max_iterations": state.get("max_iterations", 3),
            "retrieval_attempt": state.get("retrieval_attempt", 0),
            "max_retrieval_attempts": state.get("max_retrieval_attempts", 3),
            "eval_config": _eval_config,
        }

    graph.add_node("initialize", _initialize_state)
    # ... rest of nodes unchanged
```

Replace the `initialize_state` function reference in `graph.add_node("initialize", initialize_state)` with `_initialize_state` (defined inline inside `build_graph`). The standalone `initialize_state` function at module level can remain for backward compatibility but is no longer used by the graph.

Full replacement of `build_graph` in `src/graph.py`:

```python
def build_graph(eval_config: dict | None = None):
    graph = StateGraph(GraphState)

    _eval_config = eval_config or {}

    def _initialize_state(state: GraphState) -> GraphState:
        return {
            **state,
            "iteration": state.get("iteration", 0),
            "max_iterations": state.get("max_iterations", 3),
            "retrieval_attempt": state.get("retrieval_attempt", 0),
            "max_retrieval_attempts": state.get("max_retrieval_attempts", 3),
            "eval_config": _eval_config,
        }

    graph.add_node("initialize", _initialize_state)
    graph.add_node("router", route_question)
    graph.add_node("contradiction_check", run_contradiction_check)
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("grade_docs", grade_documents)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("increment_retrieval_attempt", increment_retrieval_attempt)
    graph.add_node("web_search", web_search_fallback)
    graph.add_node("generate", generate_answer)
    graph.add_node("llm_direct_generate", generate_direct_answer)
    graph.add_node("grade_hallucination", grade_hallucination)
    graph.add_node("grade_quality", grade_answer_quality)
    graph.add_node("increment_iteration", increment_iteration)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "contradiction": "contradiction_check",
            "llm_direct": "llm_direct_generate",
            "retrieve": "retrieve",
        },
    )

    graph.add_edge("contradiction_check", END)
    graph.add_edge("retrieve", "grade_docs")

    graph.add_conditional_edges(
        "grade_docs",
        route_after_doc_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fallback": "web_search",
        },
    )

    graph.add_edge("rewrite", "increment_retrieval_attempt")
    graph.add_edge("increment_retrieval_attempt", "retrieve")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", "grade_hallucination")
    graph.add_edge("llm_direct_generate", "grade_quality")

    graph.add_conditional_edges(
        "grade_hallucination",
        route_after_hallucination,
        {
            "quality": "grade_quality",
            "retry": "increment_iteration",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "grade_quality",
        route_after_quality,
        {
            "retry": "increment_iteration",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "increment_iteration",
        retry_route,
        {
            "web_search": "web_search",
            "end": END,
        },
    )

    return graph.compile()
```

- [ ] **Step 3: Update `route_question` in `src/nodes/router.py` to short-circuit when `router=False`**

Add the eval_config check at the top of `route_question`, before the `force_route` check:

```python
def route_question(state: GraphState) -> GraphState:
    eval_config = state.get("eval_config") or {}
    if eval_config.get("router") is False:
        # Naive RAG: bypass LLM router, default to filings (vector search)
        return {
            **state,
            "route": "filings",
            "initial_route": "filings",
            "route_reason": "eval: router disabled (naive RAG baseline)",
        }

    question = state["question"]
    route_hint = state.get("route_hint", "none") or "none"
    force_route = state.get("force_route", False)

    if force_route and route_hint in _VALID_ROUTES:
        return {
            **state,
            "route": route_hint,
            "initial_route": route_hint,
            "route_reason": f"Forced by planner route_hint: {route_hint}",
        }

    llm = get_router_llm().with_structured_output(RouteDecision)
    decision = llm.invoke([
        {"role": "system", "content": _SYSTEM_PROMPT.format(route_hint=route_hint)},
        {"role": "user", "content": question},
    ])

    return {
        **state,
        "route": decision.route,
        "initial_route": decision.route,
        "route_reason": decision.reason,
    }
```

- [ ] **Step 4: Add `use_reranker` param to `raptor_retrieve` in `src/filings/raptor_retrieval.py`**

Change the function signature from:

```python
def raptor_retrieve(
    query: str,
    top_k: int = 10,
    final_top_k: int = 8,
    namespace: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    max_children_per_summary: Optional[int] = None,
    use_simple_rerank: bool = False,
) -> Dict[str, Any]:
```

To:

```python
def raptor_retrieve(
    query: str,
    top_k: int = 10,
    final_top_k: int = 8,
    namespace: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    max_children_per_summary: Optional[int] = None,
    use_simple_rerank: bool = False,
    use_reranker: bool = True,
) -> Dict[str, Any]:
```

Then find the ranking section (after `deduplicate_by_id`) and wrap the `hybrid_rrf_rank` call:

```python
    dense_ranked_nodes = sort_nodes_by_score(combined_nodes)
    if use_reranker:
        ranked_nodes = hybrid_rrf_rank(
            query,
            dense_ranked_nodes,
            text_getter=lambda node: str(node.get("text", "")),
            key=lambda node: str(node.get("id") or ""),
            top_k=final_top_k,
        )
    else:
        ranked_nodes = dense_ranked_nodes[:final_top_k]
```

- [ ] **Step 5: Update `retrieve_docs` in `src/nodes/retriever.py` to pass `use_reranker`**

In the `elif route == "filings":` block, change the `raptor_retrieve` call:

```python
    elif route == "filings":
        try:
            from src.filings.raptor_retrieval import raptor_retrieve
            eval_config = state.get("eval_config") or {}
            use_reranker = eval_config.get("reranker", True)
            result = raptor_retrieve(query, top_k=10, final_top_k=6, use_reranker=use_reranker)
            docs = [
                {
                    "content": c.get("text", ""),
                    "metadata": {
                        "source": "SEC Filing",
                        "form_type": c.get("form_type"),
                        "rank": c.get("rank"),
                    },
                }
                for c in result.get("contexts", [])
            ]
        except Exception as exc:
            logger.warning("Filings retrieval failed: %s", exc)
```

- [ ] **Step 6: Update `grade_documents` in `src/nodes/grader.py` to short-circuit when `corrective=False`**

Add at the top of `grade_documents`, before the `if not retrieved_docs` check:

```python
def grade_documents(state: GraphState) -> GraphState:
    eval_config = state.get("eval_config") or {}
    if eval_config.get("corrective") is False:
        # Ablation: skip corrective grading, pass all docs through
        docs = state.get("retrieved_docs", [])
        return {
            **state,
            "filtered_docs": docs,
            "doc_relevance": ["yes"] * len(docs),
            "relevant_doc_count": len(docs),
        }

    question = state.get("rewritten_question") or state["question"]
    # ... rest of original function unchanged
```

- [ ] **Step 7: Update `grade_hallucination` and `grade_answer_quality` in `src/nodes/grader.py` to short-circuit when `self_rag=False`**

Add at the top of `grade_hallucination`:

```python
def grade_hallucination(state: GraphState) -> GraphState:
    eval_config = state.get("eval_config") or {}
    if eval_config.get("self_rag") is False:
        return {**state, "hallucination_grade": "yes"}

    answer = state.get("answer", "")
    # ... rest of original function unchanged
```

Add at the top of `grade_answer_quality`:

```python
def grade_answer_quality(state: GraphState) -> GraphState:
    eval_config = state.get("eval_config") or {}
    if eval_config.get("self_rag") is False:
        return {**state, "answer_quality_grade": "yes"}

    question = state["question"]
    # ... rest of original function unchanged
```

- [ ] **Step 8: Verify the graph builds without errors for all 4 configs**

```bash
uv run python -c "
from evals.configs.ablation_configs import ABLATION_CONFIGS
from src.graph import build_graph
for name, cfg in ABLATION_CONFIGS.items():
    g = build_graph(eval_config=cfg)
    print(f'{name}: OK')
"
```

Expected:
```
naive_rag: OK
plus_router: OK
plus_reranker: OK
full_system: OK
```

- [ ] **Step 9: Run existing tests to ensure nothing is broken**

```bash
uv run pytest -q
```

Expected: all existing tests PASS.

- [ ] **Step 10: Commit**

```bash
git add src/state.py src/graph.py src/nodes/router.py src/nodes/retriever.py src/nodes/grader.py src/filings/raptor_retrieval.py
git commit -m "feat(evals): add eval_config flags to graph nodes for ablation study"
```

---

## Task 8: Golden queries dataset

**Files:**
- Create: `evals/dataset/golden_queries.json`

All 30 queries are scoped to AAPL, MSFT, NVDA only. `expected_numbers` values are drawn from publicly reported financials — verify against the actual XBRL tables before running evals (`SELECT metric_name, value FROM xbrl_facts WHERE ticker='AAPL' AND fiscal_year=2024`).

- [ ] **Step 1: Create `evals/dataset/golden_queries.json`**

```json
[
  {
    "id": "t1_001",
    "tier": 1,
    "query": "What was Apple's total revenue in fiscal year 2024?",
    "expected_answer": "Apple's total revenue for fiscal year 2024 was approximately $391.0 billion.",
    "expected_sources": ["AAPL 10-K 2024", "AAPL XBRL FY2024"],
    "expected_entities": ["Apple", "FY2024", "391"],
    "expected_numbers": [391.0],
    "route": "sql"
  },
  {
    "id": "t1_002",
    "tier": 1,
    "query": "What are the primary risk factors disclosed in Apple's most recent 10-K filing?",
    "expected_answer": "Apple's 10-K discloses risks including supply chain concentration, geopolitical tensions, competition in smartphones and services, and regulatory scrutiny in app stores.",
    "expected_sources": ["AAPL 10-K 2024"],
    "expected_entities": ["Apple", "supply chain", "competition", "regulatory"],
    "expected_numbers": [],
    "route": "filings"
  },
  {
    "id": "t1_003",
    "tier": 1,
    "query": "What was Microsoft's total revenue for fiscal year 2024?",
    "expected_answer": "Microsoft's total revenue for fiscal year 2024 was approximately $245.1 billion.",
    "expected_sources": ["MSFT 10-K 2024", "MSFT XBRL FY2024"],
    "expected_entities": ["Microsoft", "FY2024", "245"],
    "expected_numbers": [245.1],
    "route": "sql"
  },
  {
    "id": "t1_004",
    "tier": 1,
    "query": "What material events did Nvidia disclose in 8-K filings during fiscal year 2024?",
    "expected_answer": "Nvidia's 8-K filings in FY2024 disclosed events including record data center revenue, H100 GPU supply agreements, and updated financial guidance.",
    "expected_sources": ["NVDA 8-K 2024"],
    "expected_entities": ["Nvidia", "H100", "data center"],
    "expected_numbers": [],
    "route": "filings"
  },
  {
    "id": "t1_005",
    "tier": 1,
    "query": "What was Nvidia's data center revenue for fiscal year 2025?",
    "expected_answer": "Nvidia's data center revenue for fiscal year 2025 (ending January 2025) was approximately $115.2 billion.",
    "expected_sources": ["NVDA XBRL FY2025"],
    "expected_entities": ["Nvidia", "data center", "FY2025", "115"],
    "expected_numbers": [115.2],
    "route": "sql"
  },
  {
    "id": "t1_006",
    "tier": 1,
    "query": "Who are the independent directors on Microsoft's board of directors?",
    "expected_answer": "Microsoft's independent directors include Satya Nadella (CEO/Chair), Reid Hoffman, Hugh Johnston, and other independent members disclosed in the proxy statement.",
    "expected_sources": ["MSFT DEF 14A 2024"],
    "expected_entities": ["Microsoft", "board", "directors"],
    "expected_numbers": [],
    "route": "filings"
  },
  {
    "id": "t1_007",
    "tier": 1,
    "query": "What is Apple's revenue breakdown by product segment for fiscal year 2024?",
    "expected_answer": "Apple FY2024 segment revenue: iPhone ~$201.2B, Services ~$96.2B, Mac ~$29.9B, iPad ~$26.7B, Wearables/Home/Accessories ~$37.0B.",
    "expected_sources": ["AAPL 10-K 2024", "AAPL XBRL FY2024"],
    "expected_entities": ["iPhone", "Services", "Mac", "iPad"],
    "expected_numbers": [201.2, 96.2, 29.9, 26.7],
    "route": "sql"
  },
  {
    "id": "t2_001",
    "tier": 2,
    "query": "Compare Apple's revenue recognition policy language across 10-K filings from 2021 to 2024. Has anything material changed?",
    "expected_answer": "Apple's revenue recognition policy has remained largely consistent from 2021-2024, following ASC 606. Minor updates in 2023-2024 reflect expanded services revenue disclosures.",
    "expected_sources": ["AAPL 10-K 2021", "AAPL 10-K 2022", "AAPL 10-K 2023", "AAPL 10-K 2024"],
    "expected_entities": ["Apple", "revenue recognition", "ASC 606"],
    "expected_numbers": [],
    "route": "filings"
  },
  {
    "id": "t2_002",
    "tier": 2,
    "query": "What was the quarter-over-quarter change in Apple's gross margin for each quarter in fiscal 2024?",
    "expected_answer": "Apple FY2024 gross margin by quarter: Q1 45.9%, Q2 46.6%, Q3 46.3%, Q4 46.2%. Services gross margin consistently above 70%.",
    "expected_sources": ["AAPL XBRL FY2024"],
    "expected_entities": ["Apple", "gross margin", "Q1", "Q2", "Q3", "Q4"],
    "expected_numbers": [45.9, 46.6, 46.3, 46.2],
    "route": "sql"
  },
  {
    "id": "t2_003",
    "tier": 2,
    "query": "How has Microsoft's risk factor language about 'cybersecurity' evolved from 2021 to 2024?",
    "expected_answer": "Microsoft's cybersecurity risk disclosures expanded significantly from 2021-2024, with 2023-2024 filings adding specific language about nation-state threats following the SolarWinds and Exchange incidents.",
    "expected_sources": ["MSFT 10-K 2021", "MSFT 10-K 2022", "MSFT 10-K 2023", "MSFT 10-K 2024"],
    "expected_entities": ["Microsoft", "cybersecurity", "nation-state"],
    "expected_numbers": [],
    "route": "filings"
  },
  {
    "id": "t2_004",
    "tier": 2,
    "query": "What is the trend in Microsoft's R&D spending as a percentage of revenue over the last 4 fiscal years?",
    "expected_answer": "Microsoft R&D as % of revenue: FY2021 ~12.5%, FY2022 ~12.0%, FY2023 ~12.8%, FY2024 ~12.2%. Relatively stable around 12%.",
    "expected_sources": ["MSFT XBRL FY2021", "MSFT XBRL FY2022", "MSFT XBRL FY2023", "MSFT XBRL FY2024"],
    "expected_entities": ["Microsoft", "R&D", "research and development"],
    "expected_numbers": [12.5, 12.0, 12.8, 12.2],
    "route": "sql"
  },
  {
    "id": "t2_005",
    "tier": 2,
    "query": "What did Nvidia's CEO say about data center demand on the fiscal Q4 2025 earnings call?",
    "expected_answer": "On the Q4 FY2025 earnings call, Nvidia's CEO Jensen Huang emphasized strong demand for Blackwell GPUs, citing hyperscaler capacity expansion and enterprise AI adoption.",
    "expected_sources": ["NVDA Earnings Transcript Q4 FY2025"],
    "expected_entities": ["Nvidia", "Jensen Huang", "Blackwell", "data center"],
    "expected_numbers": [],
    "route": "transcripts"
  },
  {
    "id": "t2_006",
    "tier": 2,
    "query": "Which specific Item 1A risk factors were added new in Apple's most recent 10-K versus the prior year?",
    "expected_answer": "Apple's FY2024 10-K added new risk factors around EU Digital Markets Act compliance and generative AI regulatory uncertainty not present in the FY2023 10-K.",
    "expected_sources": ["AAPL 10-K 2024", "AAPL 10-K 2023"],
    "expected_entities": ["Apple", "Digital Markets Act", "generative AI", "regulatory"],
    "expected_numbers": [],
    "route": "filings"
  },
  {
    "id": "t2_007",
    "tier": 2,
    "query": "How has Nvidia's gross margin evolved from fiscal year 2022 to fiscal year 2025?",
    "expected_answer": "Nvidia gross margin: FY2022 ~64.9%, FY2023 ~56.9% (crypto downturn), FY2024 ~72.7%, FY2025 ~75.0% (data center mix shift).",
    "expected_sources": ["NVDA XBRL FY2022", "NVDA XBRL FY2023", "NVDA XBRL FY2024", "NVDA XBRL FY2025"],
    "expected_entities": ["Nvidia", "gross margin", "data center"],
    "expected_numbers": [64.9, 56.9, 72.7, 75.0],
    "route": "sql"
  },
  {
    "id": "t2_008",
    "tier": 2,
    "query": "What percentage of Nvidia's recent patent filings are related to AI inference optimization?",
    "expected_answer": "Based on CPC code analysis, approximately 30-40% of Nvidia's patents filed in 2022-2024 relate to AI/ML inference optimization (G06N, H04L).",
    "expected_sources": ["NVDA Patents 2022-2024"],
    "expected_entities": ["Nvidia", "AI", "inference", "patents"],
    "expected_numbers": [],
    "route": "patents"
  },
  {
    "id": "t3_001",
    "tier": 3,
    "query": "Identify instances where Nvidia's CEO statements on earnings calls about data center demand contradicted guidance in 10-K filings from 2023-2024.",
    "expected_answer": "Analysis required: compare CEO statements on consecutive earnings calls with the risk factor language and guidance in subsequent 10-K/10-Q filings for any divergence in data center demand outlook.",
    "expected_sources": ["NVDA Earnings Transcripts 2023-2024", "NVDA 10-K 2023", "NVDA 10-K 2024"],
    "expected_entities": ["Nvidia", "data center", "contradiction"],
    "expected_numbers": [],
    "expected_contradictions": ["supply constraint language in 10-K vs. demand rhetoric on calls"],
    "route": "contradiction"
  },
  {
    "id": "t3_002",
    "tier": 3,
    "query": "Cross-reference Nvidia's patent filing activity in AI/ML with their data center revenue disclosures. Is patent investment proportional to revenue growth?",
    "expected_answer": "Nvidia data center revenue grew ~10x from FY2022 to FY2025. Patent filings in AI/ML grew approximately 3-4x in the same period, suggesting revenue growth outpaced patent investment as existing IP scaled.",
    "expected_sources": ["NVDA XBRL FY2022", "NVDA XBRL FY2025", "NVDA Patents 2022-2025"],
    "expected_entities": ["Nvidia", "patents", "data center", "AI/ML"],
    "expected_numbers": [],
    "expected_contradictions": [],
    "route": "contradiction"
  },
  {
    "id": "t3_003",
    "tier": 3,
    "query": "Compare AI-related risk language in the 10-K filings of Apple, Microsoft, and Nvidia. Which company has the most specific disclosures?",
    "expected_answer": "Microsoft has the most specific AI risk disclosures, including named regulatory bodies and specific product liability language. Apple focuses on competitive and regulatory risk. Nvidia focuses on export controls and compute access.",
    "expected_sources": ["AAPL 10-K 2024", "MSFT 10-K 2024", "NVDA 10-K 2024"],
    "expected_entities": ["Apple", "Microsoft", "Nvidia", "AI risk", "regulatory"],
    "expected_numbers": [],
    "expected_contradictions": [],
    "route": "filings"
  },
  {
    "id": "t3_004",
    "tier": 3,
    "query": "For Microsoft's Azure segment, correlate material 8-K disclosures in 2023-2024 with changes in risk factor language in subsequent 10-Q filings.",
    "expected_answer": "Microsoft's 8-K disclosures around Azure outages in 2023 were followed by strengthened risk factor language around service reliability and cyber incident response in subsequent 10-Q filings.",
    "expected_sources": ["MSFT 8-K 2023", "MSFT 8-K 2024", "MSFT 10-Q 2023", "MSFT 10-Q 2024"],
    "expected_entities": ["Microsoft", "Azure", "outage", "risk factors"],
    "expected_numbers": [],
    "expected_contradictions": [],
    "route": "filings"
  },
  {
    "id": "t3_005",
    "tier": 3,
    "query": "Compare customer concentration risk disclosures of Apple, Microsoft, and Nvidia with their actual revenue concentration from XBRL data.",
    "expected_answer": "Apple discloses no customer >10% of revenue; XBRL confirms broad distribution. Microsoft discloses no single customer >10%. Nvidia discloses significant hyperscaler concentration with top customers representing >40% of data center revenue.",
    "expected_sources": ["AAPL 10-K 2024", "MSFT 10-K 2024", "NVDA 10-K 2024", "NVDA XBRL FY2024"],
    "expected_entities": ["Apple", "Microsoft", "Nvidia", "customer concentration"],
    "expected_numbers": [40.0],
    "expected_contradictions": [],
    "route": "contradiction"
  },
  {
    "id": "t3_006",
    "tier": 3,
    "query": "Identify instances where Apple's earnings call guidance was materially inconsistent with risk factor disclosures in subsequent quarterly filings.",
    "expected_answer": "Analysis of Apple earnings calls vs. 10-Q risk factors: management consistently cited strong iPhone demand on calls while risk factors disclosed supply chain and demand uncertainty — a standard hedge inconsistency.",
    "expected_sources": ["AAPL Earnings Transcripts 2023-2024", "AAPL 10-Q 2023", "AAPL 10-Q 2024"],
    "expected_entities": ["Apple", "iPhone", "guidance", "risk factors"],
    "expected_numbers": [],
    "expected_contradictions": ["demand optimism on calls vs. supply risk in filings"],
    "route": "contradiction"
  },
  {
    "id": "t3_007",
    "tier": 3,
    "query": "Analyze Microsoft's Activision acquisition: compare risk factors in the proxy statement versus discussion on earnings calls before deal close.",
    "expected_answer": "The Activision proxy statement flagged regulatory approval risk prominently. Earnings call discussions were more optimistic about deal close timing, creating a measured divergence in tone before the FTC challenge.",
    "expected_sources": ["MSFT DEF 14A Activision", "MSFT Earnings Transcripts 2022-2023"],
    "expected_entities": ["Microsoft", "Activision", "FTC", "regulatory"],
    "expected_numbers": [],
    "expected_contradictions": ["regulatory optimism on calls vs. extensive risk disclosure in proxy"],
    "route": "contradiction"
  },
  {
    "id": "t3_008",
    "tier": 3,
    "query": "For Nvidia, trace the relationship between patent filings in automotive AI and automotive segment revenue over 2022-2024.",
    "expected_answer": "Nvidia's automotive segment revenue grew from ~$903M in FY2022 to ~$1.09B in FY2024. Patent filings in automotive AI increased during the same period, consistent with DRIVE platform investment.",
    "expected_sources": ["NVDA XBRL FY2022", "NVDA XBRL FY2024", "NVDA Patents 2022-2024"],
    "expected_entities": ["Nvidia", "automotive", "DRIVE", "patents"],
    "expected_numbers": [903.0, 1090.0],
    "expected_contradictions": [],
    "route": "patents"
  },
  {
    "id": "t4_001",
    "tier": 4,
    "query": "Build a complete due diligence risk profile for Apple: financial health trajectory from XBRL, IP portfolio strength from patents, and management credibility from earnings call versus filing consistency.",
    "expected_answer": "Apple due diligence profile: Financial — revenue CAGR ~7% FY2021-2024, gross margin expanding to 46%+, strong FCF. IP — 50,000+ active patents, strong in mobile/services. Management credibility — generally consistent between calls and filings with minor hedging differences on China exposure.",
    "expected_sources": ["AAPL XBRL FY2021-2024", "AAPL Patents", "AAPL Earnings Transcripts", "AAPL 10-K 2024"],
    "expected_entities": ["Apple", "financial health", "patents", "management credibility"],
    "expected_numbers": [46.0],
    "expected_contradictions": ["China revenue optimism on calls vs. geographic risk in filings"],
    "route": "contradiction"
  },
  {
    "id": "t4_002",
    "tier": 4,
    "query": "For Microsoft, analyze the last 3 years of 10-K filings: identify all related-party transactions from footnotes, cross-reference with executive affiliations, and compute total as percentage of revenue.",
    "expected_answer": "Microsoft 10-K footnotes disclose related-party transactions primarily through board member affiliations with portfolio companies. Total disclosed related-party transaction value is well below 1% of revenue in all three years.",
    "expected_sources": ["MSFT 10-K 2022", "MSFT 10-K 2023", "MSFT 10-K 2024"],
    "expected_entities": ["Microsoft", "related-party", "transactions", "footnotes"],
    "expected_numbers": [],
    "expected_contradictions": [],
    "route": "contradiction"
  },
  {
    "id": "t4_003",
    "tier": 4,
    "query": "For a hypothetical acquisition of Nvidia: identify top 10 risk factors from filings, assess IP defensibility from patent portfolio, compute key financial ratios from XBRL, and analyze management tone shift over 4 earnings calls.",
    "expected_answer": "Nvidia acquisition analysis: Top risks — export controls, hyperscaler concentration, competitive threat from AMD/custom silicon. IP — strong H100/Blackwell patent moat. Financials — P/E elevated, ROIC >60%. Management tone — increasingly confident Q1-Q4 FY2025 on AI demand.",
    "expected_sources": ["NVDA 10-K 2024", "NVDA Patents", "NVDA XBRL FY2025", "NVDA Earnings Transcripts FY2025"],
    "expected_entities": ["Nvidia", "export controls", "hyperscaler", "AMD"],
    "expected_numbers": [60.0],
    "expected_contradictions": [],
    "route": "contradiction"
  },
  {
    "id": "t4_004",
    "tier": 4,
    "query": "Identify whether Apple, Microsoft, or Nvidia disclosed any material weakness in internal controls in filings from 2022-2024, and if so, was it remediated in subsequent filings?",
    "expected_answer": "None of Apple, Microsoft, or Nvidia disclosed material weaknesses in internal controls in their 10-K or 10-Q filings from 2022-2024. All three maintained effective ICFR per SOX 404 assessments.",
    "expected_sources": ["AAPL 10-K 2022-2024", "MSFT 10-K 2022-2024", "NVDA 10-K 2022-2024"],
    "expected_entities": ["material weakness", "internal controls", "SOX 404"],
    "expected_numbers": [],
    "expected_contradictions": [],
    "route": "filings"
  },
  {
    "id": "t4_005",
    "tier": 4,
    "query": "Compare full due diligence risk profiles of Apple and Microsoft: financial trajectory, IP portfolio, management credibility, and litigation exposure.",
    "expected_answer": "Apple vs Microsoft: Financial — both strong FCF generators, MSFT higher revenue CAGR from cloud. IP — AAPL mobile/chip patents vs MSFT software/cloud patents. Management credibility — both consistent. Litigation — AAPL faces app store antitrust, MSFT faces gaming/cloud competition scrutiny.",
    "expected_sources": ["AAPL 10-K 2024", "MSFT 10-K 2024", "AAPL Patents", "MSFT Patents", "AAPL XBRL", "MSFT XBRL"],
    "expected_entities": ["Apple", "Microsoft", "financial trajectory", "IP portfolio", "litigation"],
    "expected_numbers": [],
    "expected_contradictions": [],
    "route": "contradiction"
  },
  {
    "id": "t4_006",
    "tier": 4,
    "query": "For Apple, compute LTM financial metrics from XBRL, identify IP portfolio concentration risks from patent data, and flag any management statement inconsistencies from earnings calls.",
    "expected_answer": "Apple LTM (FY2024): Revenue $391.0B, Gross Margin 46.2%, Net Income $93.7B, FCF ~$108B. IP concentration: 65%+ patents in mobile/consumer electronics — moderate concentration risk. Inconsistencies: China demand narrative on calls vs. geographic risk disclosure.",
    "expected_sources": ["AAPL XBRL FY2024", "AAPL Patents", "AAPL Earnings Transcripts 2024"],
    "expected_entities": ["Apple", "LTM", "gross margin", "FCF", "China"],
    "expected_numbers": [391.0, 46.2, 93.7, 108.0],
    "expected_contradictions": ["China demand optimism vs. geographic concentration risk in filings"],
    "route": "contradiction"
  },
  {
    "id": "t4_007",
    "tier": 4,
    "query": "Analyze Nvidia's forward guidance accuracy: compare CEO statements on AI chip demand across 4 consecutive earnings calls with actual reported revenue in subsequent 10-Q filings.",
    "expected_answer": "Nvidia's Jensen Huang consistently guided data center demand higher on Q1-Q4 FY2025 earnings calls. Actual reported revenue exceeded guidance in all 4 quarters — guidance was conservative, not optimistic, which is the inverse of typical management credibility risk.",
    "expected_sources": ["NVDA Earnings Transcripts FY2025", "NVDA XBRL FY2025"],
    "expected_entities": ["Nvidia", "guidance", "Jensen Huang", "data center", "revenue"],
    "expected_numbers": [],
    "expected_contradictions": ["guidance consistently below actuals — conservative signaling pattern"],
    "route": "contradiction"
  }
]
```

- [ ] **Step 2: Verify JSON is valid**

```bash
uv run python -c "import json; data=json.load(open('evals/dataset/golden_queries.json')); print(f'{len(data)} queries loaded, tiers: {set(q[\"tier\"] for q in data)}')"
```

Expected: `30 queries loaded, tiers: {1, 2, 3, 4}`

- [ ] **Step 3: Commit**

```bash
git add evals/dataset/golden_queries.json
git commit -m "feat(evals): add 30 golden ground truth queries (AAPL/MSFT/NVDA)"
```

---

## Task 9: Eval runner

**Files:**
- Create: `evals/runner.py`
- Modify: `tests/test_evals.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evals.py`:

```python
def test_runner_produces_result_file(tmp_path):
    from unittest.mock import patch, MagicMock
    from evals.runner import EvalRunner

    # 2-query mini dataset
    mini_dataset = [
        {
            "id": "t1_001",
            "tier": 1,
            "query": "What was Apple's total revenue in FY2024?",
            "expected_answer": "Apple FY2024 revenue was $391.0 billion.",
            "expected_sources": ["AAPL XBRL FY2024"],
            "expected_entities": ["Apple", "FY2024"],
            "expected_numbers": [391.0],
            "expected_contradictions": [],
            "route": "sql",
        }
    ]
    mini_configs = {
        "naive_rag": {"router": False, "reranker": False, "corrective": False, "self_rag": False}
    }

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "answer": "Apple's FY2024 revenue was $391.0 billion.",
        "citations": ["AAPL XBRL FY2024"],
        "retrieved_docs": [{"content": "Revenue 391B", "metadata": {"source": "AAPL XBRL FY2024"}}],
    }

    with patch("evals.runner.build_graph", return_value=mock_graph):
        runner = EvalRunner(output_dir=str(tmp_path))
        run_id = runner.run(configs=mini_configs, dataset=mini_dataset, skip_llm_metrics=True)

    result_file = tmp_path / f"{run_id}.json"
    assert result_file.exists()
    import json
    data = json.loads(result_file.read_text())
    assert "naive_rag" in data["configs"]
    assert "tier_1" in data["configs"]["naive_rag"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_evals.py::test_runner_produces_result_file -v
```

Expected: `ModuleNotFoundError: No module named 'evals.runner'`

- [ ] **Step 3: Implement `evals/runner.py`**

```python
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.graph import build_graph
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"


class EvalRunner:
    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        configs: dict[str, dict],
        dataset: list[dict],
        skip_llm_metrics: bool = False,
    ) -> str:
        run_id = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
        result: dict[str, Any] = {
            "run_id": run_id,
            "completed_at": None,
            "configs": {},
            "baseline_delta": {},
        }

        for config_name, eval_config in configs.items():
            logger.info("Running ablation config: %s", config_name)
            config_results = self._run_config(
                config_name=config_name,
                eval_config=eval_config,
                dataset=dataset,
                skip_llm_metrics=skip_llm_metrics,
            )
            result["configs"][config_name] = config_results
            # Incremental write after each config completes
            self._write(run_id, result)

        result["completed_at"] = datetime.utcnow().isoformat()
        result["baseline_delta"] = self._compute_deltas(result["configs"])
        self._write(run_id, result)

        logger.info("Eval run %s complete. Results at %s/%s.json", run_id, self.output_dir, run_id)
        return run_id

    def _run_config(
        self,
        config_name: str,
        eval_config: dict,
        dataset: list[dict],
        skip_llm_metrics: bool,
    ) -> dict[str, dict]:
        graph = build_graph(eval_config=eval_config)
        tier_scores: dict[str, list[dict]] = defaultdict(list)

        for item in dataset:
            tier_key = f"tier_{item['tier']}"
            try:
                scores = self._evaluate_item(graph, item, skip_llm_metrics)
                tier_scores[tier_key].append(scores)
            except Exception as exc:
                logger.warning("Item %s failed in config %s: %s", item["id"], config_name, exc)
                tier_scores[tier_key].append({})

        return {
            tier: _average_scores(score_list)
            for tier, score_list in tier_scores.items()
        }

    def _evaluate_item(self, graph, item: dict, skip_llm_metrics: bool) -> dict[str, float]:
        response = graph.invoke({
            "question": item["query"],
            "company": _extract_company(item["query"]),
        })

        answer = response.get("answer", "")
        citations = response.get("citations", [])
        retrieved_docs = response.get("retrieved_docs", [])
        retrieved_sources = [
            d.get("metadata", {}).get("source", "") for d in retrieved_docs
        ]

        scores: dict[str, float] = {}

        # --- Retrieval ranking metrics (no LLM) ---
        from evals.metrics.retrieval import MRRMetric, NDCGMetric
        scores["mrr"] = MRRMetric().compute(retrieved_sources, item.get("expected_sources", []))
        scores["ndcg"] = NDCGMetric().compute(retrieved_sources, item.get("expected_sources", []))

        # --- Domain metrics (no LLM) ---
        from evals.metrics.due_diligence import (
            NumericalAccuracyMetric,
            EntityMatchPrecisionMetric,
            CitationAccuracyMetric,
        )
        scores["numerical_accuracy"] = NumericalAccuracyMetric().compute(
            answer, item.get("expected_numbers", [])
        )
        scores["entity_match_precision"] = EntityMatchPrecisionMetric().compute(
            answer, item.get("expected_entities", [])
        )
        scores["citation_accuracy"] = CitationAccuracyMetric().compute(
            citations, item.get("expected_sources", [])
        )

        if not skip_llm_metrics:
            scores.update(self._run_llm_metrics(answer, item, retrieved_sources, scores))

        return scores

    def _run_llm_metrics(
        self,
        answer: str,
        item: dict,
        retrieved_sources: list[str],
        prior_scores: dict[str, float],
    ) -> dict[str, float]:
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            HallucinationMetric,
        )
        from evals.metrics.gemini_judge import GeminiJudge
        from evals.metrics.due_diligence import (
            CompletenessMetric,
            ContradictionDetectionRate,
            DueDiligenceConfidenceScore,
        )

        judge = GeminiJudge()
        test_case = LLMTestCase(
            input=item["query"],
            actual_output=answer,
            expected_output=item.get("expected_answer", ""),
            retrieval_context=retrieved_sources,
            context=retrieved_sources,
        )

        scores: dict[str, float] = {}

        tier = item.get("tier", 1)

        faithfulness_metric = FaithfulnessMetric(model=judge, threshold=0.7)
        faithfulness_metric.measure(test_case)
        scores["faithfulness"] = faithfulness_metric.score

        relevancy_metric = AnswerRelevancyMetric(model=judge, threshold=0.7)
        relevancy_metric.measure(test_case)
        scores["answer_relevancy"] = relevancy_metric.score

        hallucination_metric = HallucinationMetric(model=judge, threshold=0.3)
        hallucination_metric.measure(test_case)
        scores["hallucination_rate"] = hallucination_metric.score

        if tier in (1, 2, 3):
            from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
            cp = ContextualPrecisionMetric(model=judge, threshold=0.7)
            cp.measure(test_case)
            scores["contextual_precision"] = cp.score

            cr = ContextualRecallMetric(model=judge, threshold=0.7)
            cr.measure(test_case)
            scores["contextual_recall"] = cr.score

        completeness = CompletenessMetric(model=judge)
        scores["completeness"] = completeness.compute(answer, item.get("expected_answer", ""))

        if tier == 4:
            cdr = ContradictionDetectionRate(model=judge)
            scores["contradiction_detection_rate"] = cdr.compute(
                answer, item.get("expected_contradictions", [])
            )

        scores["due_diligence_confidence"] = DueDiligenceConfidenceScore.weighted_score(
            faithfulness=scores.get("faithfulness", 0.0),
            completeness=scores.get("completeness", 0.0),
            numerical_accuracy=prior_scores.get("numerical_accuracy", 0.0),
            citation_accuracy=prior_scores.get("citation_accuracy", 0.0),
        )

        return scores

    def _compute_deltas(self, configs: dict) -> dict:
        deltas: dict = {}
        if "naive_rag" not in configs:
            return deltas
        baseline = configs["naive_rag"]
        for config_name, config_data in configs.items():
            if config_name == "naive_rag":
                continue
            delta: dict = {}
            for tier, tier_metrics in config_data.items():
                tier_baseline = baseline.get(tier, {})
                delta[tier] = {
                    metric: round(val - tier_baseline.get(metric, 0.0), 4)
                    for metric, val in tier_metrics.items()
                }
            deltas[f"{config_name}_vs_naive"] = delta
        return deltas

    def _write(self, run_id: str, result: dict) -> None:
        path = self.output_dir / f"{run_id}.json"
        path.write_text(json.dumps(result, indent=2))


def _average_scores(score_list: list[dict]) -> dict[str, float]:
    if not score_list:
        return {}
    keys = {k for d in score_list for k in d}
    averages: dict[str, float] = {}
    for key in keys:
        vals = [d[key] for d in score_list if key in d]
        averages[key] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return averages


def _extract_company(query: str) -> str | None:
    query_lower = query.lower()
    if "apple" in query_lower or "aapl" in query_lower:
        return "Apple"
    if "microsoft" in query_lower or "msft" in query_lower:
        return "Microsoft"
    if "nvidia" in query_lower or "nvda" in query_lower:
        return "Nvidia"
    return None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_evals.py::test_runner_produces_result_file -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/test_evals.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add evals/runner.py tests/test_evals.py
git commit -m "feat(evals): add EvalRunner orchestrator with incremental JSON output"
```

---

## Task 10: FastAPI eval API

**Files:**
- Create: `src/eval_api.py`
- Modify: `src/app.py`
- Modify: `tests/test_evals.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evals.py`:

```python
def test_eval_api_list_runs(tmp_path):
    import json
    from fastapi.testclient import TestClient

    # Write a fixture result file
    fixture = {
        "run_id": "2026-04-25T12-00-00",
        "completed_at": "2026-04-25T12:06:00",
        "configs": {"naive_rag": {"tier_1": {"mrr": 0.5}}},
        "baseline_delta": {},
    }
    (tmp_path / "2026-04-25T12-00-00.json").write_text(json.dumps(fixture))

    import evals.eval_api as eval_api_module
    eval_api_module._RESULTS_DIR = tmp_path  # override for test

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(eval_api_module.router)
    client = TestClient(app)

    resp = client.get("/eval/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert "2026-04-25T12-00-00" in data["run_ids"]


def test_eval_api_get_run(tmp_path):
    import json
    import evals.eval_api as eval_api_module
    eval_api_module._RESULTS_DIR = tmp_path

    fixture = {
        "run_id": "2026-04-25T12-00-00",
        "completed_at": "2026-04-25T12:06:00",
        "configs": {"full_system": {"tier_1": {"mrr": 0.8}}},
        "baseline_delta": {},
    }
    (tmp_path / "2026-04-25T12-00-00.json").write_text(json.dumps(fixture))

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(eval_api_module.router)
    client = TestClient(app)

    resp = client.get("/eval/runs/2026-04-25T12-00-00")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "2026-04-25T12-00-00"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_evals.py -k "eval_api" -v
```

Expected: `ModuleNotFoundError: No module named 'evals.eval_api'`

Note: The test imports from `evals.eval_api`, not `src.eval_api`. Create the file at `src/eval_api.py` and also create `evals/eval_api.py` as a re-export shim so tests can import it cleanly. Actually, create the file at `src/eval_api.py` and update the test import to `src.eval_api`.

**Correction:** Update the tests to use `src.eval_api`:

Replace the two test functions above (overwrite in test file) using:
```python
import src.eval_api as eval_api_module
```
instead of `import evals.eval_api as eval_api_module`.

- [ ] **Step 3: Implement `src/eval_api.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

_RESULTS_DIR = Path(__file__).parent.parent / "evals" / "results"

router = APIRouter(prefix="/eval", tags=["evaluation"])


def _results_dir() -> Path:
    return _RESULTS_DIR


def _load_run(run_id: str) -> dict[str, Any]:
    path = _results_dir() / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    import json
    return json.loads(path.read_text())


@router.get("/runs")
def list_runs() -> dict[str, list[str]]:
    d = _results_dir()
    if not d.exists():
        return {"run_ids": []}
    run_ids = sorted(
        p.stem for p in d.glob("*.json") if p.stem != ".gitkeep"
    )
    return {"run_ids": run_ids}


@router.get("/runs/latest")
def get_latest_run() -> dict[str, Any]:
    d = _results_dir()
    if not d.exists():
        raise HTTPException(status_code=404, detail="No runs found")
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise HTTPException(status_code=404, detail="No runs found")
    import json
    return json.loads(files[0].read_text())


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    return _load_run(run_id)


@router.get("/runs/{run_id}/summary")
def get_run_summary(run_id: str) -> dict[str, Any]:
    data = _load_run(run_id)
    summary: dict[str, Any] = {"run_id": run_id, "configs": {}}
    for config_name, tiers in data.get("configs", {}).items():
        config_summary: dict[str, float] = {}
        all_metrics: dict[str, list[float]] = {}
        for tier_metrics in tiers.values():
            for metric, val in tier_metrics.items():
                all_metrics.setdefault(metric, []).append(val)
        config_summary = {
            m: round(sum(vals) / len(vals), 4) for m, vals in all_metrics.items()
        }
        summary["configs"][config_name] = config_summary
    return summary


@router.get("/runs/{run_id}/ablation")
def get_run_ablation(run_id: str) -> dict[str, Any]:
    data = _load_run(run_id)
    return {
        "run_id": run_id,
        "configs": data.get("configs", {}),
        "baseline_delta": data.get("baseline_delta", {}),
    }


@router.post("/trigger")
def trigger_eval(background_tasks: BackgroundTasks) -> dict[str, str]:
    import json
    from evals.configs.ablation_configs import ABLATION_CONFIGS
    from evals.runner import EvalRunner

    dataset_path = Path(__file__).parent.parent / "evals" / "dataset" / "golden_queries.json"
    if not dataset_path.exists():
        raise HTTPException(status_code=500, detail="Golden queries dataset not found")

    dataset = json.loads(dataset_path.read_text())

    from datetime import datetime
    run_id = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")

    def _run():
        runner = EvalRunner()
        runner.run(configs=ABLATION_CONFIGS, dataset=dataset)

    background_tasks.add_task(_run)
    return {"run_id": run_id, "status": "started"}
```

- [ ] **Step 4: Mount the eval router in `src/app.py`**

Add after the existing imports:

```python
from src.eval_api import router as eval_router
```

Add after `app.add_middleware(...)`:

```python
app.include_router(eval_router)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_evals.py -k "eval_api" -v
```

Expected: both PASS.

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/eval_api.py src/app.py tests/test_evals.py
git commit -m "feat(evals): add FastAPI eval dashboard endpoints at /eval"
```

---

## Task 11: Wire results `.gitkeep` and final integration check

**Files:**
- Modify: `.gitignore` — ignore result JSON files but keep the dir

- [ ] **Step 1: Update `.gitignore` to ignore eval result files**

Add to `.gitignore`:

```
# Eval results (large, machine-generated)
evals/results/*.json
```

- [ ] **Step 2: Run the full test suite one last time**

```bash
uv run pytest -q
```

Expected: all tests PASS, no warnings about missing modules.

- [ ] **Step 3: Verify the eval CLI entry point works (dry run, no live calls)**

```bash
uv run python -c "
from evals.configs.ablation_configs import ABLATION_CONFIGS
from evals.runner import EvalRunner
from evals.metrics.retrieval import MRRMetric, NDCGMetric
from evals.metrics.due_diligence import NumericalAccuracyMetric, DueDiligenceConfidenceScore
print('All eval modules import cleanly')
print('ABLATION_CONFIGS:', list(ABLATION_CONFIGS))
print('MRR spot check:', MRRMetric().compute(['AAPL XBRL FY2024'], ['AAPL XBRL FY2024']))
print('NDCG spot check:', NDCGMetric().compute(['AAPL XBRL FY2024', 'X'], ['AAPL XBRL FY2024']))
print('NumericalAccuracy spot check:', NumericalAccuracyMetric().compute('Revenue was 391 billion', [391.0]))
print('DueDiligenceConfidence spot check:', DueDiligenceConfidenceScore.weighted_score(1,1,1,1))
"
```

Expected:
```
All eval modules import cleanly
ABLATION_CONFIGS: ['naive_rag', 'plus_router', 'plus_reranker', 'full_system']
MRR spot check: 1.0
NDCG spot check: 1.0
NumericalAccuracy spot check: 1.0
DueDiligenceConfidence spot check: 1.0
```

- [ ] **Step 4: Final commit**

```bash
git add .gitignore evals/results/.gitkeep
git commit -m "chore(evals): ignore result JSON files, keep results dir tracked"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Task(s) covering it |
|---|---|
| DeepEval framework | Task 1 (dep), Task 2 (judge), Tasks 3-5 (metrics) |
| 30 golden queries (AAPL/MSFT/NVDA) | Task 8 |
| Retrieval metrics: precision, recall, MRR, NDCG | Task 3, Task 9 (runner) |
| Generation metrics: faithfulness, relevancy, hallucination | Task 9 (runner, LLM branch) |
| Domain metrics: numerical accuracy, entity match, citation, completeness, contradiction, DD confidence | Tasks 4-5 |
| 4 ablation configs | Task 6 |
| Graph config flags (router/reranker/corrective/self_rag) | Task 7 |
| Eval runner with incremental JSON writes | Task 9 |
| FastAPI dashboard endpoints (6 routes) | Task 10 |
| Unit tests for all custom metrics + runner + API | Tasks 2-5, 9, 10 |
| `.gitignore` for result files | Task 11 |

All spec requirements covered. No gaps found.
