# Deep Evals — Evaluation Framework Design

**Date:** 2026-04-25  
**Branch:** `feature/RG-Implement-Deep-Evals-for-RAG-evaluation`  
**Author:** Ruby Gunna

---

## 1. Purpose

Build a rigorous, automated evaluation framework for M&A Oracle that measures retrieval quality,
generation quality, and domain-specific correctness. The framework must produce an ablation study
showing the contribution of each pipeline component and expose results via FastAPI endpoints for
a future React dashboard.

---

## 2. Scope

- **In scope:** evaluation harness, ground truth dataset, all metrics, ablation runner, FastAPI dashboard API
- **Out of scope:** React frontend (next feature), KG ablation config (plugged in when teammate's branch merges), human evaluation loop
- **Data constraint:** Ground truth queries cover AAPL, MSFT, NVDA only — the only companies ingested into Pinecone and PostgreSQL

---

## 3. Architecture

```
evals/
├── dataset/
│   └── golden_queries.json        # 30 ground truth Q&A pairs
├── configs/
│   └── ablation_configs.py        # 4 ablation config dicts + KG placeholder
├── metrics/
│   ├── gemini_judge.py            # DeepEvalBaseLLM wrapper for Gemini 2.5 Flash (Vertex AI)
│   ├── retrieval.py               # Custom MRRMetric, NDCGMetric (pure math, no LLM)
│   └── due_diligence.py           # Custom domain metrics (numerical accuracy, entity match,
│                                  #   citation accuracy, DD confidence, contradiction rate,
│                                  #   completeness)
├── runner.py                      # Orchestrator: dataset → configs → metrics → JSON
└── results/                       # Timestamped run files: evals/results/{run_id}.json

src/eval_api.py                    # FastAPI router mounted at /eval
```

**Data flow:**

```
golden_queries.json
       │
       ▼
runner.py  ──▶  ablation config (4 configs)
                       │
                       ▼
              graph.py (config flags disable/enable nodes)
                       │
              ┌────────┴────────┐
              │  DeepEval       │  ← Gemini 2.5 Flash judge (Vertex AI)
              │  metrics        │
              └────────┬────────┘
                       │
              evals/results/{timestamp}.json
                       │
                       ▼
              GET /eval/runs/{run_id}  →  React dashboard (future)
```

---

## 4. Ground Truth Dataset

**File:** `evals/dataset/golden_queries.json`  
**Size:** 30 queries across 4 tiers. All queries scoped to AAPL, MSFT, NVDA.

Each record:
```json
{
  "id": "t1_001",
  "tier": 1,
  "query": "What was Apple's total revenue in fiscal year 2024?",
  "expected_answer": "Apple's total revenue for fiscal year 2024 was $391.0 billion.",
  "expected_sources": ["AAPL 10-K 2024", "AAPL XBRL FY2024"],
  "expected_entities": ["Apple", "FY2024", "$391.0 billion"],
  "expected_numbers": [391.0],
  "route": "sql"
}
```

**Tier distribution:**

| Tier | Count | Query type | Primary route |
|------|-------|-----------|---------------|
| 1 | 7 | Simple factual lookup | `sql` / `llm_direct` |
| 2 | 8 | Version/specificity, multi-filing | `filings` / `transcripts` |
| 3 | 8 | Multi-source cross-reference | hybrid (sql + filings + patents) |
| 4 | 7 | Full agentic due diligence | `contradiction` |

---

## 5. Metrics

### 5.1 DeepEval Built-in (Gemini judge)

| Metric | Class | Applied to tiers |
|--------|-------|-----------------|
| Contextual Precision | `ContextualPrecisionMetric` | 1–3 |
| Contextual Recall | `ContextualRecallMetric` | 1–3 |
| Faithfulness | `FaithfulnessMetric` | all |
| Answer Relevancy | `AnswerRelevancyMetric` | all |
| Hallucination | `HallucinationMetric` | all |

### 5.2 Custom `BaseMetric` — Retrieval Ranking (no LLM)

**`MRRMetric`** — Mean Reciprocal Rank across retrieved contexts vs. expected sources.  
**`NDCGMetric`** — NDCG@k (default k=5) using binary relevance from expected sources.

Both take `retrieved_sources: list[str]` and `expected_sources: list[str]` from the test case.

### 5.3 Custom `BaseMetric` — Domain-Specific (Gemini judge where noted)

| Metric | Judge | Description |
|--------|-------|-------------|
| `NumericalAccuracyMetric` | no | Extracts numbers from answer, checks within 1% tolerance of `expected_numbers` |
| `EntityMatchPrecisionMetric` | no | Case-insensitive string match: count of `expected_entities` found in answer text / total expected entities |
| `CitationAccuracyMetric` | no | Source string overlap: cited sources vs. `expected_sources` |
| `CompletenessMetric` | Gemini | LLM judge: does the answer cover all required points from `expected_answer`? |
| `ContradictionDetectionRate` | Gemini | Tier 4 only: did the system surface the expected contradictions? |
| `DueDiligenceConfidenceScore` | Gemini | Composite: faithfulness × completeness × numerical accuracy. Weighted 0–1. |

**`DueDiligenceConfidenceScore` weights:**
- Faithfulness: 0.35
- Completeness: 0.30
- Numerical accuracy: 0.25
- Citation accuracy: 0.10

---

## 6. Ablation Configurations

**File:** `evals/configs/ablation_configs.py`

```python
ABLATION_CONFIGS = {
    "naive_rag":     {"router": False, "reranker": False, "corrective": False, "self_rag": False},
    "plus_router":   {"router": True,  "reranker": False, "corrective": False, "self_rag": False},
    "plus_reranker": {"router": True,  "reranker": True,  "corrective": False, "self_rag": False},
    "full_system":   {"router": True,  "reranker": True,  "corrective": True,  "self_rag": True},
    # KG config: plug in when teammate's branch merges
    # "plus_kg":     {"router": True,  "reranker": True,  "corrective": True,  "self_rag": True, "kg": True},
}
```

**Naive RAG** bypasses the semantic router and sends all queries directly to vector search (Pinecone
cosine similarity, no reranking). This is the denominator for all improvement measurements.

The config dict is passed as `eval_config` into `graph.py`'s `build_graph(config)`. Graph nodes
check flags to short-circuit: e.g., `retrieve_docs` skips reranking if `config["reranker"]` is False.

---

## 7. Runner

**File:** `evals/runner.py`

```
EvalRunner.run(configs, dataset, output_dir)
  for each config:
    for each query in dataset:
      response = invoke graph with config flags
      build LLMTestCase(input, actual_output, retrieval_context, expected_output)
      compute all applicable metrics
    aggregate per-tier averages
  write evals/results/{run_id}.json
```

The runner calls `deepeval.evaluate(test_cases, metrics)` per config. Total run time estimate:
30 queries × 4 configs × ~3s per LLM judge call ≈ ~6 minutes per full eval run.

Results are also written incrementally (one config at a time) so a partial run is recoverable.

---

## 8. Results JSON Schema

**File:** `evals/results/{YYYY-MM-DDTHH-MM-SS}.json`

```json
{
  "run_id": "2026-04-25T14-00-00",
  "completed_at": "2026-04-25T14:06:12",
  "configs": {
    "naive_rag": {
      "tier_1": {
        "contextual_precision": 0.61,
        "contextual_recall": 0.54,
        "faithfulness": 0.72,
        "answer_relevancy": 0.68,
        "hallucination_rate": 0.18,
        "mrr": 0.44,
        "ndcg": 0.51,
        "numerical_accuracy": 0.80,
        "entity_match_precision": 0.65,
        "citation_accuracy": 0.55,
        "due_diligence_confidence": 0.62
      },
      "tier_2": {},
      "tier_3": {},
      "tier_4": {}
    },
    "plus_router": {},
    "plus_reranker": {},
    "full_system": {}
  },
  "baseline_delta": {
    "plus_router_vs_naive": {},
    "full_system_vs_naive": {}
  }
}
```

---

## 9. FastAPI Dashboard Endpoints

**File:** `src/eval_api.py` — mounted at `/eval` in `src/app.py`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/eval/runs` | List all run IDs in `evals/results/` |
| `GET` | `/eval/runs/{run_id}` | Full result JSON for one run |
| `GET` | `/eval/runs/{run_id}/summary` | Per-tier metric averages (all configs) |
| `GET` | `/eval/runs/{run_id}/ablation` | Side-by-side config comparison |
| `GET` | `/eval/runs/latest` | Shortcut to most recent run |
| `POST` | `/eval/trigger` | Kick off new eval run as background task; returns `run_id` |

All endpoints return JSON. The React dashboard will consume these directly.

---

## 10. Gemini Judge Configuration

**File:** `evals/metrics/gemini_judge.py`

Implements `DeepEvalBaseLLM` using `google.genai.Client(vertexai=True, project=..., location=...)`.
Model: `gemini-2.5-flash`. Uses `get_genai_client()` and `get_model_name()` from `src/model_config.py`
so GCP constants are not hardcoded.

---

## 11. Graph Config Flags

`graph.py` gains a `build_graph(eval_config: dict | None = None)` signature. `None` means
production defaults (all features on). When `eval_config` is provided:

- `router=False` → skip `route_question`, hardcode route to `"filings"` (vector search fallback)
- `reranker=False` → `retrieve_docs` skips Pinecone reranker, returns raw cosine results
- `corrective=False` → `grade_docs` always returns `"generate"` (no rewrite loop)
- `self_rag=False` → skip `grade_hallucination` and `grade_quality`, go straight to END

---

## 12. Testing

Unit tests in `tests/test_evals.py`:
- Test each custom metric (`MRRMetric`, `NDCGMetric`, `NumericalAccuracyMetric`, `EntityMatchPrecisionMetric`, `CitationAccuracyMetric`) with synthetic inputs — no LLM calls
- Test `EvalRunner` with a 2-query mock dataset and mocked DeepEval evaluate — no live graph calls
- Test FastAPI endpoints with a fixture result file

Integration test (not in CI, manual trigger):
- `uv run python -m evals.runner --configs naive_rag,full_system --dry-run` runs 2 queries to verify end-to-end wiring

---

## 13. Dependencies to Add

```toml
"deepeval>=1.0.0",
```

`scipy` is already a transitive dependency of `scikit-learn` (already in `pyproject.toml`) — no explicit add needed. NDCG uses `sklearn.metrics.ndcg_score`.
