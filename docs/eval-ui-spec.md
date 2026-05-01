# Eval UI — Frontend Spec

**For:** Nithu (frontend)
**Backend contact:** Ruby
**Base URL (deployed):** `https://ma-oracle-508519534978.us-central1.run.app`
**Base URL (local dev):** `http://localhost:8000`

---

## What to build

A new **Evaluation tab** in the existing React app. Match the current design system exactly — same card containers, same teal gradient buttons, same font/spacing as the existing workspace.

---

## Endpoints

All endpoints are live and tested. No auth required.

### 1. Trigger a new eval run
```
POST /eval/trigger
Body: (none)
```
**Response:**
```json
{
  "run_id": "2026-05-01T14-32-10",
  "status": "started"
}
```
Runs in the background. Returns immediately with a `run_id`. Poll for results using endpoint 3.

---

### 2. List all past runs
```
GET /eval/runs
```
**Response:**
```json
{
  "run_ids": ["2026-05-01T14-32-10", "2026-04-30T09-15-44"]
}
```
Sorted oldest → newest. Use for the run history dropdown.

---

### 3. Get a specific run (poll this after trigger)
```
GET /eval/runs/{run_id}
```
**Response shape:**
```json
{
  "run_id": "2026-05-01T14-32-10",
  "completed_at": "2026-05-01T14-45-22" | null,
  "configs": {
    "naive_rag":    { "tier_1": { "faithfulness": 0.61, "numerical_accuracy": 0.42, ... } },
    "plus_router":  { "tier_1": { ... } },
    "plus_reranker":{ "tier_1": { ... } },
    "plus_kg":      { "tier_1": { ... } },
    "full_system":  { "tier_1": { ... }, "tier_2": { ... } }
  },
  "baseline_delta": { ... }
}
```
While `completed_at` is `null` the run is still in progress. Poll every 10 seconds.
Configs appear incrementally as each finishes — render partial results as they arrive.

---

### 4. Get averaged summary per config (use for bar chart)
```
GET /eval/runs/{run_id}/summary
```
**Response:**
```json
{
  "run_id": "...",
  "configs": {
    "naive_rag":     { "faithfulness": 0.58, "numerical_accuracy": 0.40, "answer_relevancy": 0.62, "due_diligence_confidence": 0.51 },
    "plus_router":   { "faithfulness": 0.67, ... },
    "plus_reranker": { "faithfulness": 0.73, ... },
    "plus_kg":       { "faithfulness": 0.74, ... },
    "full_system":   { "faithfulness": 0.79, ... }
  }
}
```
All scores are 0–1. Higher = better for all metrics except `hallucination_rate` (lower = better).

---

### 5. Get ablation delta vs baseline (use for delta table)
```
GET /eval/runs/{run_id}/ablation
```
**Response:**
```json
{
  "run_id": "...",
  "configs": { ... },
  "baseline_delta": {
    "plus_router_vs_naive":   { "tier_1": { "faithfulness": +0.09, "numerical_accuracy": +0.05 } },
    "plus_reranker_vs_naive": { "tier_1": { ... } },
    "plus_kg_vs_naive":       { "tier_1": { ... } },
    "full_system_vs_naive":   { "tier_1": { ... }, "tier_2": { ... } }
  }
}
```
Positive delta = improvement over `naive_rag` baseline. Negative = regression.

---

### 6. Get latest run (load on page open)
```
GET /eval/runs/latest
```
Same shape as endpoint 3. Use this on page load so results are pre-populated.

---

## Page layout

### Left sidebar (same style as existing "Run diagnostics" + "Controls" cards)

**Card 1 — Run controls**
- Button: **"Run Evaluation"** (teal gradient, same as "Run query")
  - On click → `POST /eval/trigger`
  - Button becomes disabled + shows spinner while `completed_at` is null
  - Below button: show `run_id` as monospace text once started

**Card 2 — Run history**
- Label: "Past runs"
- Dropdown/select: populated from `GET /eval/runs`
- On select → fetch that run via `GET /eval/runs/{run_id}/summary` and `/ablation`
- Default: load latest run on page open via `GET /eval/runs/latest`

**Card 3 — Run status** (small, below history)
- "Status" row: `running…` (spinner) or `complete ✓` (green)
- "Dataset" row: `35 queries`
- "Configs" row: `5 configs`

---

### Main panel

**Section 1 — Metric bar chart** (top, same card container as "Ask a question")

Title: **"Ablation — Score by Config"**

- Grouped bar chart, one group per metric
- X-axis: metric names (`faithfulness`, `numerical_accuracy`, `answer_relevancy`, `due_diligence_confidence`)
- Each group has 5 bars (one per config), color-coded:
  - `naive_rag` → grey
  - `plus_router` → light teal
  - `plus_reranker` → medium teal
  - `plus_kg` → blue
  - `full_system` → dark teal (matches brand)
- Y-axis: 0.0 → 1.0
- Data source: `GET /eval/runs/{run_id}/summary`
- Recommended library: **Recharts** (likely already in the project) or Chart.js

---

**Section 2 — Delta table** (below chart)

Title: **"Delta vs Naive RAG Baseline"**

Tier tabs: `Tier 1` | `Tier 2` | `Tier 3` | `Tier 4` (filter rows by tier)

| Metric | +Router | +Reranker | +KG | Full System |
|---|---|---|---|---|
| faithfulness | +0.09 | +0.15 | +0.16 | +0.21 |
| numerical_accuracy | +0.05 | +0.08 | +0.08 | +0.11 |
| ... | | | | |

- Positive delta → green text (`#22c55e`)
- Negative delta → red text (`#ef4444`)
- Zero/null → grey dash
- Data source: `GET /eval/runs/{run_id}/ablation` → `baseline_delta`

---

## Polling logic

```js
// After POST /eval/trigger
const poll = setInterval(async () => {
  const run = await fetch(`/eval/runs/${runId}`).then(r => r.json())
  updateChartWithPartialData(run)           // render as configs complete
  if (run.completed_at !== null) {
    clearInterval(poll)
    setStatus('complete')
  }
}, 10_000)  // every 10 seconds
```

Render partial results as each config finishes — the runner writes incrementally, so configs appear one by one in the response.

---

## Available metrics (all 0–1 unless noted)

| Metric | Description |
|---|---|
| `faithfulness` | Answer grounded in retrieved context |
| `answer_relevancy` | Answer addresses the question |
| `hallucination_rate` | Lower is better |
| `contextual_precision` | Retrieved docs relevant to query |
| `contextual_recall` | Relevant docs actually retrieved |
| `numerical_accuracy` | Financial figures correct within 1% |
| `entity_match_precision` | Named entities correct |
| `citation_accuracy` | Citations match expected sources |
| `completeness` | Answer covers all expected points |
| `contradiction_detection_rate` | Tier 4 only |
| `due_diligence_confidence` | Composite weighted score |
| `mrr` | Mean Reciprocal Rank (retrieval) |
| `ndcg` | Normalized Discounted Cumulative Gain |

Show `faithfulness`, `numerical_accuracy`, `answer_relevancy`, `due_diligence_confidence` by default in the chart. Let others be visible via a "Show all metrics" toggle.

---

## Notes

- All eval runs can take 10–30 minutes depending on dataset size. The spinner + polling handles this.
- The backend is already deployed and all 6 endpoints are live — no backend changes needed.
- `run_id` format is always `YYYY-MM-DDTHH-MM-SS` — safe to display as a timestamp.
- If `GET /eval/runs/latest` returns 404, show an empty state with just the "Run Evaluation" button.
