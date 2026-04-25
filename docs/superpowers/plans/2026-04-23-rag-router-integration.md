# RAG Router Integration — M&A Oracle

**Date**: 2026-04-23
**Branch**: `featureRG_Integrate_RAG_Router`
**Author**: Ruby Gunna
**Scope**: Wire LangGraph router/graph to M&A Oracle data sources

---

## Context

The `src/nodes/` directory and `src/graph.py` were ported from a prior Claude certification Q&A assignment. They need to be re-wired to serve M&A Oracle due diligence queries. The five M&A retrieval modules (`transcripts`, `patents`, `litigation`, `filings/raptor`, `nl_sql`) are already working and use Gemini via google-genai SDK. The graph nodes use LangChain + OpenAI — a missing dependency that isn't in `pyproject.toml` — making them currently broken.

**Goal**: Wire the LangGraph router/graph to dispatch M&A queries to the correct data sources (XBRL/SQL, 10-K filings, transcripts, patents, litigation, contradiction detection), satisfying the project spec's minimum 4-route RAG Router and Agentic Pipeline requirements.

**Reference spec**: `C:\Users\roopm\Downloads\ma-oracle.md`
**Architecture diagram**: `docs/architecture.md`

---

## Critical Blockers (must fix first)

| Problem | Impact |
|---|---|
| `model_config.py` uses `langchain_openai.ChatOpenAI` — not in `pyproject.toml` | Graph can't run |
| `nodes/retriever.py` imports `src.ingestion.indexer` — doesn't exist | Graph crashes on retrieve step |
| `api.py` imports `src.cache.semantic_cache.SemanticCache` — doesn't exist | App won't start |
| Router/planner hard-coded for Claude certification topics | Wrong route decisions for M&A queries |

---

## Key Decisions

| Decision | Choice | Reason |
|---|---|---|
| LLM backend for nodes | `langchain-google-vertexai` (ChatVertexAI) | Uses GCP creds (follows CLAUDE.md), keeps `.with_structured_output()` intact across 7 node files — avoids full node rewrite |
| Cache | Copy `Downloads/cache/cache/` → `src/cache/`, fix imports, add `semantic_cache.py` adapter | 3-tier production cache (exact + semantic + retrieval), real Redis backend, replaces the missing `SemanticCache` that breaks api.py |
| `graph` route | Stub returning empty docs | KG not built yet — separate deliverable |
| Web search | Remove from router edges, keep as corrective fallback only | M&A Oracle uses 5 internal sources; web search not relevant for due diligence |
| `contradiction` route | Bypasses retrieve→grade→generate, calls `run_due_diligence()` directly | Contradiction detection is a multi-step sub-pipeline (XBRL SQL + transcripts + Gemini) — doesn't fit the single-retrieve model |

---

## Architecture Mapping

From `docs/architecture.md`:

```
RAG Router (4 routes)
├─ Vector Search  →  transcripts / patents / litigation / filings
├─ Graph Query    →  graph (stub)
├─ SQL/Struct     →  sql
└─ Agentic        →  contradiction
```

---

## Implementation Steps

### Step 0 — Migrate Cache Modules: `src/cache/`

Copy `C:\Users\roopm\Downloads\cache\cache\` into `src/cache/` and fix the import wiring so it works in this project.

**Files to create:**
- `src/cache/__init__.py` — fixed factory (the downloaded `__init__.py` has dead code after a `return` — fix the syntax)
- `src/cache/base.py` — copy as-is (no external imports)
- `src/cache/embedding_cache.py` — fix `from ..config import PINECONE_API_KEY, PINECONE_EMBED_MODEL` → use `src.utils.secrets.get_secret("PINECONE_API_KEY")` and hardcode `PINECONE_EMBED_MODEL = "llama-text-embed-v2"`
- `src/cache/redis_backend.py` — fix `from ..config import REDIS_URL` → `os.getenv("REDIS_URL", "redis://localhost:6379")`
- `src/cache/semantic_cache.py` — **new adapter file** that wraps `RedisCacheBackend` with the interface `api.py` expects:

```python
# Adapter exposing the interface api.py imports:
#   SemanticCache, compute_corpus_version, is_time_sensitive_question

from src.cache.redis_backend import RedisCacheBackend
from src.cache.embedding_cache import embed_query

_RECENCY_TERMS = {"current", "latest", "recent", "today", "now", "live"}

def is_time_sensitive_question(question: str) -> bool:
    return any(t in question.lower() for t in _RECENCY_TERMS)

def compute_corpus_version(chunking_strategy: str = "hierarchical") -> int:
    return _get_backend().get_doc_version()

def _get_backend() -> RedisCacheBackend:
    ...  # singleton

class SemanticCache:
    def get_similar(self, question, corpus_version, chunking_strategy, similarity_threshold) -> dict | None:
        embedding = embed_query(question)
        result = _get_backend().get_semantic(embedding, threshold=similarity_threshold,
                                             source_filter=chunking_strategy)
        if result:
            result["cache_similarity"] = result.pop("similarity", None)
            result["cache_source_question"] = result.pop("question", None)
        return result

    def store(self, question, result, corpus_version, chunking_strategy) -> None:
        embedding = embed_query(question)
        import json
        _get_backend().set_semantic(
            question=question, embedding=embedding,
            answer=result.get("final_answer", ""),
            sources_json=json.dumps(result.get("citations", [])),
            doc_version=corpus_version, ttl_seconds=3600,
            source_filter=chunking_strategy,
        )
```

**Also add to `pyproject.toml`:** `redis>=5.0.0`

---

### Step 1 — Fix LLM Layer: `pyproject.toml` + `src/model_config.py`

Add `langchain-google-vertexai>=2.0.0` to `pyproject.toml`. Rewrite `model_config.py`:

```python
from langchain_google_vertexai import ChatVertexAI

_PROJECT = "codelab-2-485215"
_LOCATION = "us-central1"
_MODEL = "gemini-2.5-flash"

def _get_llm(**kwargs) -> ChatVertexAI:
    return ChatVertexAI(model=_MODEL, project=_PROJECT, location=_LOCATION,
                        temperature=0, **kwargs)

def get_router_llm(): return _get_llm()
def get_planner_llm(): return _get_llm()
def get_rewriter_llm(): return _get_llm()
def get_grader_llm(): return _get_llm()
def get_generation_llm(): return _get_llm()
def get_direct_generation_llm(): return _get_llm()
def get_merge_llm(): return _get_llm()
```

Remove all OpenAI/embedding functions (Pinecone hosted inference handles embeddings).

---

### Step 2 — Extend GraphState: `src/state.py`

Add M&A-specific fields:
```python
company: str | None          # target company (e.g. "Apple Inc.")
period: str | None           # fiscal period (e.g. "Q4 2024")
source_type: str | None      # which M&A source answered
data_source_result: dict     # raw result from retrieval module
contradiction_report: list   # findings from run_due_diligence()
```

Update `route` Literal:
```python
route: Literal["sql", "filings", "transcripts", "patents", "litigation",
               "graph", "contradiction", "llm_direct"]
```

---

### Step 3 — Rewrite Router: `src/nodes/router.py`

**Architecture mapping:**
| Arch Route | Implementation routes |
|---|---|
| Vector Search | `transcripts`, `patents`, `litigation`, `filings` |
| SQL / Structured | `sql` |
| Graph Query | `graph` (stub — KG not built yet) |
| Agentic | `contradiction` |
| — | `llm_direct` |

System prompt classifies M&A queries:
- `sql` — financial metrics, revenue, profit, margins, ratios, XBRL numbers
- `filings` — 10-K/10-Q narrative: risk factors, MD&A, footnotes, disclosures
- `transcripts` — what management said on earnings calls
- `patents` — IP portfolio, patent claims, CPC codes, citation analysis
- `litigation` — court cases, lawsuits, settlements, legal exposure
- `graph` — entity relationships, subsidiaries, board connections (stub)
- `contradiction` — compare management statements vs. filed disclosures
- `llm_direct` — general M&A/finance concepts needing no retrieval

Keep `force_route` / `route_hint` logic for planner-controlled sub-questions.

---

### Step 4 — Rewrite Retriever: `src/nodes/retriever.py`

Replace `src.ingestion.indexer` with M&A retrieval dispatch. Normalize all source outputs into `{"content": str, "metadata": dict}` so grader/generator work unchanged.

```python
from src.nl_sql.pipeline import ask as nl_sql_ask
from src.filings.raptor_retrieval import raptor_retrieve
from src.transcripts.retrieval import retrieve_transcripts
from src.patents.retrieval import retrieve_patents
from src.litigation.retrieval import retrieve_litigation

def retrieve_docs(state: GraphState) -> GraphState:
    route = state["route"]
    query = state.get("rewritten_question") or state["question"]
    company = state.get("company")

    if route == "sql":
        result = nl_sql_ask(query)
        docs = [{"content": result["answer"], "metadata": {"source": "XBRL/SQL", "sql": result["sql"]}}]

    elif route == "filings":
        result = raptor_retrieve(query, top_k=10, final_top_k=6)
        docs = [{"content": c["text"], "metadata": {"source": "SEC Filing",
                 "form_type": c.get("form_type"), "rank": c["rank"]}}
                for c in result.get("contexts", [])]

    elif route == "transcripts":
        hits = retrieve_transcripts(query, company=company)
        docs = [{"content": h["fields"].get("text", ""),
                 "metadata": {"source": "Earnings Transcript",
                              "company": h["fields"].get("company_title"),
                              "period": h["fields"].get("period_of_report"),
                              "accession_no": h["fields"].get("accession_no")}}
                for h in hits]

    elif route == "patents":
        hits = retrieve_patents(query, company=company)
        docs = [{"content": h["fields"].get("text", ""),
                 "metadata": {"source": "Patent",
                              "patent_id": h["fields"].get("patent_id"),
                              "patent_title": h["fields"].get("patent_title"),
                              "grant_date": h["fields"].get("grant_date")}}
                for h in hits]

    elif route == "litigation":
        hits = retrieve_litigation(query, company=company)
        docs = [{"content": h["fields"].get("text", ""),
                 "metadata": {"source": "Litigation",
                              "case_name": h["fields"].get("case_name"),
                              "court": h["fields"].get("court"),
                              "date_filed": h["fields"].get("date_filed")}}
                for h in hits]

    else:  # graph (stub), llm_direct, contradiction — handled elsewhere
        docs = []

    return {**state, "retrieval_query": query, "retrieved_docs": docs,
            "filtered_docs": [], "doc_relevance": [], "relevant_doc_count": 0}
```

---

### Step 5 — Add Contradiction Node + Update Graph: `src/graph.py`

Add `contradiction_check` node — bypasses retrieve→grade→generate:

```python
from src.contradictions.detector import run_due_diligence

def run_contradiction_check(state: GraphState) -> GraphState:
    company = state.get("company", "")
    period = state.get("period", "")
    fiscal_year = int(period[-4:]) if period and period[-4:].isdigit() else 2024
    findings = run_due_diligence(company=company, transcript_company=company,
                                  fiscal_year=fiscal_year)
    summary = _format_contradiction_report(findings)
    return {**state, "contradiction_report": findings, "answer": summary,
            "citations": [f"{f['metric']} ({f['period']})" for f in findings]}
```

Update `route_after_router`:
- `contradiction` → `contradiction_check` → END
- `llm_direct` → `llm_direct_generate`
- all others → `retrieve`

Remove web_search from router conditional edges (keep as corrective fallback only when 0 relevant docs after retrieval).

---

### Step 6 — Update Generator System Prompt: `src/nodes/generator.py`

```python
# New system prompt for generate_answer:
"You are a forensic financial analyst assistant for M&A due diligence. "
"Answer using ONLY the provided context from SEC filings, transcripts, patents, or litigation. "
"Cite every factual claim with [source | date] format. "
"Flag material risks and anomalies explicitly. Do not speculate."
```

Keep `generate_answer` / `generate_direct_answer` structure intact.

---

### Step 7 — Rewrite Planner: `src/nodes/planner.py`

Replace `SOURCE_TOPIC_SPECS` (Claude courses) with M&A source specs:

```python
SOURCE_TOPIC_SPECS = {
    "financial_metrics": {
        "phrases": ("revenue", "net income", "gross margin", "operating income", "cash", "ebitda"),
        "summary_question": "What are the key financial metrics?",
        "route_hint": "sql",
    },
    "risk_factors": {
        "phrases": ("risk factor", "10-k", "10-q", "item 1a", "md&a", "material weakness"),
        "summary_question": "What are the key risk disclosures?",
        "route_hint": "filings",
    },
    "management_statements": {
        "phrases": ("earnings call", "management said", "ceo", "guidance", "transcript"),
        "summary_question": "What did management say?",
        "route_hint": "transcripts",
    },
    "patents": {
        "phrases": ("patent", "ip portfolio", "cpc", "invention", "claim"),
        "summary_question": "What is the patent portfolio?",
        "route_hint": "patents",
    },
    "litigation": {
        "phrases": ("lawsuit", "litigation", "court", "settlement", "legal"),
        "summary_question": "What is the litigation exposure?",
        "route_hint": "litigation",
    },
}
```

Update multi-hop decomposition for M&A patterns:
- "compare earnings call vs filing" → [transcripts + sql + contradiction]
- "revenue trend over time" → [sql + filings]
- "patent portfolio AND litigation" → [patents + litigation]

Remove all Anthropic/Claude-specific heuristics, recency markers, comparison patterns.

---

### Step 8 — Wire API Layer: `src/api.py`

No major changes needed once Step 0 creates `src/cache/semantic_cache.py` — the imports already match. Minor change: update `run_single_question()` to forward `company` and `period` into graph state when provided by caller.

---

### Step 9 — Update App Entry Point: `src/app.py`

```python
app = FastAPI(title="M&A Oracle — Due Diligence Intelligence API", version="0.1.0")

class DueDiligenceRequest(BaseModel):
    company: str
    transcript_company: str | None = None
    fiscal_year: int
    quarter: str = "FY"

@app.post("/due-diligence")
def due_diligence(payload: DueDiligenceRequest):
    from src.contradictions.detector import run_due_diligence
    return run_due_diligence(
        company=payload.company,
        transcript_company=payload.transcript_company or payload.company,
        fiscal_year=payload.fiscal_year,
        quarter=payload.quarter,
    )
```

Keep `/query`, `/adaptive-query`, `/health`. Update CORS origins for M&A Oracle frontend.

---

## Out of Scope for This Branch (Future PRs)

Per architecture diagram — noted but not implemented here:
- **Knowledge Graph** (`graph` route stubs to empty docs) — separate deliverable
- **LangSmith / LangFuse observability** — separate integration PR
- **Audit log** (query_id, tenant_id, sources, tokens/cost) — separate PR
- **Slack webhook notifications** — separate PR

> **Note:** BM25 hybrid search was delivered early in this branch via `src/utils/hybrid.py` — integrated into filings, litigation, and patents retrieval.

---

## Files Modified Summary

| File | Change |
|---|---|
| `pyproject.toml` | Add `langchain-google-vertexai>=2.0.0`, `redis>=5.0.0` |
| `src/cache/__init__.py` | New — factory for `RedisCacheBackend` (fix syntax bug from download) |
| `src/cache/base.py` | New — copy `CacheBackend` ABC as-is |
| `src/cache/embedding_cache.py` | New — copy + fix `..config` imports → `src.utils.secrets` |
| `src/cache/redis_backend.py` | New — copy + fix `REDIS_URL` import → `os.getenv`; `__init__` degrades gracefully when Redis unavailable |
| `src/cache/semantic_cache.py` | New — adapter: `SemanticCache` / `compute_corpus_version` / `is_time_sensitive_question`; stores full result payload (not just answer string); handles Redis unavailability without crashing |
| `src/utils/hybrid.py` | New — BM25 + RRF hybrid ranking utility used across filings, litigation, patents |
| `src/model_config.py` | Replace OpenAI/LangChain with ChatVertexAI (GCP creds) |
| `src/state.py` | Add `company`, `period`, `source_type`, `data_source_result`, `contradiction_report`; update route Literal |
| `src/nodes/router.py` | Full rewrite for M&A route types |
| `src/nodes/retriever.py` | Replace `ingestion.indexer` with M&A retrieval dispatch; all 5 branches wrapped in `try/except` with empty-docs fallback |
| `src/graph.py` | Add `contradiction_check` node; extract quarter token from `period` and forward to `run_due_diligence`; update conditional edges |
| `src/nodes/generator.py` | Update system prompt for M&A context; update `_answer_style_instructions` with M&A-relevant phrases |
| `src/nodes/planner.py` | Replace Claude course heuristics with M&A source topics |
| `src/nodes/fallback.py` | Rewritten — replace Brave Search with Google CSE; replace Anthropic domain trust with SEC/USPTO/CourtListener domain scoring; route-aware query rewriter |
| `src/api.py` | Forward `company`/`period` to graph state; cache-scoped keys; `contradiction_report` in output |
| `src/app.py` | Add `/due-diligence` endpoint with input validation (`fiscal_year` in [2000,2030], non-empty `company`); update title |
| `src/filings/raptor_retrieval.py` | Integrate `hybrid_rrf_rank` replacing rerank fallback chain |
| `src/litigation/retrieval.py` | Integrate `hybrid_rrf_rank` replacing `bge-reranker-v2-m3` API calls |
| `src/patents/retrieval.py` | Integrate `hybrid_rrf_rank` replacing citation-boost reranker |
| `src/transcripts/retrieval.py` | Minor updates |

**New demo scripts** (call live services — not unit tests, not collected by pytest):
- `src/nodes/demo_adaptive.py`, `demo_graph.py`, `demo_retriever.py`, `demo_router.py`
- `src/patents/demo_patents.py`, `src/transcripts/demo_transcript.py`

**New tests:**
- `tests/test_hybrid.py` — unit tests for `hybrid_rrf_rank`

**Files unchanged** (already correct for M&A Oracle):
- `src/nodes/grader.py`, `src/nodes/merge.py`, `src/nodes/rewriter.py`
- `src/tiering.py`
- `src/nl_sql/pipeline.py`, `src/contradictions/detector.py`
- `src/xbrl/`

**Deleted** (were live-service demos masquerading as tests, invisible to pytest anyway):
- `src/nodes/adaptive_test.py`, `graph_test.py`, `retriever_test.py`, `router_test.py`
- `src/patents/test_patents.py`, `src/transcripts/test_transcript.py`

---

## Post-Review Fixes (applied 2026-04-24)

Issues caught by code review before merge, fixed on the same branch:

| # | File | Fix |
|---|---|---|
| 1 | `src/cache/semantic_cache.py` | `store()` serializes full result dict; `get_similar()` restores it — cache hits return complete payloads, not sparse `{answer, sources_json}` |
| 2 | `src/graph.py` | `run_contradiction_check` extracts quarter token (Q1/Q2/Q3/Q4/FY) from `period` and passes it to `run_due_diligence()` |
| 3 | `src/nodes/retriever.py` | All 5 retrieval branches wrapped in `try/except`; failures log warning and return `docs = []` so corrective-RAG rewrite/fallback path proceeds |
| 4 | `src/nodes/generator.py` | Removed stale Claude-course phrases (`"syllabus"`, `"covers"`, `"what courses"`) from `_answer_style_instructions`; replaced with M&A-relevant phrases |
| 5 | `src/cache/redis_backend.py` + `semantic_cache.py` | Redis connection failure in `__init__` no longer crashes; `_get_backend()` returns `None`; all cache methods degrade gracefully |
| 6 | `src/app.py` | `DueDiligenceRequest` validates `fiscal_year` in [2000, 2030] and `company` non-empty via Pydantic `Field` constraints |

---

## Verification

```bash
# 1. Install updated dependencies
uv sync

# 2. Smoke-test LLM layer
python -c "from src.model_config import get_router_llm; print(get_router_llm().invoke('hello').content)"

# 3. Test M&A router decisions
python -c "
from src.nodes.router import route_question
tests = [
    'What was Apple revenue Q4 2024?',
    'What did Apple CEO say about iPhone on the earnings call?',
    'Does Apple have patents related to on-device inference?',
    'Compare what management said about revenue growth vs the 10-K filing',
]
for q in tests:
    r = route_question({'question': q})
    print(r['route'], '|', q[:60])
"

# 4. End-to-end graph run
python -c "
from src.graph import build_graph
g = build_graph()
result = g.invoke({'question': 'What was Apple total revenue fiscal year 2024?',
                   'company': 'Apple Inc.', 'max_iterations': 1, 'max_retrieval_attempts': 1})
print('Route:', result.get('route'))
print('Answer:', result.get('answer', '')[:300])
"

# 5. Contradiction detection
python -c "
from src.contradictions.detector import detect_contradiction
r = detect_contradiction(
    company='Apple Inc.', metric_label='Total Revenue',
    xbrl_question='What is Apple total revenue for fiscal Q4 2024?',
    transcript_question='What did Apple say about revenue in fiscal Q4 2024?',
    transcript_company='Apple Inc.', period_label='fiscal Q4 2024',
    transcript_period_start='2024-07-01', transcript_period_end='2025-01-31',
)
print('Score:', r['contradiction_score'], '| Severity:', r['severity'])
"

# 6. FastAPI endpoint
uvicorn src.app:app --port 8000
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What patents does Apple have in on-device AI?"}'
```
