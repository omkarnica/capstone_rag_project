# M&A Oracle

**Automated due diligence intelligence — detecting contradictions between management narrative and formal records across SEC filings, XBRL financials, earnings transcripts, patents, and litigation.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Cloud%20Run-blue)](https://ma-oracle-508519534978.us-central1.run.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-agentic-orange)](https://langchain-ai.github.io/langgraph/)
[![Vertex AI](https://img.shields.io/badge/Vertex%20AI-Gemini%202.5-blue)](https://cloud.google.com/vertex-ai)
[![Pinecone](https://img.shields.io/badge/Pinecone-ragcapstone-purple)](https://www.pinecone.io)
[![Neo4j](https://img.shields.io/badge/Neo4j-knowledge%20graph-teal)](https://neo4j.com)
[![LangSmith](https://img.shields.io/badge/LangSmith-tracing-yellow)](https://smith.langchain.com)

---

## Demo

https://github.com/omkarnica/capstone_rag_project/raw/dev/docs/MandA%20Oracle_demo.mp4

---

## The Scenario

A private equity firm is considering acquiring a tech company for $2B. Analysts need to answer: *Is the CEO's public optimism consistent with what the financial filings actually say? Are there hidden liabilities in the footnotes?*

M&A Oracle automates this process — routing due diligence questions to the right evidence source, cross-referencing multiple filing types, and flagging contradictions with citations.

**Example query:**
> Identify all instances where Meta's CEO's earnings call statements about metaverse investment contradicted the CapEx guidance in their 10-K filings from 2022–2024.

The system retrieves all relevant Meta earnings transcripts, extracts CEO statements, fetches CapEx figures from XBRL data, compares narrative claims against actual numbers, and flags specific contradictions.

---

## System Architecture

### Request Flow

```
User Query
    │
    ▼
FastAPI  (/query, /due-diligence)
    │
    ▼
Semantic Cache ──► Cache Hit → Return Cached Answer
    │ Miss
    ▼
Planner  (single-hop Tier 1 / multi-hop Tier 2 / llm_direct Tier 0)
    │
    ▼
LangGraph Workflow
    │
    ├─► Router ──► sql          → NL-to-SQL → PostgreSQL XBRL
    │             ├─► filings   → RAPTOR retrieval → Pinecone
    │             ├─► transcripts → Pinecone hybrid search
    │             ├─► patents   → Pinecone hybrid search
    │             ├─► litigation → Pinecone hybrid search
    │             ├─► graph     → Cypher generation → Neo4j
    │             ├─► contradiction → multi-source sweep
    │             └─► llm_direct → Gemini (no retrieval)
    │
    ├─► Rewriter  (query reformulation)
    ├─► Retriever → Grader  (relevance filtering)
    ├─► Merge     (multi-hop sub-question fusion)
    ├─► Generator (cited answer synthesis)
    └─► Fallback  (Gemini Google Search grounding)
         │
         ▼
Audit Logger → BigQuery  +  LangSmith trace
```

### Hybrid Retrieval

All vector routes use Reciprocal Rank Fusion over dense (Pinecone) and sparse (BM25) rankings before passing top-k chunks to the grader. Implementation: `src/utils/hybrid.py`.

### Knowledge Graph

Neo4j graph covers Companies → Filings → Sections, Subsidiaries, BoardMembers, Patents → TechnologyDomains. `src/graph_retrieval.py` generates read-only Cypher via Gemini, executes it, and formats rows as evidence. `src/Knowledge graph/kg.py` handles ingestion.

### Agentic Multi-hop Pipeline

`src/nodes/planner.py` decomposes complex questions into ordered sub-questions, each with a route hint. Sub-questions execute through the full LangGraph workflow and results are fused by `src/nodes/merge.py` before final generation.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic |
| Orchestration | LangGraph |
| LLM | Gemini 2.5 Flash via Vertex AI |
| Vector store | Pinecone (`ragcapstone` index) |
| Embeddings | Pinecone `llama-text-embed-v2` |
| Reranker | Pinecone `bge-reranker-v2-m3` |
| Knowledge graph | Neo4j |
| Relational DB | PostgreSQL (Cloud SQL) |
| Cache | Redis (Cloud Memorystore) |
| Secrets | GCP Secret Manager |
| Observability | LangSmith + BigQuery audit log |
| Evaluation | deepeval + custom metrics |
| Frontend | React + Vite |
| Package manager | uv |

---

## Data Sources

| Source | Content | Volume |
|---|---|---|
| SEC EDGAR | 10-K, 10-Q, 8-K filings | Millions of documents |
| XBRL | Structured financial facts | Hundreds of tags per filing |
| Earnings transcripts | CEO/CFO Q&A, guidance statements | Thousands of calls |
| USPTO patents | Claims, CPC codes, assignees | 13M+ patents |
| CourtListener | Litigation records, court cases | Federal + state courts |
| SEC DEF 14A | Proxy statements, board composition | Per company per year |
| SEC Exhibit 21 | Subsidiary listings | Per filing year |

---

## Retrieval Strategies

| Route | Trigger | Module |
|---|---|---|
| `sql` | Financial metrics, XBRL numbers, balance sheet | `src/nl_sql/pipeline.py` |
| `filings` | 10-K/10-Q narrative, risk factors, MD&A | `src/filings/raptor_retrieval.py` |
| `transcripts` | Earnings call statements, CEO/CFO quotes | `src/transcripts/retrieval.py` |
| `patents` | IP portfolio, patent claims, CPC codes | `src/patents/retrieval.py` |
| `litigation` | Court cases, settlements, legal exposure | `src/litigation/retrieval.py` |
| `graph` | Subsidiaries, board members, ownership structure | `src/graph_retrieval.py` |
| `contradiction` | Cross-source due diligence sweep | `src/contradictions/detector.py` |
| `llm_direct` | General M&A concepts, no retrieval needed | Gemini direct |

---

## Evaluation Framework

### Metrics

- **NumericalAccuracy** — fraction of expected financial figures found within 1% tolerance
- **ContradictionDetection** — precision/recall on flagged narrative vs. filing discrepancies
- **GeminiJudge** — LLM-as-judge scoring for answer faithfulness and citation quality
- **RetrievalPrecision/Recall** — chunk-level relevance evaluation

### Ablation Study

`evals/configs/ablation_configs.py` defines retrieval configurations (dense-only, hybrid, with/without reranker, with/without RAPTOR). `evals/runner.py` runs all configs against `evals/dataset/` and outputs a `baseline_delta` comparison to `evals/results/`.

### Observability

All LangGraph runs are traced in LangSmith (project `RAG-Capstone`). Every query writes a structured `AuditRecord` with UUID correlation ID, tenant, user, route, latency, sources, and confidence score to BigQuery (`codelab-2-485215.ma_oracle.audit_log`).

---

## Setup

### Prerequisites

- Python 3.11+
- `uv` package manager
- GCP Application Default Credentials (`gcloud auth application-default login`)
- Pinecone API key
- Neo4j instance (URI, user, password)
- Redis instance
- PostgreSQL database `ma_oracle` on port `5433` (local dev)

### Environment

Create a `.env` file at the project root:

```env
PINECONE_API_KEY=your-pinecone-key
NEO4J_URI=bolt://your-neo4j-host:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password
REDIS_URL=redis://localhost:6379
LANGSMITH_API_KEY=your-langsmith-key   # optional, auto-fetched from Secret Manager
```

Secrets not in `.env` are resolved from GCP Secret Manager at startup.

### Install

```bash
uv sync
```

### Run

```bash
uv run uvicorn src.app:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`.

### Tests

```bash
uv run pytest -q
```

---

## Local Dev Setup

### Prerequisites
- GCP access to project `codelab-2-485215`
- `gcloud` CLI installed and authenticated

### Start Redis Tunnel (required before running app)

The app connects to Cloud Memorystore Redis via an SSH tunnel through the bastion host.
Keep this terminal open while running the app.

```powershell
gcloud compute ssh redis-bastion `
  --zone=us-central1-a `
  --project=codelab-2-485215 `
  --ssh-flag="-L 6379:10.101.67.12:6379" `
  --ssh-flag="-N"
```

### Start App

In a new terminal:

```bash
uv run uvicorn src.app:app --reload --port 8000
```

---

## Deliverables

| Deliverable | Status |
|---|---|
| Deployed web application | ✅ [Live](https://ma-oracle-508519534978.us-central1.run.app) |
| Authentication with role-based access | ✅ Tenant + user ID isolation on all requests |
| Multi-tenant architecture with data isolation | ✅ `tenant_id` scoped across query, audit, and cache layers |
| Document lifecycle management | ✅ Ingestion pipelines for SEC, patents, litigation, transcripts |
| React frontend — query interface + admin dashboard | ✅ Vite/React app |
| Knowledge graph with visualization | ✅ Neo4j graph, Cypher generation, entity relationship queries |
| RAG router with minimum 4 retrieval strategies | ✅ 7 routes (sql, filings, transcripts, patents, litigation, graph, contradiction) |
| Agentic pipeline for multi-hop queries | ✅ Planner → sub-questions → merge → generate |
| Multimodal handling | ✅ XBRL structured data + filing text (RAPTOR) + earnings transcripts + patents |
| Evaluation framework (RAGAS + custom metrics) | ✅ deepeval + NumericalAccuracy + ContradictionDetection + GeminiJudge |
| Ablation study | ✅ Dense vs hybrid vs reranker configs with baseline delta scoring |
| LLM observability | ✅ LangSmith tracing + BigQuery audit log with correlation IDs |
| Slack/webhook notifications | ⚠️ In progress |
| Structured logging with correlation IDs | ✅ UUID `query_id` on every request, BigQuery sink |
| GitHub repository with documentation | ✅ This repository |

---

## Team

| Name | Role |
|---|---|
| Karnica Jain | Backend, Authentication,  ingestion pipelines |
| Ruby Gunna | RAG pipeline, LangGraph orchestration, evaluation |
| Nithu Arjunan | Frontend,knowledge graph,raptor tree,ingestion, deployment |
