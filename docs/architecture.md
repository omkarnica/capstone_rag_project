# M&A Oracle Architecture

## Overview
M&A Oracle answers due diligence questions by routing each query to the evidence
source most likely to contain the answer, then generating a concise cited answer.

## Request Flow
1. `src/app.py` receives `/query` or `/due-diligence`.
2. `src/api.py` checks the semantic cache and plans single-hop or multi-hop work
   using `src/nodes/planner.py`. Queries are tiered: Tier 0 (llm_direct), Tier 1
   (single-route RAG), Tier 2 (multi-hop RAG) — see `src/tiering.py`.
3. `src/graph.py` executes the LangGraph workflow.
4. `src/nodes/router.py` classifies the question and chooses a route.
   `src/nodes/graph_topics.py` assists by detecting knowledge-graph-specific intent.
5. `src/nodes/rewriter.py` optionally rewrites the query for retrieval.
6. `src/nodes/retriever.py` dispatches to the relevant data module.
7. `src/nodes/grader.py` filters retrieved evidence for relevance.
8. `src/nodes/generator.py` produces the final answer and citations.
9. For multi-hop plans `src/nodes/merge.py` combines sub-question results before
   generation.
10. `src/nodes/fallback.py` uses Gemini Google Search grounding when internal
    retrieval cannot provide sufficient evidence.

## Routes
- `sql`: PostgreSQL structured facts through `src/nl_sql/pipeline.py`
- `filings`: RAPTOR filing retrieval through `src/filings/raptor_retrieval.py`
- `transcripts`: Pinecone transcript retrieval through `src/transcripts/retrieval.py`
- `patents`: Pinecone patent retrieval through `src/patents/retrieval.py`
- `litigation`: Pinecone litigation retrieval through `src/litigation/retrieval.py`
- `graph`: Neo4j knowledge graph retrieval through `src/graph_retrieval.py` —
  Gemini generates a read-only Cypher query against the filing/patent/board schema,
  executes it against Neo4j, and formats the rows as evidence documents
- `contradiction`: multi-source comparison through `src/contradictions/detector.py`
- `llm_direct`: general M&A/finance explanations without retrieval

## Hybrid Retrieval
Vector evidence routes use hybrid ranking:

1. Dense vector retrieval returns semantic candidates from Pinecone.
2. BM25 scores the same candidate texts for keyword relevance.
3. Reciprocal Rank Fusion combines dense and BM25 rankings.
4. The final top-k fused chunks are passed to grading and generation.

The shared implementation is `src/utils/hybrid.py`.

## Knowledge Graph
`src/graph_retrieval.py` exposes the `retrieve_graph_docs` function backed by Neo4j.
The graph schema covers Companies, Filings, Sections, Subsidiaries, BoardMembers,
Patents, and TechnologyDomains. `src/Knowledge graph/kg.py` handles graph
construction and ingestion. `src/nodes/graph_topics.py` provides intent detection to
pre-screen graph-routable questions before the LLM router runs.

## Agentic Multi-hop Pipeline
`src/nodes/planner.py` decomposes complex questions into ordered sub-questions and
assigns each a route hint and tier. Sub-questions execute sequentially through the
same LangGraph workflow. `src/nodes/merge.py` combines the sub-results before the
final generation step.

## Storage
- Supabase DB in Cloud stores structured SEC/XBRL facts, transcript metadata, patent metadata,
  and litigation metadata.
- Pinecone stores text chunks for filings, transcripts, patents, and litigation.
- Neo4j stores the knowledge graph: company–filing–section–subsidiary–board–patent
  relationships.
- Redis stores semantic cache entries and corpus version metadata.
- BigQuery stores structured audit log records (via `src/audit/logger.py`).

## Observability
`src/observability.py` configures LangSmith tracing for all LangGraph runs.
`src/audit/logger.py` writes a per-query `AuditRecord` (with a UUID `query_id`,
`tenant_id`, `user_id`, route, latency, sources, and confidence score) to BigQuery
in a background thread. Structured application logs use `src/utils/logger.py`.

## Evaluation
`evals/runner.py` drives ablation experiments defined in
`evals/configs/ablation_configs.py`. Custom metrics live in `evals/metrics/`:
- `due_diligence.py` — NumericalAccuracy and contradiction-detection metrics
- `gemini_judge.py` — LLM-as-judge scoring via Gemini
- `retrieval.py` — retrieval precision/recall metrics

## External Services
- Vertex AI Gemini for routing, planning, rewriting, grading, generation, SQL
  generation, Cypher generation, and contradiction analysis.
- Pinecone for vector search and hosted embeddings (`llama-text-embed-v2`) and
  reranking (`bge-reranker-v2-m3`).
- Neo4j for knowledge graph storage and Cypher query execution.
- LangSmith for LLM call tracing and observability.
- GCP Secret Manager for storing secrets and  runtime secret resolution.
- Gemini Google Search grounding for web fallback when internal retrieval is
  insufficient.

## Known Gaps
- Slack/webhook notifications are not yet implemented.
- Live demo scripts under `src/` require external credentials and should be run
  manually, not as unit tests.
