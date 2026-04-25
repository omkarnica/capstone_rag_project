# M&A Oracle Architecture

## Overview
M&A Oracle answers due diligence questions by routing each query to the evidence
source most likely to contain the answer, then generating a concise cited answer.

## Request Flow
1. `src/app.py` receives `/query`, `/adaptive-query`, or `/due-diligence`.
2. `src/api.py` checks the semantic cache and plans single-hop or multi-hop work.
3. `src/graph.py` executes the LangGraph workflow.
4. `src/nodes/router.py` chooses the route.
5. `src/nodes/retriever.py` dispatches to the relevant data module.
6. `src/nodes/grader.py` filters retrieved evidence.
7. `src/nodes/generator.py` produces the final answer and citations.
8. `src/nodes/fallback.py` uses Gemini Google Search grounding first, then
   Google Custom Search only when internal retrieval cannot provide evidence.

## Routes
- `sql`: PostgreSQL structured facts through `src/nl_sql/pipeline.py`
- `filings`: RAPTOR filing retrieval through `src/filings/raptor_retrieval.py`
- `transcripts`: Pinecone transcript retrieval through `src/transcripts/retrieval.py`
- `patents`: Pinecone patent retrieval through `src/patents/retrieval.py`
- `litigation`: Pinecone litigation retrieval through `src/litigation/retrieval.py`
- `contradiction`: multi-source comparison through `src/contradictions/detector.py`
- `graph`: placeholder for future knowledge graph work
- `llm_direct`: general M&A/finance explanations without retrieval

## Hybrid Retrieval
Vector evidence routes use hybrid ranking:

1. Dense vector retrieval returns semantic candidates from Pinecone.
2. BM25 scores the same candidate texts for keyword relevance.
3. Reciprocal Rank Fusion combines dense and BM25 rankings.
4. The final top-k fused chunks are passed to grading and generation.

The shared implementation is `src/utils/hybrid.py`.

## Storage
- PostgreSQL stores structured SEC/XBRL, transcript metadata, patent metadata, and
  litigation metadata.
- Pinecone stores text chunks for filings, transcripts, patents, and litigation.
- Redis stores semantic cache entries and corpus version metadata.

## External Services
- Vertex AI Gemini for routing, planning, grading, generation, SQL generation, and
  contradiction analysis.
- Pinecone for vector search and hosted embeddings.
- Google Custom Search for web fallback against trusted M&A source domains.

## Known Gaps
- The `graph` route is a placeholder.
- Web fallback depends on `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID`.
- Live demo scripts under `src/` require external credentials and should be run
  manually, not as unit tests.
