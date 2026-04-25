# M&A Oracle - Project Guide

## Purpose
M&A Oracle is an agentic RAG system for due diligence. Its core value is finding
evidence-backed discrepancies between management narrative and formal records:
SEC filings, XBRL financial facts, earnings transcript excerpts, patents, and
litigation records.

## Runtime Stack
- API: FastAPI in `src/app.py`
- Orchestration: LangGraph in `src/graph.py`
- LLM: Gemini on Vertex AI through `google-genai` or LangChain Vertex wrappers
- Project: `codelab-2-485215`
- Region: `us-central1`
- Database: PostgreSQL, local dev commonly on port `5433`, database `ma_oracle`
- Vector store: Pinecone index `ragcapstone`
- Embeddings: Pinecone hosted `llama-text-embed-v2`
- Reranker: Pinecone hosted `bge-reranker-v2-m3`
- Cache: Redis through `src/cache`
- Package manager: `uv`

## Main Routes
The router in `src/nodes/router.py` classifies questions into:
- `sql`: structured XBRL and database questions
- `filings`: SEC 10-K/10-Q narrative retrieval through RAPTOR
- `transcripts`: management statements and earnings excerpts
- `patents`: patent/IP questions
- `litigation`: legal exposure and court records
- `graph`: knowledge graph placeholder
- `contradiction`: due diligence comparison workflow
- `llm_direct`: general M&A concepts that do not require retrieval

## Entry Points
- `src/app.py`: FastAPI app
- `src/api.py`: adaptive query wrapper, semantic cache, single/multi-hop execution
- `src/graph.py`: LangGraph wiring
- `src/contradictions/detector.py`: contradiction due diligence sweep

## Data Source Modules
- `src/xbrl`: SEC structured financial facts in PostgreSQL
- `src/nl_sql`: natural language to SQL over XBRL and structured tables
- `src/filings`: SEC filing ingestion, chunking, RAPTOR indexing/retrieval
- `src/transcripts`: SEC 8-K earnings press release ingestion and retrieval
- `src/patents`: patent loading and Pinecone retrieval
- `src/litigation`: litigation loading and Pinecone retrieval
- `src/cache`: Redis-backed semantic/retrieval cache

## Required Environment
- GCP Application Default Credentials for Vertex AI and Secret Manager
- `PINECONE_API_KEY` in `.env`, environment, or GCP Secret Manager
- `REDIS_URL` if Redis is not on `redis://localhost:6379`
- `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID` for web fallback

## LLM Usage
Use Vertex AI credentials, not raw Gemini API keys.

```python
from google import genai

client = genai.Client(
    vertexai=True,
    project="codelab-2-485215",
    location="us-central1",
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="your prompt here",
)
result = response.text
```

## Conventions
- Use `get_logger(__name__)`, not `print()`, in production modules.
- Use `download_error_boundary` and `db_error_boundary` from `src/utils/exceptions.py`.
- Use PostgreSQL `COPY` for bulk loads.
- Use `ON CONFLICT DO NOTHING` for idempotent inserts where applicable.
- Use `src.xbrl.loader.get_connection()` for database connections.
- Use Pinecone integrated inference APIs already established in the project.

## Testing
Run unit tests with:

```bash
uv run pytest -q
```

Pytest is configured to collect only `tests/`. Demo scripts under `src/` may call
live external services and should not be treated as unit tests.
