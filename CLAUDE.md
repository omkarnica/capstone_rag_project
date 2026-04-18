cat > CLAUDE.md << 'ENDOFFILE'
# M&A Oracle — Claude Code Navigation Guide

## What This Project Is
Agentic RAG system for M&A due diligence. Finds contradictions between
what management says (earnings calls) vs. what filings reveal (SEC data).
The contradiction detection across quantitative and qualitative sources
is the core value proposition.

## Tech Stack
- LLM: Gemini via GCP Vertex AI, google-genai SDK (NO API keys, GCP creds only)
- GCP Project: codelab-2-485215, region: us-central1
- Deployment: Google Cloud Run
- DB: PostgreSQL — local dev on port 5433, database name ma_oracle
- Vector store: Pinecone — index ragcapstone (NITHU's Org), hosted inference
- Embeddings: llama-text-embed-v2 via Pinecone hosted inference (upsert_records())
- Language: Python
- Package manager: uv

## Project Structure
xbrl/               <- DONE. SEC financial data pipeline (lives at root, not src/)
src/
  transcripts/      <- DONE. Apple earnings press releases from SEC 8-K filings
  patents/          <- IN PROGRESS. USPTO PatentsView API pipeline
  utils/            <- Shared: exceptions.py (error boundaries), logging helpers
tests/
docs/
data/               <- SEC bulk files, downloaded data
logs/

## LLM Usage Pattern
ALWAYS call LLMs like this — never use raw API keys:

from litellm import completion
import google.auth
import google.auth.transport.requests

creds, project = google.auth.default()
creds.refresh(google.auth.transport.requests.Request())

response = completion(
    model="vertex_ai/gemini-2.0-flash",
    messages=[{"role": "user", "content": "your prompt"}],
    vertex_project="codelab-2-485215",
    vertex_location="us-central1",
)

## Coding Conventions — ALWAYS FOLLOW THESE
- Logging: get_logger(__name__), never print()
- HTTP: tenacity retry, 3 attempts, exponential backoff, 0.15s delay
- HTTP errors: download_error_boundary from src/utils/exceptions.py
- DB errors: db_error_boundary from src/utils/exceptions.py
- DB inserts: ON CONFLICT DO NOTHING
- Bulk loads: PostgreSQL COPY, not row-by-row INSERT
- DB connection: get_connection() from xbrl/loader.py
- GCP credentials only for LLM — no API keys, no .env for Gemini
- Pinecone API key: stored in .env as PINECONE_API_KEY, loaded via load_dotenv()

## Pinecone Vector Store
- Index: ragcapstone (NITHU's Org)
- Embedding model: llama-text-embed-v2 (Pinecone hosted inference)
- ALWAYS use index.upsert_records(namespace=..., records=[...]) — NOT upsert()
- Each record must have: id (UUID str), text (chunk), plus flat metadata fields
- Namespace strategy:
  - transcripts — all earnings transcript chunks (filter by company_title or cik)
  - patents      — all patent claim chunks (filter by assignee_organization or cik)
- NO pgvector — do not add embedding columns to PostgreSQL tables

## MCP Tools: code-review-graph
ALWAYS use these tools BEFORE Grep/Glob/Read — faster, cheaper, structural context.

| Tool | Use when |
|------|----------|
| semantic_search_nodes | Finding functions/classes by name or keyword |
| query_graph | Tracing callers, callees, imports, tests, dependencies |
| detect_changes | Reviewing code changes — gives risk-scored analysis |
| get_review_context | Need source snippets — token-efficient |
| get_impact_radius | Understanding blast radius of a change |
| get_affected_flows | Finding which execution paths are impacted |
| get_architecture_overview | Understanding high-level codebase structure |
| refactor_tool | Planning renames, finding dead code |

Fall back to Grep/Glob/Read only when the graph does not cover it.
Graph auto-updates on file changes via hooks.

## Database Schema
filings             <- XBRL: filing metadata (adsh, cik, name, form, period)
facts               <- XBRL: numeric values (tag, value, ddate)
tag_normalization   <- XBRL: raw to canonical tag mapping
transcripts         <- 8-K metadata (cik, accession_no, filed_date, exhibit_url)
transcript_sections <- extracted earnings text (transcript_id, section_text)
patents             <- patent metadata [TO BE CREATED]
patent_claims       <- patent claim text [TO BE CREATED]

Key facts:
- adsh is the join key between filings and facts (XBRL)
- transcript_id is the join key between transcripts and transcript_sections
- num.txt has ~3.8M rows — always use DB, never Excel
- Multiple XBRL tags map to the same concept — always use canonical tags

## Status
- [x] XBRL pipeline — structured financial data from SEC bulk files
- [x] Transcripts pipeline — earnings press releases from 8-K Exhibit 99.1
- [x] Patents pipeline — USPTO bulk TSV files (reader.py + parser.py + loader.py)
- [x] NL-to-SQL — natural language query over XBRL data (src/nl_sql/pipeline.py)
- [x] Transcripts vector store — Pinecone upsert via pinecone_loader.py
- [ ] Patents vector store — Pinecone upsert (patents namespace)
- [ ] RAG router
- [ ] Knowledge graph
- [ ] Agentic pipeline

## Active Branch
karnica/transcripts-pipeline — used for all data pipelines

## Team
- A: manages GCP/IAM. Required role: roles/aiplatform.user
- C (you): data pipelines
ENDOFFILE
## LLM Usage Pattern (UPDATED - use google-genai SDK, NOT LiteLLM)
ALWAYS call LLMs like this:
```python
from google import genai

client = genai.Client(
    vertexai=True,
    project="codelab-2-485215",
    location="us-central1"
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="your prompt here"
)
result = response.text
```
Never use LiteLLM. Never use raw API keys.
