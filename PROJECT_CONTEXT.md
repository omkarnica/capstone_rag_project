# M&A Oracle — Project Context for Claude Code

## What We're Building
An Agentic RAG system for M&A due diligence. Analyzes SEC filings, 
earnings transcripts, patents, and financial data to surface hidden risks.

## Tech Stack
- LLM: Gemini via Vertex AI, accessed through LiteLLM (NO raw API keys)
- GCP Project ID: codelab-2-485215
- GCP Region: us-central1
- Deployment target: Google Cloud Run
- Database: PostgreSQL (local dev → Cloud SQL in production)
- Language: Python

## LiteLLM Usage Pattern
ALWAYS call LLMs like this — never use raw API keys:
```python
from litellm import completion
import google.auth
import google.auth.transport.requests

# Get GCP credentials
creds, project = google.auth.default()
creds.refresh(google.auth.transport.requests.Request())

response = completion(
    model="vertex_ai/gemini-2.0-flash",
    messages=[{"role": "user", "content": "your prompt"}],
    vertex_project="codelab-2-485215",
    vertex_location="us-central1",
)
```

## Team Split (Week 1)
- Teammate A: HTML parser for 10-K/8-K filings
- Teammate B: Book RAG indexing
- Me (C): XBRL loader → Postgres ← YOU ARE HERE

## Data Location
- SEC XBRL bulk data downloaded to: [UPDATE WITH YOUR PATH]
- Files: num.txt, sub.txt, tag.txt, pre.txt (tab-separated despite .txt extension)

## Database Schema
Two core tables:
- `filings` — from sub.txt (company metadata, form type, period)
- `facts` — from num.txt (tag, value, date, units)
- `tag_normalization` — maps raw XBRL tags to canonical names

## Key Facts About the Data
- num.txt has ~3.8M rows (Excel's 1M row limit truncates it — always use DB)
- 62,165 unique XBRL tags exist
- Multiple tags mean the same concept (RevenueFromContractWith... AND Revenues both = Revenue)
- adsh is the join key between sub.txt and num.txt