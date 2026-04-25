#This is the main retrieval entry point for Transcipts dat
"""
Transcript retrieval pipeline for M&A Oracle RAG system.

retrieve_transcripts:
    Searches the Pinecone 'transcripts' namespace using integrated inference
    (index.search() with raw text). Optionally filters by company_title and
    period_of_report date range. Final ranking uses dense vector rank plus
    BM25 keyword rank fused with Reciprocal Rank Fusion (RRF).

generate_transcript_answer:
    Calls retrieve_transcripts, builds a numbered context string (max 10000
    chars), and calls Gemini to produce a grounded answer with citations
    (accession_no + period_of_report per chunk).

Pinecone index: ragcapstone, namespace: transcripts
Embedding: llama-text-embed-v2 (Pinecone hosted inference)
Hybrid ranking: dense vector search + BM25 + RRF
LLM: gemini-2.5-flash via google-genai SDK + GCP Vertex AI
"""

from __future__ import annotations

import re

from google import genai
from google.genai import types
from pinecone import Pinecone

from src.utils.hybrid import hybrid_rrf_rank
from src.utils.logger import get_logger
from src.utils.secrets import get_secret

logger = get_logger(__name__)

_INDEX_NAME   = "ragcapstone"
_NAMESPACE    = "transcripts"
_GCP_PROJECT  = "codelab-2-485215"
_GCP_LOCATION = "us-central1"
_LLM_MODEL    = "gemini-2.5-flash"
_MAX_CONTEXT_CHARS = 10_000

_SYSTEM_PROMPT = """\
You are a financial analyst assistant for M&A due diligence.
Answer the question strictly using the provided context excerpts from SEC earnings filings.
Rules:
- Cite the source of every factual claim using [accession_no | period_of_report].
- If the context is insufficient to answer, say so explicitly — do not speculate.
- Be concise and precise. Use numbers and dates from the context directly."""

_COMPANY_CANONICAL: dict[str, str] = {
    "apple":                  "Apple Inc.",
    "apple inc":              "Apple Inc.",
    "microsoft":              "MICROSOFT CORP",
    "microsoft corp":         "MICROSOFT CORP",
    "microsoft corporation":  "MICROSOFT CORP",
}

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_company(name: str) -> str:
    """Map a free-form company name to the exact stored Pinecone value."""
    key = _PUNCT_RE.sub("", name.lower()).strip()
    return _COMPANY_CANONICAL.get(key, name)


_pc: Pinecone | None = None
_genai_client: genai.Client | None = None


def _get_pinecone() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=get_secret("PINECONE_API_KEY"))
    return _pc


def _get_genai() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(
            vertexai=True,
            project=_GCP_PROJECT,
            location=_GCP_LOCATION,
        )
    return _genai_client


def _build_filter(company: str | None, period_start: str | None, period_end: str | None) -> dict | None:
    """
    Build a Pinecone metadata filter dict for the transcript search.

    Filters on company_title (exact match) and/or period_of_report date range.
    Returns None if no filters are requested.
    """
    conditions: list[dict] = []

    if company:
        conditions.append({"company_title": {"$eq": _normalize_company(company)}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve_transcripts(
    query: str,
    company: str | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Search the transcripts namespace and return hybrid-ranked hits.

    Uses Pinecone integrated inference for vector search (no local embedding).
    Applies metadata filters on company_title and period_of_report.
    Final ranking combines Pinecone dense rank and BM25 keyword rank with RRF.

    Args:
        query:        Natural-language question to search for.
        company:      Exact company_title value, e.g. "Apple Inc." or None for all.
        period_start: ISO date lower bound on period_of_report, e.g. "2022-01-01".
        period_end:   ISO date upper bound on period_of_report, e.g. "2024-12-31".
        top_k:        Number of candidates to fetch and return after RRF.

    Returns:
        List of hit dicts, each containing '_id', '_score', and 'fields' (metadata).
        Ordered by dense-vector + BM25 RRF score.
    """
    pc    = _get_pinecone()
    index = pc.Index(_INDEX_NAME)

    metadata_filter = _build_filter(company, period_start, period_end)

    search_kwargs: dict = {
        "namespace": _NAMESPACE,
        "query": {
            "top_k": top_k,
            "inputs": {"text": query},
        },
        "fields": [
            "company_title", "content_type", "form_type", "section_item",
            "filed_date", "period_of_report", "accession_no", "chunk_index", "cik",
            "text",
        ],
    }
    if metadata_filter:
        search_kwargs["query"]["filter"] = metadata_filter

    response = index.search(**search_kwargs)
    hits: list[dict] = response.get("result", {}).get("hits", [])

    logger.info(
        "Transcript vector search complete",
        extra={
            "namespace": _NAMESPACE,
            "query": query,
            "company": company,
            "period_start": period_start,
            "period_end": period_end,
            "hit_count": len(hits),
            "filter_applied": metadata_filter is not None,
        },
    )

    if not hits:
        return []

    # Post-retrieval date filter (period_of_report stored as "YYYY-MM-DD" string)
    if period_start or period_end:
        hits = [
            h for h in hits
            if (not period_start or h["fields"].get("period_of_report", "") >= period_start)
            and (not period_end or h["fields"].get("period_of_report", "") <= period_end)
        ]
        if not hits:
            return []

    dense_ranked_hits = sorted(hits, key=lambda h: h.get("_score", 0.0), reverse=True)
    hits = hybrid_rrf_rank(
        query,
        dense_ranked_hits,
        text_getter=lambda h: h.get("fields", {}).get("text", ""),
        key=lambda h: h.get("_id") or h.get("id") or "",
        top_k=top_k,
    )
    rerank_method = "dense-vector + BM25 RRF"

    logger.info(
        "Transcript retrieval complete",
        extra={
            "namespace": _NAMESPACE,
            "query": query,
            "final_hit_count": len(hits),
            "rerank_method": rerank_method,
        },
    )
    return hits


def _build_transcript_context(hits: list[dict]) -> str:
    """
    Build a numbered context string from retrieved hits, capped at 10000 chars.

    Each chunk is prefixed with a metadata header:
    [N] Company | Period | Accession
    """
    parts: list[str] = []
    total_chars = 0

    for i, hit in enumerate(hits, start=1):
        fields = hit.get("fields", {})
        header = (
            f"[{i}] {fields.get('company_title', 'Unknown')} | "
            f"Period: {fields.get('period_of_report', 'N/A')} | "
            f"Accession: {fields.get('accession_no', 'N/A')}"
        )
        body = fields.get("text", "").strip()
        chunk_str = f"{header}\n{body}\n"

        if total_chars + len(chunk_str) > _MAX_CONTEXT_CHARS:
            break
        parts.append(chunk_str)
        total_chars += len(chunk_str)

    return "\n".join(parts)


def generate_transcript_answer(
    query: str,
    company: str | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
) -> dict:
    """
    Retrieve relevant transcript chunks and generate a grounded answer via Gemini.

    Args:
        query:        Natural-language question.
        company:      Filter by company_title, or None for all companies.
        period_start: ISO date lower bound on period_of_report.
        period_end:   ISO date upper bound on period_of_report.

    Returns:
        Dict with keys:
            query        (str)  — original question
            answer       (str)  — Gemini-generated answer with citations
            hits         (list) — raw retrieval results
            hit_count    (int)  — number of chunks retrieved
    """
    hits = retrieve_transcripts(
        query=query,
        company=company,
        period_start=period_start,
        period_end=period_end,
    )

    if not hits:
        return {
            "query":     query,
            "answer":    "No relevant transcript chunks found for the given filters.",
            "hits":      [],
            "hit_count": 0,
        }

    context = _build_transcript_context(hits)

    citations = "; ".join(
        f"{h['fields'].get('accession_no', '?')} ({h['fields'].get('period_of_report', '?')})"
        for h in hits
    )

    user_prompt = (
        f"Context (earnings filing excerpts):\n{context}\n\n"
        f"Available sources: {citations}\n\n"
        f"Question: {query}"
    )

    client = _get_genai()
    response = client.models.generate_content(
        model=_LLM_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=_SYSTEM_PROMPT),
    )
    answer = (response.text or "").strip()

    logger.info(
        "Transcript answer generated",
        extra={"query": query, "company": company, "hit_count": len(hits)},
    )

    return {
        "query":     query,
        "answer":    answer,
        "hits":      hits,
        "hit_count": len(hits),
    }


if __name__ == "__main__":
    result = generate_transcript_answer(
        query="What did Apple say about iPhone revenue in fiscal 2024?",
        company="Apple Inc.",
        period_start="2023-01-01",
        period_end="2024-12-31",
    )
    print(f"Query: {result['query']}")
    print(f"Hits:  {result['hit_count']}")
    print(f"\nAnswer:\n{result['answer']}")
