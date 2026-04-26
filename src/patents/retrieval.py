#This is the main entry point for patents retrival
"""
Patent retrieval pipeline for M&A Oracle RAG system.

retrieve_patents:
    Searches the Pinecone 'patents' namespace using integrated inference
    (index.search() with raw text). Optionally filters by company_title and
    grant_date range. Final ranking uses dense vector rank plus BM25 keyword
    rank fused with Reciprocal Rank Fusion (RRF).

generate_patent_answer:
    Calls retrieve_patents, builds a numbered context string (max 10000 chars),
    and calls Gemini to produce a grounded answer with citations
    (patent_id + patent_title per chunk).

Pinecone index: ragcapstone, namespace: patents
Embedding: llama-text-embed-v2 (Pinecone hosted inference)
  Indexed text format per chunk:
    Patent: {title}\\nCPC Codes: {cpc_codes}\\nGrant Date: {grant_date}\\nClaim 1: {claim_text}
Hybrid ranking: dense vector search + BM25 + RRF
LLM: gemini-2.5-flash via google-genai SDK + GCP Vertex AI
"""

from __future__ import annotations

import re

from google import genai
from google.genai import types
from pinecone import Pinecone

from src.model_config import get_genai_client, get_model_name
from src.utils.hybrid import hybrid_rrf_rank
from src.utils.logger import get_logger
from src.utils.secrets import get_secret

logger = get_logger(__name__)

_INDEX_NAME   = "ragcapstone"
_NAMESPACE    = "patents"
_MAX_CONTEXT_CHARS = 10_000

_SYSTEM_PROMPT = """\
You are a patent analyst assistant for M&A due diligence and IP strategy.
Answer the question strictly using the provided patent claim excerpts.
Rules:
- Cite every factual claim with [patent_id | patent_title].
- If the context is insufficient to answer, say so explicitly — do not speculate.
- Be precise about claim scope, CPC classifications, and grant dates.
- Summarize patterns across multiple patents where relevant."""

_COMPANY_CANONICAL: dict[str, str] = {
    "apple":                 "Apple Inc",
    "apple inc":             "Apple Inc",
    "apple inc.":            "Apple Inc",
    "microsoft":             "MICROSOFT CORP",
    "microsoft corp":        "MICROSOFT CORP",
    "microsoft corporation": "MICROSOFT CORP",
}

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_company(name: str) -> str:
    """Map a free-form company name to the exact stored Pinecone value."""
    key = _PUNCT_RE.sub("", name.lower()).strip()
    return _COMPANY_CANONICAL.get(key, name)


_pc: Pinecone | None = None


def _get_pinecone() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=get_secret("PINECONE_API_KEY"))
    return _pc


def _build_filter(
    company: str | None,
    grant_date_start: str | None,
    grant_date_end: str | None,
) -> dict | None:
    """
    Build a Pinecone metadata filter for the patent search.

    Filters on company_title (exact match) and/or grant_date date range.
    Returns None if no filters are requested.
    """
    conditions: list[dict] = []

    if company:
        conditions.append({"company_title": {"$eq": _normalize_company(company)}})

    if grant_date_start and grant_date_end:
        conditions.append({"grant_date": {"$gte": grant_date_start}})
        conditions.append({"grant_date": {"$lte": grant_date_end}})
    elif grant_date_start:
        conditions.append({"grant_date": {"$gte": grant_date_start}})
    elif grant_date_end:
        conditions.append({"grant_date": {"$lte": grant_date_end}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve_patents(
    query: str,
    company: str | None = None,
    grant_date_start: str | None = None,
    grant_date_end: str | None = None,
    top_k: int = 15,
    final_top_k: int = 8,
) -> list[dict]:
    """
    Search the patents namespace and return hybrid-ranked hits.

    Step 1: Vector search via Pinecone integrated inference.
    Step 2: BM25 keyword rank over candidate text.
    Step 3: Reciprocal Rank Fusion over dense and BM25 rankings.

    Args:
        query:            Natural-language question or technology description.
        company:          Exact company_title value. Defaults to "Apple Inc".
        grant_date_start: ISO date lower bound on grant_date, e.g. "2020-01-01".
        grant_date_end:   ISO date upper bound on grant_date, e.g. "2024-12-31".
        top_k:            Candidates fetched from Pinecone before RRF.
        final_top_k:      Final number of hits returned after RRF.

    Returns:
        List of hit dicts (up to final_top_k), ordered by final_score descending.
        Each dict has '_id', '_score', and 'fields' (metadata).
    """
    pc    = _get_pinecone()
    index = pc.Index(_INDEX_NAME)

    metadata_filter = _build_filter(company, grant_date_start, grant_date_end)

    search_kwargs: dict = {
        "namespace": _NAMESPACE,
        "query": {
            "top_k": top_k,
            "inputs": {"text": query},
        },
        "fields": [
            "company_title", "content_type", "patent_id", "claim_number",
            "patent_title", "assignee_organization", "cpc_codes",
            "citation_count", "grant_date", "text",
        ],
    }
    if metadata_filter:
        search_kwargs["query"]["filter"] = metadata_filter

    response = index.search(**search_kwargs)
    hits: list[dict] = response.get("result", {}).get("hits", [])

    logger.info(
        "Patent vector search complete",
        extra={
            "namespace": _NAMESPACE,
            "query": query,
            "company": company,
            "grant_date_start": grant_date_start,
            "grant_date_end": grant_date_end,
            "hit_count": len(hits),
            "filter_applied": metadata_filter is not None,
        },
    )

    if not hits:
        return []

    dense_ranked_hits = sorted(hits, key=lambda h: h.get("_score", 0.0), reverse=True)
    hits = hybrid_rrf_rank(
        query,
        dense_ranked_hits,
        text_getter=lambda h: h.get("fields", {}).get("text", ""),
        key=lambda h: h.get("_id") or h.get("id") or "",
        top_k=final_top_k,
    )
    rerank_method = "dense-vector + BM25 RRF"

    logger.info(
        "Patent retrieval complete",
        extra={
            "namespace": _NAMESPACE,
            "query": query,
            "final_hit_count": len(hits),
            "rerank_method": rerank_method,
        },
    )
    return hits


def _build_patent_context(hits: list[dict]) -> str:
    """
    Build a numbered context string from retrieved patent hits, capped at 10000 chars.

    Each chunk is prefixed with a metadata header:
    [N] patent_id | patent_title | Grant date | CPC | Citations
    """
    parts: list[str] = []
    total_chars = 0

    for i, hit in enumerate(hits, start=1):
        fields = hit.get("fields", {})
        header = (
            f"[{i}] Patent ID: {fields.get('patent_id', 'N/A')} | "
            f"Title: {fields.get('patent_title', 'N/A')}"
        )
        body = (
            f"Grant Date: {fields.get('grant_date', 'N/A')} | "
            f"CPC: {fields.get('cpc_codes', 'N/A')} | "
            f"Citations: {fields.get('citation_count', 0)} | "
            f"Claim #{fields.get('claim_number', 'N/A')}"
        )
        chunk_str = f"{header}\n{body}\n"

        if total_chars + len(chunk_str) > _MAX_CONTEXT_CHARS:
            break
        parts.append(chunk_str)
        total_chars += len(chunk_str)

    return "\n".join(parts)


def generate_patent_answer(
    query: str,
    company: str | None = None,
    grant_date_start: str | None = None,
    grant_date_end: str | None = None,
) -> dict:
    """
    Retrieve relevant patent claim chunks and generate a grounded answer via Gemini.

    Args:
        query:            Natural-language question about patents or IP.
        company:          Filter by company_title. Defaults to "Apple Inc".
        grant_date_start: ISO date lower bound on grant_date.
        grant_date_end:   ISO date upper bound on grant_date.

    Returns:
        Dict with keys:
            query        (str)  — original question
            answer       (str)  — Gemini-generated answer with citations
            hits         (list) — raw retrieval results (post-boost)
            hit_count    (int)  — number of chunks returned
    """
    hits = retrieve_patents(
        query=query,
        company=company,
        grant_date_start=grant_date_start,
        grant_date_end=grant_date_end,
    )

    if not hits:
        return {
            "query":     query,
            "answer":    "No relevant patent chunks found for the given filters.",
            "hits":      [],
            "hit_count": 0,
        }

    context = _build_patent_context(hits)

    citations = "; ".join(
        f"{h['fields'].get('patent_id', '?')} — {h['fields'].get('patent_title', '?')}"
        for h in hits
    )

    user_prompt = (
        f"Context (patent claim excerpts):\n{context}\n\n"
        f"Available sources: {citations}\n\n"
        f"Question: {query}"
    )

    client = get_genai_client()
    response = client.models.generate_content(
        model=get_model_name(),
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=_SYSTEM_PROMPT),
    )
    answer = (response.text or "").strip()

    logger.info(
        "Patent answer generated",
        extra={"query": query, "company": company, "hit_count": len(hits)},
    )

    return {
        "query":     query,
        "answer":    answer,
        "hits":      hits,
        "hit_count": len(hits),
    }


if __name__ == "__main__":
    result = generate_patent_answer(
        query="What machine learning patents does Apple have related to on-device inference?",
        company="Apple Inc",
        grant_date_start="2020-01-01",
        grant_date_end="2025-12-31",
    )
    print(f"Query: {result['query']}")
    print(f"Hits:  {result['hit_count']}")
    print(f"\nAnswer:\n{result['answer']}")
