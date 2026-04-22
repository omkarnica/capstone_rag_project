"""
Litigation retrieval pipeline for M&A Oracle RAG system.

retrieve_litigation:
    Searches the Pinecone 'litigation' namespace using integrated inference.
    Optionally filters by company_name and date_filed range (Python post-filter).
    Reranks with bge-reranker-v2-m3, falling back to score-sorted order.

generate_litigation_answer:
    Calls retrieve_litigation, builds a numbered context string (max 10000 chars),
    and calls Gemini to produce a grounded answer with citations
    (case_name + date_filed + court per chunk).

Pinecone index: ragcapstone, namespace: litigation
Embedding: llama-text-embed-v2 (Pinecone hosted inference)
Reranker: bge-reranker-v2-m3 via pc.inference.rerank()
LLM: gemini-2.5-flash via google-genai SDK + GCP Vertex AI
"""

from __future__ import annotations

import re

from google import genai
from google.genai import types
from pinecone import Pinecone

from src.utils.logger import get_logger
from src.utils.secrets import get_secret

logger = get_logger(__name__)

_INDEX_NAME        = "ragcapstone"
_NAMESPACE         = "litigation"
_RERANK_MODEL      = "bge-reranker-v2-m3"
_GCP_PROJECT       = "codelab-2-485215"
_GCP_LOCATION      = "us-central1"
_LLM_MODEL         = "gemini-2.5-flash"
_MAX_CONTEXT_CHARS = 10_000

_SYSTEM_PROMPT = """\
You are a legal analyst assistant for M&A due diligence.
Answer the question strictly using the provided court opinion excerpts.
Rules:
- Cite every factual claim with [case_name | court | date_filed].
- If the context is insufficient to answer, say so explicitly — do not speculate.
- Note case outcomes, damages awarded, and ongoing vs. terminated cases where relevant.
- Flag patterns across multiple cases (e.g. recurring IP disputes, regulatory actions)."""

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


def _build_filter(company: str | None) -> dict | None:
    """Build a Pinecone metadata filter for company_name exact match."""
    if not company:
        return None
    return {"company_name": {"$eq": _normalize_company(company)}}


def retrieve_litigation(
    query: str,
    company: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Search the litigation namespace and return reranked hits.

    Args:
        query:      Natural-language question about litigation.
        company:    Exact company_name value, or None for all.
        date_start: ISO date lower bound on date_filed, e.g. "2018-01-01".
        date_end:   ISO date upper bound on date_filed, e.g. "2024-12-31".
        top_k:      Number of candidates to fetch before reranking.

    Returns:
        List of hit dicts ordered by rerank score descending.
        Each dict has '_id', '_score', and 'fields' (metadata).
    """
    pc    = _get_pinecone()
    index = pc.Index(_INDEX_NAME)

    metadata_filter = _build_filter(company)

    search_kwargs: dict = {
        "namespace": _NAMESPACE,
        "query": {
            "top_k": top_k,
            "inputs": {"text": query},
        },
        "fields": [
            "company_name", "content_type", "case_id", "case_name",
            "court", "court_citation", "date_filed", "status",
            "cite_count", "opinion_id", "chunk_index", "url", "text",
        ],
    }
    if metadata_filter:
        search_kwargs["query"]["filter"] = metadata_filter

    response = index.search(**search_kwargs)
    hits: list[dict] = response.get("result", {}).get("hits", [])

    logger.info(
        "Litigation vector search complete",
        extra={
            "namespace": _NAMESPACE,
            "query": query,
            "company": company,
            "hit_count": len(hits),
        },
    )

    if not hits:
        return []

    # Post-retrieval date filter (date_filed stored as "YYYY-MM-DD")
    if date_start or date_end:
        hits = [
            h for h in hits
            if (not date_start or h["fields"].get("date_filed", "") >= date_start)
            and (not date_end or h["fields"].get("date_filed", "") <= date_end)
        ]
        if not hits:
            return []

    # Rerank
    try:
        documents = [
            f"Case: {h['fields'].get('case_name', '')} | "
            f"Court: {h['fields'].get('court_citation', '')} | "
            f"Date: {h['fields'].get('date_filed', '')} | "
            f"{h['fields'].get('text', '')[:300]}"
            for h in hits
        ]
        rerank_result = pc.inference.rerank(
            model=_RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=top_k,
            return_documents=False,
        )
        reranked_indices = [item.index for item in rerank_result.data]
        hits = [hits[i] for i in reranked_indices]
        rerank_method = "bge-reranker-v2-m3"
    except Exception as exc:
        logger.warning(
            "Reranking failed — using score-sorted order",
            extra={"error": str(exc)},
        )
        hits = sorted(hits, key=lambda h: h.get("_score", 0.0), reverse=True)
        rerank_method = "score-sorted (fallback)"

    logger.info(
        "Litigation retrieval complete",
        extra={"query": query, "final_hit_count": len(hits), "rerank_method": rerank_method},
    )
    return hits


def _build_litigation_context(hits: list[dict]) -> str:
    """Build a numbered context string from retrieved hits, capped at 10000 chars."""
    parts: list[str] = []
    total_chars = 0

    for i, hit in enumerate(hits, start=1):
        fields = hit.get("fields", {})
        header = (
            f"[{i}] {fields.get('case_name', 'Unknown')} | "
            f"{fields.get('court_citation', fields.get('court', 'N/A'))} | "
            f"Filed: {fields.get('date_filed', 'N/A')} | "
            f"Status: {fields.get('status', 'N/A')}"
        )
        body = fields.get("text", "").strip()
        chunk_str = f"{header}\n{body}\n"

        if total_chars + len(chunk_str) > _MAX_CONTEXT_CHARS:
            break
        parts.append(chunk_str)
        total_chars += len(chunk_str)

    return "\n".join(parts)


def generate_litigation_answer(
    query: str,
    company: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> dict:
    """
    Retrieve relevant litigation chunks and generate a grounded answer via Gemini.

    Args:
        query:      Natural-language question about litigation.
        company:    Filter by company_name, or None for all.
        date_start: ISO date lower bound on date_filed.
        date_end:   ISO date upper bound on date_filed.

    Returns:
        Dict with keys:
            query      (str)  — original question
            answer     (str)  — Gemini-generated answer with citations
            hits       (list) — raw retrieval results
            hit_count  (int)  — number of chunks retrieved
    """
    hits = retrieve_litigation(
        query=query,
        company=company,
        date_start=date_start,
        date_end=date_end,
    )

    if not hits:
        return {
            "query":     query,
            "answer":    "No relevant litigation chunks found for the given filters.",
            "hits":      [],
            "hit_count": 0,
        }

    context = _build_litigation_context(hits)

    citations = "; ".join(
        f"{h['fields'].get('case_name', '?')} ({h['fields'].get('date_filed', '?')})"
        for h in hits
    )

    user_prompt = (
        f"Context (court opinion excerpts):\n{context}\n\n"
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
        "Litigation answer generated",
        extra={"query": query, "company": company, "hit_count": len(hits)},
    )

    return {
        "query":     query,
        "answer":    answer,
        "hits":      hits,
        "hit_count": len(hits),
    }


if __name__ == "__main__":
    result = generate_litigation_answer(
        query="What patent infringement cases has Apple been involved in?",
        company="Apple Inc",
        date_start="2018-01-01",
        date_end="2024-12-31",
    )
    print(f"Query: {result['query']}")
    print(f"Hits:  {result['hit_count']}")
    print(f"\nAnswer:\n{result['answer']}")
