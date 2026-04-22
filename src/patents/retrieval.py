"""
Patent retrieval pipeline for M&A Oracle RAG system.

retrieve_patents:
    Searches the Pinecone 'patents' namespace using integrated inference
    (index.search() with raw text). Optionally filters by company_title and
    grant_date range. Applies a citation-boost reranking step:
        final_score = 0.7 * semantic_score + 0.3 * (citation_count / max_citation_count)
    Falls back to score-sorted order if bge-reranker-v2-m3 reranking fails.

generate_patent_answer:
    Calls retrieve_patents, builds a numbered context string (max 10000 chars),
    and calls Gemini to produce a grounded answer with citations
    (patent_id + patent_title per chunk).

Pinecone index: ragcapstone, namespace: patents
Embedding: llama-text-embed-v2 (Pinecone hosted inference)
  Indexed text format per chunk:
    Patent: {title}\\nCPC Codes: {cpc_codes}\\nGrant Date: {grant_date}\\nClaim 1: {claim_text}
Reranker: bge-reranker-v2-m3 via pc.inference.rerank() + citation boost
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

_INDEX_NAME   = "ragcapstone"
_NAMESPACE    = "patents"
_RERANK_MODEL = "bge-reranker-v2-m3"
_GCP_PROJECT  = "codelab-2-485215"
_GCP_LOCATION = "us-central1"
_LLM_MODEL    = "gemini-2.5-flash"
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


def _citation_boost_rerank(
    hits: list[dict],
    semantic_scores: list[float],
    final_top_k: int,
) -> list[dict]:
    """
    Blend semantic score with citation count into a final ranking score.

    Formula: final_score = 0.7 * semantic_score + 0.3 * (citation_count / max_citation_count)

    Citation count is read from the 'citation_count' metadata field.
    If max_citation_count is 0, the citation term is omitted (weight falls to semantic only).

    Args:
        hits:           Hit dicts from Pinecone search or initial reranking.
        semantic_scores: Normalized semantic scores aligned with hits list.
        final_top_k:    Number of hits to return after boosting.

    Returns:
        Top final_top_k hits sorted by final_score descending.
    """
    citation_counts = [
        float(h["fields"].get("citation_count") or 0) for h in hits
    ]
    max_citations = max(citation_counts) if citation_counts else 0.0

    scored: list[tuple[float, dict]] = []
    for i, hit in enumerate(hits):
        sem = semantic_scores[i] if i < len(semantic_scores) else 0.0
        if max_citations > 0:
            cite_norm = citation_counts[i] / max_citations
            final = 0.7 * sem + 0.3 * cite_norm
        else:
            final = sem
        scored.append((final, hit))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored[:final_top_k]]


def retrieve_patents(
    query: str,
    company: str | None = None,
    grant_date_start: str | None = None,
    grant_date_end: str | None = None,
    top_k: int = 15,
    final_top_k: int = 8,
) -> list[dict]:
    """
    Search the patents namespace and return citation-boosted, reranked hits.

    Step 1: Vector search via Pinecone integrated inference.
    Step 2: Semantic rerank with bge-reranker-v2-m3 (fallback: score-sorted).
    Step 3: Citation-count boost applied to reranked scores.

    Args:
        query:            Natural-language question or technology description.
        company:          Exact company_title value. Defaults to "Apple Inc".
        grant_date_start: ISO date lower bound on grant_date, e.g. "2020-01-01".
        grant_date_end:   ISO date upper bound on grant_date, e.g. "2024-12-31".
        top_k:            Candidates fetched from Pinecone before reranking.
        final_top_k:      Final number of hits returned after citation boost.

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

    # Step 2: Semantic reranking
    rerank_method = "score-sorted (fallback)"
    semantic_scores: list[float] = [h.get("_score", 0.0) for h in hits]

    try:
        documents = [
            (
                f"Patent: {h['fields'].get('patent_title', '')} | "
                f"CPC: {h['fields'].get('cpc_codes', '')} | "
                f"Grant: {h['fields'].get('grant_date', '')} | "
                f"Claim {h['fields'].get('claim_number', '')}"
            )
            for h in hits
        ]
        rerank_result = pc.inference.rerank(
            model=_RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=len(hits),
            return_documents=False,
        )
        reranked_items = sorted(rerank_result.data, key=lambda x: x.index)
        semantic_scores = [item.score for item in rerank_result.data]
        hits = [hits[item.index] for item in rerank_result.data]
        rerank_method = "bge-reranker-v2-m3 + citation-boost"
    except Exception as exc:
        logger.warning(
            "Reranking failed — using score-sorted order with citation boost",
            extra={"error": str(exc)},
        )
        hits = sorted(hits, key=lambda h: h.get("_score", 0.0), reverse=True)
        semantic_scores = [h.get("_score", 0.0) for h in hits]
        rerank_method = "score-sorted + citation-boost (fallback)"

    # Step 3: Citation-count boost
    hits = _citation_boost_rerank(hits, semantic_scores, final_top_k)

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

    client = _get_genai()
    response = client.models.generate_content(
        model=_LLM_MODEL,
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
