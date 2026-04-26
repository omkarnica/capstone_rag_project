"""
Query-time retrieval over RAPTOR vectors stored in Pinecone.
This is entry-point for RAPTOR-based retreival 

Calling modules:

    application code
      |
      |-- src.filings.raptor_retrieval.raptor_retrieve()
      `-- src.filings.retrieve_context_for_llm()

Retrieval flow:

    user query
      |
      |-- embed_query()
      |-- query_pinecone()
      |-- convert_match_to_node()
      |-- expand_summary_nodes()
      |-- expand_to_leaf_chunks()
      |-- rerank_nodes_pinecone() or rerank_nodes_simple()
      `-- build_context_string()

This module is not called by run_filings_pipeline(). The pipeline builds and
verifies the index; this module reads from that index for downstream LLM use.
"""

import os
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

from dotenv import load_dotenv

from .config_loader import load_config_yaml
from src.utils.secrets import get_secret
from src.utils.hybrid import hybrid_rrf_rank

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
_CONFIG = load_config_yaml(BASE_DIR / "config.yaml")


# =========================================================
# CONFIG
# =========================================================

def _clean_config_str(value: Any, default: str) -> str:
    raw = str(value if value is not None else default).strip()
    if "=" in raw:
        raw = raw.split("=", 1)[1].strip()
    return raw.strip('"').strip("'")


PINECONE_API_KEY_ENV = _clean_config_str(_CONFIG.get("PINECONE_API_KEY_ENV"), "PINECONE_API_KEY")
PINECONE_INDEX_NAME = _clean_config_str(_CONFIG.get("PINECONE_INDEX_NAME"), "ragcapstone")
PINECONE_NAMESPACE = _clean_config_str(_CONFIG.get("PINECONE_NAMESPACE"), "user1")
PINECONE_EMBED_MODEL = _clean_config_str(_CONFIG.get("PINECONE_EMBED_MODEL"), "llama-text-embed-v2")
PINECONE_RERANK_MODEL = _clean_config_str(_CONFIG.get("PINECONE_RERANK_MODEL"), "bge-reranker-v2-m3")


# =========================================================
# PINECONE INIT
# =========================================================

_PINECONE_CLIENT = None
_PINECONE_INDEX = None


def get_pinecone_client():
    global _PINECONE_CLIENT
    if _PINECONE_CLIENT is not None:
        return _PINECONE_CLIENT

    from pinecone import Pinecone

    pinecone_api_key = get_secret(PINECONE_API_KEY_ENV)

    _PINECONE_CLIENT = Pinecone(api_key=pinecone_api_key)
    return _PINECONE_CLIENT


def get_pinecone_index():
    global _PINECONE_INDEX
    if _PINECONE_INDEX is not None:
        return _PINECONE_INDEX

    _PINECONE_INDEX = get_pinecone_client().Index(PINECONE_INDEX_NAME)
    return _PINECONE_INDEX


# =========================================================
# EMBEDDING HELPERS
# =========================================================

def embed_query(query: str) -> List[float]:
    """
    Create query embedding using Pinecone inference model from config.yaml.
    """
    response = get_pinecone_client().inference.embed(
        model=PINECONE_EMBED_MODEL,
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"},
    )
    if not response:
        raise RuntimeError("Pinecone inference returned no query embedding.")
    return list(response[0].values)


# =========================================================
# PINECONE QUERY HELPERS
# =========================================================

def query_pinecone(
    query_vector: List[float],
    top_k: int = 10,
    namespace: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    include_metadata: bool = True,
) -> List[Dict[str, Any]]:
    """
    Query Pinecone and return matches as list of dicts.
    """
    namespace = namespace if namespace is not None else PINECONE_NAMESPACE

    response = get_pinecone_index().query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_values=False,
        include_metadata=include_metadata,
        filter=metadata_filter
    )

    matches = response.get("matches", [])
    return matches


def fetch_vectors_by_ids(
    ids: List[str],
    namespace: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch vectors by IDs from Pinecone.
    Returns dict keyed by vector id.
    """
    namespace = namespace if namespace is not None else PINECONE_NAMESPACE

    if not ids:
        return {}

    response = get_pinecone_index().fetch(ids=ids, namespace=namespace)
    return response.get("vectors", {})


# =========================================================
# UTILS
# =========================================================

def safe_get_metadata(match: Dict[str, Any]) -> Dict[str, Any]:
    return match.get("metadata", {}) or {}


def normalize_score(score: float) -> float:
    """
    Optional score normalization.
    If Pinecone similarity is cosine, score may already be usable.
    This keeps it bounded for downstream blending/reranking if needed.
    """
    if score is None:
        return 0.0
    return float(score)


def deduplicate_by_id(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for node in nodes:
        node_id = node.get("id")
        if node_id and node_id not in seen:
            seen.add(node_id)
            deduped.append(node)
    return deduped


def sort_nodes_by_score(nodes: List[Dict[str, Any]], descending: bool = True) -> List[Dict[str, Any]]:
    return sorted(nodes, key=lambda x: x.get("score", 0.0), reverse=descending)


# =========================================================
# MATCH -> INTERNAL NODE FORMAT
# =========================================================

def convert_match_to_node(match: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardize Pinecone match into internal format.
    """
    metadata = safe_get_metadata(match)

    return {
        "id": match.get("id"),
        "score": normalize_score(match.get("score", 0.0)),
        "text": metadata.get("text", ""),
        "node_type": metadata.get("node_type", "leaf"),
        "level": metadata.get("level"),
        "child_ids": metadata.get("child_ids", []),
        "parent_id": metadata.get("parent_id"),
        "source_doc_id": metadata.get("source_doc_id"),
        "form_type": metadata.get("form_type"),
        "year": metadata.get("year"),
        "metadata": metadata
    }


def convert_fetched_vector_to_node(vector_id: str, vector_obj: Dict[str, Any], inherited_score: float = 0.0) -> Dict[str, Any]:
    """
    Standardize Pinecone fetch result into internal node format.
    Fetch results don't include similarity score, so we inherit
    the score from the summary parent or use a fallback.
    """
    metadata = (vector_obj or {}).get("metadata", {}) or {}

    return {
        "id": vector_id,
        "score": inherited_score,
        "text": metadata.get("text", ""),
        "node_type": metadata.get("node_type", "leaf"),
        "level": metadata.get("level"),
        "child_ids": metadata.get("child_ids", []),
        "parent_id": metadata.get("parent_id"),
        "source_doc_id": metadata.get("source_doc_id"),
        "form_type": metadata.get("form_type"),
        "year": metadata.get("year"),
        "metadata": metadata
    }


# =========================================================
# SUMMARY EXPANSION
# =========================================================

def expand_summary_nodes(
    nodes: List[Dict[str, Any]],
    namespace: Optional[str] = None,
    max_children_per_summary: Optional[int] = None,
    propagate_parent_score: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Separate leaf and summary nodes.
    For summary nodes, fetch child leaf nodes using child_ids.

    Returns:
        direct_leaf_nodes,
        expanded_child_nodes
    """
    direct_leaf_nodes = []
    expanded_child_nodes = []

    for node in nodes:
        node_type = node.get("node_type", "leaf")
        if node_type == "leaf":
            direct_leaf_nodes.append(node)
            continue

        if node_type == "summary":
            child_ids = node.get("child_ids", []) or []
            if max_children_per_summary is not None:
                child_ids = child_ids[:max_children_per_summary]

            if not child_ids:
                logger.warning(f"Summary node {node.get('id')} has no child_ids.")
                continue

            fetched = fetch_vectors_by_ids(child_ids, namespace=namespace)

            for child_id, child_vector in fetched.items():
                inherited_score = node["score"] if propagate_parent_score else 0.0
                child_node = convert_fetched_vector_to_node(
                    vector_id=child_id,
                    vector_obj=child_vector,
                    inherited_score=inherited_score
                )

                # preserve linkage to summary hit
                child_node["retrieved_via_summary_id"] = node["id"]
                child_node["retrieved_via_summary_score"] = node["score"]

                expanded_child_nodes.append(child_node)

    return direct_leaf_nodes, expanded_child_nodes


def _expand_summary_to_leaf_nodes(
    summary_node: Dict[str, Any],
    namespace: Optional[str] = None,
    max_children_per_summary: Optional[int] = None,
    propagate_parent_score: bool = True,
    visited_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Recursively expand a summary node to leaf chunk nodes.
    Handles multi-level RAPTOR trees where summary children may also be summaries.
    """
    visited_ids = visited_ids or set()
    summary_id = summary_node.get("id")
    if summary_id:
        visited_ids.add(summary_id)

    child_ids = summary_node.get("child_ids", []) or []
    if max_children_per_summary is not None:
        child_ids = child_ids[:max_children_per_summary]

    if not child_ids:
        logger.warning(f"Summary node {summary_id} has no child_ids.")
        return []

    fetched = fetch_vectors_by_ids(child_ids, namespace=namespace)
    leaf_descendants: List[Dict[str, Any]] = []

    for child_id, child_vector in fetched.items():
        inherited_score = summary_node["score"] if propagate_parent_score else 0.0
        child_node = convert_fetched_vector_to_node(
            vector_id=child_id,
            vector_obj=child_vector,
            inherited_score=inherited_score
        )

        child_node["retrieved_via_summary_id"] = summary_id
        child_node["retrieved_via_summary_score"] = summary_node["score"]

        child_type = child_node.get("node_type", "leaf")
        if child_type == "leaf":
            leaf_descendants.append(child_node)
            continue

        if child_type == "summary":
            if child_id in visited_ids:
                logger.warning("Cycle detected while expanding summary descendants at node_id=%s", child_id)
                continue
            nested_leaf_nodes = _expand_summary_to_leaf_nodes(
                summary_node=child_node,
                namespace=namespace,
                max_children_per_summary=max_children_per_summary,
                propagate_parent_score=propagate_parent_score,
                visited_ids=set(visited_ids),
            )
            for nested in nested_leaf_nodes:
                if not nested.get("retrieved_via_summary_id"):
                    nested["retrieved_via_summary_id"] = summary_id
                    nested["retrieved_via_summary_score"] = summary_node["score"]
            leaf_descendants.extend(nested_leaf_nodes)

    return leaf_descendants


def expand_to_leaf_chunks(
    nodes: List[Dict[str, Any]],
    namespace: Optional[str] = None,
    max_children_per_summary: Optional[int] = None,
    propagate_parent_score: bool = True,
) -> List[Dict[str, Any]]:
    """
    Ensure retrieval candidates are leaf chunk nodes only.
    - Keep direct leaf matches
    - Recursively expand summary matches to their leaf descendants
    """
    leaf_nodes: List[Dict[str, Any]] = []
    expanded_leaf_nodes: List[Dict[str, Any]] = []

    for node in nodes:
        node_type = node.get("node_type", "leaf")
        if node_type == "leaf":
            leaf_nodes.append(node)
            continue

        if node_type == "summary":
            expanded_leaf_nodes.extend(
                _expand_summary_to_leaf_nodes(
                    summary_node=node,
                    namespace=namespace,
                    max_children_per_summary=max_children_per_summary,
                    propagate_parent_score=propagate_parent_score,
                )
            )

    # Keep only leaf nodes that represent chunks.
    all_leafs = leaf_nodes + expanded_leaf_nodes
    chunk_leafs = [n for n in all_leafs if n.get("node_type") == "leaf" and n.get("metadata", {}).get("chunk_id")]
    if chunk_leafs:
        return chunk_leafs
    return [n for n in all_leafs if n.get("node_type") == "leaf"]


# =========================================================
# OPTIONAL SCORE ADJUSTMENT
# =========================================================

def blend_scores_for_expanded_children(
    child_nodes: List[Dict[str, Any]],
    boost: float = 0.95
) -> List[Dict[str, Any]]:
    """
    Children retrieved through a summary do not have their own query similarity score.
    So we propagate summary score and slightly discount it.
    """
    adjusted = []
    for node in child_nodes:
        score = node.get("score", 0.0)
        if node.get("retrieved_via_summary_id"):
            node["score"] = score * boost
        adjusted.append(node)
    return adjusted


# =========================================================
# OPTIONAL SECOND-PASS RERANKING
# =========================================================

def lexical_overlap_score(query: str, text: str) -> float:
    """
    Lightweight reranking fallback if you don't have a reranker yet.
    """
    if not query or not text:
        return 0.0

    q_tokens = set(query.lower().split())
    t_tokens = set(text.lower().split())

    if not q_tokens or not t_tokens:
        return 0.0

    overlap = len(q_tokens.intersection(t_tokens))
    return overlap / max(len(q_tokens), 1)


def rerank_nodes_simple(query: str, nodes: List[Dict[str, Any]], alpha: float = 0.8) -> List[Dict[str, Any]]:
    """
    Simple reranker combining Pinecone score + lexical overlap.
    Replace this later with bge-reranker / cross-encoder if needed.
    """
    reranked = []

    for node in nodes:
        semantic_score = node.get("score", 0.0)
        lexical_score = lexical_overlap_score(query, node.get("text", ""))
        final_score = alpha * semantic_score + (1 - alpha) * lexical_score

        node_copy = dict(node)
        node_copy["semantic_score"] = semantic_score
        node_copy["lexical_score"] = lexical_score
        node_copy["final_score"] = final_score
        reranked.append(node_copy)

    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked


def rerank_nodes_pinecone(query: str, nodes: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    """
    Rerank using Pinecone inference reranker model (e.g. bge-reranker-v2-m3).
    Falls back to empty list if rerank API returns nothing.
    """
    if not nodes:
        return []

    documents = [str(node.get("text", "")) for node in nodes]
    if not any(documents):
        return []

    rr = get_pinecone_client().inference.rerank(
        model=PINECONE_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
        return_documents=False,
    )

    ranked_nodes: List[Dict[str, Any]] = []
    rr_items = rr.data if hasattr(rr, "data") else rr
    for item in rr_items:
        idx = item.index if hasattr(item, "index") else item.get("index")
        score = item.score if hasattr(item, "score") else item.get("score", 0.0)
        if idx is None or idx >= len(nodes):
            continue
        node = dict(nodes[idx])
        node["rerank_score"] = float(score)
        node["final_score"] = float(score)
        ranked_nodes.append(node)
    return ranked_nodes


# =========================================================
# MAIN RAPTOR RETRIEVAL
# =========================================================

def raptor_retrieve(
    query: str,
    top_k: int = 10,
    final_top_k: int = 8,
    namespace: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    max_children_per_summary: Optional[int] = None,
    use_simple_rerank: bool = False,
    use_reranker: bool = True,
) -> Dict[str, Any]:
    """
    Full RAPTOR-aware retrieval:
      1. Embed query
      2. Query Pinecone
      3. Convert matches to nodes
      4. Expand summary nodes using child_ids
      5. Merge leaf nodes + expanded children
      6. Deduplicate
      7. Rerank
      8. Return final contexts
    """
    logger.info(f"RAPTOR retrieval started for query: {query}")

    query_vector = embed_query(query)

    raw_matches = query_pinecone(
        query_vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        metadata_filter=metadata_filter,
        include_metadata=True
    )

    logger.info(f"Initial Pinecone matches: {len(raw_matches)}")

    initial_nodes = [convert_match_to_node(m) for m in raw_matches]

    combined_nodes = expand_to_leaf_chunks(
        nodes=initial_nodes,
        namespace=namespace,
        max_children_per_summary=max_children_per_summary,
        propagate_parent_score=True,
    )

    combined_nodes = blend_scores_for_expanded_children(combined_nodes, boost=0.95)
    combined_nodes = deduplicate_by_id(combined_nodes)

    leaf_chunk_count = sum(1 for n in combined_nodes if n.get("node_type") == "leaf")
    logger.info(f"Combined leaf chunk nodes after summary expansion + dedup: {leaf_chunk_count}")

    dense_ranked_nodes = sort_nodes_by_score(combined_nodes)
    if use_reranker:
        ranked_nodes = hybrid_rrf_rank(
            query,
            dense_ranked_nodes,
            text_getter=lambda node: str(node.get("text", "")),
            key=lambda node: str(node.get("id") or ""),
            top_k=final_top_k,
        )
    else:
        ranked_nodes = dense_ranked_nodes[:final_top_k]

    contexts = []
    for rank, node in enumerate(ranked_nodes, start=1):
        context_metadata = dict(node.get("metadata", {}) or {})
        if context_metadata.get("form_type") in (None, ""):
            context_metadata["form_type"] = node.get("form_type")
        if context_metadata.get("year") in (None, ""):
            context_metadata["year"] = node.get("year")

        contexts.append({
            "rank": rank,
            "id": node.get("id"),
            "text": node.get("text", ""),
            "node_type": node.get("node_type"),
            "form_type": node.get("form_type"),
            "year": node.get("year"),
            "score": node.get("score"),
            "final_score": node.get("final_score", node.get("score")),
            "source_doc_id": node.get("source_doc_id"),
            "parent_id": node.get("parent_id"),
            "retrieved_via_summary_id": node.get("retrieved_via_summary_id"),
            "metadata": context_metadata
        })

    return {
        "query": query,
        "initial_match_count": len(initial_nodes),
        "direct_leaf_count": sum(1 for n in initial_nodes if n.get("node_type") == "leaf"),
        "expanded_child_count": max(0, len(combined_nodes) - sum(1 for n in initial_nodes if n.get("node_type") == "leaf")),
        "final_context_count": len(contexts),
        "contexts": contexts
    }


# =========================================================
# CONTEXT FORMATTER FOR LLM
# =========================================================

def build_context_string(contexts: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    """
    Convert retrieved contexts into a single string for the LLM prompt.
    """
    parts = []
    total_chars = 0

    for ctx in contexts:
        header = (
            f"[Rank: {ctx['rank']}] "
            f"[Node ID: {ctx['id']}] "
            f"[Type: {ctx['node_type']}] "
            f"[Form Type: {ctx.get('form_type')}] "
            f"[Source Doc: {ctx.get('source_doc_id')}]"
        )

        text = ctx.get("text", "").strip()
        block = f"{header}\n{text}\n"

        if total_chars + len(block) > max_chars:
            break

        parts.append(block)
        total_chars += len(block)

    return "\n---\n".join(parts)


def retrieve_context_for_llm(
    query: str,
    top_k: int = 10,
    final_top_k: int = 8,
    namespace: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    max_children_per_summary: Optional[int] = None,
    use_simple_rerank: bool = False,
    max_chars: int = 12000,
) -> str:
    """
    Convenience wrapper for direct LLM usage.
    Runs RAPTOR retrieval and returns only the formatted context string.
    """
    results = raptor_retrieve(
        query=query,
        top_k=top_k,
        final_top_k=final_top_k,
        namespace=namespace,
        metadata_filter=metadata_filter,
        max_children_per_summary=max_children_per_summary,
        use_simple_rerank=use_simple_rerank,
    )
    return build_context_string(results.get("contexts", []), max_chars=max_chars)


# =========================================================
# OPTIONAL: HELPERS FOR DEBUGGING
# =========================================================

def print_retrieval_results(results: Dict[str, Any]) -> None:
    print("\n========== RAPTOR RETRIEVAL RESULTS ==========")
    print(f"Query: {results['query']}")
    print(f"Initial matches: {results['initial_match_count']}")
    print(f"Direct leaf nodes: {results['direct_leaf_count']}")
    print(f"Expanded child nodes: {results['expanded_child_count']}")
    print(f"Final contexts: {results['final_context_count']}")
    print("=============================================\n")

    for ctx in results["contexts"]:
        print(f"Rank: {ctx['rank']}")
        print(f"ID: {ctx['id']}")
        print(f"Type: {ctx['node_type']}")
        print(f"Form Type: {ctx.get('form_type')}")
        print(f"Score: {ctx['score']}")
        print(f"Final Score: {ctx.get('final_score')}")
        print(f"Source Doc ID: {ctx.get('source_doc_id')}")
        print(f"Retrieved via summary: {ctx.get('retrieved_via_summary_id')}")
        print(f"Text Preview: {ctx['text'][:300]}")
        print("-" * 60)


# =========================================================
# EXAMPLE USAGE
# =========================================================

if __name__ == "__main__":
    query = "What was disclosed under Item 2.02 in Apple's January 2022 8-K filing?"

    context_str = retrieve_context_for_llm(
        query=query,
        top_k=10,
        final_top_k=6,
        namespace=PINECONE_NAMESPACE,
        metadata_filter=None,
        max_children_per_summary=10,
        use_simple_rerank=True,
        max_chars=12000,
    )
    print("\n========== CONTEXT STRING FOR LLM ==========\n")
    print(context_str[:3000])
