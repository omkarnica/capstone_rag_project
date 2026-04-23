from __future__ import annotations

from langchain_core.documents import Document

from src.ingestion.indexer import load_vectorstore, load_parent_store
from src.state import GraphState


def _doc_to_dict(doc: Document) -> dict:
    return {
        "content": doc.page_content,
        "metadata": doc.metadata,
    }


def _hydrate_parents_from_children(child_docs: list[Document]) -> list[dict]:
    """
    Convert retrieved child chunks into unique parent documents using parent_id.
    """
    parent_store = load_parent_store()
    hydrated_parents = []
    seen_parent_ids = set()

    for child_doc in child_docs:
        metadata = child_doc.metadata or {}
        parent_id = metadata.get("parent_id")

        if not parent_id:
            # Fallback: return the child itself if no parent_id exists
            hydrated_parents.append(_doc_to_dict(child_doc))
            continue

        if parent_id in seen_parent_ids:
            continue

        seen_parent_ids.add(parent_id)

        parent_payload = parent_store.get(parent_id)
        if not parent_payload:
            # Fallback: child chunk if parent store entry missing
            hydrated_parents.append(_doc_to_dict(child_doc))
            continue

        hydrated_parents.append(
            {
                "content": parent_payload["page_content"],
                "metadata": parent_payload["metadata"],
            }
        )

    return hydrated_parents


def retrieve_docs(state: GraphState) -> GraphState:
    """
    Retrieve documents from the vectorstore.

    recursive mode:
        returns retrieved chunks directly

    hierarchical mode:
        retrieves child chunks, then hydrates parent documents
    """
    query = state.get("rewritten_question") or state["question"]
    strategy = state.get("chunking_strategy", "hierarchical")

    vectorstore = load_vectorstore(strategy=strategy)
    docs = vectorstore.similarity_search(query, k=4)

    if strategy == "hierarchical":
        retrieved_docs = _hydrate_parents_from_children(docs)
    else:
        retrieved_docs = [_doc_to_dict(doc) for doc in docs]

    return {
        **state,
        "retrieval_query": query,
        "retrieved_docs": retrieved_docs,
        "filtered_docs": [],
        "doc_relevance": [],
        "relevant_doc_count": 0,
    }