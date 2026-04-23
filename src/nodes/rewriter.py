from __future__ import annotations

from src.model_config import get_rewriter_llm
from src.state import GraphState


def _failed_context_summary(state: GraphState) -> str:
    docs = state.get("retrieved_docs", [])
    if not docs:
        return "No documents were retrieved."

    summaries = []
    for i, doc in enumerate(docs, start=1):
        metadata = doc.get("metadata", {})
        title = metadata.get("title") or metadata.get("file_name") or metadata.get("source", "unknown")
        preview = doc.get("content", "")[:250].replace("\n", " ")
        summaries.append(f"[{i}] Title: {title}\nPreview: {preview}")

    return "\n\n".join(summaries)


def rewrite_query(state: GraphState) -> GraphState:
    """
    Corrective RAG:
    Rewrite the question to improve retrieval after weak document grading.
    """
    question = state["question"]
    previous_query = state.get("retrieval_query", question)
    failed_context = _failed_context_summary(state)

    llm = get_rewriter_llm()

    prompt = f"""
You are rewriting a user query to improve retrieval from a local document corpus.

Original user question:
{question}

Previous retrieval query:
{previous_query}

Previously retrieved but weak/misaligned chunks:
{failed_context}

Rewrite the query so it is:
- more specific
- better aligned to likely document vocabulary
- still faithful to the user's original meaning

Do not answer the question.
Return only the rewritten retrieval query.
""".strip()

    response = llm.invoke(prompt)

    return {
        **state,
        "rewritten_question": response.content.strip(),
    }