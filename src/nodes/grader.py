from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.model_config import get_grader_llm
from src.state import GraphState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BinaryGrade(BaseModel):
    score: Literal["yes", "no"] = Field(description="Binary grading decision.")


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return exc.__class__.__name__ == "ResourceExhausted" or "429" in text or "resource exhausted" in text

def _format_context_for_grading(state: GraphState) -> str:
    docs = state.get("filtered_docs") or state.get("retrieved_docs") or []
    web_results = state.get("web_results") or []

    if docs:
        blocks = []
        for i, doc in enumerate(docs, start=1):
            blocks.append(f"[Doc {i}]\n{doc.get('content', '')}")
        return "\n\n".join(blocks)

    if web_results:
        blocks = []
        for i, item in enumerate(web_results, start=1):
            blocks.append(
                f"[Web {i}] Title: {item.get('title', '')}\n"
                f"URL: {item.get('url', '')}\n"
                f"{item.get('content', '')}"
            )
        return "\n\n".join(blocks)

    return ""


def _grade_graph_documents(state: GraphState) -> GraphState:
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {
            **state,
            "filtered_docs": [],
            "doc_relevance": [],
            "relevant_doc_count": 0,
        }

    doc_relevance = ["yes"] * len(docs)
    return {
        **state,
        "filtered_docs": docs,
        "doc_relevance": doc_relevance,
        "relevant_doc_count": len(docs),
    }


def grade_documents(state: GraphState) -> GraphState:
    """
    Corrective RAG:
    Grade each retrieved chunk for relevance to the question.
    Keep only relevant chunks.
    """
    eval_config = state.get("eval_config") or {}
    if eval_config.get("corrective") is False:
        # Ablation: skip corrective grading, pass all docs through
        docs = state.get("retrieved_docs", [])
        return {
            **state,
            "filtered_docs": docs,
            "doc_relevance": ["yes"] * len(docs),
            "relevant_doc_count": len(docs),
        }

    question = state.get("rewritten_question") or state["question"]
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        return {
            **state,
            "filtered_docs": [],
            "doc_relevance": [],
            "relevant_doc_count": 0,
        }

    if state.get("route") == "graph":
        return _grade_graph_documents(state)

    llm = get_grader_llm().with_structured_output(BinaryGrade)

    filtered_docs = []
    doc_relevance = []

    for doc in retrieved_docs:
        chunk = doc.get("content", "")

        prompt = f"""
You are grading whether a retrieved document chunk is relevant to a user's question.

Question:
{question}

Document chunk:
{chunk}

Rules:
- Reply "yes" if the chunk contains useful evidence for answering the question.
- Reply "no" if it is irrelevant, too broad, or too weak.

Return a binary decision only.
""".strip()

        try:
            result = llm.invoke(prompt)
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise
            logger.warning("Document grading skipped due to rate limit: %s", exc)
            return {
                **state,
                "filtered_docs": retrieved_docs,
                "doc_relevance": ["yes"] * len(retrieved_docs),
                "relevant_doc_count": len(retrieved_docs),
            }
        doc_relevance.append(result.score)

        if result.score == "yes":
            filtered_docs.append(doc)

    return {
        **state,
        "filtered_docs": filtered_docs,
        "doc_relevance": doc_relevance,
        "relevant_doc_count": len(filtered_docs),
    }


def grade_hallucination(state: GraphState) -> GraphState:
    """
    Self-RAG:
    Check whether the generated answer is supported by available context.
    """
    eval_config = state.get("eval_config") or {}
    if eval_config.get("self_rag") is False:
        return {**state, "hallucination_grade": "yes"}

    answer = state.get("answer", "")
    context = _format_context_for_grading(state)

    if not context or not answer:
        return {
            **state,
            "hallucination_grade": "no",
        }

    llm = get_grader_llm().with_structured_output(BinaryGrade)

    prompt = f"""
You are checking whether an answer is grounded in the provided context.

Context:
{context}

Answer:
{answer}

Is the answer fully supported by the context?
Return only:
yes
or
no
""".strip()

    try:
        result = llm.invoke(prompt)
    except Exception as exc:
        if not _is_rate_limit_error(exc):
            raise
        logger.warning("Hallucination grading skipped due to rate limit: %s", exc)
        return {
            **state,
            "hallucination_grade": "yes",
        }

    return {
        **state,
        "hallucination_grade": result.score,
    }


def grade_answer_quality(state: GraphState) -> GraphState:
    """
    Self-RAG:
    Check whether the answer actually addresses the user's question.
    """
    eval_config = state.get("eval_config") or {}
    if eval_config.get("self_rag") is False:
        return {**state, "answer_quality_grade": "yes"}

    question = state["question"]
    answer = state.get("answer", "")

    if not answer:
        return {
            **state,
            "answer_quality_grade": "no",
        }

    llm = get_grader_llm().with_structured_output(BinaryGrade)

    prompt = f"""
You are checking whether an answer is useful and responsive.

Question:
{question}

Answer:
{answer}

Does the answer adequately address the user's question?
Return only:
yes
or
no
""".strip()

    try:
        result = llm.invoke(prompt)
    except Exception as exc:
        if not _is_rate_limit_error(exc):
            raise
        logger.warning("Answer quality grading skipped due to rate limit: %s", exc)
        return {
            **state,
            "answer_quality_grade": "yes",
        }

    return {
        **state,
        "answer_quality_grade": result.score,
    }
