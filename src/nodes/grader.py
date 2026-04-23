from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.model_config import get_grader_llm
from src.state import GraphState


class BinaryGrade(BaseModel):
    score: Literal["yes", "no"] = Field(description="Binary grading decision.")

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


def grade_documents(state: GraphState) -> GraphState:
    """
    Corrective RAG:
    Grade each retrieved chunk for relevance to the question.
    Keep only relevant chunks.
    """
    question = state.get("rewritten_question") or state["question"]
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        return {
            **state,
            "filtered_docs": [],
            "doc_relevance": [],
            "relevant_doc_count": 0,
        }

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

        result = llm.invoke(prompt)
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

    result = llm.invoke(prompt)

    return {
        **state,
        "hallucination_grade": result.score,
    }


def grade_answer_quality(state: GraphState) -> GraphState:
    """
    Self-RAG:
    Check whether the answer actually addresses the user's question.
    """
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

    result = llm.invoke(prompt)

    return {
        **state,
        "answer_quality_grade": result.score,
    }