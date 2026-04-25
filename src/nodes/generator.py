from __future__ import annotations

from typing import List

from src.model_config import get_direct_generation_llm, get_generation_llm
from src.state import GraphState


def _format_doc_context(docs: List[dict]) -> str:
    blocks = []

    for i, doc in enumerate(docs, start=1):
        metadata = doc.get("metadata", {})
        title = metadata.get("title") or metadata.get("file_name") or metadata.get("source", "unknown")
        page = metadata.get("page_label") or metadata.get("page")
        source = metadata.get("source_url") or metadata.get("source", "unknown")

        blocks.append(
            f"""[Document {i}]
Title: {title}
Page: {page}
Source: {source}

Content:
{doc.get("content", "").strip()}
"""
        )

    return "\n\n".join(blocks)


def _format_web_context(results: List[dict]) -> str:
    blocks = []

    for i, item in enumerate(results, start=1):
        blocks.append(
            f"""[Web Result {i}]
Title: {item.get("title", "unknown")}
URL: {item.get("url", "unknown")}

Content:
{item.get("content", "").strip()}
"""
        )

    return "\n\n".join(blocks)


def _doc_citations(docs: List[dict]) -> List[str]:
    citations = []
    seen = set()

    for doc in docs:
        metadata = doc.get("metadata", {})
        title = metadata.get("title") or metadata.get("file_name") or metadata.get("source", "unknown")
        page = metadata.get("page_label") or metadata.get("page")

        citation = f"{title} (page {page})" if page is not None else str(title)
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)

    return citations


def _web_citations(results: List[dict]) -> List[str]:
    citations = []
    seen = set()

    for item in results:
        title = item.get("title", "unknown")
        url = item.get("url", "unknown")
        citation = f"{title} - {url}"

        if citation not in seen:
            seen.add(citation)
            citations.append(citation)

    return citations


def _answer_style_instructions(question: str, *, direct: bool = False) -> str:
    lowered = question.lower().strip()
    asks_yes_no = lowered.startswith(
        ("is ", "are ", "has ", "have ", "can ", "does ", "do ", "did ", "was ", "were ")
    )
    asks_for_multiple_items = any(
        phrase in lowered
        for phrase in (
            "what are",
            "list",
            "what patents",
            "what lawsuits",
            "what risk factors",
            "what metrics",
            "what did management",
            "enumerate",
            "summarize all",
        )
    )

    instructions = [
        "- Start with the direct answer immediately.",
        "- Do not use headings or labels such as 'Short answer', 'Status', 'Details', or 'Limitations'.",
        "- Do not repeat the question.",
        "- Do not narrate the retrieval process or say 'based on the provided context' unless context is insufficient.",
    ]

    if asks_yes_no:
        instructions.append("- If supported by the evidence, begin with 'Yes.' or 'No.'.")

    if direct:
        instructions.extend(
            [
                "- Keep the answer to 2 or 3 sentences and under 90 words.",
                "- Prefer plain prose over bullets.",
            ]
        )
    else:
        instructions.extend(
            [
                "- Keep the answer under 140 words.",
                "- Default to one short paragraph.",
            ]
        )
        if asks_for_multiple_items:
            instructions.append(
                "- If the question asks for several items, use one short opening sentence followed by at most 4 flat bullets."
            )
        else:
            instructions.append("- Use bullets only when they materially improve clarity.")

    instructions.append(
        "- If the context is insufficient, say that in one short sentence instead of adding a separate caveat section."
    )

    return "\n".join(instructions)


def generate_answer(state: GraphState) -> GraphState:
    """
    Generate a grounded answer from:
    1. filtered_docs
    2. retrieved_docs
    3. web_results
    """
    question = state["question"]
    llm = get_generation_llm()

    docs = state.get("filtered_docs") or state.get("retrieved_docs") or []
    web_results = state.get("web_results") or []
    web_search_error = state.get("web_search_error")

    if docs:
        context = _format_doc_context(docs)
        citations = _doc_citations(docs)
        source_mode = "documents"
    elif web_results:
        context = _format_web_context(web_results)
        citations = _web_citations(web_results)
        source_mode = "web"
    elif web_search_error:
        return {
            **state,
            "answer": f"{web_search_error} I could not verify the latest information right now.",
            "citations": [],
        }
    else:
        return {
            **state,
            "answer": "I do not have enough supporting context to answer this question.",
            "citations": [],
        }

    prompt = f"""
You are a forensic financial analyst assistant for M&A due diligence.

Answer the user's question using ONLY the provided context from SEC filings, earnings transcripts, patents, or litigation records.
Do not invent facts or speculate beyond the evidence.
Cite every factual claim with [source | date] format.
Flag any material risks or anomalies explicitly.
If the context is insufficient, say so clearly in one sentence.

Output rules:
{_answer_style_instructions(question)}

Question:
{question}

Context source type: {source_mode}

Context:
{context}
""".strip()

    response = llm.invoke(prompt)

    return {
        **state,
        "answer": response.content.strip(),
        "citations": citations,
    }


def generate_direct_answer(state: GraphState) -> GraphState:
    """
    Direct LLM answer for conceptual/general questions that do not need retrieval.
    """
    question = state["question"]
    llm = get_direct_generation_llm()

    prompt = f"""
You are a knowledgeable M&A and finance expert.

Answer the question directly using your knowledge of M&A processes, financial analysis, and due diligence best practices.
Be accurate, concise, and clear.
Do not claim to use documents or external search.

Output rules:
{_answer_style_instructions(question, direct=True)}

Question:
{question}
""".strip()

    response = llm.invoke(prompt)

    return {
        **state,
        "answer": response.content.strip(),
        "citations": [],
    }