from __future__ import annotations

from src.model_config import get_merge_llm


def _dedupe_citations(sub_results: list[dict]) -> list[str]:
    seen = set()
    citations = []

    for result in sub_results:
        for citation in result.get("citations", []):
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)

    return citations


def _merge_style_instructions(question: str) -> str:
    lowered = question.lower().strip()
    asks_yes_no = lowered.startswith(
        ("is ", "are ", "has ", "have ", "can ", "does ", "do ", "did ", "was ", "were ")
    )
    asks_for_multiple_items = any(
        phrase in lowered
        for phrase in (
            "what does",
            "what do",
            "what topics",
            "what courses",
            "what is the syllabus",
            "cover",
            "covers",
            "list",
        )
    )

    instructions = [
        "- Synthesize the sub-answers into one compact response, not a stitched transcript.",
        "- Do not use headings or labels such as 'Short answer', 'Status', 'Details', or 'Limitations'.",
        "- Do not answer each sub-question separately.",
        "- Avoid repeating overlapping facts.",
        "- Keep the final answer under 160 words.",
        "- Default to one short paragraph.",
    ]

    if asks_yes_no:
        instructions.append("- If supported by the evidence, begin with 'Yes.' or 'No.'.")

    if asks_for_multiple_items:
        instructions.append(
            "- If needed for clarity, use one short opening sentence followed by at most 4 flat bullets."
        )
    else:
        instructions.append("- Prefer prose over bullets unless bullets clearly improve readability.")

    instructions.append("- If any part is uncertain, keep that uncertainty to one short clause or sentence.")

    return "\n".join(instructions)


def merge_answers(original_question: str, sub_results: list[dict]) -> dict:
    """
    Merge multiple sub-question answers into one final answer.
    Keeps citations from all sub-results.
    """
    llm = get_merge_llm()

    parts = []
    for i, result in enumerate(sub_results, start=1):
        sub_question = result.get("question", "")
        answer = result.get("answer", "")
        route = result.get("final_route", "")
        citations = result.get("citations", [])

        parts.append(
            f"""[Sub-answer {i}]
Sub-question: {sub_question}
Route used: {route}
Answer:
{answer}

Citations:
{citations}
"""
        )

    prompt = f"""
You are merging answers from multiple routed sub-questions into one final answer.

Original user question:
{original_question}

Sub-answers:
{chr(10).join(parts)}

Instructions:
- Produce one coherent final answer
- Preserve factual distinctions
- Do not invent any new facts
- If one sub-answer is uncertain, keep that uncertainty
- Avoid repetition
- Do not output citations inline; they will be attached separately
{_merge_style_instructions(original_question)}
""".strip()

    response = llm.invoke(prompt)

    return {
        "answer": response.content.strip(),
        "citations": _dedupe_citations(sub_results),
        "sub_results": sub_results,
    }