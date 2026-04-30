from __future__ import annotations


def test_build_answer_prompt_adds_graph_specific_readability_rules() -> None:
    from src.nodes.generator import _build_answer_prompt

    docs = [
        {
            "content": "Apple board member in 2024: Al Gore | Title: Former Vice President of the United States",
            "metadata": {"source": "Knowledge Graph", "title": "Neo4j Graph Result"},
        }
    ]

    prompt = _build_answer_prompt(
        "Who are the board members of Apple in 2024?",
        docs=docs,
        web_results=[],
    )

    assert "knowledge graph" in prompt.lower()
    assert "treat each graph row as a factual assertion" in prompt.lower()
    assert "do not say 'the context lists'" in prompt.lower()
    assert "do not include inline citation tags" in prompt.lower()
    assert "write a natural, user-friendly answer" in prompt.lower()
    assert "do not just echo raw retrieved lines" in prompt.lower()


def test_build_answer_prompt_keeps_general_document_mode_generic() -> None:
    from src.nodes.generator import _build_answer_prompt

    docs = [
        {
            "content": "Risk factors include supply-chain disruption.",
            "metadata": {"source": "SEC Filing", "title": "10-K"},
        }
    ]

    prompt = _build_answer_prompt(
        "What are Apple's risk factors?",
        docs=docs,
        web_results=[],
    )

    assert "knowledge graph" not in prompt.lower()
    assert "treat each graph row as a factual assertion" not in prompt.lower()
