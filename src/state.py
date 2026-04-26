from typing import Any, Dict, List, Literal, TypedDict


class GraphState(TypedDict, total=False):
    chunking_strategy: str
    question: str
    rewritten_question: str
    retrieval_query: str
    web_search_error: str

    route: Literal[
        "sql",
        "filings",
        "transcripts",
        "patents",
        "litigation",
        "graph",
        "contradiction",
        "llm_direct",
    ]
    route_reason: str
    route_hint: str
    initial_route: str
    force_route: bool

    # M&A context
    company: str | None
    period: str | None
    source_type: str | None
    data_source_result: dict
    contradiction_report: list

    retrieved_docs: List[dict]
    filtered_docs: List[dict]
    web_results: List[dict]

    answer: str
    citations: List[str]

    doc_relevance: List[str]
    relevant_doc_count: int

    hallucination_grade: str
    answer_quality_grade: str

    iteration: int
    max_iterations: int

    retrieval_attempt: int
    max_retrieval_attempts: int

    # Evaluation ablation config — None in production, dict during eval runs
    eval_config: Dict[str, Any]
