from typing import List, Literal, TypedDict


class GraphState(TypedDict, total=False):
    chunking_strategy: str
    question: str
    rewritten_question: str
    retrieval_query: str
    web_search_error: str

    route: Literal["llm_direct", "vectorstore", "web_search"]
    route_reason: str
    route_hint: str
    initial_route: str
    force_route: bool

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