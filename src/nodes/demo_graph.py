from src.graph import build_graph
from src.tiering import tier_for_single_route, tier_label

graph = build_graph()

questions = [
    {
        "question": "What was Apple total revenue for fiscal year 2024?",
        "company": "Apple Inc.",
        "period": "FY 2024",
    },
    {
        "question": "What patents does Apple have related to on-device AI?",
        "company": "Apple Inc",
        "period": "2024",
    },
]

for item in questions:
    print("=" * 100)
    print("QUESTION:", item["question"])

    result = graph.invoke(
        {
            "question": item["question"],
            "company": item["company"],
            "period": item["period"],
            "max_iterations": 3,
            "max_retrieval_attempts": 1,
            "chunking_strategy": "hierarchical",
        }
    )

    single_tier = tier_for_single_route(result.get("route"))
    print("TIER:", tier_label(single_tier))
    print("ROUTE:", result.get("route"))
    print("ROUTE REASON:", result.get("route_reason"))
    print("RETRIEVAL QUERY:", result.get("retrieval_query"))
    print("REWRITTEN QUESTION:", result.get("rewritten_question"))
    print("RELEVANT DOC COUNT:", result.get("relevant_doc_count"))
    print("RETRIEVAL ATTEMPT:", result.get("retrieval_attempt"))
    print("ANSWER:", result.get("answer"))
    print("CITATIONS:", result.get("citations"))
    print("HALLUCINATION GRADE:", result.get("hallucination_grade"))
    print("ANSWER QUALITY GRADE:", result.get("answer_quality_grade"))
    print()
