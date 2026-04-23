from src.graph import build_graph
from src.tiering import tier_for_single_route, tier_label

graph = build_graph()

questions = [
    "What is a professional certification?",
    "What does the Claude certification exam guide cover?",
    "Has there been any Professional Claude certification launched recently ? What is the syllabus for this exam?",
]

for q in questions:
    print("=" * 100)
    print("QUESTION:", q)

    result = graph.invoke(
        {
            "question": q,
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