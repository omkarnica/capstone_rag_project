from src.api import run_adaptive_query

questions = [
    "What is a professional certification?",
    "What does the Claude certification exam guide cover?",
    "Has there been any Professional Claude certification launched recently? What is the syllabus for this exam?",
]

for q in questions:
    print("=" * 100)
    print("QUESTION:", q)

    result = run_adaptive_query(q)

    print("TIER:", result["tier_label"])
    print("PLAN TYPE:", result["plan_type"])
    print("PLAN REASON:", result["plan_reason"])
    print("SUBQUESTIONS:", result["subquestions"])
    print("FINAL ANSWER:", result["final_answer"])
    print("CITATIONS:", result["citations"])

    for i, sub in enumerate(result["sub_results"], start=1):
        print(f"\n--- SUB RESULT {i} ---")
        print("QUESTION:", sub["question"])
        print("TIER:", sub.get("tier_label"))
        print("ROUTE:", sub.get("final_route") or sub.get("route"))
        print("ANSWER:", sub["answer"])
        print("CITATIONS:", sub["citations"])