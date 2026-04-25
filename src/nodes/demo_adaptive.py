from src.api import run_adaptive_query

questions = [
    {
        "question": "What was Apple total revenue for fiscal year 2024?",
        "company": "Apple Inc.",
        "period": "FY 2024",
    },
    {
        "question": "What did Microsoft management say about cloud growth in 2024?",
        "company": "MICROSOFT CORP",
        "period": "FY 2024",
    },
    {
        "question": "Compare management commentary against SEC filings for Apple revenue.",
        "company": "Apple Inc.",
        "period": "FY 2024",
    },
]

for item in questions:
    print("=" * 100)
    print("QUESTION:", item["question"])

    result = run_adaptive_query(
        item["question"],
        company=item["company"],
        period=item["period"],
    )

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
