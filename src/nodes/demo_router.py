from src.nodes.router import route_question

questions = [
    "What was Apple revenue in fiscal year 2024?",
    "What did Microsoft management say about cloud growth?",
    "What patents does Apple have related to on-device AI?",
    "What litigation exposure does Microsoft have?",
    "Compare Apple management commentary against SEC filings.",
]

for q in questions:
    state = {"question": q}
    result = route_question(state)

    print("=" * 80)
    print("QUESTION:", q)
    print("ROUTE:", result["route"])
    print("REASON:", result["route_reason"])
    print("=" * 80)
