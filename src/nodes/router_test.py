from src.nodes.router import route_question

questions = [
    "What is a professional certification?",
    "What does the Claude certification exam guide cover?",
    "Have any new Claude certifications launched recently?",
]

for q in questions:
    state = {"question": q}
    result = route_question(state)

    print("=" * 80)
    print("QUESTION:", q)
    print("ROUTE:", result["route"])
    print("=" * 80)