from src.patents.retrieval import generate_patent_answer

queries = [
    "Apple patents related to neural engine and on-device machine learning",
    "Apple patents on health monitoring and biometric sensors",
    "Apple patents related to battery optimization and power management",
    "Apple patents on augmented reality display technology",
    "Apple patents on secure enclave and biometric authentication",
]

for i, query in enumerate(queries, start=1):
    print(f"\n{'='*60}")
    print(f"QUERY {i}: {query}")
    print("=" * 60)
    result = generate_patent_answer(query=query, company="Apple Inc")
    print(f"Hits: {result['hit_count']}")
    print(result['answer'])
