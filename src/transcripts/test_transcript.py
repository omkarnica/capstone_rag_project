from src.transcripts.retrieval import generate_transcript_answer

queries = [
    {
        "query": "What did Apple report about iPhone revenue in fiscal Q4 2024?",
        "company": "Apple Inc.",
        "period_start": "2024-07-01",
        "period_end": "2024-12-31",
    },
    {
        "query": "How did Microsoft describe its cloud business performance in 2023?",
        "company": "Microsoft Corporation",
        "period_start": "2023-01-01",
        "period_end": "2023-12-31",
    },
    {
        "query": "What were Apple's Services revenue figures across 2022 and 2023?",
        "company": "Apple Inc.",
        "period_start": "2022-01-01",
        "period_end": "2023-12-31",
    },
    {
        "query": "What geographic segments did Apple report weakness in during 2024?",
        "company": "Apple Inc.",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
    },
    {
        "query": "How did Microsoft characterize its AI-related revenue growth?",
        "company": "Microsoft Corporation",
        "period_start": "2023-01-01",
        "period_end": "2024-12-31",
    },
]

for i, q in enumerate(queries, start=1):
    print(f"\n{'='*60}")
    print(f"QUERY {i}: {q['query']}")
    print(f"Company: {q['company']} | {q['period_start']} to {q['period_end']}")
    print("=" * 60)
    result = generate_transcript_answer(**q)
    print(f"Hits: {result['hit_count']}")
    print(result['answer'])
