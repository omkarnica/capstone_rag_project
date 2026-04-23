from src.nodes.retriever import retrieve_docs

question = "What does the Claude certification page say about the duration and format of the exam?"

state = {
    "question": question,
    "chunking_strategy": "hierarchical",
}

result = retrieve_docs(state)

print("=" * 80)
print("QUESTION:")
print(question)
print("=" * 80)

print(f"\nRetrieved: {len(result['retrieved_docs'])} docs\n")

for i, doc in enumerate(result["retrieved_docs"], start=1):
    print(f"\n--- DOC {i} ---")
    print("Metadata:", doc["metadata"])
    print("Content preview:", doc["content"][:500])
    print("-" * 80)