# Graph-Specific Grading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a graph-only grading path that preserves relevant Neo4j entity rows for list-style graph answers without changing grading behavior for XBRL, Pinecone, filings, transcripts, patents, litigation, or any other non-graph source.

**Architecture:** Keep `grade_documents(...)` as the single corrective-RAG entrypoint, but branch on `route == "graph"` into deterministic graph-aware relevance helpers. Non-graph routes continue to use the existing LLM chunk-grading prompt unchanged.

**Tech Stack:** Python, Pydantic, pytest, unittest.mock

---

## File Structure

- Modify: `capstone_rag_project/src/nodes/grader.py`
- Modify: `capstone_rag_project/tests/test_retriever.py`
- Create or modify only if needed: `capstone_rag_project/tests/test_grader.py`

### Task 1: Add Failing Tests For Graph-Specific Preservation

**Files:**
- Modify: `capstone_rag_project/tests/test_retriever.py`
- Modify: `capstone_rag_project/src/nodes/grader.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_grade_documents_keeps_all_graph_board_member_rows_for_list_query():
    from src.nodes.grader import grade_documents

    docs = [
        {"content": "Apple board member in 2024: Alex Gorsky | Title: Former Chair and CEO, Johnson & Johnson", "metadata": {"source": "Knowledge Graph"}},
        {"content": "Apple board member in 2024: Andrea Jung | Title: President and CEO, Grameen America", "metadata": {"source": "Knowledge Graph"}},
        {"content": "Apple board member in 2024: Tim Cook | Title: CEO, Apple", "metadata": {"source": "Knowledge Graph"}},
    ]

    result = grade_documents(
        {
            "route": "graph",
            "question": "Who are the board members of Apple in 2024",
            "rewritten_question": "Apple Inc. board of directors 2024",
            "retrieved_docs": docs,
        }
    )

    assert result["doc_relevance"] == ["yes", "yes", "yes"]
    assert result["relevant_doc_count"] == 3
    assert result["filtered_docs"] == docs
```

```python
def test_grade_documents_keeps_all_graph_subsidiary_rows_for_list_query():
    from src.nodes.grader import grade_documents

    docs = [
        {"content": "Apple subsidiary: Apple Operations International Limited", "metadata": {"source": "Knowledge Graph"}},
        {"content": "Apple subsidiary: Apple Sales International", "metadata": {"source": "Knowledge Graph"}},
    ]

    result = grade_documents(
        {
            "route": "graph",
            "question": "List Apple's subsidiaries",
            "rewritten_question": "Apple subsidiaries",
            "retrieved_docs": docs,
        }
    )

    assert result["doc_relevance"] == ["yes", "yes"]
    assert result["relevant_doc_count"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retriever.py -k "graph_board_member_rows or graph_subsidiary_rows" -v`
Expected: FAIL because the current generic grading path still uses the LLM per row and does not preserve graph list rows deterministically.

- [ ] **Step 3: Write minimal implementation**

```python
def _grade_graph_documents(state: GraphState) -> GraphState:
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {
            **state,
            "filtered_docs": [],
            "doc_relevance": [],
            "relevant_doc_count": 0,
        }

    doc_relevance = ["yes"] * len(docs)
    return {
        **state,
        "filtered_docs": docs,
        "doc_relevance": doc_relevance,
        "relevant_doc_count": len(docs),
    }
```

```python
if state.get("route") == "graph":
    return _grade_graph_documents(state)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retriever.py -k "graph_board_member_rows or graph_subsidiary_rows" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add capstone_rag_project/src/nodes/grader.py capstone_rag_project/tests/test_retriever.py
git commit -m "feat: preserve graph list rows during grading"
```

### Task 2: Add Deterministic Graph-Aware Relevance Checks

**Files:**
- Modify: `capstone_rag_project/src/nodes/grader.py`
- Modify: `capstone_rag_project/tests/test_retriever.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_graph_grading_keeps_filing_evidence_rows():
    from src.nodes.grader import grade_documents

    docs = [
        {"content": "Apple filing: 2024 10-K | Filing ID: AAPL_10-K_2024 | Source file: aapl_10-k_docling.json", "metadata": {"source": "Knowledge Graph"}},
        {"content": "Apple filing: 2024 10-Q | Filing ID: AAPL_10-Q_2024 | Source file: aapl_10-q_docling.json", "metadata": {"source": "Knowledge Graph"}},
    ]

    result = grade_documents(
        {
            "route": "graph",
            "question": "What Apple filings are available for 2024?",
            "rewritten_question": "Apple filings 2024",
            "retrieved_docs": docs,
        }
    )

    assert result["doc_relevance"] == ["yes", "yes"]
```

```python
def test_graph_grading_keeps_patent_evidence_rows():
    from src.nodes.grader import grade_documents

    docs = [
        {"content": "Apple patent: US1234567 | Title: Wireless security system | Grant date: 2024-03-15 | Domain: G06", "metadata": {"source": "Knowledge Graph"}},
    ]

    result = grade_documents(
        {
            "route": "graph",
            "question": "Which Apple patents were granted in 2024?",
            "rewritten_question": "Apple patents granted 2024",
            "retrieved_docs": docs,
        }
    )

    assert result["doc_relevance"] == ["yes"]
```

```python
def test_graph_grading_fails_open_for_unknown_graph_row_shape():
    from src.nodes.grader import grade_documents

    docs = [
        {"content": "mystery graph evidence row", "metadata": {"source": "Knowledge Graph"}},
    ]

    result = grade_documents(
        {
            "route": "graph",
            "question": "What does the Apple graph show?",
            "rewritten_question": "Apple graph evidence",
            "retrieved_docs": docs,
        }
    )

    assert result["doc_relevance"] == ["yes"]
    assert result["relevant_doc_count"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retriever.py -k "filing_evidence_rows or patent_evidence_rows or unknown_graph_row_shape" -v`
Expected: FAIL after tightening graph grading beyond the simple passthrough if any graph row family is not yet handled.

- [ ] **Step 3: Write minimal implementation**

```python
GRAPH_EVIDENCE_MARKERS = (
    "board member",
    "subsidiary",
    "filing:",
    "filing section",
    "patent:",
    "technology domain",
)


def _is_graph_evidence_doc(doc: dict) -> bool:
    metadata = doc.get("metadata", {})
    content = doc.get("content", "").lower()
    if metadata.get("source") != "Knowledge Graph":
        return False
    return any(marker in content for marker in GRAPH_EVIDENCE_MARKERS)
```

```python
def _grade_graph_documents(state: GraphState) -> GraphState:
    docs = state.get("retrieved_docs", [])
    filtered_docs = []
    doc_relevance = []

    for doc in docs:
        if _is_graph_evidence_doc(doc):
            doc_relevance.append("yes")
            filtered_docs.append(doc)
        else:
            # Fail open for graph rows to avoid over-pruning valid entity evidence.
            doc_relevance.append("yes")
            filtered_docs.append(doc)

    return {
        **state,
        "filtered_docs": filtered_docs,
        "doc_relevance": doc_relevance,
        "relevant_doc_count": len(filtered_docs),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retriever.py -k "filing_evidence_rows or patent_evidence_rows or unknown_graph_row_shape" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add capstone_rag_project/src/nodes/grader.py capstone_rag_project/tests/test_retriever.py
git commit -m "feat: add graph-aware grading helpers"
```

### Task 3: Prove Non-Graph Grading Is Unchanged

**Files:**
- Modify: `capstone_rag_project/tests/test_retriever.py`
- Modify: `capstone_rag_project/src/nodes/grader.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_non_graph_routes_still_use_llm_grading():
    from src.nodes.grader import BinaryGrade, grade_documents

    class StubGrader:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, _prompt):
            return BinaryGrade(score="no")

    docs = [
        {"content": "Some filing chunk", "metadata": {"source": "SEC Filing"}},
    ]

    with patch("src.nodes.grader.get_grader_llm", return_value=StubGrader()):
        result = grade_documents(
            {
                "route": "filings",
                "question": "What are Apple's risk factors?",
                "retrieved_docs": docs,
            }
        )

    assert result["doc_relevance"] == ["no"]
    assert result["relevant_doc_count"] == 0
```

```python
def test_graph_routes_do_not_call_generic_llm_grader():
    from src.nodes.grader import grade_documents

    docs = [
        {"content": "Apple board member in 2024: Tim Cook | Title: CEO, Apple", "metadata": {"source": "Knowledge Graph"}},
    ]

    with patch("src.nodes.grader.get_grader_llm", side_effect=AssertionError("generic grader should not run for graph")):
        result = grade_documents(
            {
                "route": "graph",
                "question": "Who are the board members of Apple in 2024",
                "retrieved_docs": docs,
            }
        )

    assert result["doc_relevance"] == ["yes"]
    assert result["relevant_doc_count"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retriever.py -k "still_use_llm_grading or do_not_call_generic_llm_grader" -v`
Expected: FAIL until `grade_documents(...)` branches correctly between graph and non-graph routes.

- [ ] **Step 3: Write minimal implementation**

```python
def grade_documents(state: GraphState) -> GraphState:
    eval_config = state.get("eval_config") or {}
    if eval_config.get("corrective") is False:
        ...

    if state.get("route") == "graph":
        return _grade_graph_documents(state)

    question = state.get("rewritten_question") or state["question"]
    retrieved_docs = state.get("retrieved_docs", [])
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retriever.py -k "still_use_llm_grading or do_not_call_generic_llm_grader" -v`
Expected: PASS

- [ ] **Step 5: Run the focused verification suite**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add capstone_rag_project/src/nodes/grader.py capstone_rag_project/tests/test_retriever.py
git commit -m "test: isolate graph-specific grading from other routes"
```
