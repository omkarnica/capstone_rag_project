# Neo4j Routing Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic routing for board-member and subsidiary questions, and make filings retrieval fall back once to Neo4j when Pinecone retrieval fails or produces no useful evidence.

**Architecture:** Keep query-time Neo4j access centralized in `src/graph_retrieval.py`. Add graph-topic detection in planner and router, annotate filings retrieval outcomes in the retriever, and teach the LangGraph orchestration to try `graph` once before rewrite or web fallback.

**Tech Stack:** Python, LangGraph, Pydantic, Neo4j Python driver, dotenv, pytest

---

## File Structure

- Modify: `capstone_rag_project/src/graph_retrieval.py`
- Modify: `capstone_rag_project/src/Knowledge graph/kg.py`
- Modify: `capstone_rag_project/src/nodes/router.py`
- Modify: `capstone_rag_project/src/nodes/planner.py`
- Modify: `capstone_rag_project/src/nodes/retriever.py`
- Modify: `capstone_rag_project/src/graph.py`
- Modify: `capstone_rag_project/tests/test_graph_retrieval.py`
- Create or modify: `capstone_rag_project/tests/test_router.py`
- Create or modify: `capstone_rag_project/tests/test_planner.py`
- Create or modify: `capstone_rag_project/tests/test_retriever.py`

### Task 1: Make Neo4j Env Loading Robust

**Files:**
- Modify: `capstone_rag_project/src/graph_retrieval.py`
- Modify: `capstone_rag_project/src/Knowledge graph/kg.py`
- Test: `capstone_rag_project/tests/test_graph_retrieval.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_neo4j_settings_prefers_env(monkeypatch):
    from src import graph_retrieval

    monkeypatch.setenv("NEO4J_URI", "neo4j+s://db.example.io")
    monkeypatch.setenv("NEO4J_USER", "neo4j-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")

    assert graph_retrieval._resolve_neo4j_settings() == (
        "neo4j+s://db.example.io",
        "neo4j-user",
        "secret",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest capstone_rag_project/tests/test_graph_retrieval.py -k neo4j_settings -v`
Expected: FAIL until `.env` values are loaded before Neo4j settings are resolved.

- [ ] **Step 3: Write minimal implementation**

```python
from dotenv import load_dotenv

load_dotenv()

def _resolve_neo4j_settings() -> tuple[str, str, str]:
    neo4j_uri = os.getenv("NEO4J_URI") or str(_CONFIG.get("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user = os.getenv("NEO4J_USER") or str(_CONFIG.get("NEO4J_USER", "neo4j"))
    neo4j_password = os.getenv("NEO4J_PASSWORD") or str(_CONFIG.get("NEO4J_PASSWORD", "password"))
    return neo4j_uri, neo4j_user, neo4j_password
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest capstone_rag_project/tests/test_graph_retrieval.py -k neo4j_settings -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add capstone_rag_project/src/graph_retrieval.py capstone_rag_project/src/Knowledge\ graph/kg.py capstone_rag_project/tests/test_graph_retrieval.py
git commit -m "fix: load neo4j settings from env"
```

### Task 2: Add Shared Graph-Topic Detection In Planner And Router

**Files:**
- Modify: `capstone_rag_project/src/nodes/router.py`
- Modify: `capstone_rag_project/src/nodes/planner.py`
- Test: `capstone_rag_project/tests/test_router.py`
- Test: `capstone_rag_project/tests/test_planner.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_router_routes_board_member_query_to_graph():
    from src.nodes.router import route_question

    result = route_question({"question": "Who are Apple's board members?"})

    assert result["route"] == "graph"


def test_router_routes_subsidiary_query_to_graph():
    from src.nodes.router import route_question

    result = route_question({"question": "List Apple's subsidiaries"})

    assert result["route"] == "graph"


def test_planner_hints_graph_for_board_member_query():
    from src.nodes.planner import _guess_route_hint

    assert _guess_route_hint("Who are Apple's board members?") == "graph"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest capstone_rag_project/tests/test_router.py capstone_rag_project/tests/test_planner.py -v`
Expected: FAIL because there is no deterministic graph-topic detection yet.

- [ ] **Step 3: Write minimal implementation**

```python
GRAPH_MARKERS = {
    "board member",
    "board members",
    "board of directors",
    "director",
    "directors",
    "subsidiary",
    "subsidiaries",
    "corporate hierarchy",
    "ownership structure",
}


def _is_graph_topic(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in GRAPH_MARKERS)
```

```python
if _is_graph_topic(question):
    return {
        **state,
        "route": "graph",
        "initial_route": "graph",
        "route_reason": "Deterministic graph-topic routing",
    }
```

```python
if _is_graph_topic(question):
    return "graph"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest capstone_rag_project/tests/test_router.py capstone_rag_project/tests/test_planner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add capstone_rag_project/src/nodes/router.py capstone_rag_project/src/nodes/planner.py capstone_rag_project/tests/test_router.py capstone_rag_project/tests/test_planner.py
git commit -m "feat: route board and subsidiary queries to graph"
```

### Task 3: Record Filings Retrieval Failures And Empty Results

**Files:**
- Modify: `capstone_rag_project/src/nodes/retriever.py`
- Test: `capstone_rag_project/tests/test_retriever.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_retrieve_docs_marks_filings_failure():
    from src.nodes.retriever import retrieve_docs

    result = retrieve_docs({"route": "filings", "question": "Summarize Apple's risk factors"})

    assert "filings_error" in result


def test_retrieve_docs_marks_filings_empty_result():
    from src.nodes.retriever import retrieve_docs

    with patch("src.filings.raptor_retrieval.raptor_retrieve", return_value={"contexts": []}):
        result = retrieve_docs({"route": "filings", "question": "Summarize Apple's risk factors"})

    assert result["filings_empty"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest capstone_rag_project/tests/test_retriever.py -v`
Expected: FAIL because filings retrieval state is not annotated yet.

- [ ] **Step 3: Write minimal implementation**

```python
filings_error = False
filings_empty = False
```

```python
except Exception as exc:
    filings_error = True
    logger.warning("Filings retrieval failed: %s", exc)
```

```python
filings_empty = len(docs) == 0
```

```python
return {
    **state,
    "retrieval_query": query,
    "retrieved_docs": docs,
    "filtered_docs": [],
    "doc_relevance": [],
    "relevant_doc_count": 0,
    "filings_error": filings_error,
    "filings_empty": filings_empty,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest capstone_rag_project/tests/test_retriever.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add capstone_rag_project/src/nodes/retriever.py capstone_rag_project/tests/test_retriever.py
git commit -m "feat: annotate filings retrieval outcomes"
```

### Task 4: Add One-Time Filings-To-Graph Fallback In LangGraph

**Files:**
- Modify: `capstone_rag_project/src/graph.py`
- Test: `capstone_rag_project/tests/test_retriever.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_route_after_doc_grading_falls_back_to_graph_for_empty_filings():
    from src.graph import route_after_doc_grading

    state = {
        "route": "filings",
        "relevant_doc_count": 0,
        "retrieval_attempt": 0,
        "max_retrieval_attempts": 3,
        "filings_empty": True,
        "graph_fallback_attempted": False,
    }

    assert route_after_doc_grading(state) == "graph_fallback"
```

```python
def test_route_after_doc_grading_falls_back_to_graph_for_filings_error():
    from src.graph import route_after_doc_grading

    state = {
        "route": "filings",
        "relevant_doc_count": 0,
        "retrieval_attempt": 0,
        "max_retrieval_attempts": 3,
        "filings_error": True,
        "graph_fallback_attempted": False,
    }

    assert route_after_doc_grading(state) == "graph_fallback"
```

```python
def test_route_after_doc_grading_only_falls_back_once():
    from src.graph import route_after_doc_grading

    state = {
        "route": "filings",
        "relevant_doc_count": 0,
        "retrieval_attempt": 0,
        "max_retrieval_attempts": 3,
        "filings_empty": True,
        "graph_fallback_attempted": True,
    }

    assert route_after_doc_grading(state) == "rewrite"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest capstone_rag_project/tests/test_retriever.py -k graph_fallback -v`
Expected: FAIL because there is no graph fallback branch yet.

- [ ] **Step 3: Write minimal implementation**

```python
def use_graph_fallback(state: GraphState) -> GraphState:
    return {
        **state,
        "route": "graph",
        "graph_fallback_attempted": True,
        "retrieved_docs": [],
        "filtered_docs": [],
        "doc_relevance": [],
        "relevant_doc_count": 0,
    }
```

```python
if (
    state.get("route") == "filings"
    and not state.get("graph_fallback_attempted", False)
    and (state.get("filings_error") or state.get("filings_empty"))
):
    return "graph_fallback"
```

```python
graph.add_node("graph_fallback", use_graph_fallback)
graph.add_edge("graph_fallback", "retrieve")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest capstone_rag_project/tests/test_retriever.py -k graph_fallback -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add capstone_rag_project/src/graph.py capstone_rag_project/tests/test_retriever.py
git commit -m "feat: fall back from filings to graph once"
```

### Task 5: Cover Zero-Relevance Post-Grading Fallback

**Files:**
- Modify: `capstone_rag_project/src/graph.py`
- Test: `capstone_rag_project/tests/test_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
def test_route_after_doc_grading_falls_back_to_graph_after_irrelevant_filings_docs():
    from src.graph import route_after_doc_grading

    state = {
        "route": "filings",
        "relevant_doc_count": 0,
        "retrieval_attempt": 0,
        "max_retrieval_attempts": 3,
        "filings_error": False,
        "filings_empty": False,
        "graph_fallback_attempted": False,
    }

    assert route_after_doc_grading(state) == "graph_fallback"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest capstone_rag_project/tests/test_retriever.py -k irrelevant_filings_docs -v`
Expected: FAIL because zero-relevance grading does not yet trigger graph fallback.

- [ ] **Step 3: Write minimal implementation**

```python
if (
    state.get("route") == "filings"
    and not state.get("graph_fallback_attempted", False)
    and state.get("relevant_doc_count", 0) == 0
):
    return "graph_fallback"
```
 
- [ ] **Step 4: Run test to verify it passes**

Run: `pytest capstone_rag_project/tests/test_retriever.py -k irrelevant_filings_docs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add capstone_rag_project/src/graph.py capstone_rag_project/tests/test_retriever.py
git commit -m "feat: use graph fallback after irrelevant filings results"
```

### Task 6: Verify Existing Graph Retrieval Path Still Works

**Files:**
- Modify: `capstone_rag_project/tests/test_graph_retrieval.py`

- [ ] **Step 1: Add regression tests for graph dispatch**

```python
def test_retrieve_docs_dispatches_graph_route():
    from src.nodes.retriever import retrieve_docs

    graph_docs = [
        {
            "content": "subsidiary: Apple Operations International Limited",
            "metadata": {"source": "Knowledge Graph"},
        }
    ]

    with patch("src.nodes.retriever.retrieve_graph_docs", return_value=graph_docs) as mock_retrieve:
        result = retrieve_docs(
            {
                "route": "graph",
                "question": "List Apple's subsidiaries",
                "company": "AAPL",
            }
        )

    mock_retrieve.assert_called_once_with("List Apple's subsidiaries", company="AAPL")
    assert result["retrieved_docs"] == graph_docs
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest capstone_rag_project/tests/test_graph_retrieval.py -k dispatches_graph_route -v`
Expected: PASS

- [ ] **Step 3: Run the focused verification suite**

Run: `pytest capstone_rag_project/tests/test_graph_retrieval.py capstone_rag_project/tests/test_router.py capstone_rag_project/tests/test_planner.py capstone_rag_project/tests/test_retriever.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add capstone_rag_project/tests/test_graph_retrieval.py capstone_rag_project/tests/test_router.py capstone_rag_project/tests/test_planner.py capstone_rag_project/tests/test_retriever.py
git commit -m "test: cover graph routing and fallback behavior"
```
