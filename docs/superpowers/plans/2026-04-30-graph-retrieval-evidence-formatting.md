# Graph Retrieval Evidence Formatting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Neo4j retrieval so graph-based answers are grounded in explicit natural-language evidence for board members, subsidiaries, filings, filing sections, patents, and technology domains.

**Architecture:** Keep `src/graph_retrieval.py` as the single query-time Neo4j entrypoint. First strengthen the Cypher-generation prompt so the graph LLM prefers richer semantic columns, then replace generic row serialization with row-shape-aware evidence formatting helpers and a safe generic fallback.

**Tech Stack:** Python, Neo4j Python driver, dotenv, pytest, unittest.mock

---

## File Structure

- Modify: `capstone_rag_project/src/graph_retrieval.py`
- Modify: `capstone_rag_project/tests/test_graph_retrieval.py`
- Modify: `capstone_rag_project/docs/superpowers/specs/2026-04-30-graph-retrieval-evidence-formatting-design.md` only if implementation uncovers an ambiguity that must be clarified in the spec

### Task 1: Strengthen The Graph Cypher Prompt

**Files:**
- Modify: `capstone_rag_project/src/graph_retrieval.py`
- Test: `capstone_rag_project/tests/test_graph_retrieval.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_graph_prompt_instructs_semantic_aliases_and_year_fields() -> None:
    from src.graph_retrieval import _graph_prompt

    prompt = _graph_prompt("Who are the board members of Apple in 2024?", company="AAPL")

    assert "return semantic aliases" in prompt.lower()
    assert "include company and year fields when available" in prompt.lower()
    assert "for board-member questions" in prompt.lower()
    assert "for subsidiary questions" in prompt.lower()
    assert "for filing questions" in prompt.lower()
    assert "for patent questions" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_retrieval.py -k semantic_aliases_and_year_fields -v`
Expected: FAIL because the current graph prompt only asks for read-only Cypher and human-readable aliases in a generic way.

- [ ] **Step 3: Write minimal implementation**

```python
def _graph_prompt(question: str, *, company: str | None = None) -> str:
    company_hint = f"Preferred company ticker: {company.upper()}" if company else "No preferred company ticker"
    return f"""
You translate M&A due diligence questions into Cypher for Neo4j.

Use only the schema below. Produce exactly one read-only Cypher query.
Allowed clauses: MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT.
Never use CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP, LOAD, FOREACH, or CALL.
Return at most 10 rows.
If the user asks about a company and a ticker hint is available, use it.
Prefer returning named columns with human-readable aliases.
Return semantic aliases instead of opaque abbreviations where possible.
Include company and year fields when available.
For board-member questions, prefer years_present, is_current, and explicit year filtering when the question mentions a year.
For subsidiary questions, include company and source-form context when available.
For filing questions, include form type, year, filing_id, source_file, and section context when available.
For patent questions, include patent_id, patent_title, grant_date, grant_year, and domain fields when available.
Return only Cypher. No explanation. No markdown unless it is a single cypher fence.

Schema:
{_GRAPH_SCHEMA}

{company_hint}

Question:
{question}
""".strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_retrieval.py -k semantic_aliases_and_year_fields -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add capstone_rag_project/src/graph_retrieval.py capstone_rag_project/tests/test_graph_retrieval.py
git commit -m "feat: strengthen graph cypher prompt for semantic returns"
```

### Task 2: Add Generic Fallback Formatting Helpers

**Files:**
- Modify: `capstone_rag_project/src/graph_retrieval.py`
- Test: `capstone_rag_project/tests/test_graph_retrieval.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_generic_row_to_doc_falls_back_to_key_value_lines() -> None:
    from src.graph_retrieval import _row_to_doc

    doc = _row_to_doc({"custom_field": "value", "count": 3}, cypher="MATCH ... RETURN ...")

    assert "custom_field: value" in doc["content"]
    assert "count: 3" in doc["content"]
    assert doc["metadata"]["cypher"] == "MATCH ... RETURN ..."
```

```python
def test_list_values_are_preserved_in_fallback_formatting() -> None:
    from src.graph_retrieval import _row_to_doc

    doc = _row_to_doc({"years_present": [2024, 2025]}, cypher="MATCH ... RETURN ...")

    assert "years_present: [2024, 2025]" in doc["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph_retrieval.py -k fallback_formatting -v`
Expected: FAIL after renaming or restructuring the row-formatting code but before the generic fallback is restored.

- [ ] **Step 3: Write minimal implementation**

```python
def _generic_row_lines(row: dict[str, Any]) -> list[str]:
    return [f"{key}: {_format_value(value)}" for key, value in row.items()]
```

```python
def _generic_row_to_doc(row: dict[str, Any], *, cypher: str) -> dict[str, Any]:
    return {
        "content": "\n".join(_generic_row_lines(row)),
        "metadata": {
            "source": "Knowledge Graph",
            "title": "Neo4j Graph Result",
            "cypher": cypher,
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph_retrieval.py -k fallback_formatting -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add capstone_rag_project/src/graph_retrieval.py capstone_rag_project/tests/test_graph_retrieval.py
git commit -m "refactor: add generic graph row fallback formatting"
```

### Task 3: Add Board-Member And Subsidiary Evidence Formatting

**Files:**
- Modify: `capstone_rag_project/src/graph_retrieval.py`
- Test: `capstone_rag_project/tests/test_graph_retrieval.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_board_member_row_formats_as_explicit_evidence() -> None:
    from src.graph_retrieval import _row_to_doc

    row = {
        "company": "Apple",
        "year": 2024,
        "board_member": "Art Levinson",
        "title": "Founder and CEO, Calico",
    }

    doc = _row_to_doc(row, cypher="MATCH ... RETURN ...")

    assert "Apple board member in 2024: Art Levinson" in doc["content"]
    assert "Title: Founder and CEO, Calico" in doc["content"]
```

```python
def test_subsidiary_row_formats_as_explicit_evidence() -> None:
    from src.graph_retrieval import _row_to_doc

    row = {
        "company": "Apple",
        "subsidiary": "Apple Operations International Limited",
        "year": 2024,
        "source_form_type": "10-K",
    }

    doc = _row_to_doc(row, cypher="MATCH ... RETURN ...")

    assert "Apple subsidiary in 2024: Apple Operations International Limited" in doc["content"]
    assert "Source form: 10-K" in doc["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph_retrieval.py -k "board_member_row_formats or subsidiary_row_formats" -v`
Expected: FAIL because rows are still rendered as generic key-value lines.

- [ ] **Step 3: Write minimal implementation**

```python
def _looks_like_board_member_row(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    return ("board_member" in keys or "boardmembername" in keys or "member_name" in keys) and (
        "title" in keys or "boardmembertitle" in keys
    )
```

```python
def _format_board_member_row(row: dict[str, Any]) -> str:
    company = row.get("company") or row.get("ticker") or "Company"
    year = row.get("year")
    name = row.get("board_member") or row.get("BoardMemberName") or row.get("member_name") or "Unknown"
    title = row.get("title") or row.get("BoardMemberTitle")
    years_present = row.get("years_present")

    head = f"{company} board member in {year}: {name}" if year else f"{company} board member: {name}"
    details = []
    if title:
        details.append(f"Title: {title}")
    if years_present:
        details.append(f"Years present: {_format_value(years_present)}")
    return " | ".join([head, *details]) if details else head
```

```python
def _looks_like_subsidiary_row(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    return "subsidiary" in keys or "subsidiary_name" in keys
```

```python
def _format_subsidiary_row(row: dict[str, Any]) -> str:
    company = row.get("company") or row.get("ticker") or "Company"
    year = row.get("year")
    name = row.get("subsidiary") or row.get("subsidiary_name") or "Unknown"
    source_form_type = row.get("source_form_type")

    head = f"{company} subsidiary in {year}: {name}" if year else f"{company} subsidiary: {name}"
    details = [f"Source form: {source_form_type}"] if source_form_type else []
    return " | ".join([head, *details]) if details else head
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph_retrieval.py -k "board_member_row_formats or subsidiary_row_formats" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add capstone_rag_project/src/graph_retrieval.py capstone_rag_project/tests/test_graph_retrieval.py
git commit -m "feat: format board and subsidiary graph evidence"
```

### Task 4: Add Filing, Section, Patent, And Domain Evidence Formatting

**Files:**
- Modify: `capstone_rag_project/src/graph_retrieval.py`
- Test: `capstone_rag_project/tests/test_graph_retrieval.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_filing_row_formats_as_explicit_evidence() -> None:
    from src.graph_retrieval import _row_to_doc

    row = {
        "company": "Apple",
        "year": 2024,
        "form_type": "10-K",
        "filing_id": "AAPL_10-K_2024",
        "source_file": "aapl_10-k_docling.json",
    }

    doc = _row_to_doc(row, cypher="MATCH ... RETURN ...")

    assert "Apple filing: 2024 10-K" in doc["content"]
    assert "Filing ID: AAPL_10-K_2024" in doc["content"]
    assert "Source file: aapl_10-k_docling.json" in doc["content"]
```

```python
def test_section_row_formats_as_explicit_evidence() -> None:
    from src.graph_retrieval import _row_to_doc

    row = {
        "company": "Apple",
        "year": 2024,
        "form_type": "10-K",
        "section_title": "Risk Factors",
        "section_id": "AAPL_10-K_2024_section_12",
        "text": "Supply chain constraints may materially affect operations.",
    }

    doc = _row_to_doc(row, cypher="MATCH ... RETURN ...")

    assert "Apple 2024 10-K section: Risk Factors" in doc["content"]
    assert "Section ID: AAPL_10-K_2024_section_12" in doc["content"]
    assert "Supply chain constraints may materially affect operations." in doc["content"]
```

```python
def test_patent_row_formats_as_explicit_evidence() -> None:
    from src.graph_retrieval import _row_to_doc

    row = {
        "company": "Apple",
        "patent_id": "US1234567",
        "patent_title": "Wireless security system",
        "grant_date": "2024-03-15",
        "domain": "G06",
    }

    doc = _row_to_doc(row, cypher="MATCH ... RETURN ...")

    assert "Apple patent: US1234567" in doc["content"]
    assert "Title: Wireless security system" in doc["content"]
    assert "Grant date: 2024-03-15" in doc["content"]
    assert "Domain: G06" in doc["content"]
```

```python
def test_domain_row_formats_as_explicit_evidence() -> None:
    from src.graph_retrieval import _row_to_doc

    row = {
        "company": "Apple",
        "cpc_prefix": "G06",
        "label": "Computing & Data Processing",
    }

    doc = _row_to_doc(row, cypher="MATCH ... RETURN ...")

    assert "Apple technology domain: G06 | Computing & Data Processing" in doc["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph_retrieval.py -k "filing_row_formats or section_row_formats or patent_row_formats or domain_row_formats" -v`
Expected: FAIL because these row shapes still fall through to generic formatting.

- [ ] **Step 3: Write minimal implementation**

```python
def _looks_like_filing_row(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    return "filing_id" in keys and "form_type" in keys
```

```python
def _format_filing_row(row: dict[str, Any]) -> str:
    company = row.get("company") or row.get("ticker") or "Company"
    year = row.get("year")
    form_type = row.get("form_type") or "Filing"
    filing_id = row.get("filing_id")
    source_file = row.get("source_file")
    parts = [f"{company} filing: {year} {form_type}" if year else f"{company} filing: {form_type}"]
    if filing_id:
        parts.append(f"Filing ID: {filing_id}")
    if source_file:
        parts.append(f"Source file: {source_file}")
    return " | ".join(parts)
```

```python
def _looks_like_section_row(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    return ("section_id" in keys and "text" in keys) or "section_title" in keys or "title" in keys
```

```python
def _format_section_row(row: dict[str, Any]) -> str:
    company = row.get("company") or row.get("ticker") or "Company"
    year = row.get("year")
    form_type = row.get("form_type")
    title = row.get("section_title") or row.get("title") or "Untitled section"
    section_id = row.get("section_id")
    text = row.get("text")
    head = f"{company} {year} {form_type} section: {title}" if year and form_type else f"{company} filing section: {title}"
    detail_parts = [f"Section ID: {section_id}"] if section_id else []
    body = f"{head} | {' | '.join(detail_parts)}" if detail_parts else head
    return f"{body}\n{text}" if text else body
```

```python
def _looks_like_patent_row(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    return "patent_id" in keys or "patent_title" in keys
```

```python
def _format_patent_row(row: dict[str, Any]) -> str:
    company = row.get("company") or row.get("ticker") or "Company"
    patent_id = row.get("patent_id") or "Unknown patent"
    title = row.get("patent_title") or row.get("title")
    grant_date = row.get("grant_date")
    domain = row.get("domain") or row.get("cpc_prefix")
    parts = [f"{company} patent: {patent_id}"]
    if title:
        parts.append(f"Title: {title}")
    if grant_date:
        parts.append(f"Grant date: {grant_date}")
    if domain:
        parts.append(f"Domain: {domain}")
    return " | ".join(parts)
```

```python
def _looks_like_domain_row(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    return "cpc_prefix" in keys and "label" in keys and "patent_id" not in keys
```

```python
def _format_domain_row(row: dict[str, Any]) -> str:
    company = row.get("company") or row.get("ticker") or "Company"
    prefix = row.get("cpc_prefix") or "Unknown"
    label = row.get("label") or "Unknown domain"
    return f"{company} technology domain: {prefix} | {label}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph_retrieval.py -k "filing_row_formats or section_row_formats or patent_row_formats or domain_row_formats" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add capstone_rag_project/src/graph_retrieval.py capstone_rag_project/tests/test_graph_retrieval.py
git commit -m "feat: format filing and patent graph evidence"
```

### Task 5: Wire The Formatter Into Retrieval And Verify End-To-End Graph Docs

**Files:**
- Modify: `capstone_rag_project/src/graph_retrieval.py`
- Modify: `capstone_rag_project/tests/test_graph_retrieval.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_retrieve_graph_docs_returns_semantic_subsidiary_evidence() -> None:
    from src.graph_retrieval import retrieve_graph_docs

    llm = MagicMock()
    llm.invoke.return_value.content = (
        "MATCH (c:Company {ticker: 'AAPL'})-[:HAS_SUBSIDIARY]->(s:Subsidiary) "
        "RETURN 'Apple' AS company, s.name AS subsidiary, 2024 AS year LIMIT 1"
    )

    result_cursor = MagicMock()
    result_cursor.data.return_value = [
        {"company": "Apple", "subsidiary": "Apple Operations International Limited", "year": 2024}
    ]

    session = MagicMock()
    session.__enter__.return_value = session
    session.run.return_value = result_cursor

    driver = MagicMock()
    driver.session.return_value = session

    with (
        patch("src.graph_retrieval.get_graph_llm", return_value=llm),
        patch("src.graph_retrieval.get_neo4j_driver", return_value=driver),
    ):
        docs = retrieve_graph_docs("List Apple's subsidiaries", company="AAPL")

    assert docs[0]["content"].startswith("Apple subsidiary in 2024: Apple Operations International Limited")
    assert docs[0]["metadata"]["cypher"].startswith("MATCH")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_retrieval.py -k semantic_subsidiary_evidence -v`
Expected: FAIL because retrieval still emits generic row serialization.

- [ ] **Step 3: Write minimal implementation**

```python
def _row_to_doc(row: dict[str, Any], *, cypher: str) -> dict[str, Any]:
    if _looks_like_board_member_row(row):
        content = _format_board_member_row(row)
    elif _looks_like_subsidiary_row(row):
        content = _format_subsidiary_row(row)
    elif _looks_like_section_row(row):
        content = _format_section_row(row)
    elif _looks_like_filing_row(row):
        content = _format_filing_row(row)
    elif _looks_like_patent_row(row):
        content = _format_patent_row(row)
    elif _looks_like_domain_row(row):
        content = _format_domain_row(row)
    else:
        content = "\n".join(_generic_row_lines(row))

    return {
        "content": content,
        "metadata": {
            "source": "Knowledge Graph",
            "title": "Neo4j Graph Result",
            "cypher": cypher,
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_retrieval.py -k semantic_subsidiary_evidence -v`
Expected: PASS

- [ ] **Step 5: Run the focused verification suite**

Run: `uv run pytest tests/test_graph_retrieval.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add capstone_rag_project/src/graph_retrieval.py capstone_rag_project/tests/test_graph_retrieval.py
git commit -m "feat: return semantic graph evidence docs"
```
