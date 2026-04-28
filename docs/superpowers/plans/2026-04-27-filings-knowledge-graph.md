# Filings Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `kg.py` script that reads five Apple filings JSON artifacts plus a filtered patents CSV from `data/`, constructs a Neo4j knowledge graph with company, filing, section, subsidiary, board-member, patent, and technology-domain nodes, and logs a graph summary.

**Architecture:** Keep `kg.py` as a standalone Neo4j writer with no Pinecone dependency. Use pure helpers for year extraction, section extraction, subsidiary normalization, board-member aggregation, patent CSV filtering, and CPC parsing so the graph logic is testable without Neo4j. Ingest the three docling files for filing and section nodes, attach subsidiaries and board members from their sidecar JSON files, then ingest patents from a separately filtered `2021-2025` CSV into `Patent` and `TechnologyDomain`.

**Tech Stack:** Python, Neo4j Python driver, `json`, `logging`, `argparse`, `re`, `pathlib`, `csv`, `pytest`

---

## File Structure

- Modify: `capstone_rag_project/src/Knowledge graph/kg.py`
  - Replace the current mixed-source builder with a standalone filings-only KG builder.
  - Remove Pinecone runtime dependency from the execution path.
  - Add pure helpers for year extraction, ID generation, section filtering, section extraction, subsidiary normalization, board-member normalization, and aggregation.
  - Add patents CSV filtering, CPC parsing, patent/domain ingestion, Neo4j schema creation, graph summary logging, and CLI entry point.

- Create: `capstone_rag_project/tests/test_kg.py`
  - Add pure-function unit tests for:
    - filing year extraction
    - filing ID generation
    - title detection
    - page-stamp detection
    - section extraction filtering
    - board-member normalization
    - board-member aggregation
    - subsidiary ID normalization
    - patents CSV filtering
    - CPC parsing
    - CPC section extraction

## Task 1: Replace `kg.py` Skeleton With Standalone Filings KG Structure

**Files:**
- Modify: `capstone_rag_project/src/Knowledge graph/kg.py`
- Test: `capstone_rag_project/tests/test_kg.py`

- [ ] **Step 1: Write the failing test for filing year extraction and filing ID generation**

```python
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_kg_module():
    kg_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "Knowledge graph"
        / "kg.py"
    )
    spec = spec_from_file_location("kg_module", kg_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_year_from_report_date_preferred():
    kg = load_kg_module()
    entry = {
        "report_date": "2025-12-31",
        "filing_date": "2026-02-05",
    }

    assert kg.extract_year(entry) == 2025


def test_build_filing_id_uses_report_year():
    kg = load_kg_module()

    assert kg.build_filing_id("AAPL", "10-K", 2025) == "AAPL_10-K_2025"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest capstone_rag_project\tests\test_kg.py -k "year_from_report_date_preferred or build_filing_id_uses_report_year" -v
```

Expected:

```text
FAIL ... AttributeError: module 'kg_module' has no attribute 'extract_year'
```

- [ ] **Step 3: Write minimal standalone helper structure in `kg.py`**

```python
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


logger = logging.getLogger("filings_kg")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def load_json(path: str) -> list[Any] | dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc


def extract_year(entry: dict[str, Any]) -> int | None:
    report_date = entry.get("report_date")
    if report_date:
        return int(report_date[:4])

    filing_date = entry.get("filing_date")
    if filing_date:
        return int(filing_date[:4])

    return None


def build_filing_id(ticker: str, form_type: str, year: int) -> str:
    return f"{ticker.upper()}_{form_type}_{year}"
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
.venv\Scripts\python.exe -m pytest capstone_rag_project\tests\test_kg.py -k "year_from_report_date_preferred or build_filing_id_uses_report_year" -v
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```powershell
git add capstone_rag_project/src/Knowledge\ graph/kg.py capstone_rag_project/tests/test_kg.py
git commit -m "feat: scaffold standalone filings knowledge graph helpers"
```

## Task 2: Add Section-Filtering Helpers and Extraction Logic

**Files:**
- Modify: `capstone_rag_project/src/Knowledge graph/kg.py`
- Test: `capstone_rag_project/tests/test_kg.py`

- [ ] **Step 1: Write the failing tests for page stamps, TOC links, short blocks, and title detection**

```python
def test_is_page_stamp_detects_docling_footer():
    kg = load_kg_module()
    assert kg._is_page_stamp("Apple Inc. | 2021 Form 10-K | 26") is True


def test_is_title_block_for_short_heading():
    kg = load_kg_module()
    assert kg._is_title_block("Foreign Currency Risk") is True


def test_is_title_block_for_long_paragraph():
    kg = load_kg_module()
    text = "This is a long narrative paragraph that explains a risk in detail and should not be treated as a heading because it clearly reads like prose."
    assert kg._is_title_block(text) is False


def test_extract_sections_skips_toc_short_and_page_stamp():
    kg = load_kg_module()
    entry = {
        "ticker": "AAPL",
        "form_type": "10-K",
        "report_date": "2021-09-25",
        "filing_date": "2021-10-29",
        "source": "aapl_2021.html",
        "docling": {
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "content_layer": "body",
                    "hyperlink": "#item1",
                    "text": "Item 1. Business",
                },
                {
                    "self_ref": "#/texts/1",
                    "content_layer": "body",
                    "text": "Short text",
                },
                {
                    "self_ref": "#/texts/2",
                    "content_layer": "body",
                    "text": "Apple Inc. | 2021 Form 10-K | 26",
                },
                {
                    "self_ref": "#/texts/3",
                    "content_layer": "body",
                    "text": "Foreign Currency Risk",
                },
                {
                    "self_ref": "#/texts/4",
                    "content_layer": "body",
                    "text": "The Company is exposed to movements in foreign currency exchange rates because a significant portion of revenue is generated outside the United States.",
                },
            ]
        },
    }

    sections = kg.extract_sections_from_docling_entry(entry, "aapl_10-k_docling.json")

    assert len(sections) == 1
    assert sections[0]["section_id"] == "AAPL_10-K_2021_section_4"
    assert sections[0]["title"] == "Foreign Currency Risk"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest capstone_rag_project\tests\test_kg.py -k "page_stamp or title_block or extract_sections_skips_toc_short_and_page_stamp" -v
```

Expected:

```text
FAIL ... AttributeError: module 'kg_module' has no attribute 'extract_sections_from_docling_entry'
```

- [ ] **Step 3: Implement section helpers and extraction**

```python
PAGE_STAMP_RE = re.compile(r"^Apple Inc\. \| \d{4} Form \d+.*\| \d+$")
ITEM_HEADING_RE = re.compile(r"^Item\s+\d+[\w\.]*", re.IGNORECASE)


def _extract_ordinal(self_ref: str) -> int:
    match = re.search(r"#/texts/(\d+)$", self_ref or "")
    if not match:
        return -1
    return int(match.group(1))


def _is_page_stamp(text: str) -> bool:
    return bool(PAGE_STAMP_RE.match(text.strip()))


def _is_title_block(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned or len(cleaned) > 120:
        return False
    if cleaned.endswith(".") or cleaned.endswith(","):
        return False
    if ITEM_HEADING_RE.match(cleaned):
        return True
    return cleaned == cleaned.title()


def extract_sections_from_docling_entry(
    entry: dict[str, Any],
    source_file: str,
) -> list[dict[str, Any]]:
    ticker = str(entry.get("ticker", "")).upper()
    form_type = str(entry.get("form_type", "")).upper()
    year = extract_year(entry)
    if not ticker or not form_type or year is None:
        return []

    filing_id = build_filing_id(ticker, form_type, year)
    texts = entry.get("docling", {}).get("texts", [])

    sections: list[dict[str, Any]] = []
    pending_title: str | None = None

    for item in texts:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if item.get("content_layer") != "body":
            continue
        if item.get("hyperlink"):
            continue
        if _is_page_stamp(text):
            continue
        if len(text) < 50:
            if _is_title_block(text):
                pending_title = text
            continue

        ordinal = _extract_ordinal(str(item.get("self_ref", "")))
        sections.append(
            {
                "section_id": f"{filing_id}_section_{ordinal}",
                "filing_id": filing_id,
                "ticker": ticker,
                "form_type": form_type,
                "year": year,
                "ordinal": ordinal,
                "title": pending_title,
                "text": text,
                "source_file": source_file,
            }
        )
        pending_title = None

    return sections
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
.venv\Scripts\python.exe -m pytest capstone_rag_project\tests\test_kg.py -k "page_stamp or title_block or extract_sections_skips_toc_short_and_page_stamp" -v
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Commit**

```powershell
git add capstone_rag_project/src/Knowledge\ graph/kg.py capstone_rag_project/tests/test_kg.py
git commit -m "feat: add docling section extraction helpers"
```

## Task 3: Add Board-Member Normalization and Aggregation Helpers

**Files:**
- Modify: `capstone_rag_project/src/Knowledge graph/kg.py`
- Test: `capstone_rag_project/tests/test_kg.py`

- [ ] **Step 1: Write the failing tests for board-member normalization and aggregation**

```python
def test_board_name_normalization():
    kg = load_kg_module()
    assert kg.normalize_board_member_name("Art Levinson Board Chair") == "Art Levinson"


def test_board_name_no_change():
    kg = load_kg_module()
    assert kg.normalize_board_member_name("Tim Cook") == "Tim Cook"


def test_board_title_conflict_uses_most_recent():
    kg = load_kg_module()
    records = [
        {"name": "Art Levinson Chair", "title": "Chair", "years_present": [2023, 2024]},
        {"name": "Art Levinson Board Chair", "title": "Board Chair", "years_present": [2025]},
    ]

    aggregated = kg._aggregate_board_members(records, current_year=2025)

    assert aggregated == [
        {
            "id": "AAPL_art_levinson",
            "name": "Art Levinson",
            "ticker": "AAPL",
            "title": "Board Chair",
            "years_present": [2023, 2024, 2025],
            "is_current": True,
        }
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest capstone_rag_project\tests\test_kg.py -k "board_name_normalization or board_name_no_change or board_title_conflict_uses_most_recent" -v
```

Expected:

```text
FAIL ... AttributeError: module 'kg_module' has no attribute 'normalize_board_member_name'
```

- [ ] **Step 3: Implement board-member normalization and aggregation**

```python
ROLE_SUFFIXES = [
    "Lead Independent Director",
    "Board Chair",
    "Chairman",
    "Director",
    "President",
    "Chair",
    "CEO",
    "CFO",
    "COO",
]


def normalize_board_member_name(raw_name: str) -> str:
    name = re.sub(r"\s+", " ", raw_name.strip())
    for suffix in sorted(ROLE_SUFFIXES, key=len, reverse=True):
        name = re.sub(rf"\s+{re.escape(suffix)}$", "", name, flags=re.IGNORECASE)
    return name.strip()


def _aggregate_board_members(
    records: list[dict[str, Any]],
    ticker: str = "AAPL",
    current_year: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for record in records:
        normalized_name = normalize_board_member_name(str(record.get("name", "")))
        if not normalized_name:
            continue
        years = sorted({int(year) for year in record.get("years_present", [])})
        if not years:
            continue

        current = grouped.get(normalized_name)
        most_recent_year = max(years)

        if current is None:
            grouped[normalized_name] = {
                "name": normalized_name,
                "title": str(record.get("title", "")).strip(),
                "latest_year": most_recent_year,
                "years_present": set(years),
            }
            continue

        current["years_present"].update(years)
        if most_recent_year >= current["latest_year"]:
            current["title"] = str(record.get("title", "")).strip()
            current["latest_year"] = most_recent_year

    if current_year is None:
        current_year = max(
            (details["latest_year"] for details in grouped.values()),
            default=0,
        )

    aggregated: list[dict[str, Any]] = []
    for normalized_name, details in sorted(grouped.items()):
        slug = re.sub(r"[^\w]+", "_", normalized_name.lower()).strip("_")
        years_present = sorted(details["years_present"])
        aggregated.append(
            {
                "id": f"{ticker.upper()}_{slug}",
                "name": normalized_name,
                "ticker": ticker.upper(),
                "title": details["title"],
                "years_present": years_present,
                "is_current": max(years_present) == current_year,
            }
        )

    return aggregated
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
.venv\Scripts\python.exe -m pytest capstone_rag_project\tests\test_kg.py -k "board_name_normalization or board_name_no_change or board_title_conflict_uses_most_recent" -v
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

```powershell
git add capstone_rag_project/src/Knowledge\ graph/kg.py capstone_rag_project/tests/test_kg.py
git commit -m "feat: add board member normalization and aggregation"
```

## Task 4: Add Subsidiary Normalization Helper and Subsidiary Payload Coverage

**Files:**
- Modify: `capstone_rag_project/src/Knowledge graph/kg.py`
- Test: `capstone_rag_project/tests/test_kg.py`

- [ ] **Step 1: Write the failing tests for subsidiary ID normalization and yearly edge payload**

```python
def test_subsidiary_id():
    kg = load_kg_module()
    result = kg.normalize_subsidiary_id("AAPL", "Apple Operations International Limited")
    assert result == "AAPL_apple_operations_international_limited"


def test_subsidiary_one_edge_per_year():
    kg = load_kg_module()
    payload = {
        "ticker": "AAPL",
        "company_title": "Apple Inc.",
        "form_type": "10-K",
        "subsidiaries_by_year": [
            {"year": 2021, "subsidiaries": ["Apple Asia Limited"]},
            {"year": 2022, "subsidiaries": ["Apple Asia Limited"]},
            {"year": 2024, "subsidiaries": ["Apple Asia Limited"]},
        ],
    }

    relationships = kg.build_subsidiary_relationship_payload(payload, "AAPL")

    assert relationships == [
        {
            "subsidiary_id": "AAPL_apple_asia_limited",
            "name": "Apple Asia Limited",
            "ticker": "AAPL",
            "year": 2021,
            "source_form_type": "10-K",
        },
        {
            "subsidiary_id": "AAPL_apple_asia_limited",
            "name": "Apple Asia Limited",
            "ticker": "AAPL",
            "year": 2022,
            "source_form_type": "10-K",
        },
        {
            "subsidiary_id": "AAPL_apple_asia_limited",
            "name": "Apple Asia Limited",
            "ticker": "AAPL",
            "year": 2024,
            "source_form_type": "10-K",
        },
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest capstone_rag_project\tests\test_kg.py -k "subsidiary_id or subsidiary_one_edge_per_year" -v
```

Expected:

```text
FAIL ... AttributeError: module 'kg_module' has no attribute 'normalize_subsidiary_id'
```

- [ ] **Step 3: Implement subsidiary normalization and relationship payload helper**

```python
def normalize_subsidiary_id(ticker: str, name: str) -> str:
    normalized = name.lower()
    normalized = re.sub(r"[^\w]", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return f"{ticker.upper()}_{normalized}"


def build_subsidiary_relationship_payload(
    payload: dict[str, Any],
    ticker: str,
) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    form_type = str(payload.get("form_type", "")).strip()

    for yearly_record in payload.get("subsidiaries_by_year", []):
        year = yearly_record.get("year")
        for subsidiary_name in yearly_record.get("subsidiaries", []):
            relationships.append(
                {
                    "subsidiary_id": normalize_subsidiary_id(ticker, subsidiary_name),
                    "name": subsidiary_name,
                    "ticker": ticker.upper(),
                    "year": year,
                    "source_form_type": form_type,
                }
            )

    return relationships
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
.venv\Scripts\python.exe -m pytest capstone_rag_project\tests\test_kg.py -k "subsidiary_id or subsidiary_one_edge_per_year" -v
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```powershell
git add capstone_rag_project/src/Knowledge\ graph/kg.py capstone_rag_project/tests/test_kg.py
git commit -m "feat: add subsidiary normalization helpers"
```

## Task 5: Add Neo4j Schema, Ingestion Functions, and CLI Entrypoint

**Files:**
- Modify: `capstone_rag_project/src/Knowledge graph/kg.py`
- Test: `capstone_rag_project/tests/test_kg.py`

- [ ] **Step 1: Write the failing smoke-shape test for top-level builder wiring**

```python
def test_run_filings_kg_build_requires_docling_sources(tmp_path):
    kg = load_kg_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    company_file = data_dir / "aapl_10-k_docling.json"
    company_file.write_text("[]", encoding="utf-8")
    (data_dir / "aapl_10-q_docling.json").write_text("[]", encoding="utf-8")
    (data_dir / "aapl_8-k_docling.json").write_text("[]", encoding="utf-8")
    (data_dir / "aapl_10-k_subsidiaries.json").write_text('{"subsidiaries_by_year": []}', encoding="utf-8")
    (data_dir / "aapl_def14a_board_members.json").write_text('{"board_members": []}', encoding="utf-8")

    class DummyDriver:
        pass

    summary = kg.run_filings_kg_build("AAPL", str(data_dir), driver=DummyDriver())

    assert summary["ticker"] == "AAPL"
    assert summary["data_dir"] == str(data_dir)
    assert summary["docling_files"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest capstone_rag_project\tests\test_kg.py -k "run_filings_kg_build_requires_docling_sources" -v
```

Expected:

```text
FAIL ... TypeError: run_filings_kg_build() got an unexpected keyword argument 'driver'
```

- [ ] **Step 3: Implement schema creation, ingestion functions, summary logging, and CLI**

```python
def create_schema(driver) -> None:
    statements = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Filing) REQUIRE f.filing_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Subsidiary) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (b:BoardMember) REQUIRE b.id IS UNIQUE",
    ]
    with driver.session() as session:
        for statement in statements:
            session.run(statement)


def upsert_company(driver, ticker: str, name: str) -> None:
    with driver.session() as session:
        session.run(
            """
            MERGE (c:Company {ticker: $ticker})
            SET c.name = $name
            """,
            ticker=ticker.upper(),
            name=name,
        )


def ingest_docling_filings(driver, path: str, ticker: str) -> None:
    entries = load_json(path)
    if not isinstance(entries, list):
        raise ValueError(f"Expected list in {path}")
    source_file = Path(path).name

    with driver.session() as session:
        for entry in entries:
            year = extract_year(entry)
            if year is None:
                logger.warning("Skipping docling entry without year in %s", path)
                continue

            filing_id = build_filing_id(ticker, entry["form_type"], year)
            sections = extract_sections_from_docling_entry(entry, source_file)

            session.run(
                """
                MATCH (c:Company {ticker: $ticker})
                MERGE (f:Filing {filing_id: $filing_id})
                SET f.ticker = $ticker,
                    f.form_type = $form_type,
                    f.year = $year,
                    f.report_date = $report_date,
                    f.filing_date = $filing_date,
                    f.source_file = $source_file,
                    f.section_count = $section_count
                MERGE (c)-[:HAS_FILING]->(f)
                """,
                ticker=ticker.upper(),
                filing_id=filing_id,
                form_type=entry["form_type"],
                year=year,
                report_date=entry.get("report_date"),
                filing_date=entry.get("filing_date"),
                source_file=source_file,
                section_count=len(sections),
            )

            for section in sections:
                session.run(
                    """
                    MATCH (f:Filing {filing_id: $filing_id})
                    MERGE (s:Section {section_id: $section_id})
                    SET s.filing_id = $filing_id,
                        s.form_type = $form_type,
                        s.year = $year,
                        s.ordinal = $ordinal,
                        s.title = $title,
                        s.text = $text,
                        s.source_file = $source_file,
                        s.ticker = $ticker
                    MERGE (f)-[:HAS_SECTION]->(s)
                    """,
                    **section,
                )


def ingest_subsidiaries(driver, path: str, ticker: str) -> None:
    payload = load_json(path)
    relationships = build_subsidiary_relationship_payload(payload, ticker)

    with driver.session() as session:
        for relationship in relationships:
            session.run(
                """
                MATCH (c:Company {ticker: $ticker})
                MERGE (s:Subsidiary {id: $subsidiary_id})
                SET s.name = $name,
                    s.ticker = $ticker
                MERGE (c)-[r:HAS_SUBSIDIARY {year: $year}]->(s)
                SET r.source_form_type = $source_form_type
                """,
                **relationship,
            )


def ingest_board_members(driver, path: str, ticker: str) -> None:
    payload = load_json(path)
    aggregated = _aggregate_board_members(payload.get("board_members", []), ticker=ticker.upper())

    with driver.session() as session:
        for member in aggregated:
            session.run(
                """
                MATCH (c:Company {ticker: $ticker})
                MERGE (b:BoardMember {id: $id})
                SET b.name = $name,
                    b.ticker = $ticker,
                    b.title = $title,
                    b.years_present = $years_present,
                    b.is_current = $is_current
                MERGE (c)-[r:HAS_BOARD_MEMBER]->(b)
                SET r.is_current = $is_current
                """,
                **member,
            )


def log_graph_summary(driver, ticker: str) -> None:
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Company {ticker: $ticker})
            OPTIONAL MATCH (c)-[:HAS_FILING]->(f:Filing)
            OPTIONAL MATCH (f)-[:HAS_SECTION]->(s:Section)
            OPTIONAL MATCH (c)-[:HAS_SUBSIDIARY]->(sub:Subsidiary)
            OPTIONAL MATCH (c)-[:HAS_BOARD_MEMBER]->(bm:BoardMember)
            RETURN count(DISTINCT f) AS filings,
                   count(DISTINCT s) AS sections,
                   count(DISTINCT sub) AS subsidiaries,
                   count(DISTINCT bm) AS board_members
            """,
            ticker=ticker.upper(),
        ).single()
    logger.info("Graph summary for %s: %s", ticker, dict(result or {}))


def run_filings_kg_build(ticker: str, data_dir: str, driver=None) -> dict[str, Any]:
    ticker = ticker.upper()
    data_path = Path(data_dir)
    own_driver = driver is None
    if own_driver:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        create_schema(driver)
        upsert_company(driver, ticker, "Apple Inc.")
        ingest_docling_filings(driver, str(data_path / "aapl_10-k_docling.json"), ticker)
        ingest_docling_filings(driver, str(data_path / "aapl_10-q_docling.json"), ticker)
        ingest_docling_filings(driver, str(data_path / "aapl_8-k_docling.json"), ticker)
        ingest_subsidiaries(driver, str(data_path / "aapl_10-k_subsidiaries.json"), ticker)
        ingest_board_members(driver, str(data_path / "aapl_def14a_board_members.json"), ticker)
        log_graph_summary(driver, ticker)
    finally:
        if own_driver and driver is not None:
            driver.close()

    return {
        "ticker": ticker,
        "data_dir": str(data_path),
        "docling_files": 3,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    run_filings_kg_build(ticker=args.ticker, data_dir=args.data_dir)
```

- [ ] **Step 4: Run targeted tests to verify they pass**

Run:

```powershell
.venv\Scripts\python.exe -m pytest capstone_rag_project\tests\test_kg.py -v
```

Expected:

```text
all tests in test_kg.py pass
```

- [ ] **Step 5: Run a manual smoke test against real Apple data**

Run:

```powershell
.venv\Scripts\python.exe "capstone_rag_project\src\Knowledge graph\kg.py" --ticker AAPL --data-dir "capstone_rag_project\data"
```

Expected log shape:

```text
... INFO ... Graph summary for AAPL: {'filings': ..., 'sections': ..., 'subsidiaries': ..., 'board_members': ...}
```

- [ ] **Step 6: Commit**

```powershell
git add capstone_rag_project/src/Knowledge\ graph/kg.py capstone_rag_project/tests/test_kg.py
git commit -m "feat: implement standalone filings knowledge graph builder"
```

## Self-Review

Spec coverage check:

- standalone `kg.py`: covered in Task 5
- five local JSON data sources only: covered in Task 5
- year from `report_date` first: covered in Task 1
- section skip rules and title detection: covered in Task 2
- board-member normalization and `is_current`: covered in Task 3 and Task 5
- one `HAS_SUBSIDIARY` edge per year: covered in Task 4 and Task 5
- Neo4j schema and smoke test: covered in Task 5

Placeholder scan:

- no `TODO`, `TBD`, or “implement later” placeholders remain

Type consistency:

- `extract_year`, `build_filing_id`, `extract_sections_from_docling_entry`, `normalize_board_member_name`, `_aggregate_board_members`, `normalize_subsidiary_id`, and `run_filings_kg_build` are used consistently across tasks

## Task 6: Add Filtered Patents Source and Patent Domain Ingestion

**Files:**
- Modify: `capstone_rag_project/src/Knowledge graph/kg.py`
- Test: `capstone_rag_project/tests/test_kg.py`

- [ ] **Step 1: Write the failing tests for patent filtering and CPC parsing helpers**

```python
def test_parse_cpc_codes_parses_postgres_array_string():
    kg = load_kg_module("kg_module_parse_cpc_codes")

    assert kg.parse_cpc_codes("{A61B5/02,G01C22/006}") == [
        "A61B5/02",
        "G01C22/006",
    ]


def test_extract_cpc_sections_deduplicates_three_char_prefixes():
    kg = load_kg_module("kg_module_extract_cpc_sections")

    assert kg.extract_cpc_sections(
        ["A61B5/02", "H04W72/00", "H04L9/32", "G06F3/01"]
    ) == {"A61", "H04", "G06"}
```

- [ ] **Step 2: Add the failing test for filtering out 2020 rows**

```python
def test_filter_patents_csv_excludes_2020(tmp_path):
    kg = load_kg_module("kg_module_filter_patents_csv")
    raw_path = tmp_path / "patents.csv"
    filtered_path = tmp_path / "patents_2021_2025.csv"
    raw_path.write_text(
        "\\n".join(
            [
                "patent_id,patent_title,grant_date,assignee_organization,cpc_codes,citation_count,created_at",
                '1,Older patent,2020-01-07,Apple Inc.,"{A61B5/02}",0,2026-04-18',
                '2,Window patent,2021-01-05,Apple Inc.,"{H04W72/00}",0,2026-04-18',
            ]
        ),
        encoding="utf-8",
    )

    output_path = kg.filter_patents_csv(str(raw_path), str(filtered_path))

    assert Path(output_path) == filtered_path
    lines = filtered_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "Older patent" not in lines[1]
    assert "Window patent" in lines[1]
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_kg.py -k "parse_cpc_codes or extract_cpc_sections or filter_patents_csv_excludes_2020" -v
```

Expected:

```text
FAIL ... AttributeError: module 'kg_module' has no attribute 'parse_cpc_codes'
```

- [ ] **Step 4: Implement patent helpers in `kg.py`**

```python
import csv

CPC_SECTION_LABELS = {
    "A": "Human Necessities",
    "B": "Operations & Transport",
    "C": "Chemistry & Metallurgy",
    "D": "Textiles & Paper",
    "E": "Fixed Constructions",
    "F": "Mechanical Engineering",
    "G": "Physics & Computing",
    "H": "Electricity & Electronics",
}


def parse_cpc_codes(raw: str) -> list[str]:
    if not raw:
        return []
    return [code.strip() for code in raw.strip("{}").split(",") if code.strip()]


def extract_cpc_sections(codes: list[str]) -> set[str]:
    return {code[:3] for code in codes if code}


def filter_patents_csv(raw_path: str | Path, filtered_path: str | Path) -> str:
    raw_file = Path(raw_path)
    filtered_file = Path(filtered_path)

    with raw_file.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames or []
        rows = []
        for row in reader:
            grant_date = str(row.get("grant_date") or "").strip()
            if len(grant_date) < 4 or not grant_date[:4].isdigit():
                continue
            grant_year = int(grant_date[:4])
            if 2021 <= grant_year <= 2025:
                rows.append(row)

    with filtered_file.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return str(filtered_file)
```

- [ ] **Step 5: Add the failing test for patent/domain payload wiring**

```python
def test_patent_domain_payload_is_deduplicated_per_prefix():
    kg = load_kg_module("kg_module_patent_domains")
    codes = kg.parse_cpc_codes("{H04W72/00,H04L9/32,G06F3/01}")
    sections = kg.extract_cpc_sections(codes)

    assert codes == ["H04W72/00", "H04L9/32", "G06F3/01"]
    assert sections == {"H04", "G06"}
```

- [ ] **Step 6: Implement `ingest_patents(...)` and extend schema**

```python
def ingest_patents(driver, patents_csv_path: str | Path, ticker: str) -> None:
    patents_path = Path(patents_csv_path)
    with patents_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    ticker_value = ticker.upper()
    with driver.session() as session:
        for row in rows:
            patent_id = str(row.get("patent_id") or "").strip()
            title = str(row.get("patent_title") or "").strip()
            grant_date = str(row.get("grant_date") or "").strip()
            if not patent_id or not title or len(grant_date) < 4:
                continue

            grant_year = int(grant_date[:4])
            codes = parse_cpc_codes(str(row.get("cpc_codes") or ""))
            sections = extract_cpc_sections(codes)

            session.run(
                """
                MATCH (c:Company {ticker: $ticker})
                MERGE (p:Patent {patent_id: $patent_id})
                SET p.title = $title,
                    p.grant_date = $grant_date,
                    p.grant_year = $grant_year,
                    p.cpc_codes = $cpc_codes,
                    p.ticker = $ticker
                MERGE (c)-[:HAS_PATENT]->(p)
                """,
                patent_id=patent_id,
                title=title,
                grant_date=grant_date,
                grant_year=grant_year,
                cpc_codes=codes,
                ticker=ticker_value,
            )

            for prefix in sections:
                section_letter = prefix[0]
                label = CPC_SECTION_LABELS.get(section_letter, "Other")
                session.run(
                    """
                    MERGE (d:TechnologyDomain {cpc_prefix: $prefix})
                    SET d.label = $label,
                        d.section = $section_letter
                    WITH d
                    MATCH (p:Patent {patent_id: $patent_id})
                    MERGE (p)-[:BELONGS_TO_DOMAIN {cpc_prefix: $prefix}]->(d)
                    """,
                    prefix=prefix,
                    label=label,
                    section_letter=section_letter,
                    patent_id=patent_id,
                )
```

- [ ] **Step 7: Extend `run_filings_kg_build(...)` to filter and ingest patents**

```python
raw_patents_path = data_path / "patents.csv"
filtered_patents_path = data_path / "patents_2021_2025.csv"
filter_patents_csv(raw_patents_path, filtered_patents_path)
ingest_patents(driver, filtered_patents_path, ticker_value)
```

Also extend `create_schema(...)` with:

```python
"CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patent) REQUIRE p.patent_id IS UNIQUE",
"CREATE CONSTRAINT IF NOT EXISTS FOR (d:TechnologyDomain) REQUIRE d.cpc_prefix IS UNIQUE",
```

- [ ] **Step 8: Add a wiring test for patents in `run_filings_kg_build(...)`**

```python
def test_run_filings_kg_build_filters_and_ingests_patents(tmp_path):
    kg = load_kg_module("kg_module_run_builder_patents")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    for file_name, content in {
        "aapl_10-k_docling.json": "[]",
        "aapl_10-q_docling.json": "[]",
        "aapl_8-k_docling.json": "[]",
        "aapl_10-k_subsidiaries.json": '{"form_type":"10-K","subsidiaries_by_year":[]}',
        "aapl_def14a_board_members.json": '{"board_members_by_year":[]}',
        "patents.csv": "patent_id,patent_title,grant_date,assignee_organization,cpc_codes,citation_count,created_at\\n",
    }.items():
        (data_dir / file_name).write_text(content, encoding="utf-8")

    calls = []

    class DummyDriver:
        def close(self):
            raise AssertionError("Injected driver should not be closed")

    def record(name):
        def _recorder(*args, **kwargs):
            calls.append((name, args, kwargs))
        return _recorder

    with (
        patch.object(kg, "create_schema", side_effect=record("create_schema")),
        patch.object(kg, "upsert_company", side_effect=record("upsert_company")),
        patch.object(kg, "ingest_docling_filings", side_effect=record("ingest_docling_filings")),
        patch.object(kg, "ingest_subsidiaries", side_effect=record("ingest_subsidiaries")),
        patch.object(kg, "ingest_board_members", side_effect=record("ingest_board_members")),
        patch.object(kg, "filter_patents_csv", side_effect=record("filter_patents_csv")),
        patch.object(kg, "ingest_patents", side_effect=record("ingest_patents")),
        patch.object(kg, "log_graph_summary", side_effect=record("log_graph_summary")),
    ):
        summary = kg.run_filings_kg_build("AAPL", str(data_dir), driver=DummyDriver())

    assert summary["ticker"] == "AAPL"
    assert any(name == "filter_patents_csv" for name, _, _ in calls)
    assert any(name == "ingest_patents" for name, _, _ in calls)
```

- [ ] **Step 9: Run the full test file**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_kg.py -v
```

Expected:

```text
all tests in test_kg.py pass
```

- [ ] **Step 10: Update the manual smoke-test checklist**

After running:

```powershell
.venv\Scripts\python.exe "src\Knowledge graph\kg.py" --ticker AAPL --data-dir ".\data"
```

verify in Neo4j:

```cypher
MATCH (c:Company {ticker:"AAPL"})-[:HAS_PATENT]->(p:Patent)
RETURN count(p);
```

```cypher
MATCH (p:Patent)-[:BELONGS_TO_DOMAIN]->(d:TechnologyDomain)
RETURN d.cpc_prefix, d.label, count(*) AS patents
ORDER BY patents DESC;
```
