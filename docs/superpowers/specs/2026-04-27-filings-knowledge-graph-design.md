# Knowledge Graph Builder Design

File: `kg.py`
Date: 2026-04-27
Scope: Apple Inc. (`AAPL`) knowledge graph from five filings JSON sources plus one filtered patents CSV source, persisted to Neo4j.

## Goal

Build the Apple knowledge graph in [kg.py](C:/Users/nithu/Desktop/GenAI/RAG%20Capstone%20-%20Finance/capstone_rag_project/src/Knowledge%20graph/kg.py) from five local filings JSON sources plus one filtered patents CSV source and persist it to Neo4j.

`kg.py` runs as a standalone script. It is not called by `ingestion_filing()` and does not depend on Pinecone at runtime.

## In-Scope Data Sources

| File | Produces |
| --- | --- |
| `data/aapl_10-k_docling.json` | Filing + Section nodes (`10-K`) |
| `data/aapl_10-q_docling.json` | Filing + Section nodes (`10-Q`) |
| `data/aapl_8-k_docling.json` | Filing + Section nodes (`8-K`) |
| `data/aapl_10-k_subsidiaries.json` | Subsidiary nodes + `HAS_SUBSIDIARY` edges |
| `data/aapl_def14a_board_members.json` | BoardMember nodes + `HAS_BOARD_MEMBER` edges |
| `data/patents_2021_2025.csv` | Patent nodes + TechnologyDomain nodes |

## Out of Scope (v1)

- Pinecone namespaces: `litigation`, `transcripts`, `filings`
- RAPTOR summary nodes
- XBRL financial facts
- Any company other than `AAPL` in the first run

## Graph Model

### Node Labels

- `Company`
- `Filing`
- `Section`
- `Subsidiary`
- `BoardMember`
- `Patent`
- `TechnologyDomain`

### Relationships

- `(:Company)-[:HAS_FILING]->(:Filing)`
- `(:Filing)-[:HAS_SECTION]->(:Section)`
- `(:Company)-[:HAS_SUBSIDIARY {year: int, source_form_type: str}]->(:Subsidiary)`
- `(:Company)-[:HAS_BOARD_MEMBER {is_current: bool}]->(:BoardMember)`
- `(:Company)-[:HAS_PATENT]->(:Patent)`
- `(:Patent)-[:BELONGS_TO_DOMAIN {cpc_prefix: str}]->(:TechnologyDomain)`

## Canonical ID Design

### Company

- Key: `ticker`
- Example: `AAPL`

### Filing

Pattern: `{TICKER}_{FORM_TYPE}_{YEAR}`

Examples:

- `AAPL_10-K_2021`
- `AAPL_10-Q_2024`
- `AAPL_8-K_2025`

Year is derived from `report_date`.

### Section

Pattern: `{filing_id}_section_{ordinal}`

Examples:

- `AAPL_10-K_2024_section_0`
- `AAPL_10-K_2024_section_12`

Ordinal is the zero-based position of the text block within the filing entry text stream.

### Subsidiary

Pattern: `{ticker}_{normalized_name}`

Examples:

- `AAPL_apple_operations_international_limited`
- `AAPL_apple_distribution_international_limited`

Display `name` retains original casing.

### BoardMember

Pattern: `{ticker}_{normalized_person_name}`

Examples:

- `AAPL_art_levinson`
- `AAPL_tim_cook`

### Patent

Use the raw patent identifier as the unique graph key:

- `10524670`
- `12000001`

### TechnologyDomain

Use the 3-character CPC section prefix:

- `A61`
- `G06`
- `H04`

## Year Extraction

Year is derived from `report_date`, not `filing_date`.

Rule:

```python
if entry.get("report_date"):
    year = int(entry["report_date"][:4])
elif entry.get("filing_date"):
    year = int(entry["filing_date"][:4])
else:
    year = None
```

If both dates are missing, log a warning and skip the entry.

Why this matters:

- `filing_date` is when the SEC received the document
- `report_date` is the period the filing covers

Using `report_date` keeps fiscal labeling correct even when a filing is submitted in the following calendar year.

## Docling JSON Structure

Each docling file is a list of yearly filing entries. Each entry includes:

- `source`
- `ticker`
- `year`
- `form_type`
- `report_date`
- `filing_date`
- `company_title`
- `docling`

The `docling.texts` stream is the source for `Section` nodes.

### Key Text Fields

| Field | Purpose |
| --- | --- |
| `text` | canonical content to store |
| `label` | used with heuristics for title detection |
| `self_ref` | used to derive the ordinal |
| `parent.$ref` | distinguishes body items from grouped content |
| `hyperlink` | present on table-of-contents items and should be skipped |

### Section Extraction Rules

Walk `entry["docling"]["texts"]` in order.

Include a text item only if all are true:

- `len(item["text"].strip()) >= 50`
- `item.get("content_layer") == "body"`
- `item.get("hyperlink")` is absent
- text does not match the page-stamp pattern:
  - `r"^Apple Inc\. \| \d{4} Form \d+.*\| \d+$"`

Title detection heuristic:

- `len(text) <= 120`
- text does not end with `.` or `,`
- text matches `r"^Item\s+\d+[\w\.]*"` or is short and title-cased

Tables are excluded from `Section` nodes in v1.

## Normalization Rules

### Board Member Name Cleanup

Apply deterministic cleanup before ID generation and merge:

```python
ROLE_SUFFIXES = [
    "Chair", "Board Chair", "Chairman", "Director",
    "CEO", "CFO", "COO", "President", "Lead Independent Director"
]

def normalize_board_member_name(raw_name: str) -> str:
    name = raw_name.strip()
    name = re.sub(r"\s+", " ", name)
    for suffix in sorted(ROLE_SUFFIXES, key=len, reverse=True):
        pattern = rf"\s+{re.escape(suffix)}$"
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    return name.strip()
```

Examples:

- `Art Levinson Chair` -> `Art Levinson`
- `Art Levinson Board Chair` -> `Art Levinson`
- `Tim Cook` -> `Tim Cook`
- `James Bell Lead Independent Director` -> `James Bell`

### Board Member Title Conflict Resolution

If the same normalized person appears with different titles across years:

- sort records by year descending
- take the title from the most recent year
- accumulate all `years_present`
- compute `is_current` as `max(years_present) == current_year`

Example aggregation result:

```python
{
    "name": "Art Levinson",
    "title": "Board Chair",
    "years_present": [2021, 2022, 2023, 2024, 2025],
    "is_current": True,
}
```

### Subsidiary ID Normalization

```python
def normalize_subsidiary_id(ticker: str, name: str) -> str:
    normalized = name.lower()
    normalized = re.sub(r"[^\w]", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    return f"{ticker.upper()}_{normalized}"
```

Display `name` keeps the original casing and punctuation.

### Subsidiary Edge Multiplicity

Create one `HAS_SUBSIDIARY` relationship per year per subsidiary.

If a subsidiary appears in 2021, 2022, and 2024, it gets three edges.

### Patent CSV Filtering

Keep the raw source file unchanged:

- `data/patents.csv`

Create a filtered file separately:

- `data/patents_2021_2025.csv`

Filtering rule:

- include only rows where `grant_date` year is between `2021` and `2025`, inclusive
- exclude all `2020` rows

This preserves the original patent source while making the KG ingestion window explicit.

### CPC Parsing

The `cpc_codes` column is stored as a PostgreSQL-array-like string:

```python
"{A61B5/02,G01C22/006}"
```

It should be parsed into a list:

```python
["A61B5/02", "G01C22/006"]
```

Recommended helpers:

```python
def parse_cpc_codes(raw: str) -> list[str]:
    if not raw:
        return []
    return [c.strip() for c in raw.strip("{}").split(",") if c.strip()]


def extract_cpc_sections(codes: list[str]) -> set[str]:
    return {code[:3] for code in codes if code}
```

One patent may have many raw CPC codes, but it should create only one domain edge per unique 3-character CPC prefix.

### Technology Domain Labels

Map the first CPC section letter to a human-readable label:

```python
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
```

## Neo4j Constraints

```cypher
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company)     REQUIRE c.ticker IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (f:Filing)      REQUIRE f.filing_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section)     REQUIRE s.section_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (s:Subsidiary)  REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (b:BoardMember) REQUIRE b.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patent)      REQUIRE p.patent_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:TechnologyDomain) REQUIRE d.cpc_prefix IS UNIQUE;
```

## Node Properties

### Company

| Property | Type | Example |
| --- | --- | --- |
| `ticker` | `str` | `AAPL` |
| `name` | `str` | `Apple Inc.` |

### Filing

| Property | Type | Example |
| --- | --- | --- |
| `filing_id` | `str` | `AAPL_10-K_2021` |
| `ticker` | `str` | `AAPL` |
| `form_type` | `str` | `10-K` |
| `year` | `int` | `2021` |
| `report_date` | `str` | `2021-09-25` |
| `filing_date` | `str` | `2021-10-29` |
| `source_file` | `str` | `aapl_10-k_docling.json` |
| `section_count` | `int` | `47` |

### Section

| Property | Type | Example |
| --- | --- | --- |
| `section_id` | `str` | `AAPL_10-K_2021_section_12` |
| `filing_id` | `str` | `AAPL_10-K_2021` |
| `form_type` | `str` | `10-K` |
| `year` | `int` | `2021` |
| `ordinal` | `int` | `12` |
| `title` | `str \| None` | `Foreign Currency Risk` |
| `text` | `str` | full paragraph text |
| `source_file` | `str` | `aapl_10-k_docling.json` |

### Subsidiary

| Property | Type | Example |
| --- | --- | --- |
| `id` | `str` | `AAPL_apple_operations_international_limited` |
| `name` | `str` | `Apple Operations International Limited` |
| `ticker` | `str` | `AAPL` |
| `jurisdiction` | `str` | `Ireland` |

### `HAS_SUBSIDIARY`

| Property | Type | Example |
| --- | --- | --- |
| `year` | `int` | `2023` |
| `source_form_type` | `str` | `10-K` |

### BoardMember

| Property | Type | Example |
| --- | --- | --- |
| `id` | `str` | `AAPL_art_levinson` |
| `name` | `str` | `Art Levinson` |
| `ticker` | `str` | `AAPL` |
| `title` | `str` | `Board Chair` |
| `years_present` | `list[int]` | `[2021, 2022, 2023, 2024, 2025]` |
| `is_current` | `bool` | `True` |

### `HAS_BOARD_MEMBER`

| Property | Type | Example |
| --- | --- | --- |
| `is_current` | `bool` | `True` |

### Patent

| Property | Type | Example |
| --- | --- | --- |
| `patent_id` | `str` | `10524670` |
| `title` | `str` | `Accurate calorimetry for intermittent exercises` |
| `grant_date` | `str` | `2021-01-05` |
| `grant_year` | `int` | `2021` |
| `cpc_codes` | `list[str]` | `["A61B5/02", "G01C22/006"]` |
| `ticker` | `str` | `AAPL` |

Dropped patent fields:

- `citation_count`
- `created_at`
- `assignee_organization`

### TechnologyDomain

| Property | Type | Example |
| --- | --- | --- |
| `cpc_prefix` | `str` | `H04` |
| `label` | `str` | `Electricity & Electronics` |
| `section` | `str` | `H` |

## Module Design for `kg.py`

`kg.py` is a standalone script that reads from `data/` and writes to Neo4j.

### Functions

```python
def load_json(path: str) -> list | dict

def create_schema(driver)

def upsert_company(driver, ticker: str, name: str)

def ingest_docling_filings(driver, path: str, ticker: str)

def extract_sections_from_docling_entry(entry: dict) -> list[dict]

def _extract_ordinal(self_ref: str) -> int

def _is_title_block(text: str) -> bool

def _is_page_stamp(text: str) -> bool

def ingest_subsidiaries(driver, path: str, ticker: str)

def normalize_subsidiary_id(ticker: str, name: str) -> str

def ingest_board_members(driver, path: str, ticker: str)

def normalize_board_member_name(raw_name: str) -> str

def _aggregate_board_members(records: list[dict]) -> list[dict]

def filter_patents_csv(raw_path: str, filtered_path: str) -> str

def parse_cpc_codes(raw: str) -> list[str]

def extract_cpc_sections(codes: list[str]) -> set[str]

def ingest_patents(driver, patents_csv_path: str, ticker: str)

def log_graph_summary(driver, ticker: str)

def run_filings_kg_build(ticker: str, data_dir: str)
```

## Data Flow

1. `create_schema()`
2. `upsert_company()`
3. `ingest_docling_filings()` for `10-K`, `10-Q`, and `8-K`
4. `ingest_subsidiaries()`
5. `ingest_board_members()`
6. `filter_patents_csv()`
7. `ingest_patents()`
8. `log_graph_summary()`

More explicitly:

```text
create_schema()
    -> upsert_company()
    -> ingest_docling_filings() x3
       -> derive year from report_date
       -> build filing_id = AAPL_{FORM}_{YEAR}
       -> upsert Filing
       -> create HAS_FILING
       -> extract_sections_from_docling_entry()
          -> skip short items, TOC links, page stamps
          -> detect title blocks
          -> upsert Section + HAS_SECTION
    -> ingest_subsidiaries()
       -> one HAS_SUBSIDIARY edge per (subsidiary, year)
    -> ingest_board_members()
       -> aggregate multi-year records
       -> normalize person names
       -> upsert BoardMember + HAS_BOARD_MEMBER {is_current}
    -> filter_patents_csv()
       -> write patents_2021_2025.csv from patents.csv
    -> ingest_patents()
       -> parse CPC arrays
       -> create Patent nodes
       -> create HAS_PATENT
       -> create TechnologyDomain nodes
       -> create BELONGS_TO_DOMAIN per unique CPC prefix
    -> log_graph_summary()
```

## Invocation

`kg.py` is run directly:

```bash
python kg.py
python kg.py --ticker AAPL --data-dir ./data
```

Environment variables:

- `NEO4J_URI=bolt://localhost:7687`
- `NEO4J_USER=neo4j`
- `NEO4J_PASSWORD=your_password`

Recommended `__main__`:

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    run_filings_kg_build(ticker=args.ticker, data_dir=args.data_dir)
```

## Error Handling

| Situation | Behavior |
| --- | --- |
| File not found | `logger.warning(...)`, skip and continue |
| Invalid JSON | `raise ValueError(f"Invalid JSON in {path}")` |
| Empty data list | `logger.info(...)`, skip and continue |
| `report_date` missing | fall back to `filing_date[:4]`; if both absent, skip entry with warning |
| patent `grant_date` outside `2021-2025` | exclude from filtered CSV |
| Neo4j write failure | surface the exception |
| Section text under 50 chars | skip silently |
| Page stamp detected | skip silently |
| TOC link detected | skip silently |

## Unit Tests

Pure-function test coverage should include:

- `test_filing_id_generation`
- `test_year_from_report_date`
- `test_board_name_normalization`
- `test_board_name_no_change`
- `test_board_title_conflict`
- `test_board_years_accumulated`
- `test_subsidiary_id`
- `test_section_extraction_skips_page_stamp`
- `test_section_extraction_skips_toc`
- `test_section_extraction_skips_short`
- `test_title_detection`
- `test_title_detection_long`
- `test_subsidiary_one_edge_per_year`
- `test_filter_patents_csv_excludes_2020`
- `test_parse_cpc_codes`
- `test_extract_cpc_sections`
- `test_patent_domain_deduplication`

## Smoke Test

After `run_filings_kg_build("AAPL", "data/")`, verify:

```python
assert node_count("Company", ticker="AAPL") == 1
assert node_count("Filing", ticker="AAPL") >= 3
assert node_count("Section", ticker="AAPL") >= 100
assert node_count("Subsidiary") >= 1
assert node_count("BoardMember") >= 1
assert edge_count("HAS_FILING", ticker="AAPL") >= 3
assert edge_count("HAS_SECTION") >= 100
assert edge_count("HAS_SUBSIDIARY") >= 1
assert edge_count("HAS_BOARD_MEMBER") >= 1
assert node_count("Patent", ticker="AAPL") >= 1
assert node_count("TechnologyDomain") >= 1
assert edge_count("HAS_PATENT", ticker="AAPL") >= 1
assert edge_count("BELONGS_TO_DOMAIN") >= 1
```

## Future Extensions

Deferred explicitly:

- XBRL financial facts -> `FinancialFact`
- RAPTOR chunk bridge -> `RaptorChunk`
- litigation ingestion
- competitor edges
- multi-company support

## Why This Design

This design keeps the first graph build factual, explainable, and debuggable.

Local JSON files are the source of truth, not Pinecone. RAPTOR summary nodes stay out of the graph. Section extraction remains intentionally simple: walk the docling text stream, skip noise, preserve usable structure, and keep the script easy to rebuild and inspect independently from the ingestion pipeline.
