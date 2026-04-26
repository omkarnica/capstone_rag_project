# DEF 14A Board Members Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `DEF 14A` sidecar flow that extracts board member names and titles, aggregates `years_present`, and saves a JSON artifact in `data/` without affecting Pinecone ingestion.

**Architecture:** Reuse the existing SEC filings manifest, add a proxy-filings selector and parser in `src/filings/ingestion.py`, build both yearly and aggregated member views, and persist one company-level JSON file under `data/`.

**Tech Stack:** Python, `requests`, `BeautifulSoup`, existing filings ingestion helpers, JSON persistence.

**Spec:** `docs/superpowers/specs/2026-04-26-def14a-board-members-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/filings/ingestion.py` | Add `DEF 14A` filtering, download, parsing, aggregation, and save helpers |
| Modify | `tests/test_filings_ingestion.py` | Add focused tests for proxy sidecar behavior |

---

## Task 1: Add `DEF 14A` filing selection

**Files:**
- Modify: `src/filings/ingestion.py`
- Modify: `tests/test_filings_ingestion.py`

- [ ] **Step 1: Write the failing test**

Add a test verifying that a helper correctly filters:

- `form == "DEF 14A"`
- `year` present
- year between `2021` and `2025`

- [ ] **Step 2: Run the targeted test to confirm it fails**

```bash
uv run pytest tests/test_filings_ingestion.py -k def14a_filter -v
```

- [ ] **Step 3: Implement the selector helper**

Add a focused helper that returns only eligible `DEF 14A` filings.

- [ ] **Step 4: Re-run the targeted test**

```bash
uv run pytest tests/test_filings_ingestion.py -k def14a_filter -v
```

---

## Task 2: Parse board members from proxy HTML

**Files:**
- Modify: `src/filings/ingestion.py`
- Modify: `tests/test_filings_ingestion.py`

- [ ] **Step 1: Write failing parser tests**

Add tests for representative `DEF 14A` HTML that verify:

- member `name` is extracted
- `title` is extracted
- duplicates are removed within a year

- [ ] **Step 2: Run the targeted tests**

```bash
uv run pytest tests/test_filings_ingestion.py -k def14a_parser -v
```

- [ ] **Step 3: Implement proxy parsing helpers**

Add helpers that:

1. fetch proxy filing HTML
2. parse likely board-member rows conservatively
3. return a list of `{name, title}`

- [ ] **Step 4: Re-run the targeted tests**

```bash
uv run pytest tests/test_filings_ingestion.py -k def14a_parser -v
```

---

## Task 3: Aggregate `years_present` and save JSON

**Files:**
- Modify: `src/filings/ingestion.py`
- Modify: `tests/test_filings_ingestion.py`

- [ ] **Step 1: Write failing aggregation tests**

Add tests that verify:

- the artifact is written to `data/<ticker>_def14a_board_members.json`
- `board_members_by_year` preserves per-year source data
- `board_members` aggregates `years_present`

- [ ] **Step 2: Run the targeted tests**

```bash
uv run pytest tests/test_filings_ingestion.py -k def14a_aggregation -v
```

- [ ] **Step 3: Implement aggregation and save helpers**

Add logic that:

1. downloads each `DEF 14A`
2. extracts yearly members
3. merges members by normalized name
4. builds `years_present`
5. writes final JSON in `data/`

- [ ] **Step 4: Re-run the targeted tests**

```bash
uv run pytest tests/test_filings_ingestion.py -k def14a_aggregation -v
```

---

## Task 4: Integrate the sidecar into ingestion flow

**Files:**
- Modify: `src/filings/ingestion.py`
- Modify: `tests/test_filings_ingestion.py`

- [ ] **Step 1: Write the failing orchestration test**

Add a test proving:

- `DEF 14A` extraction runs as a sidecar
- returned ingestion metadata can include a proxy board JSON path
- no Pinecone-facing flow changes are introduced

- [ ] **Step 2: Run the targeted tests**

```bash
uv run pytest tests/test_filings_ingestion.py -k def14a_sidecar -v
```

- [ ] **Step 3: Integrate into `ingestion_filing()`**

Hook the proxy sidecar in without changing Docling, chunking, RAPTOR, or Pinecone inputs.

- [ ] **Step 4: Re-run the targeted tests**

```bash
uv run pytest tests/test_filings_ingestion.py -k def14a_sidecar -v
```

---

## Task 5: Verification

**Files:**
- Modify: none unless tests uncover issues

- [ ] **Step 1: Run the full focused filings tests**

```bash
uv run pytest tests/test_filings_ingestion.py tests/test_filings_pipeline.py -v
```

- [ ] **Step 2: Manual smoke check**

Run a company proxy extraction and confirm:

- JSON is written in `data/`
- `board_members` includes `name`, `title`, `years_present`
- no new proxy artifact feeds into Pinecone

