# Exhibit 21 Subsidiaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add non-disruptive Exhibit 21 extraction to the filings ingestion flow and save a multi-year subsidiaries JSON artifact in `data/`.

**Architecture:** Keep the current filing download, XBRL, and Docling pipeline unchanged. Add sidecar helpers in `src/filings/ingestion.py` to discover Exhibit 21 attachments from SEC filing indexes, fetch and parse subsidiary lists, aggregate them by fiscal year, and persist one company-level JSON file.

**Tech Stack:** Python, `requests`, `BeautifulSoup`, existing SEC ingestion utilities, local JSON persistence.

**Spec:** `docs/superpowers/specs/2026-04-26-exhibit-21-subsidiaries-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/filings/ingestion.py` | Add Exhibit 21 discovery, fetch, parse, aggregate, save, and lifecycle logging |
| Create or Modify | `tests/test_filings_ingestion.py` | Unit tests for Exhibit 21 helpers and JSON aggregation |
| Modify | `.gitignore` | Ignore generated subsidiaries output if generated artifacts are not tracked |

---

## Task 1: Extend filing metadata for Exhibit 21 lookup

**Files:**
- Modify: `src/filings/ingestion.py`

- [ ] **Step 1: Add a failing test for filing metadata**

Add a test that verifies each filing record preserves:

- `year`
- `accession`
- `primary_doc`
- `filing_url`

and also exposes the accession in a way the Exhibit 21 lookup can reuse directly.

- [ ] **Step 2: Run the targeted test to confirm the current gap**

Run:

```bash
uv run pytest tests/test_filings_ingestion.py -k filings_metadata -v
```

- [ ] **Step 3: Update `get_filings(cik)`**

Keep the current return format, but make sure each filing entry includes stable fields needed for Exhibit 21 lookup, such as:

- `accession`
- `filing_url`

No existing keys should be removed or renamed.

- [ ] **Step 4: Re-run the targeted test**

Run:

```bash
uv run pytest tests/test_filings_ingestion.py -k filings_metadata -v
```

- [ ] **Step 5: Commit**

```bash
git add src/filings/ingestion.py tests/test_filings_ingestion.py
git commit -m "test(filings): preserve filing metadata for Exhibit 21 lookup"
```

---

## Task 2: Add Exhibit 21 index discovery helper

**Files:**
- Modify: `src/filings/ingestion.py`
- Modify: `tests/test_filings_ingestion.py`

- [ ] **Step 1: Write failing tests for index lookup**

Add tests that verify:

- the SEC index JSON is queried with `{accession_nodash}-index.json`
- the first attachment with `type` starting with `EX-21` is selected
- `None` is returned if no matching attachment exists

- [ ] **Step 2: Run the targeted tests**

Run:

```bash
uv run pytest tests/test_filings_ingestion.py -k exhibit_21_lookup -v
```

- [ ] **Step 3: Implement a helper in `ingestion.py`**

Add a focused helper that:

1. Builds the index URL.
2. Fetches the index JSON.
3. Reads `directory.item`.
4. Returns the Exhibit 21 filename or `None`.

This helper must log failures and return `None` instead of raising.

- [ ] **Step 4: Re-run the targeted tests**

Run:

```bash
uv run pytest tests/test_filings_ingestion.py -k exhibit_21_lookup -v
```

- [ ] **Step 5: Commit**

```bash
git add src/filings/ingestion.py tests/test_filings_ingestion.py
git commit -m "feat(filings): add Exhibit 21 attachment discovery"
```

---

## Task 3: Add Exhibit 21 fetch and subsidiaries parsing

**Files:**
- Modify: `src/filings/ingestion.py`
- Modify: `tests/test_filings_ingestion.py`

- [ ] **Step 1: Write failing parser tests**

Add tests using a representative Exhibit 21 HTML snippet that verify:

- subsidiary names are extracted as a list
- duplicates are removed
- empty or malformed content returns an empty list instead of failing

- [ ] **Step 2: Run the targeted tests**

Run:

```bash
uv run pytest tests/test_filings_ingestion.py -k subsidiaries_parser -v
```

- [ ] **Step 3: Implement helpers**

Add helpers that:

- build the exhibit URL from `cik`, `accession`, and the exhibit filename
- fetch the exhibit content
- parse subsidiaries from HTML/text using BeautifulSoup and conservative text cleanup

The implementation should prefer stable parsing over aggressive heuristics.

- [ ] **Step 4: Re-run the targeted tests**

Run:

```bash
uv run pytest tests/test_filings_ingestion.py -k subsidiaries_parser -v
```

- [ ] **Step 5: Commit**

```bash
git add src/filings/ingestion.py tests/test_filings_ingestion.py
git commit -m "feat(filings): parse subsidiaries from Exhibit 21"
```

---

## Task 4: Aggregate subsidiaries by fiscal year and save JSON

**Files:**
- Modify: `src/filings/ingestion.py`
- Modify: `tests/test_filings_ingestion.py`

- [ ] **Step 1: Write failing aggregation tests**

Add tests that verify:

- multiple yearly filings are aggregated into one JSON payload
- each yearly entry contains `year`, `accession`, `exhibit_21_url`, and `subsidiaries`
- output is saved under `data/`
- years with missing Exhibit 21 do not break the full output

- [ ] **Step 2: Run the targeted tests**

Run:

```bash
uv run pytest tests/test_filings_ingestion.py -k subsidiaries_aggregation -v
```

- [ ] **Step 3: Implement aggregation and save helpers**

Add logic that:

1. Iterates through the already filtered filings.
2. Looks up Exhibit 21 per year.
3. Fetches and parses subsidiaries if available.
4. Builds a company-level structure with `subsidiaries_by_year`.
5. Writes `data/<ticker>_<form-type>_subsidiaries.json`.

- [ ] **Step 4: Re-run the targeted tests**

Run:

```bash
uv run pytest tests/test_filings_ingestion.py -k subsidiaries_aggregation -v
```

- [ ] **Step 5: Commit**

```bash
git add src/filings/ingestion.py tests/test_filings_ingestion.py
git commit -m "feat(filings): save multi-year subsidiaries artifact"
```

---

## Task 5: Integrate the sidecar step into `ingestion_filing()`

**Files:**
- Modify: `src/filings/ingestion.py`
- Modify: `tests/test_filings_ingestion.py`

- [ ] **Step 1: Write a failing orchestration test**

Add a test that verifies:

- normal XBRL and Docling steps are still called
- Exhibit 21 extraction runs as an additional step
- missing Exhibit 21 data does not stop the existing flow
- the returned payload can include `subsidiaries_json_path` without breaking current fields

- [ ] **Step 2: Run the targeted tests**

Run:

```bash
uv run pytest tests/test_filings_ingestion.py -k ingestion_filing_exhibit_21 -v
```

- [ ] **Step 3: Integrate into `ingestion_filing()`**

Hook the new sidecar step into the current flow after filing discovery and before final return. Keep the existing return keys intact and add the subsidiaries artifact path only as an extra field.

- [ ] **Step 4: Re-run the targeted tests**

Run:

```bash
uv run pytest tests/test_filings_ingestion.py -k ingestion_filing_exhibit_21 -v
```

- [ ] **Step 5: Commit**

```bash
git add src/filings/ingestion.py tests/test_filings_ingestion.py
git commit -m "feat(filings): integrate Exhibit 21 extraction into ingestion flow"
```

---

## Task 6: Verification and cleanup

**Files:**
- Modify: `.gitignore` if needed

- [ ] **Step 1: Decide generated file policy**

If generated `data/*_subsidiaries.json` files should not be committed, add an ignore rule such as:

```gitignore
data/*_subsidiaries.json
```

If they should be tracked, skip this step.

- [ ] **Step 2: Run the focused filings test suite**

Run:

```bash
uv run pytest tests/test_filings_ingestion.py -v
```

- [ ] **Step 3: Run the broader repo tests that touch filings ingestion**

Run:

```bash
uv run pytest tests -k filings -v
```

- [ ] **Step 4: Manual smoke check**

Run one real ingestion flow for a known company and confirm:

- existing XBRL JSON still writes successfully
- existing Docling JSON still writes successfully
- `data/<ticker>_<form-type>_subsidiaries.json` is created
- the JSON includes yearly subsidiaries entries

- [ ] **Step 5: Final commit**

```bash
git add .gitignore src/filings/ingestion.py tests/test_filings_ingestion.py
git commit -m "test(filings): verify Exhibit 21 subsidiaries sidecar flow"
```

---

## Self-Review

- Current ingestion outputs remain unchanged.
- Exhibit 21 is additive and non-fatal.
- Output is aggregated by fiscal year for later knowledge graph work.
- Downstream chunking and RAPTOR consumers remain untouched.
