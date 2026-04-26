# Exhibit 21 Subsidiaries Extraction Design

**Date:** 2026-04-26
**Target Area:** `src/filings/ingestion.py`

---

## 1. Purpose

Add Exhibit 21 extraction to the existing SEC filings ingestion flow so the pipeline also captures subsidiaries lists for each fiscal year. The new output must be saved as JSON in `data/` and must not disturb the current 10-K ingestion, XBRL extraction, Docling conversion, validation, chunking, or RAPTOR flow.

---

## 2. Scope

- **In scope:** discover Exhibit 21 attachments from the SEC filing index, download the exhibit, parse subsidiaries, aggregate subsidiaries by fiscal year, save one JSON artifact in `data/`, add non-disruptive logging.
- **Out of scope:** using subsidiaries in chunking, Pinecone, RAPTOR, retrieval, or knowledge graph construction.

---

## 3. Existing Flow

Today `ingestion_filing()` does the following:

1. Finds the company and SEC CIK.
2. Fetches recent filings from the SEC submissions API.
3. Filters filings by form type and year.
4. Downloads filing HTML files.
5. Extracts cleaned XBRL JSON.
6. Converts filing HTML to Docling JSON.
7. Runs validation checks.

The downstream pipeline in `src/filings/pipeline.py` depends only on the current ingestion outputs, especially the Docling JSON path and XBRL JSON path.

---

## 4. Design

### 4.1 Add Exhibit 21 as a sidecar path

Exhibit 21 extraction will be added as an extra sidecar step inside the ingestion flow. It will not replace or alter the current filing HTML download path. The main filing document will continue to be downloaded from:

`https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_document}`

For the same accession, the code will also query:

`https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession_nodash}-index.json`

That index will be used to locate the `EX-21*` attachment filename, if present.

### 4.2 Preserve current ingestion behavior

The existing outputs must remain unchanged:

- `<folder>/<year>.html`
- `<folder>.json`
- `<folder>_docling.json`

Any missing Exhibit 21, malformed index JSON, or parse failure should be logged and skipped without failing the normal filings pipeline.

### 4.3 Aggregate subsidiaries across years

The new artifact should capture subsidiaries for all fiscal years already included in the filings filter. Instead of saving one output per year, the code will aggregate all available yearly results into one company-level JSON file.

Suggested filename:

- `data/<ticker>_<form-type>_subsidiaries.json`

Suggested shape:

```json
{
  "ticker": "AAPL",
  "company_title": "Apple Inc.",
  "form_type": "10-K",
  "subsidiaries_by_year": [
    {
      "year": 2025,
      "accession": "000032019325000073",
      "exhibit_21_url": "https://www.sec.gov/Archives/edgar/data/...",
      "subsidiaries": [
        "Example Subsidiary 1",
        "Example Subsidiary 2"
      ]
    }
  ]
}
```

This structure is appropriate for later knowledge graph work because it keeps the fiscal year and source exhibit tied to the extracted entities.

---

## 5. Components

### 5.1 Filing metadata enrichment

`get_filings(cik)` should keep the current fields and add enough metadata to support Exhibit 21 lookup, especially the accession without dashes.

### 5.2 Exhibit 21 lookup helper

A helper should:

1. Build the SEC filing index JSON URL.
2. Fetch the index.
3. Inspect `directory.item`.
4. Return the first filename whose `type` starts with `EX-21`.

### 5.3 Exhibit 21 fetch and parse helper

A helper should:

1. Build the exhibit URL from the discovered filename.
2. Download the exhibit HTML or text.
3. Parse likely subsidiaries from the exhibit body.
4. Normalize whitespace and remove duplicates while preserving order.

### 5.4 Data save helper

A helper should save one aggregated JSON file under `data/` for the ticker and form type.

---

## 6. Error Handling

- Missing Exhibit 21 in a filing: log and continue.
- SEC index request failure: log and continue.
- Exhibit fetch failure: log and continue.
- Parse failure for one year: log and continue.
- Empty subsidiaries list for a year: allowed.

This behavior keeps the current ingestion pipeline resilient.

---

## 7. Testing

Add focused tests for:

- Exhibit 21 filename discovery from mocked SEC index JSON.
- No-match behavior when no `EX-21` attachment exists.
- Per-year aggregation into a single output file.
- JSON structure for future knowledge graph use.
- Non-fatal handling when one year fails but others succeed.

---

## 8. Result

After the change, the filings pipeline will still produce the same current outputs, plus a new subsidiaries JSON artifact in `data/` that contains the subsidiaries list grouped by fiscal year.
