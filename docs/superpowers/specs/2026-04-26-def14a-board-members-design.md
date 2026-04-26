# DEF 14A Board Members Extraction Design

**Date:** 2026-04-26
**Target Area:** `src/filings/ingestion.py`

---

## 1. Purpose

Add a `DEF 14A` sidecar extraction flow to the filings ingestion pipeline so board member data can be captured across fiscal years and saved as JSON in `data/`. This output will support later graph-building work and must not affect the existing Pinecone/RAPTOR filings ingestion path.

---

## 2. Scope

- **In scope:** fetch `DEF 14A` filings from the SEC submissions manifest, download proxy filing content, extract board member names and titles, aggregate `years_present`, save JSON in `data/`.
- **Out of scope:** sending proxy data to Docling chunking, RAPTOR, Pinecone, or retrieval; extracting compensation, committee membership, or current-board status.

---

## 3. Existing Context

The filings ingestion pipeline already has one additive sidecar flow for Exhibit 21 subsidiaries:

- `10-K` HTML download still feeds XBRL + Docling + chunking + RAPTOR
- subsidiaries are saved separately as JSON under `data/`
- subsidiaries do **not** go into Pinecone

The new `DEF 14A` feature should follow the same pattern.

---

## 4. Design

### 4.1 Filing selection

From the existing SEC submissions manifest, collect:

```python
proxy_filings = [
    f for f in filings
    if f["form"] == "DEF 14A"
    and f["year"] is not None
    and 2021 <= f["year"] <= 2025
]
```

This selection should be independent from the main `10-K` / `10-Q` ingestion path.

### 4.2 Output shape

Save one JSON file per company:

- `data/<ticker>_def14a_board_members.json`

Recommended structure:

```json
{
  "ticker": "MSFT",
  "company_title": "Microsoft Corporation",
  "form_type": "DEF 14A",
  "board_members": [
    {
      "name": "Satya Nadella",
      "title": "Chairman & CEO",
      "years_present": [2021, 2022, 2023, 2024, 2025]
    }
  ],
  "board_members_by_year": [
    {
      "year": 2025,
      "members": [
        {
          "name": "Satya Nadella",
          "title": "Chairman & CEO"
        }
      ]
    }
  ]
}
```

### 4.3 Extracted fields

Keep only:

- `name`
- `title`
- `years_present`

No `is_current`, `independent`, or committee fields for this version.

### 4.4 Dual view

Keep both:

1. `board_members`
   - normalized member-centric view for graph construction
2. `board_members_by_year`
   - year-centric evidence view that preserves source filing context

This avoids recomputing year membership later.

---

## 5. Components

### 5.1 Proxy filings collector

Add a helper that filters `DEF 14A` rows from the existing `filings` list.

### 5.2 Proxy downloader/parser

Add a helper that:

1. downloads the `DEF 14A` filing HTML
2. identifies likely board-member tables or sections
3. extracts member names and titles conservatively

### 5.3 Aggregator

Add a helper that:

1. builds yearly member lists
2. merges members by normalized name
3. accumulates `years_present`
4. writes final JSON to `data/`

---

## 6. Non-Goals and Boundaries

- Do not include `DEF 14A` data in Docling chunking
- Do not include `DEF 14A` data in Pinecone
- Do not modify current `10-K` or `10-Q` RAPTOR behavior
- Do not add live “current board” checks

---

## 7. Error Handling

- Missing `DEF 14A` for a year: log and continue
- Parse failure for one year: log and continue
- Empty member list: allowed

This feature should be non-fatal like the subsidiaries sidecar.

---

## 8. Testing

Add tests for:

- filtering `DEF 14A` filings correctly
- parsing member names/titles from representative HTML
- aggregating `years_present`
- writing the final JSON structure in `data/`
- confirming this sidecar does not affect the existing Pinecone path
