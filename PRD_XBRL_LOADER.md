# PRD: XBRL Loader → Postgres

## Goal
Build a Python pipeline that downloads SEC XBRL bulk data, 
parses it, normalizes tags, and loads it into a local PostgreSQL 
database so financial queries like "What was Apple's revenue in FY2024?" 
can be answered with a simple SQL query.

## Input
SEC quarterly bulk ZIP files from:
https://www.sec.gov/dera/data/financial-statements
Files inside: num.txt, sub.txt (tab-separated, despite .txt extension)

## Output
A populated PostgreSQL database with 3 tables:
1. `filings` — company/filing metadata
2. `facts` — financial data points
3. `tag_normalization` — maps XBRL tags to canonical names

## Database (Local Dev)
- Use local PostgreSQL
- DB name: ma_oracle
- Create a db_config.py that reads connection params from environment 
  variables with sensible local defaults
- DO NOT hardcode passwords

## File Structure to Create

xbrl/
├── init.py
├── downloader.py      # Downloads quarterly ZIPs from SEC
├── parser.py          # Reads num.txt + sub.txt into dataframes
├── normalizer.py      # Maps raw tags to canonical_tag
├── loader.py          # Bulk inserts into Postgres
├── schema.sql         # SQL to create tables + indexes
├── tag_map.py         # The canonical tag mapping dict
└── main.py            # Orchestrates the full pipeline

## Schema
```sql
CREATE TABLE IF NOT EXISTS filings (
    adsh          TEXT PRIMARY KEY,
    cik           INTEGER NOT NULL,
    name          TEXT NOT NULL,
    sic           INTEGER,
    form          TEXT,
    period        DATE,
    fiscal_year   INTEGER,
    fiscal_period TEXT,
    filed         DATE
);

CREATE TABLE IF NOT EXISTS facts (
    id      SERIAL PRIMARY KEY,
    adsh    TEXT NOT NULL REFERENCES filings(adsh),
    tag     TEXT NOT NULL,
    version TEXT,
    ddate   DATE NOT NULL,
    qtrs    INTEGER,
    uom     TEXT,
    value   NUMERIC
);

CREATE TABLE IF NOT EXISTS tag_normalization (
    raw_tag       TEXT PRIMARY KEY,
    canonical_tag TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_adsh  ON facts(adsh);
CREATE INDEX IF NOT EXISTS idx_facts_tag   ON facts(tag);
CREATE INDEX IF NOT EXISTS idx_facts_ddate ON facts(ddate);
CREATE INDEX IF NOT EXISTS idx_filings_cik  ON filings(cik);
CREATE INDEX IF NOT EXISTS idx_filings_name ON filings(name);
```

## Tag Normalization Rules
Map these raw tags to canonical names at minimum:

Revenue:
- Revenues → Revenue
- RevenueFromContractWithCustomerExcludingAssessedTax → Revenue
- RevenueFromContractWithCustomerIncludingAssessedTax → Revenue

NetIncome:
- NetIncomeLoss → NetIncome
- ProfitLoss → NetIncome
- NetIncomeLossAvailableToCommonStockholdersBasic → NetIncome

StockholdersEquity:
- StockholdersEquity → StockholdersEquity
- StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest → StockholdersEquity

OperatingIncome:
- OperatingIncomeLoss → OperatingIncome

TotalAssets:
- Assets → TotalAssets

## Acceptance Tests
After loading, these SQL queries must return correct results:
```sql
-- Test 1: Apple revenue FY2024
SELECT f.name, n.value, n.ddate
FROM facts n JOIN filings f ON n.adsh = f.adsh
WHERE f.name ILIKE '%apple%'
  AND n.tag = 'Revenue'  -- uses canonical tag
  AND f.form = '10-K'
ORDER BY n.ddate DESC LIMIT 5;

-- Test 2: Multiple companies, same metric
SELECT f.name, n.value
FROM facts n JOIN filings f ON n.adsh = f.adsh
WHERE n.tag = 'Revenue'
  AND f.fiscal_period = 'FY'
  AND f.fiscal_year = 2024
ORDER BY n.value DESC LIMIT 10;
```

## Performance Requirements
- Loading 3.8M rows must use bulk insert (pandas + psycopg2 copy_from 
  or SQLAlchemy executemany), NOT row-by-row inserts
- Full load should complete in under 10 minutes

## What NOT to build today
- No FastAPI / no web layer
- No LLM calls
- No vector embeddings
- No Cloud SQL (local Postgres only for now)
- No Docker (yet)