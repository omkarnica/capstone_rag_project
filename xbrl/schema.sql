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

CREATE INDEX IF NOT EXISTS idx_facts_adsh ON facts(adsh);
CREATE INDEX IF NOT EXISTS idx_facts_tag ON facts(tag);
CREATE INDEX IF NOT EXISTS idx_facts_ddate ON facts(ddate);
CREATE INDEX IF NOT EXISTS idx_filings_cik ON filings(cik);
CREATE INDEX IF NOT EXISTS idx_filings_name ON filings(name);
