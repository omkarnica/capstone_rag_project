-- Litigation pipeline schema for M&A Oracle
-- Run once: psql -h localhost -p 5433 -d ma_oracle -U postgres -f src/litigation/schema.sql

CREATE TABLE IF NOT EXISTS litigation_cases (
    case_id         TEXT        PRIMARY KEY,
    case_name       TEXT,
    court           TEXT,
    court_citation  TEXT,
    date_filed      DATE,
    docket_number   TEXT,
    status          TEXT,
    company_name    TEXT,
    url             TEXT,
    cite_count      INTEGER     DEFAULT 0
);

CREATE TABLE IF NOT EXISTS litigation_opinions (
    opinion_id      TEXT        PRIMARY KEY,
    case_id         TEXT        REFERENCES litigation_cases(case_id),
    opinion_type    TEXT,
    plain_text      TEXT,
    snippet         TEXT
);

CREATE INDEX IF NOT EXISTS idx_litigation_cases_company  ON litigation_cases(company_name);
CREATE INDEX IF NOT EXISTS idx_litigation_cases_date     ON litigation_cases(date_filed);
CREATE INDEX IF NOT EXISTS idx_litigation_opinions_case  ON litigation_opinions(case_id);
