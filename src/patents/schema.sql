-- Patents pipeline schema
-- Apply with: psql -h localhost -p 5433 -U postgres -d ma_oracle -f src/patents/schema.sql

CREATE TABLE IF NOT EXISTS patents (
    patent_id             TEXT PRIMARY KEY,
    patent_title          TEXT,
    grant_date            DATE,
    assignee_organization TEXT,
    cpc_codes             TEXT[],
    citation_count        INTEGER DEFAULT 0,
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_patents_assignee_organization ON patents (assignee_organization);
CREATE INDEX IF NOT EXISTS idx_patents_grant_date            ON patents (grant_date);

CREATE TABLE IF NOT EXISTS patent_claims (
    id             SERIAL  PRIMARY KEY,
    patent_id      TEXT    NOT NULL REFERENCES patents(patent_id),
    claim_number   INTEGER NOT NULL,
    claim_text     TEXT    NOT NULL,
    is_independent BOOLEAN NOT NULL DEFAULT TRUE,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_patent_claims_patent_id ON patent_claims (patent_id);
