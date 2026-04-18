-- Transcripts pipeline schema
-- Apply with: psql -h localhost -p 5433 -U postgres -d ma_oracle -f src/transcripts/schema.sql

CREATE TABLE IF NOT EXISTS transcripts (
    id               SERIAL PRIMARY KEY,
    cik              INTEGER NOT NULL,
    accession_no     TEXT    NOT NULL,
    company_name     TEXT,
    filed_date       DATE,
    period_of_report DATE,
    form_type        TEXT    DEFAULT '8-K',
    exhibit_url      TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT transcripts_accession_no_key UNIQUE (accession_no)
);

CREATE INDEX IF NOT EXISTS idx_transcripts_cik        ON transcripts (cik);
CREATE INDEX IF NOT EXISTS idx_transcripts_filed_date ON transcripts (filed_date);

CREATE TABLE IF NOT EXISTS transcript_sections (
    id            SERIAL  PRIMARY KEY,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(id),
    section_item  TEXT    NOT NULL,
    section_text  TEXT    NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transcript_sections_transcript_id
    ON transcript_sections (transcript_id);
