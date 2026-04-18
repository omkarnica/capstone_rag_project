"""
Load patent claims from PostgreSQL into Pinecone (namespace: patents).

One chunk per independent claim.
Metadata: company_title, content_type, patent_id, claim_number, cpc_codes, assignee_organization
"""

from __future__ import annotations

import os

import psycopg2
from dotenv import load_dotenv
from pinecone import Pinecone

from src.utils.logger import get_logger
from src.utils.secrets import get_secret

load_dotenv()
logger = get_logger(__name__)

# ── Pinecone config ────────────────────────────────────────────────────────────
PINECONE_API_KEY  = os.environ["PINECONE_API_KEY"]
INDEX_NAME        = "ragcapstone"
NAMESPACE         = "patents"
EMBED_MODEL       = "llama-text-embed-v2"

# ── PostgreSQL config ──────────────────────────────────────────────────────────
DB_CONFIG = dict(
    host="localhost", port=5433, dbname="ma_oracle",
    user="postgres", password=get_secret("DB_PASSWORD")
)

# ── Batch size ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 50


def fetch_claims(conn, assignee_organization: str) -> list[dict]:
    """Query independent patent claims joined with patent metadata for a given assignee."""
    sql = """
        SELECT
            pc.patent_id,
            pc.claim_number,
            pc.claim_text,
            p.patent_title,
            p.grant_date,
            p.assignee_organization,
            p.cpc_codes,
            p.citation_count
        FROM patent_claims pc
        JOIN patents p ON p.patent_id = pc.patent_id
        WHERE pc.is_independent = TRUE
          AND p.assignee_organization ILIKE %s
        ORDER BY p.grant_date DESC, pc.patent_id, pc.claim_number
    """
    with conn.cursor() as cur:
        cur.execute(sql, (f"%{assignee_organization}%",))
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in rows]


def build_record(row: dict, company_title: str) -> dict:
    """Build a Pinecone upsert record dict from a patent claim row."""
    cpc_str = ", ".join(row["cpc_codes"]) if row["cpc_codes"] else ""
    grant_date_str = str(row["grant_date"]) if row["grant_date"] else ""

    text = (
        f"Patent: {row['patent_title']}\n"
        f"Assignee: {row['assignee_organization'] or ''}\n"
        f"CPC Codes: {cpc_str}\n"
        f"Grant Date: {grant_date_str}\n"
        f"Claim {row['claim_number']}: {row['claim_text']}"
    )

    record_id = f"{row['patent_id']}_claim_{row['claim_number']}"

    return {
        "_id": record_id,
        "text": text,
        "company_title": company_title,
        "content_type": "patent",
        "patent_id": row["patent_id"],
        "claim_number": row["claim_number"],
        "patent_title": row["patent_title"] or "",
        "assignee_organization": row["assignee_organization"] or "",
        "cpc_codes": cpc_str,
        "citation_count": row["citation_count"] or 0,
        "grant_date": grant_date_str,
    }


def upsert_claims(index, records: list[dict]) -> int:
    """Upsert patent claim records to Pinecone in batches; returns total count upserted."""
    total = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        index.upsert_records(NAMESPACE, batch)
        total += len(batch)
        logger.info(f"Upserted {total}/{len(records)} claims")
    return total


def run(company_title: str, assignee_organization: str) -> None:
    """Fetch independent claims from PostgreSQL and upsert them to the Pinecone patents namespace."""
    logger.info(f"Patents Pinecone loader start: {company_title}")

    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        rows = fetch_claims(conn, assignee_organization)
    finally:
        conn.close()

    if not rows:
        logger.warning(f"No independent claims found for: {assignee_organization}")
        return

    logger.info(f"Found {len(rows)} independent claims for {company_title}")
    records = [build_record(row, company_title) for row in rows]
    upserted = upsert_claims(index, records)
    logger.info(f"Patents Pinecone loader complete: {upserted} records upserted")


if __name__ == "__main__":
    run(
        company_title="Apple Inc",
        assignee_organization="Apple",
    )