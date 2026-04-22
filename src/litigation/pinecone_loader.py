"""
Reads litigation opinions from PostgreSQL and upserts them to Pinecone.

Flow:
1. Query litigation_opinions JOIN litigation_cases for text + metadata.
2. Chunk each plain_text (falling back to snippet) into ~1600-char segments
   with ~200-char overlap, splitting on paragraph boundaries.
3. Build one Pinecone record per chunk with flat metadata fields.
4. Upsert in batches of 96 using index.upsert_records(namespace="litigation").
   Pinecone's hosted llama-text-embed-v2 model embeds the `text` field automatically.

CLI:
    python -m src.litigation.pinecone_loader
    python -m src.litigation.pinecone_loader --company "Apple Inc"
"""

from __future__ import annotations

import argparse
import uuid
from typing import Iterator

from pinecone import Pinecone

from src.utils.exceptions import db_error_boundary
from src.utils.logger import get_logger
from src.utils.secrets import get_secret
from src.xbrl.loader import get_connection

logger = get_logger(__name__)

_INDEX_NAME    = "ragcapstone"
_NAMESPACE     = "litigation"
_BATCH_SIZE    = 96
_MAX_CHARS     = 1600
_OVERLAP_CHARS = 200

_QUERY_ALL = """
    SELECT
        lo.opinion_id,
        lo.opinion_type,
        lo.plain_text,
        lo.snippet,
        lc.case_id,
        lc.case_name,
        lc.court,
        lc.court_citation,
        lc.date_filed,
        lc.status,
        lc.company_name,
        lc.cite_count,
        lc.url
    FROM litigation_opinions lo
    JOIN litigation_cases lc ON lo.case_id = lc.case_id
    ORDER BY lc.date_filed DESC, lo.opinion_id
"""

_QUERY_BY_COMPANY = """
    SELECT
        lo.opinion_id,
        lo.opinion_type,
        lo.plain_text,
        lo.snippet,
        lc.case_id,
        lc.case_name,
        lc.court,
        lc.court_citation,
        lc.date_filed,
        lc.status,
        lc.company_name,
        lc.cite_count,
        lc.url
    FROM litigation_opinions lo
    JOIN litigation_cases lc ON lo.case_id = lc.case_id
    WHERE lc.company_name ILIKE %s
    ORDER BY lc.date_filed DESC, lo.opinion_id
"""


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        added_len = len(para) + (2 if current_parts else 0)
        if current_parts and current_len + added_len > _MAX_CHARS:
            chunk = "\n\n".join(current_parts)
            chunks.append(chunk)
            overlap = chunk[-_OVERLAP_CHARS:] if len(chunk) > _OVERLAP_CHARS else chunk
            current_parts = [overlap]
            current_len = len(overlap)
        current_parts.append(para)
        current_len += added_len

    if current_parts:
        chunks.append("\n\n".join(current_parts))
    return chunks


def _fetch_opinions(company: str | None) -> list[dict]:
    """Query PostgreSQL for opinions with case metadata."""
    conn = get_connection()
    try:
        with db_error_boundary("SELECT", table="litigation_opinions"):
            with conn.cursor() as cur:
                if company:
                    cur.execute(_QUERY_BY_COMPANY, (f"%{company}%",))
                else:
                    cur.execute(_QUERY_ALL)
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()
    logger.info("Fetched opinions from DB", extra={"company": company, "count": len(rows)})
    return rows


def _get_index():
    pc = Pinecone(api_key=get_secret("PINECONE_API_KEY"))
    return pc.Index(_INDEX_NAME)


def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def upsert_litigation(company: str | None = None) -> int:
    """Chunk litigation opinions from PostgreSQL and upsert to Pinecone."""
    opinions = _fetch_opinions(company)
    if not opinions:
        logger.info("No opinions to upsert", extra={"company": company})
        return 0

    index = _get_index()
    records: list[dict] = []

    for row in opinions:
        text = (row.get("plain_text") or row.get("snippet") or "").strip()
        if not text:
            continue

        chunks = _chunk_text(text)
        date_filed = row.get("date_filed")

        for chunk_index, chunk in enumerate(chunks):
            records.append({
                "id":            str(uuid.uuid4()),
                "text":          chunk,
                "company_name":  row.get("company_name") or "",
                "content_type":  "litigation",
                "case_id":       row.get("case_id") or "",
                "case_name":     row.get("case_name") or "",
                "court":         row.get("court") or "",
                "court_citation": row.get("court_citation") or "",
                "date_filed":    str(date_filed) if date_filed else "",
                "status":        row.get("status") or "",
                "cite_count":    row.get("cite_count") or 0,
                "opinion_id":    row.get("opinion_id") or "",
                "chunk_index":   chunk_index,
                "url":           row.get("url") or "",
            })

    if not records:
        logger.info("No chunks to upsert", extra={"company": company})
        return 0

    total = 0
    for batch in _batched(records, _BATCH_SIZE):
        index.upsert_records(namespace=_NAMESPACE, records=batch)
        total += len(batch)
        logger.info(
            "Upserted litigation batch",
            extra={"batch_size": len(batch), "total_so_far": total},
        )

    logger.info(
        "Litigation Pinecone upsert complete",
        extra={"company": company, "total_vectors": total},
    )
    return total


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description="Upsert litigation opinions from PostgreSQL to Pinecone."
    )
    p.add_argument("--company", default=None, help="Filter by company name (ILIKE match).")
    args = p.parse_args()
    total = upsert_litigation(company=args.company)
    logger.info("Run complete", extra={"total_vectors_upserted": total})


if __name__ == "__main__":
    main()
