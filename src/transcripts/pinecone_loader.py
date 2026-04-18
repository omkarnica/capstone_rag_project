"""
Reads transcript sections from PostgreSQL and upserts them to Pinecone.

Flow:
1) Query transcript_sections JOIN transcripts for text + metadata, optionally
   filtered by CIK.
2) Chunk each section_text into ~1600-char / ~400-token segments with ~200-char
   overlap, splitting on paragraph boundaries (\n\n) where possible.
3) Build one Pinecone record per chunk: id (UUID), text (the chunk), and flat
   metadata fields (company_title, cik, accession_no, filed_date,
   period_of_report, form_type, content_type, chunk_index).
4) Upsert in batches of 96 using index.upsert_records(namespace="transcripts").
   Pinecone's hosted llama-text-embed-v2 model embeds the `text` field
   automatically — no local embedding call needed.

CLI:
    python -m src.transcripts.pinecone_loader
    python -m src.transcripts.pinecone_loader --cik 320193
    python -m src.transcripts.pinecone_loader --cik 789019
"""

from __future__ import annotations

import argparse
import uuid
from typing import Iterator

from pinecone import Pinecone

from src.utils.secrets import get_secret

from src.utils.exceptions import db_error_boundary
from src.utils.logger import get_logger
from xbrl.loader import get_connection

logger = get_logger(__name__)

_INDEX_NAME   = "ragcapstone"
_NAMESPACE    = "transcripts"
_BATCH_SIZE   = 96
_MAX_CHARS    = 1600
_OVERLAP_CHARS = 200

_KNOWN_CIKS: dict[int, str] = {
    320193: "Apple Inc.",
    789019: "Microsoft Corporation",
}

_QUERY_ALL = """
    SELECT
        ts.id            AS section_id,
        ts.section_text,
        t.company_name,
        t.filed_date,
        t.period_of_report,
        t.accession_no,
        t.cik,
        t.form_type
    FROM transcript_sections ts
    JOIN transcripts t ON ts.transcript_id = t.id
    ORDER BY t.filed_date, ts.id
"""

_QUERY_BY_CIK = """
    SELECT
        ts.id            AS section_id,
        ts.section_text,
        t.company_name,
        t.filed_date,
        t.period_of_report,
        t.accession_no,
        t.cik,
        t.form_type
    FROM transcript_sections ts
    JOIN transcripts t ON ts.transcript_id = t.id
    WHERE t.cik = %s
    ORDER BY t.filed_date, ts.id
"""


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_chars: int = _MAX_CHARS, overlap_chars: int = _OVERLAP_CHARS) -> list[str]:
    """
    Split text into overlapping chunks, preferring paragraph boundaries.

    Strategy:
    1. Split on double-newline (paragraph boundaries).
    2. Accumulate paragraphs until the buffer would exceed max_chars.
    3. Emit the buffer as a chunk, then seed the next buffer with the last
       overlap_chars characters of the emitted chunk for context continuity.
    4. Any remaining buffer is emitted as the final chunk.

    Args:
        text:          Plain text to chunk.
        max_chars:     Target maximum characters per chunk (~400 tokens).
        overlap_chars: Characters from the end of each chunk prepended to
                       the next chunk (~50 tokens).

    Returns:
        List of text strings, each at most max_chars long (approximately).
        Returns a list with the original text if it is shorter than max_chars.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len: int = 0

    for para in paragraphs:
        # +2 for the "\n\n" separator when joining
        added_len = len(para) + (2 if current_parts else 0)

        if current_parts and current_len + added_len > max_chars:
            chunk = "\n\n".join(current_parts)
            chunks.append(chunk)

            # Seed next chunk with trailing overlap for context continuity
            overlap = chunk[-overlap_chars:] if len(chunk) > overlap_chars else chunk
            current_parts = [overlap]
            current_len = len(overlap)

        current_parts.append(para)
        current_len += added_len

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


# ── Pinecone helpers ──────────────────────────────────────────────────────────

def _get_index():
    """
    Create and return a Pinecone Index handle for ragcapstone.

    Fetches PINECONE_API_KEY from GCP Secret Manager, falling back to .env.
    """
    api_key = get_secret("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    return pc.Index(_INDEX_NAME)


def _batched(items: list, size: int) -> Iterator[list]:
    """Yield successive slices of `items` of length `size`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ── DB fetch ──────────────────────────────────────────────────────────────────

def _fetch_sections(cik: int | None) -> list[dict]:
    """
    Query PostgreSQL for transcript sections with their filing metadata.

    Args:
        cik: Filter to a specific company CIK, or None for all companies.

    Returns:
        List of dicts with keys: section_id, section_text, company_name,
        filed_date, period_of_report, accession_no, cik, form_type.
    """
    conn = get_connection()
    try:
        with db_error_boundary("SELECT", table="transcript_sections"):
            with conn.cursor() as cur:
                if cik is not None:
                    cur.execute(_QUERY_BY_CIK, (cik,))
                else:
                    cur.execute(_QUERY_ALL)
                cols = [desc[0] for desc in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()

    logger.info(
        "Fetched transcript sections from DB",
        extra={"cik": cik, "section_count": len(rows)},
    )
    return rows


# ── Main upsert logic ─────────────────────────────────────────────────────────

def upsert_transcripts(cik: int | None = None) -> int:
    """
    Chunk all transcript sections and upsert them to Pinecone.

    Reads sections from PostgreSQL, chunks each section_text, builds one
    Pinecone record per chunk with flat metadata, and upserts in batches of
    96. Pinecone's hosted llama-text-embed-v2 model embeds the `text` field
    — no local embedding call is needed.

    Args:
        cik: Filter to a specific company CIK (320193=Apple, 789019=Microsoft),
             or None to process all companies.

    Returns:
        Total number of vectors upserted to Pinecone.
    """
    sections = _fetch_sections(cik)
    if not sections:
        logger.info("No transcript sections found", extra={"cik": cik})
        return 0

    index = _get_index()
    records: list[dict] = []

    for section in sections:
        text = section["section_text"] or ""
        if not text.strip():
            continue

        chunks = _chunk_text(text)
        filed_date = section["filed_date"]
        period = section["period_of_report"]

        for chunk_index, chunk in enumerate(chunks):
            records.append({
                "id":               str(uuid.uuid4()),
                "text":             chunk,
                "company_title":    section["company_name"] or "",
                "content_type":     "transcript",
                "form_type":        section["form_type"] or "8-K",
                "filed_date":       str(filed_date) if filed_date else "",
                "period_of_report": str(period) if period else "",
                "accession_no":     section["accession_no"] or "",
                "chunk_index":      chunk_index,
                "cik":              section["cik"],
            })

    if not records:
        logger.info("No chunks to upsert", extra={"cik": cik})
        return 0

    total_upserted = 0
    for batch in _batched(records, _BATCH_SIZE):
        index.upsert_records(namespace=_NAMESPACE, records=batch)
        total_upserted += len(batch)
        logger.info(
            "Upserted Pinecone batch",
            extra={"batch_size": len(batch), "total_so_far": total_upserted},
        )

    logger.info(
        "Pinecone upsert complete",
        extra={"cik": cik, "total_vectors": total_upserted},
    )
    return total_upserted


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Upsert transcript sections from PostgreSQL to Pinecone."
    )
    p.add_argument(
        "--cik",
        type=int,
        default=None,
        help=(
            "Filter by company CIK (320193=Apple, 789019=Microsoft). "
            "Omit to process all companies."
        ),
    )
    return p


def main() -> None:
    """CLI entry point."""
    args = _build_arg_parser().parse_args()
    total = upsert_transcripts(cik=args.cik)
    logger.info("Run complete", extra={"total_vectors_upserted": total})


if __name__ == "__main__":
    main()
