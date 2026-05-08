"""
End-to-end filings pipeline (no retrieval):

1) Ingestion: ingestion_filing
2) Chunking: docling_json_to_pinecone_chunks + save_chunks_to_json
3) RAPTOR: run_raptor_pipeline
4) Verification: verify_raptor_tree
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

try:
    from capstone_rag_project.Filings.ingestion import ingestion_filing
    from capstone_rag_project.Filings.chunking import (
        docling_json_to_pinecone_chunks,
        is_low_value_chunk,
        save_chunks_to_json,
        should_keep_chunk,
    )
    from capstone_rag_project.Filings.raptor import (
        derive_source_doc_id,
        derive_tree_output_filename,
        load_chunks_from_json,
        run_raptor_pipeline,
    )
    from capstone_rag_project.Filings.raptor_verification import verify_raptor_tree
except ModuleNotFoundError:
    from ingestion import ingestion_filing
    from chunking import (
        docling_json_to_pinecone_chunks,
        is_low_value_chunk,
        save_chunks_to_json,
        should_keep_chunk,
    )
    from raptor import (
        derive_source_doc_id,
        derive_tree_output_filename,
        load_chunks_from_json,
        run_raptor_pipeline,
    )
    from raptor_verification import verify_raptor_tree


logger = logging.getLogger("pipeline")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_docling_entries(docling_json_path: str) -> List[Dict[str, Any]]:
    with open(docling_json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if isinstance(loaded, list):
        return [d for d in loaded if isinstance(d, dict)]
    if isinstance(loaded, dict):
        return [loaded]
    raise ValueError("Unsupported Docling JSON format. Expected dict or list of dicts.")


def run_filings_pipeline(
    company_title: str,
    form_type: str = "10-K",
    ticker_override: str | None = None,
    expected_sections: List[str] | None = None,
    namespace: str | None = None,
    max_levels: int = 3,
    n_components: int = 10,
) -> Dict[str, Any]:
    """
    Required flow:
    ingestion_filing -> chunking/save_chunks_to_json -> run_raptor_pipeline -> verify_raptor_tree
    """
    logger.info("Starting filings pipeline for company_title=%s form_type=%s", company_title, form_type)

    # 1) Ingestion
    ingestion_output = ingestion_filing(
        company_title=company_title,
        form_type=form_type,
        ticker_override=ticker_override,
        expected_sections=expected_sections,
    )
    if not ingestion_output:
        raise RuntimeError("Ingestion failed. Pipeline stopped.")

    folder = str(ingestion_output["folder"])
    docling_json_path = str(ingestion_output["docling_json_path"])

    # 2) Chunking
    docs = _load_docling_entries(docling_json_path)
    chunks: List[Dict[str, Any]] = []
    for doc in docs:
        chunks.extend(docling_json_to_pinecone_chunks(doc))

    filtered_chunks = [
        ch for ch in chunks
        if not is_low_value_chunk(ch.get("metadata", {}), ch.get("text", ""))
    ]
    final_chunks = [ch for ch in filtered_chunks if should_keep_chunk(ch)]

    chunks_output_path = str(Path(folder).with_name(f"{folder}_pinecone_ready_chunks.json"))
    save_chunks_to_json(final_chunks, chunks_output_path)
    logger.info("Chunking completed: total=%s final=%s", len(chunks), len(final_chunks))

    # 3) RAPTOR
    raptor_output = run_raptor_pipeline(
        chunks_json_path=chunks_output_path,
        namespace=namespace,
        embedder=None,
        max_levels=max_levels,
        n_components=n_components,
        run_self_test=True,
    )

    # Determine tree path saved by RAPTOR so we can verify it.
    loaded_chunks = load_chunks_from_json(chunks_output_path)
    source_doc_id = derive_source_doc_id(loaded_chunks, default_value=Path(chunks_output_path).stem)
    tree_output_filename = derive_tree_output_filename(loaded_chunks, default_stem=source_doc_id)
    tree_output_path = str((Path.cwd() / tree_output_filename).resolve())

    # 4) Verification
    verification_output = verify_raptor_tree(tree_output_path, index=None, namespace=(namespace or "user1"))

    logger.info("Filings pipeline completed successfully.")
    return {
        "ingestion": ingestion_output,
        "chunks_output_path": chunks_output_path,
        "raptor": raptor_output,
        "tree_output_path": tree_output_path,
        "verification": verification_output,
    }
