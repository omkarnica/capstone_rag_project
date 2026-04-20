"""
Public package exports for the filings pipeline.

Dependency diagram:

    caller
      |
      v
    src.filings
      |-- run_filings_pipeline -> src.filings.pipeline
      `-- retrieval helpers  -> src.filings.raptor_retrieval

This module keeps imports lazy so importing src.filings does not immediately
load heavy runtime dependencies such as Docling, scikit-learn, or Pinecone.
"""

from typing import Any

__all__ = [
    "run_filings_pipeline",
    "raptor_retrieve",
    "build_context_string",
    "retrieve_context_for_llm",
]


def __getattr__(name: str) -> Any:
    if name == "run_filings_pipeline":
        from .pipeline import run_filings_pipeline

        return run_filings_pipeline

    if name in {
        "raptor_retrieve",
        "build_context_string",
        "retrieve_context_for_llm",
    }:
        from . import raptor_retrieval

        return getattr(raptor_retrieval, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
