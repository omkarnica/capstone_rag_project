from __future__ import annotations

from unittest.mock import patch


def test_raptor_uses_shared_model_config_vertex_client() -> None:
    from src.filings import raptor

    sentinel_client = object()
    raptor._VERTEX_GENAI_CLIENT = None

    with patch("src.filings.raptor.get_genai_client", return_value=sentinel_client) as mock_get_client:
        client = raptor._get_vertex_genai_client()

    assert client is sentinel_client
    assert raptor._VERTEX_GENAI_CLIENT is sentinel_client
    mock_get_client.assert_called_once_with()


def test_run_filings_pipeline_writes_chunks_and_tree_under_data_and_uses_requested_namespace() -> None:
    from src.filings.pipeline import run_filings_pipeline

    ingestion_output = {
        "folder": "data/aapl_10-k",
        "ticker": "AAPL",
        "docling_json_path": "data/aapl_10-k_docling.json",
        "xbrl_json_path": "data/aapl_10-k.json",
        "subsidiaries_json_path": "data/aapl_10-k_subsidiaries.json",
    }
    docling_entries = [
        {
            "source": "aapl_2024.html",
            "ticker": "AAPL",
            "year": 2024,
            "docling": {},
        }
    ]
    chunk = {
        "id": "chunk-1",
        "text": "Meaningful filing chunk text",
        "metadata": {"source": "aapl_2024.html"},
    }

    with (
        patch("src.filings.pipeline.ingestion_filing", return_value=ingestion_output),
        patch("src.filings.pipeline._load_docling_entries", return_value=docling_entries),
        patch("src.filings.pipeline.docling_json_to_pinecone_chunks", return_value=[chunk]),
        patch("src.filings.pipeline.is_low_value_chunk", return_value=False),
        patch("src.filings.pipeline.should_keep_chunk", return_value=True),
        patch("src.filings.pipeline.save_chunks_to_json") as mock_save_chunks,
        patch("src.filings.pipeline.run_raptor_pipeline", return_value=("nodes", {"edges": []})) as mock_run_raptor,
        patch("src.filings.pipeline.load_chunks_from_json", return_value=[chunk]),
        patch("src.filings.pipeline.derive_source_doc_id", return_value="aapl_10-k"),
        patch("src.filings.pipeline.derive_tree_output_filename", return_value="aapl_10-k_raptor_tree.json"),
        patch("src.filings.pipeline.verify_raptor_tree", return_value={"ready_for_retrieval": True}) as mock_verify,
    ):
        result = run_filings_pipeline(
            company_title="Apple Inc.",
            form_type="10-K",
            ticker_override="AAPL",
            namespace="filings",
        )

    mock_save_chunks.assert_called_once_with([chunk], "data/aapl_10-k_pinecone_ready_chunks.json")
    mock_run_raptor.assert_called_once_with(
        chunks_json_path="data/aapl_10-k_pinecone_ready_chunks.json",
        namespace="filings",
        embedder=None,
        max_levels=3,
        n_components=10,
        run_self_test=True,
        output_dir="data",
    )
    mock_verify.assert_called_once_with("data/aapl_10-k_raptor_tree.json", index=None, namespace="filings")
    assert result["chunks_output_path"] == "data/aapl_10-k_pinecone_ready_chunks.json"
    assert result["tree_output_path"] == "data/aapl_10-k_raptor_tree.json"
