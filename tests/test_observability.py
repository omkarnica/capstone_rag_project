from __future__ import annotations

from unittest.mock import patch


def test_configure_google_genai_vertex_env_sets_expected_variables() -> None:
    from src.model_config import _configure_google_genai_vertex_env

    with patch.dict("os.environ", {}, clear=True):
        _configure_google_genai_vertex_env()

        import os

        assert os.environ["GOOGLE_GENAI_USE_VERTEXAI"] == "true"
        assert os.environ["GOOGLE_CLOUD_PROJECT"] == "codelab-2-485215"
        assert os.environ["GOOGLE_CLOUD_LOCATION"] == "us-central1"


def test_build_langsmith_metadata_includes_request_context() -> None:
    from src.observability import build_langsmith_metadata

    metadata = build_langsmith_metadata(
        question="Who are Apple's current board members?",
        company="AAPL",
        period="2025",
        route="graph",
        user_id="user-1",
        plan_type="single",
    )

    assert metadata == {
        "question": "Who are Apple's current board members?",
        "company": "AAPL",
        "period": "2025",
        "route": "graph",
        "user_id": "user-1",
        "plan_type": "single",
    }


def test_build_langsmith_runnable_config_sets_metadata_and_tags() -> None:
    from src.observability import build_langsmith_runnable_config

    metadata = {"route": "graph", "company": "AAPL"}
    config = build_langsmith_runnable_config(
        metadata=metadata,
        tags=["api", "graph"],
        run_name="single_question",
    )

    assert config["run_name"] == "single_question"
    assert config["metadata"] == metadata
    assert config["tags"] == ["api", "graph"]


def test_run_single_question_passes_langsmith_metadata_to_graph() -> None:
    from src.api import run_single_question
    from unittest.mock import MagicMock

    graph = MagicMock()
    graph.invoke.return_value = {
        "route": "graph",
        "initial_route": "graph",
        "answer": "ok",
        "citations": [],
        "company": "AAPL",
        "period": "2025",
    }

    with patch("src.api.get_graph", return_value=graph):
        result = run_single_question(
            question="Who are Apple's current board members?",
            route_hint="graph",
            force_route=True,
            company="AAPL",
            period="2025",
            tenant_id="tenant-1",
            user_id="user-1",
            trace_source="api",
        )

    assert result["final_route"] == "graph"
    assert result["company"] == "AAPL"
    config = graph.invoke.call_args.kwargs["config"]
    assert config["run_name"] == "single_question"
    assert config["metadata"]["question"] == "Who are Apple's current board members?"
    assert config["metadata"]["company"] == "AAPL"
    assert "tenant_id" not in config["metadata"]
    assert "chunking_strategy" not in config["metadata"]
    assert config["tags"] == ["api", "single", "graph"]


def test_configure_langsmith_environment_sets_defaults_and_secret_manager_key() -> None:
    from src.observability import configure_langsmith_environment

    with patch.dict("os.environ", {}, clear=True):
        with patch("src.observability.get_secret", return_value="smith-secret") as mock_get_secret:
            configure_langsmith_environment()

        import os

        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"
        assert os.environ["LANGSMITH_PROJECT"] == "RAG-Capstone"
        assert os.environ["LANGSMITH_API_KEY"] == "smith-secret"
        mock_get_secret.assert_called_once_with("LangSmith_key")
