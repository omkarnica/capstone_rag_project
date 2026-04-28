from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from langsmith import tracing_context
from src.utils.secrets import get_secret


def configure_langsmith_environment() -> None:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGSMITH_PROJECT", "RAG-Capstone")

    if os.getenv("LANGSMITH_API_KEY"):
        return

    try:
        os.environ["LANGSMITH_API_KEY"] = get_secret("LangSmith_key")
    except Exception:
        # Keep tracing opt-in friendly when the secret is unavailable locally.
        pass


def is_langsmith_tracing_enabled() -> bool:
    configure_langsmith_environment()
    value = (
        os.getenv("LANGSMITH_TRACING")
        or os.getenv("LANGCHAIN_TRACING_V2")
        or ""
    ).strip().lower()
    return value in {"1", "true", "yes"}


def get_langsmith_project_name() -> str:
    configure_langsmith_environment()
    return (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "RAG-Capstone"
    )


def build_langsmith_metadata(
    *,
    question: str | None = None,
    company: str | None = None,
    period: str | None = None,
    route: str | None = None,
    user_id: str | None = None,
    plan_type: str | None = None,
    trace_source: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "question": question,
        "company": company,
        "period": period,
        "route": route,
        "user_id": user_id,
        "plan_type": plan_type,
        "trace_source": trace_source,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def build_langsmith_runnable_config(
    *,
    metadata: dict[str, Any],
    tags: list[str],
    run_name: str,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "metadata": metadata,
        "tags": tags,
    }


@contextmanager
def langsmith_trace_scope(
    *,
    metadata: dict[str, Any],
    tags: list[str],
):
    configure_langsmith_environment()
    with tracing_context(
        project_name=get_langsmith_project_name(),
        metadata=metadata,
        tags=tags,
        enabled=is_langsmith_tracing_enabled(),
    ):
        yield
