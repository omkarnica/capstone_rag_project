from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

DEFAULT_CHAT_MODEL = os.getenv("OPENAI_DEFAULT_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_GENERATION_MODEL = os.getenv("OPENAI_GENERATION_MODEL", "gpt-5-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)


def _resolve_openai_client_kwargs(
    *,
    base_url_env: str | None = None,
    compat_api_key_env: str | None = None,
) -> dict:
    base_url = os.getenv(base_url_env) if base_url_env else None
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        if base_url:
            # Many local OpenAI-compatible servers ignore auth but require a value.
            api_key = (
                os.getenv(compat_api_key_env) if compat_api_key_env else None
            ) or os.getenv("OPENAI_COMPAT_API_KEY", "not-needed")
        else:
            raise ValueError(
                "OPENAI_API_KEY is missing. Set it, or set OPENAI_BASE_URL for an "
                "OpenAI-compatible provider and optionally OPENAI_COMPAT_API_KEY."
            )

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return kwargs


def get_openai_client_kwargs(kind: str = "chat") -> dict:
    if kind == "chat":
        return _resolve_openai_client_kwargs(
            base_url_env="OPENAI_CHAT_BASE_URL",
            compat_api_key_env="OPENAI_CHAT_COMPAT_API_KEY",
        )

    if kind == "embedding":
        return _resolve_openai_client_kwargs(
            base_url_env="OPENAI_EMBEDDING_BASE_URL",
            compat_api_key_env="OPENAI_EMBEDDING_COMPAT_API_KEY",
        )

    raise ValueError(f"Unsupported OpenAI client kind: {kind}")


def get_chat_model(env_var: str, default_model: str | None = None) -> ChatOpenAI:
    model_name = os.getenv(env_var, default_model or DEFAULT_CHAT_MODEL)
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        **get_openai_client_kwargs("chat"),
    )


def get_router_llm() -> ChatOpenAI:
    return get_chat_model("OPENAI_ROUTER_MODEL")


def get_planner_llm() -> ChatOpenAI:
    return get_chat_model("OPENAI_PLANNER_MODEL")


def get_rewriter_llm() -> ChatOpenAI:
    return get_chat_model("OPENAI_REWRITER_MODEL")


def get_grader_llm() -> ChatOpenAI:
    return get_chat_model("OPENAI_GRADER_MODEL")


def get_generation_llm() -> ChatOpenAI:
    return get_chat_model("OPENAI_GENERATION_MODEL", DEFAULT_GENERATION_MODEL)


def get_direct_generation_llm() -> ChatOpenAI:
    return get_chat_model("OPENAI_DIRECT_MODEL", DEFAULT_CHAT_MODEL)


def get_merge_llm() -> ChatOpenAI:
    return get_chat_model("OPENAI_MERGE_MODEL", DEFAULT_GENERATION_MODEL)


def get_embeddings() -> OpenAIEmbeddings:
    model_name = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return OpenAIEmbeddings(
        model=model_name,
        **get_openai_client_kwargs("embedding"),
    )