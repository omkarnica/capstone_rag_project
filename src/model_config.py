from __future__ import annotations

import os

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

_PROJECT = "codelab-2-485215"
_LOCATION = "us-central1"
_MODEL = "gemini-2.5-flash"

_genai_client: genai.Client | None = None


def _configure_google_genai_vertex_env() -> None:
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", _PROJECT)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", _LOCATION)


def get_model_name() -> str:
    return _MODEL


def get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(
            vertexai=True,
            project=_PROJECT,
            location=_LOCATION,
        )
    return _genai_client


def _get_llm(**kwargs) -> ChatGoogleGenerativeAI:
    _configure_google_genai_vertex_env()
    return ChatGoogleGenerativeAI(
        model=_MODEL,
        vertexai=True,
        project=_PROJECT,
        location=_LOCATION,
        temperature=0,
        **kwargs,
    )


def get_router_llm() -> ChatGoogleGenerativeAI:
    return _get_llm()


def get_planner_llm() -> ChatGoogleGenerativeAI:
    return _get_llm()


def get_rewriter_llm() -> ChatGoogleGenerativeAI:
    return _get_llm()


def get_grader_llm() -> ChatGoogleGenerativeAI:
    return _get_llm()


def get_generation_llm() -> ChatGoogleGenerativeAI:
    return _get_llm()


def get_graph_llm() -> ChatGoogleGenerativeAI:
    return _get_llm()


def get_direct_generation_llm() -> ChatGoogleGenerativeAI:
    return _get_llm()


def get_merge_llm() -> ChatGoogleGenerativeAI:
    return _get_llm()
