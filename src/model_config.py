from __future__ import annotations

from langchain_google_vertexai import ChatVertexAI

_PROJECT = "codelab-2-485215"
_LOCATION = "us-central1"
_MODEL = "gemini-2.5-flash"


def _get_llm(**kwargs) -> ChatVertexAI:
    return ChatVertexAI(
        model=_MODEL,
        project=_PROJECT,
        location=_LOCATION,
        temperature=0,
        **kwargs,
    )


def get_router_llm() -> ChatVertexAI:
    return _get_llm()


def get_planner_llm() -> ChatVertexAI:
    return _get_llm()


def get_rewriter_llm() -> ChatVertexAI:
    return _get_llm()


def get_grader_llm() -> ChatVertexAI:
    return _get_llm()


def get_generation_llm() -> ChatVertexAI:
    return _get_llm()


def get_direct_generation_llm() -> ChatVertexAI:
    return _get_llm()


def get_merge_llm() -> ChatVertexAI:
    return _get_llm()
