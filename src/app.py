from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api import run_adaptive_query


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chunking_strategy: str = "hierarchical"
    similarity_threshold: float = 0.92
    use_cache: bool = True


class HealthResponse(BaseModel):
    status: str


app = FastAPI(
    title="Claude Certification Knowledge Assistant API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _run_query(payload: QueryRequest) -> dict[str, Any]:
    try:
        return run_adaptive_query(
            question=payload.question,
            chunking_strategy=payload.chunking_strategy,
            similarity_threshold=payload.similarity_threshold,
            use_cache=payload.use_cache,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query")
def query(payload: QueryRequest) -> dict[str, Any]:
    return _run_query(payload)


@app.post("/adaptive-query")
def adaptive_query(payload: QueryRequest) -> dict[str, Any]:
    return _run_query(payload)


@app.post("/api/query")
def api_query(payload: QueryRequest) -> dict[str, Any]:
    return _run_query(payload)