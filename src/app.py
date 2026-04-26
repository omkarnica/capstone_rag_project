from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import time

from src.api import run_adaptive_query, run_single_question
from src.audit.logger import build_audit_record, log_query
from src.eval_api import router as eval_router
from src.utils.secrets import preload_secrets


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_secrets()
    yield


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chunking_strategy: str = "hierarchical"
    similarity_threshold: float = 0.92
    use_cache: bool = True
    company: str | None = None
    period: str | None = None
    tenant_id: str = "default"
    user_id: str = "anonymous"


class DueDiligenceRequest(BaseModel):
    company: str = Field(..., min_length=1)
    transcript_company: str | None = None
    fiscal_year: int = Field(..., ge=2000, le=2030)
    quarter: str = "FY"


class HealthResponse(BaseModel):
    status: str


app = FastAPI(
    title="M&A Oracle — Due Diligence Intelligence API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(eval_router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _run_query(payload: QueryRequest) -> dict[str, Any]:
    start = time.monotonic()
    try:
        result = run_adaptive_query(
            question=payload.question,
            chunking_strategy=payload.chunking_strategy,
            similarity_threshold=payload.similarity_threshold,
            use_cache=payload.use_cache,
            company=payload.company,
            period=payload.period,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = int((time.monotonic() - start) * 1000)
    record = build_audit_record(
        question=payload.question,
        result=result,
        latency_ms=latency_ms,
        tenant_id=payload.tenant_id,
        user_id=payload.user_id,
    )
    log_query(record)
    return result


@app.post("/query")
def query(payload: QueryRequest) -> dict[str, Any]:
    return _run_query(payload)


@app.post("/adaptive-query")
def adaptive_query(payload: QueryRequest) -> dict[str, Any]:
    return _run_query(payload)


@app.post("/api/query")
def api_query(payload: QueryRequest) -> dict[str, Any]:
    return _run_query(payload)


@app.post("/due-diligence")
def due_diligence(payload: DueDiligenceRequest) -> dict[str, Any]:
    try:
        from src.contradictions.detector import run_due_diligence
        return run_due_diligence(
            company=payload.company,
            transcript_company=payload.transcript_company or payload.company,
            fiscal_year=payload.fiscal_year,
            quarter=payload.quarter,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
