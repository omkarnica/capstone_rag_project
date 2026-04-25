from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.utils.logger import get_logger

_PROJECT_ID = "codelab-2-485215"
_DATASET = "ma_oracle"
_TABLE = "audit_log"
_TABLE_ID = f"{_PROJECT_ID}.{_DATASET}.{_TABLE}"

logger = get_logger(__name__)

_client = None
_client_lock = threading.Lock()


def _get_bq_client():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            from google.cloud import bigquery
            _client = bigquery.Client(project=_PROJECT_ID)
    return _client


@dataclass
class AuditRecord:
    query_id: str
    tenant_id: str
    user_id: str
    timestamp: datetime
    query: str
    route_taken: str | None
    sources_retrieved: list[str]
    retrieval_scores: list[float]
    graph_paths_traversed: list[str]
    generated_answer: str | None
    confidence_score: float | None
    tokens_used: int
    latency_ms: int
    user_feedback: str | None
    plan_type: str | None
    cache_hit: bool
    company: str | None
    period: str | None

    def to_bq_row(self) -> dict:
        return {
            "query_id": self.query_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "route_taken": self.route_taken,
            "sources_retrieved": self.sources_retrieved,
            "retrieval_scores": self.retrieval_scores,
            "graph_paths_traversed": self.graph_paths_traversed,
            "generated_answer": self.generated_answer,
            "confidence_score": self.confidence_score,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "user_feedback": self.user_feedback,
            "plan_type": self.plan_type,
            "cache_hit": self.cache_hit,
            "company": self.company,
            "period": self.period,
        }


def _derive_confidence_score(result: dict) -> float | None:
    """Map hallucination + quality grades to a 0-1 confidence score."""
    hallucination = result.get("hallucination_grade")
    quality = result.get("answer_quality_grade")
    if result.get("cache_hit"):
        return 0.85
    if hallucination == "yes" and quality == "yes":
        return 0.90
    if hallucination == "yes":
        return 0.60
    if hallucination == "no":
        return 0.30
    return None


def _extract_sources(result: dict) -> list[str]:
    """Pull source identifiers from retrieved_contexts headers."""
    sources = []
    for ctx in result.get("retrieved_contexts", []):
        first_line = ctx.splitlines()[0] if ctx else ""
        # Strip "[Document N] " / "[Web Result N] " prefix
        import re
        clean = re.sub(r"^\[(?:Document|Web Result)\s+\d+\]\s*", "", first_line).strip()
        if clean:
            sources.append(clean)
    return sources


def _extract_graph_paths(result: dict) -> list[str]:
    """Format contradiction report findings as traversal paths."""
    paths = []
    for finding in result.get("contradiction_report", []):
        company = finding.get("company", "")
        metric = finding.get("metric", "")
        period = finding.get("period", "")
        if company and metric:
            paths.append(f"{company} -> {metric} -> {period}")
    return paths


def build_audit_record(
    *,
    question: str,
    result: dict,
    latency_ms: int,
    tenant_id: str = "default",
    user_id: str = "anonymous",
) -> AuditRecord:
    return AuditRecord(
        query_id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        user_id=user_id,
        timestamp=datetime.now(timezone.utc),
        query=question,
        route_taken=result.get("final_route") or result.get("route"),
        sources_retrieved=_extract_sources(result),
        retrieval_scores=[],  # scores not yet propagated through result dict
        graph_paths_traversed=_extract_graph_paths(result),
        generated_answer=result.get("final_answer") or result.get("answer"),
        confidence_score=_derive_confidence_score(result),
        tokens_used=0,  # requires LLM instrumentation — tracked in future PR
        latency_ms=latency_ms,
        user_feedback=None,
        plan_type=result.get("plan_type"),
        cache_hit=bool(result.get("cache_hit", False)),
        company=result.get("company"),
        period=result.get("period"),
    )


def _write_to_bigquery(record: AuditRecord) -> None:
    try:
        client = _get_bq_client()
        errors = client.insert_rows_json(_TABLE_ID, [record.to_bq_row()])
        if errors:
            logger.warning(
                "BigQuery audit insert had errors",
                extra={"errors": errors, "query_id": record.query_id},
            )
        else:
            logger.info(
                "Audit record written",
                extra={"query_id": record.query_id, "route": record.route_taken},
            )
    except Exception as exc:
        # Never let audit failure surface to the user
        logger.warning(
            "Audit log write failed — query still succeeded",
            extra={"error": str(exc), "query_id": record.query_id},
        )


def log_query(record: AuditRecord) -> None:
    """Fire-and-forget: write audit record in a background thread."""
    thread = threading.Thread(target=_write_to_bigquery, args=(record,), daemon=True)
    thread.start()
