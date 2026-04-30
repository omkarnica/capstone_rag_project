from __future__ import annotations

from src.graph_retrieval import retrieve_graph_docs
from src.state import GraphState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve_docs(state: GraphState) -> GraphState:
    """Dispatch retrieval to the appropriate M&A data source based on route."""
    route = state.get("route", "llm_direct")
    query = state.get("rewritten_question") or state["question"]
    company = state.get("company")
    docs: list[dict] = []
    filings_error = False
    filings_empty = False

    if route == "sql":
        try:
            from src.nl_sql.pipeline import ask as nl_sql_ask
            result = nl_sql_ask(query)
            docs = [{
                "content": result.get("answer", ""),
                "metadata": {
                    "source": "XBRL/SQL",
                    "sql": result.get("sql", ""),
                },
            }]
        except Exception as exc:
            logger.warning("SQL retrieval failed: %s", exc)

    elif route == "filings":
        try:
            from src.filings.raptor_retrieval import raptor_retrieve
            eval_config = state.get("eval_config") or {}
            use_reranker = eval_config.get("reranker", True)
            result = raptor_retrieve(query, top_k=10, final_top_k=6, use_reranker=use_reranker)
            docs = [
                {
                    "content": c.get("text", ""),
                    "metadata": {
                        "source": "SEC Filing",
                        "form_type": c.get("form_type"),
                        "rank": c.get("rank"),
                    },
                }
                for c in result.get("contexts", [])
            ]
            filings_empty = len(docs) == 0
        except Exception as exc:
            filings_error = True
            filings_empty = True
            logger.warning("Filings retrieval failed: %s", exc)

    elif route == "transcripts":
        try:
            from src.transcripts.retrieval import retrieve_transcripts
            hits = retrieve_transcripts(query, company=company)
            docs = [
                {
                    "content": h.get("fields", {}).get("text", ""),
                    "metadata": {
                        "source": "Earnings Transcript",
                        "company": h.get("fields", {}).get("company_title"),
                        "period": h.get("fields", {}).get("period_of_report"),
                        "accession_no": h.get("fields", {}).get("accession_no"),
                    },
                }
                for h in hits
            ]
        except Exception as exc:
            logger.warning("Transcripts retrieval failed: %s", exc)

    elif route == "patents":
        try:
            from src.patents.retrieval import retrieve_patents
            hits = retrieve_patents(query, company=company)
            docs = [
                {
                    "content": h.get("fields", {}).get("text", ""),
                    "metadata": {
                        "source": "Patent",
                        "patent_id": h.get("fields", {}).get("patent_id"),
                        "patent_title": h.get("fields", {}).get("patent_title"),
                        "grant_date": h.get("fields", {}).get("grant_date"),
                    },
                }
                for h in hits
            ]
        except Exception as exc:
            logger.warning("Patents retrieval failed: %s", exc)

    elif route == "litigation":
        try:
            from src.litigation.retrieval import retrieve_litigation
            hits = retrieve_litigation(query, company=company)
            docs = [
                {
                    "content": h.get("fields", {}).get("text", ""),
                    "metadata": {
                        "source": "Litigation",
                        "case_name": h.get("fields", {}).get("case_name"),
                        "court": h.get("fields", {}).get("court"),
                        "date_filed": h.get("fields", {}).get("date_filed"),
                    },
                }
                for h in hits
            ]
        except Exception as exc:
            logger.warning("Litigation retrieval failed: %s", exc)

    elif route == "graph":
        try:
            docs = retrieve_graph_docs(query, company=company)
        except Exception as exc:
            logger.warning("Graph retrieval failed: %s", exc)

    # contradiction, llm_direct — no retrieval here

    return {
        **state,
        "retrieval_query": query,
        "retrieved_docs": docs,
        "filtered_docs": [],
        "doc_relevance": [],
        "relevant_doc_count": 0,
        "filings_error": filings_error,
        "filings_empty": filings_empty,
    }
