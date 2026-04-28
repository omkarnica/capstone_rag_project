from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.filings.config_loader import load_config_yaml
from src.model_config import get_graph_llm

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - exercised in environments without neo4j
    GraphDatabase = None


CONFIG_PATH = Path(__file__).resolve().parent / "filings" / "config.yaml"
_CONFIG = load_config_yaml(CONFIG_PATH)
_DRIVER = None

_GRAPH_SCHEMA = """
Nodes:
- (:Company {ticker, name})
- (:Filing {filing_id, form_type, year, source_file})
- (:Section {section_id, title, text, ordinal, year, form_type})
- (:Subsidiary {subsidiary_id, name})
- (:BoardMember {id, name, title, is_current, years_present})
- (:Patent {patent_id, patent_title, grant_date, assignee_organization, cpc_subclass})
- (:TechnologyDomain {cpc_prefix, label, section})

Relationships:
- (:Company)-[:HAS_FILING]->(:Filing)
- (:Filing)-[:HAS_SECTION]->(:Section)
- (:Company)-[:HAS_SUBSIDIARY]->(:Subsidiary)
- (:Company)-[:HAS_BOARD_MEMBER]->(:BoardMember)
- (:Company)-[:HAS_PATENT]->(:Patent)
- (:Patent)-[:BELONGS_TO_DOMAIN]->(:TechnologyDomain)
""".strip()

_WRITE_KEYWORDS = {
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "LOAD",
    "FOREACH",
    "CALL",
}


def _resolve_neo4j_settings() -> tuple[str, str, str]:
    neo4j_uri = os.getenv("NEO4J_URI") or str(_CONFIG.get("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user = os.getenv("NEO4J_USER") or str(_CONFIG.get("NEO4J_USER", "neo4j"))
    neo4j_password = os.getenv("NEO4J_PASSWORD") or str(_CONFIG.get("NEO4J_PASSWORD", "password"))
    return neo4j_uri, neo4j_user, neo4j_password


def get_neo4j_driver():
    global _DRIVER
    if _DRIVER is None:
        if GraphDatabase is None:
            raise ImportError("neo4j is required for knowledge graph retrieval")
        neo4j_uri, neo4j_user, neo4j_password = _resolve_neo4j_settings()
        _DRIVER = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    return _DRIVER


def _strip_cypher_fences(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:cypher)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def validate_read_only_cypher(cypher: str) -> str:
    normalized = _strip_cypher_fences(cypher)
    upper = normalized.upper()

    if ";" in normalized:
        raise ValueError("Cypher must be a single read-only statement")
    if "MATCH" not in upper:
        raise ValueError("Cypher must include MATCH")
    if "RETURN" not in upper:
        raise ValueError("Cypher must include RETURN")

    for keyword in _WRITE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", upper):
            raise ValueError("Cypher must be read-only")

    return normalized


def _graph_prompt(question: str, *, company: str | None = None) -> str:
    company_hint = f"Preferred company ticker: {company.upper()}" if company else "No preferred company ticker"
    return f"""
You translate M&A due diligence questions into Cypher for Neo4j.

Use only the schema below. Produce exactly one read-only Cypher query.
Allowed clauses: MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT.
Never use CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP, LOAD, FOREACH, or CALL.
Return at most 10 rows.
If the user asks about a company and a ticker hint is available, use it.
Prefer returning named columns with human-readable aliases.
Return only Cypher. No explanation. No markdown unless it is a single cypher fence.

Schema:
{_GRAPH_SCHEMA}

{company_hint}

Question:
{question}
""".strip()


def generate_cypher(question: str, *, company: str | None = None) -> str:
    llm = get_graph_llm()
    response = llm.invoke(_graph_prompt(question, company=company))
    content = response.content if hasattr(response, "content") else str(response)
    return validate_read_only_cypher(str(content))


def _format_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _row_to_doc(row: dict[str, Any], *, cypher: str) -> dict[str, Any]:
    lines = [f"{key}: {_format_value(value)}" for key, value in row.items()]
    return {
        "content": "\n".join(lines),
        "metadata": {
            "source": "Knowledge Graph",
            "title": "Neo4j Graph Result",
            "cypher": cypher,
        },
    }


def retrieve_graph_docs(question: str, *, company: str | None = None) -> list[dict[str, Any]]:
    cypher = generate_cypher(question, company=company)
    driver = get_neo4j_driver()

    with driver.session() as session:
        rows = session.run(cypher).data()

    return [_row_to_doc(row, cypher=cypher) for row in rows]
