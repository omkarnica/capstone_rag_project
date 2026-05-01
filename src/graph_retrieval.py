from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.filings.config_loader import load_config_yaml
from src.model_config import get_graph_llm
from src.utils.secrets import get_secret

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - exercised in environments without neo4j
    GraphDatabase = None


CONFIG_PATH = Path(__file__).resolve().parent / "filings" / "config.yaml"
_CONFIG = load_config_yaml(CONFIG_PATH)
_DRIVER = None

load_dotenv()

_GRAPH_SCHEMA = """
Nodes:
- (:Company {ticker, name})
- (:Filing {filing_id, form_type, year, source_file})
- (:Section {section_id, title, text, ordinal, year, form_type})
- (:Subsidiary {subsidiary_id, name, years_present})
- (:BoardMember {id, name, title, is_current, years_present})
- (:Patent {patent_id, patent_title, grant_date,grant_year, assignee_organization, cpc_subclass})
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

_LEGAL_COMPANY_SUFFIXES = (
    "Inc.",
    "Corp.",
    "Corporation",
    "Ltd.",
    "Limited",
    "LLC",
    "PLC",
    "Group",
    "Holdings",
)


def _resolve_neo4j_settings() -> tuple[str, str, str]:
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not neo4j_uri:
        secret_id = str(_CONFIG.get("NEO4J_URI_SECRET", "NEO4J_URI"))
        neo4j_uri = get_secret(secret_id)
    if not neo4j_user:
        secret_id = str(_CONFIG.get("NEO4J_USER_SECRET", "NEO4J_USER"))
        neo4j_user = get_secret(secret_id)
    if not neo4j_password:
        secret_id = str(_CONFIG.get("NEO4J_PASSWORD_SECRET", "NEO4J_PASSWORD"))
        neo4j_password = get_secret(secret_id)

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


def _extract_company_full_name(question: str) -> str | None:
    suffix_pattern = "|".join(re.escape(suffix) for suffix in _LEGAL_COMPANY_SUFFIXES)
    match = re.search(
        rf"([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*\s+(?:{suffix_pattern}))",
        question,
    )
    return match.group(1).strip() if match else None


def _graph_prompt(question: str, *, company: str | None = None) -> str:
    company_hint = f"Preferred company ticker: {company.upper()}" if company else "No preferred company ticker"
    company_full_name = _extract_company_full_name(question)
    company_name_hint = (
        f"Preferred company full name: {company_full_name}\n"
        "Preserve the full company name exactly in the Cypher match when filtering by Company.name."
        if company_full_name
        else "No preferred company full name"
    )
    return f"""
You translate M&A due diligence questions into Cypher for Neo4j. Use only the schema below. Produce exactly one read-only Cypher query.

Allowed clauses: MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT.
Never use CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP, LOAD, FOREACH, or CALL.
Return at most 10 rows.

If the user asks about a company and a ticker hint is available, use it.
If the graph stores a company under its full legal name, prefer the exact stored company name over a shortened variant.
Always use the company title if ticker is not available. (e.g. for Apple, use "Apple Inc." if that's how it's stored in the graph, not just "Apple".)

Prefer returning named columns with human-readable aliases.
Return semantic aliases instead of opaque abbreviations where possible.
Include company and year fields when available.

For board-member questions, prefer years_present, is_current, and explicit year filtering when the question mentions a year.
Treat years_present as a list/array field. For year membership use `YEAR IN node.years_present`, not string checks like `CONTAINS "2024"`.

For subsidiary questions:
- The Subsidiary node has a `years_present` field which is a list of years (e.g. [2021, 2022, 2023]).
- ALWAYS filter by `WHERE <year> IN s.years_present` when the question mentions a year.
- If no year is mentioned, return all subsidiaries without a year filter.
- Always include s.years_present in the RETURN clause.
- Always use DISTINCT to prevent duplicate rows.
- Always return company name and subsidiary name.

For filing questions, include form type, year, filing_id, source_file, and section context when available.
For patent questions, include patent_id, patent_title, grant_date, grant_year, and domain fields when available.

YEAR FILTERING RULES (critical):
- If the question mentions a year, you MUST apply a WHERE filter on the relevant node's year field.
- Never silently drop a year filter.
- For subsidiaries:   WHERE <year> IN s.years_present
- For filings:        WHERE f.year = <year>
- For board members:  WHERE <year> IN b.years_present
- For patents:        WHERE p.grant_year = <year>
- If no year field exists on the target node, add a comment: // NOTE: year filter not applicable - no year field on this node type

DEDUPLICATION RULES (critical):
- Always use DISTINCT in RETURN clauses to prevent duplicate rows.
- For subsidiaries specifically: RETURN DISTINCT c.name, s.name, s.years_present

Return only Cypher. No explanation. No markdown unless it is a single cypher fence.

Schema:
{_GRAPH_SCHEMA}

{company_hint}
{company_name_hint}

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


def _generic_row_lines(row: dict[str, Any]) -> list[str]:
    return [f"{key}: {_format_value(value)}" for key, value in row.items()]


def _row_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _looks_like_board_member_row(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    return ("board_member" in keys or "boardmembername" in keys or "member_name" in keys) and (
        "title" in keys or "boardmembertitle" in keys
    )


def _format_board_member_row(row: dict[str, Any]) -> str:
    company = _row_value(
        row,
        "company",
        "Company",
        "CompanyName",
        "company_name",
        "ticker",
        "Ticker",
    ) or "Company"
    year = _row_value(row, "year", "Year")
    name = _row_value(row, "board_member", "BoardMemberName", "member_name", "MemberName") or "Unknown"
    title = _row_value(row, "title", "BoardMemberTitle", "member_title", "MemberTitle")
    years_present = _row_value(row, "years_present", "YearsPresent")

    head = f"{company} board member in {year}: {name}" if year else f"{company} board member: {name}"
    details = []
    if title:
        details.append(f"Title: {title}")
    if years_present:
        details.append(f"Years present: {_format_value(years_present)}")
    return " | ".join([head, *details]) if details else head


def _looks_like_subsidiary_row(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    return "subsidiary" in keys or "subsidiary_name" in keys


def _format_subsidiary_row(row: dict[str, Any]) -> str:
    company = _row_value(
        row,
        "company",
        "Company",
        "CompanyName",
        "company_name",
        "ticker",
        "Ticker",
    ) or "Company"
    year = _row_value(row, "year", "Year")
    name = _row_value(
        row,
        "subsidiary",
        "SubsidiaryName",
        "Subsidiary_Name",
        "subsidiary_name",
    ) or "Unknown"
    source_form_type = _row_value(row, "source_form_type", "SourceFormType")

    head = f"{company} subsidiary in {year}: {name}" if year else f"{company} subsidiary: {name}"
    details = [f"Source form: {source_form_type}"] if source_form_type else []
    return " | ".join([head, *details]) if details else head


def _looks_like_filing_row(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    return "filing_id" in keys and "form_type" in keys


def _format_filing_row(row: dict[str, Any]) -> str:
    company = _row_value(row, "company", "CompanyName", "company_name", "ticker", "Ticker") or "Company"
    year = _row_value(row, "year", "Year")
    form_type = _row_value(row, "form_type", "FormType") or "Filing"
    filing_id = _row_value(row, "filing_id", "FilingId", "FilingID")
    source_file = _row_value(row, "source_file", "SourceFile")

    parts = [f"{company} filing: {year} {form_type}" if year else f"{company} filing: {form_type}"]
    if filing_id:
        parts.append(f"Filing ID: {filing_id}")
    if source_file:
        parts.append(f"Source file: {source_file}")
    return " | ".join(parts)


def _looks_like_section_row(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    return ("section_id" in keys and "text" in keys) or "section_title" in keys


def _format_section_row(row: dict[str, Any]) -> str:
    company = _row_value(row, "company", "CompanyName", "company_name", "ticker", "Ticker") or "Company"
    year = _row_value(row, "year", "Year")
    form_type = _row_value(row, "form_type", "FormType")
    title = _row_value(row, "section_title", "SectionTitle", "title", "Title") or "Untitled section"
    section_id = _row_value(row, "section_id", "SectionId", "SectionID")
    text = _row_value(row, "text", "Text")

    head = f"{company} {year} {form_type} section: {title}" if year and form_type else f"{company} filing section: {title}"
    detail_parts = [f"Section ID: {section_id}"] if section_id else []
    body = f"{head} | {' | '.join(detail_parts)}" if detail_parts else head
    return f"{body}\n{text}" if text else body


def _looks_like_patent_row(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    return "patent_id" in keys or "patent_title" in keys


def _format_patent_row(row: dict[str, Any]) -> str:
    company = _row_value(row, "company", "CompanyName", "company_name", "ticker", "Ticker") or "Company"
    patent_id = _row_value(row, "patent_id", "PatentId", "PatentID") or "Unknown patent"
    title = _row_value(row, "patent_title", "PatentTitle", "title", "Title")
    grant_date = _row_value(row, "grant_date", "GrantDate")
    domain = _row_value(row, "domain", "Domain", "cpc_prefix", "CpcPrefix", "CPCPrefix")

    parts = [f"{company} patent: {patent_id}"]
    if title:
        parts.append(f"Title: {title}")
    if grant_date:
        parts.append(f"Grant date: {grant_date}")
    if domain:
        parts.append(f"Domain: {domain}")
    return " | ".join(parts)


def _looks_like_domain_row(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    return "cpc_prefix" in keys and "label" in keys and "patent_id" not in keys


def _format_domain_row(row: dict[str, Any]) -> str:
    company = _row_value(row, "company", "CompanyName", "company_name", "ticker", "Ticker") or "Company"
    prefix = _row_value(row, "cpc_prefix", "CpcPrefix", "CPCPrefix") or "Unknown"
    label = _row_value(row, "label", "Label") or "Unknown domain"
    return f"{company} technology domain: {prefix} | {label}"


def _row_to_doc(row: dict[str, Any], *, cypher: str) -> dict[str, Any]:
    if _looks_like_board_member_row(row):
        content = _format_board_member_row(row)
    elif _looks_like_subsidiary_row(row):
        content = _format_subsidiary_row(row)
    elif _looks_like_section_row(row):
        content = _format_section_row(row)
    elif _looks_like_filing_row(row):
        content = _format_filing_row(row)
    elif _looks_like_patent_row(row):
        content = _format_patent_row(row)
    elif _looks_like_domain_row(row):
        content = _format_domain_row(row)
    else:
        content = "\n".join(_generic_row_lines(row))
    return {
        "content": content,
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
