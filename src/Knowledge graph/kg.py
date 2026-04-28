import argparse
import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from src.filings.config_loader import load_config_yaml

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - exercised in environments without neo4j
    GraphDatabase = None


logger = logging.getLogger("filings_kg")


CONFIG_PATH = Path(__file__).resolve().parents[1] / "filings" / "config.yaml"
_CONFIG = load_config_yaml(CONFIG_PATH)


def _resolve_neo4j_settings(config: dict[str, Any] | None = None) -> tuple[str, str, str]:
    resolved_config = config or _CONFIG
    neo4j_uri = os.getenv("NEO4J_URI") or str(
        resolved_config.get("NEO4J_URI", "bolt://localhost:7687")
    )
    neo4j_user = os.getenv("NEO4J_USER") or str(
        resolved_config.get("NEO4J_USER", "neo4j")
    )
    neo4j_password = os.getenv("NEO4J_PASSWORD") or str(
        resolved_config.get("NEO4J_PASSWORD", "password")
    )
    return neo4j_uri, neo4j_user, neo4j_password


NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD = _resolve_neo4j_settings()

PAGE_STAMP_RE = re.compile(r"^.+\|\s*\d{4}\s+Form\s+[A-Za-z0-9-]+\s*\|\s*\d+\s*$")
ITEM_HEADING_RE = re.compile(r"^Item\s+\d+[A-Z]?(?:\.\d+)?[.:]?\s+.+$", re.IGNORECASE)
ROLE_SUFFIXES = (
    "Lead Independent Director",
    "Independent Director",
    "Board Chair",
    "Board Chairman",
    "Board Chairperson",
    "Chairman",
    "Chairperson",
    "Chair",
    "Director",
)

CPC_CLASS_LABELS = {
    "H04": "Telecommunications & Signal Processing",
    "H01": "Electronic Components",
    "H03": "Electronic Circuitry",
    "H05": "Electric Techniques",
    "G06": "Computing & Data Processing",
    "G02": "Optics",
    "G01": "Measurement & Testing",
    "G09": "Displays & Advertising",
    "G10": "Acoustics",
    "G11": "Information Storage",
    "A61": "Medical & Health Devices",
    "A63": "Sports & Entertainment",
    "B32": "Layered Products & Materials",
    "C03": "Glass & Ceramics",
    "C09": "Dyes & Coatings",
    "F21": "Lighting",
}
SKIP_CPC_SECTIONS = {"Y"}


def load_json(path: str | Path) -> Any:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_year(entry: dict[str, Any]) -> int | None:
    for key in ("report_date", "filing_date"):
        value = entry.get(key)
        if not value:
            continue
        match = re.match(r"(\d{4})", str(value))
        if match:
            return int(match.group(1))
    year = entry.get("year")
    if isinstance(year, int):
        return year
    return None


def build_filing_id(ticker: str, form_type: str, year: int) -> str:
    return f"{ticker.upper()}_{form_type}_{year}"


def normalize_board_member_name(raw_name: Any) -> str:
    normalized = re.sub(r"\s+", " ", str(raw_name or "")).strip()
    if not normalized:
        return ""

    for suffix in sorted(ROLE_SUFFIXES, key=len, reverse=True):
        if normalized.lower() == suffix.lower():
            return ""
        if normalized.lower().endswith(f" {suffix.lower()}"):
            return normalized[: -(len(suffix) + 1)].strip()

    return normalized


def _aggregate_board_members(
    records: list[dict[str, Any]], ticker: str = "AAPL", current_year: int | None = None
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for record in records:
        year = record.get("year")
        if not isinstance(year, int):
            continue

        for member in record.get("members", []):
            normalized_name = normalize_board_member_name(member.get("name"))
            if not normalized_name:
                continue
            title = str(member.get("title") or "").strip()

            entry = grouped.setdefault(
                normalized_name,
                {
                    "name": normalized_name,
                    "latest_year": year,
                    "title": title,
                    "years_present": set(),
                },
            )
            entry["years_present"].add(year)
            if year > entry["latest_year"]:
                entry["latest_year"] = year
                entry["title"] = title
            elif year == entry["latest_year"]:
                # For same-year duplicates, keep the longest non-empty title seen.
                existing_title = str(entry.get("title") or "").strip()
                if len(title) > len(existing_title):
                    entry["title"] = title

    if current_year is None:
        latest_years = [entry["latest_year"] for entry in grouped.values()]
        current_year = max(latest_years) if latest_years else None

    ticker_value = ticker.upper()
    aggregated_members = []
    for name in sorted(grouped):
        entry = grouped[name]
        years_present = sorted(entry["years_present"])
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        aggregated_members.append(
            {
                "id": f"{ticker_value}_{slug}",
                "name": name,
                "ticker": ticker_value,
                "title": entry["title"],
                "years_present": years_present,
                "is_current": bool(years_present) and years_present[-1] == current_year,
            }
        )

    return aggregated_members


def normalize_subsidiary_id(ticker: str, name: Any) -> str:
    normalized_name = re.sub(r"\W+", "_", str(name or "").lower())
    normalized_name = re.sub(r"_+", "_", normalized_name).strip("_")
    return f"{ticker.upper()}_{normalized_name}"


def parse_cpc_codes(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    codes: list[str] = []
    for code in text.strip("{}").split(","):
        normalized = code.strip().strip('"').strip("'")
        if normalized:
            codes.append(normalized)
    return codes


def extract_cpc_sections(codes: list[str]) -> set[str]:
    return {
        code[:3]
        for code in codes
        if len(code) >= 3 and code[0].upper() not in SKIP_CPC_SECTIONS
    }


def filter_patents_csv(
    raw_path: str | Path,
    filtered_path: str | Path,
    assignee_organization: str | None = "Apple Inc.",
) -> str:
    raw_file = Path(raw_path)
    filtered_file = Path(filtered_path)
    expected_assignee = str(assignee_organization or "").strip().lower()

    with raw_file.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames or []
        rows = []
        for row in reader:
            assignee = str(row.get("assignee_organization") or "").strip().lower()
            if expected_assignee and assignee and assignee != expected_assignee:
                continue
            grant_date = str(row.get("grant_date") or "").strip()
            if len(grant_date) < 4 or not grant_date[:4].isdigit():
                continue
            grant_year = int(grant_date[:4])
            if 2021 <= grant_year <= 2025:
                rows.append(row)

    with filtered_file.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return str(filtered_file)


def build_subsidiary_relationship_payload(
    payload: dict[str, Any], ticker: str
) -> list[dict[str, Any]]:
    ticker_value = ticker.upper()
    source_form_type = payload.get("form_type")
    relationships: list[dict[str, Any]] = []

    for record in payload.get("subsidiaries_by_year", []):
        year = record.get("year")

        for subsidiary in record.get("subsidiaries", []):
            name = str(subsidiary or "").strip()
            if not name:
                continue

            relationships.append(
                {
                    "subsidiary_id": normalize_subsidiary_id(ticker_value, name),
                    "name": name,
                    "ticker": ticker_value,
                    "year": year,
                    "source_form_type": source_form_type,
                }
            )

    return relationships


def _extract_ordinal(self_ref: Any) -> int | None:
    if not self_ref:
        return None
    match = re.search(r"(\d+)$", str(self_ref))
    if not match:
        return None
    return int(match.group(1))


def _is_page_stamp(text: str) -> bool:
    return bool(PAGE_STAMP_RE.match(text.strip()))


def _is_title_block(text: str) -> bool:
    normalized = text.strip()
    if not normalized or _is_page_stamp(normalized):
        return False
    if len(normalized) > 80 or normalized.endswith((".", ",", "!", "?", ";", ":")):
        return False

    if ITEM_HEADING_RE.match(normalized):
        return True

    words = re.findall(r"\b[\w&/-]+\b", normalized)
    if not words or len(words) > 10:
        return False

    alpha_words = [word for word in words if any(char.isalpha() for char in word)]
    if not alpha_words:
        return False

    return all(word[:1].isupper() or word.isupper() for word in alpha_words)


def extract_sections_from_docling_entry(
    entry: dict[str, Any], source_file: str
) -> list[dict[str, Any]]:
    ticker = entry.get("ticker")
    form_type = entry.get("form_type")
    year = extract_year(entry)
    if not ticker or not form_type or year is None:
        return []

    filing_id = build_filing_id(str(ticker), str(form_type), year)
    texts = entry.get("docling", {}).get("texts", [])
    sections: list[dict[str, Any]] = []
    pending_title: str | None = None
    used_ordinals: set[int] = set()

    for block in texts:
        text = str(block.get("text", "")).strip()
        if not text:
            continue
        if block.get("content_layer") != "body":
            continue
        if block.get("hyperlink"):
            continue
        if _is_page_stamp(text):
            continue
        if len(text) < 50:
            if _is_title_block(text):
                pending_title = text
            continue

        ordinal = _extract_ordinal(block.get("self_ref"))
        ordinal_value = ordinal
        if ordinal_value is None or ordinal_value in used_ordinals:
            ordinal_value = 1
            while ordinal_value in used_ordinals:
                ordinal_value += 1

        used_ordinals.add(ordinal_value)
        sections.append(
            {
                "section_id": f"{filing_id}_section_{ordinal_value}",
                "filing_id": filing_id,
                "ticker": str(ticker).upper(),
                "form_type": str(form_type),
                "year": year,
                "ordinal": ordinal_value,
                "title": pending_title,
                "text": text,
                "source_file": source_file,
            }
        )
        pending_title = None

    return sections


def create_schema(driver) -> None:
    statements = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Filing) REQUIRE f.filing_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Subsidiary) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (b:BoardMember) REQUIRE b.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patent) REQUIRE p.patent_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:TechnologyDomain) REQUIRE d.cpc_prefix IS UNIQUE",
    ]
    with driver.session() as session:
        for statement in statements:
            session.run(statement)


def upsert_company(driver, ticker: str, name: str) -> None:
    with driver.session() as session:
        session.run(
            """
            MERGE (c:Company {ticker: $ticker})
            SET c.name = $name
            """,
            ticker=ticker.upper(),
            name=name,
        )


def ingest_docling_filings(driver, path: str | Path, ticker: str) -> None:
    entries = load_json(path)
    if not isinstance(entries, list):
        raise ValueError(f"Expected list of docling entries in {path}")

    ticker_value = ticker.upper()
    source_file = Path(path).name

    with driver.session() as session:
        for entry in entries:
            if not isinstance(entry, dict):
                continue

            year = extract_year(entry)
            form_type = str(entry.get("form_type") or "").strip()
            if year is None or not form_type:
                logger.warning("Skipping docling entry without year/form_type in %s", path)
                continue

            filing_id = build_filing_id(ticker_value, form_type, year)
            filing_payload = {
                "filing_id": filing_id,
                "ticker": ticker_value,
                "form_type": form_type,
                "year": year,
                "report_date": entry.get("report_date"),
                "filing_date": entry.get("filing_date"),
                "source": entry.get("source"),
                "source_file": source_file,
            }
            sections = extract_sections_from_docling_entry(
                {**entry, "ticker": ticker_value, "form_type": form_type, "year": year},
                source_file,
            )

            session.run(
                """
                MATCH (c:Company {ticker: $ticker})
                MERGE (f:Filing {filing_id: $filing_id})
                SET f.ticker = $ticker,
                    f.form_type = $form_type,
                    f.year = $year,
                    f.report_date = $report_date,
                    f.filing_date = $filing_date,
                    f.source = $source,
                    f.source_file = $source_file,
                    f.section_count = $section_count
                MERGE (c)-[:HAS_FILING]->(f)
                """,
                **filing_payload,
                section_count=len(sections),
            )

            for section in sections:
                session.run(
                    """
                    MATCH (f:Filing {filing_id: $filing_id})
                    MERGE (s:Section {section_id: $section_id})
                    SET s.filing_id = $filing_id,
                        s.ticker = $ticker,
                        s.form_type = $form_type,
                        s.year = $year,
                        s.ordinal = $ordinal,
                        s.title = $title,
                        s.text = $text,
                        s.source_file = $source_file
                    MERGE (f)-[:HAS_SECTION]->(s)
                    """,
                    **section,
                )


def ingest_subsidiaries(driver, path: str | Path, ticker: str) -> None:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected subsidiary payload dict in {path}")

    relationships = build_subsidiary_relationship_payload(payload, ticker)
    with driver.session() as session:
        for relationship in relationships:
            session.run(
                """
                MATCH (c:Company {ticker: $ticker})
                MERGE (s:Subsidiary {id: $subsidiary_id})
                SET s.name = $name,
                    s.ticker = $ticker
                MERGE (c)-[r:HAS_SUBSIDIARY {year: $year}]->(s)
                SET r.source_form_type = $source_form_type
                """,
                **relationship,
            )


def ingest_board_members(driver, path: str | Path, ticker: str) -> None:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected board-member payload dict in {path}")

    records = payload.get("board_members_by_year")
    if not isinstance(records, list):
        records = []
        for member in payload.get("board_members", []):
            years_present = member.get("years_present", [])
            for year in years_present:
                if isinstance(year, int):
                    records.append(
                        {
                            "year": year,
                            "members": [
                                {
                                    "name": member.get("name"),
                                    "title": member.get("title"),
                                }
                            ],
                        }
                    )

    aggregated = _aggregate_board_members(records, ticker=ticker.upper())
    with driver.session() as session:
        for member in aggregated:
            session.run(
                """
                MATCH (c:Company {ticker: $ticker})
                MERGE (b:BoardMember {id: $id})
                SET b.name = $name,
                    b.ticker = $ticker,
                    b.title = $title,
                    b.years_present = $years_present,
                    b.is_current = $is_current
                MERGE (c)-[r:HAS_BOARD_MEMBER]->(b)
                SET r.is_current = $is_current
                """,
                **member,
            )


def ingest_patents(driver, patents_csv_path: str | Path, ticker: str) -> None:
    patents_path = Path(patents_csv_path)
    ticker_value = ticker.upper()

    with patents_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        with driver.session() as session:
            company_exists = session.run(
                "MATCH (c:Company {ticker: $ticker}) RETURN c.ticker AS ticker LIMIT 1",
                ticker=ticker_value,
            ).single()
            if not company_exists:
                raise ValueError(f"Company node for ticker {ticker_value} must exist before patent ingestion")

            for row in reader:
                patent_id = str(row.get("patent_id") or "").strip()
                title = str(row.get("patent_title") or "").strip()
                grant_date = str(row.get("grant_date") or "").strip()
                if (
                    not patent_id
                    or not title
                    or len(grant_date) < 4
                    or not grant_date[:4].isdigit()
                ):
                    continue

                grant_year = int(grant_date[:4])
                codes = parse_cpc_codes(row.get("cpc_codes"))
                prefixes = sorted(extract_cpc_sections(codes))

                session.run(
                    """
                    MATCH (c:Company {ticker: $ticker})
                    MERGE (p:Patent {patent_id: $patent_id})
                    SET p.title = $title,
                        p.grant_date = $grant_date,
                        p.grant_year = $grant_year,
                        p.cpc_codes = $cpc_codes,
                        p.ticker = $ticker
                    MERGE (c)-[:HAS_PATENT]->(p)
                    """,
                    patent_id=patent_id,
                    title=title,
                    grant_date=grant_date,
                    grant_year=grant_year,
                    cpc_codes=codes,
                    ticker=ticker_value,
                )

                for prefix in prefixes:
                    section_letter = prefix[0]
                    session.run(
                        """
                        MERGE (d:TechnologyDomain {cpc_prefix: $prefix})
                        SET d.label = $label,
                            d.section = $section_letter
                        WITH d
                        MATCH (p:Patent {patent_id: $patent_id})
                        MERGE (p)-[:BELONGS_TO_DOMAIN {cpc_prefix: $prefix}]->(d)
                        """,
                        prefix=prefix,
                        label=CPC_CLASS_LABELS.get(prefix, f"Other ({prefix})"),
                        section_letter=section_letter,
                        patent_id=patent_id,
                    )


def log_graph_summary(driver, ticker: str) -> None:
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Company {ticker: $ticker})
            OPTIONAL MATCH (c)-[:HAS_FILING]->(f:Filing)
            OPTIONAL MATCH (f)-[:HAS_SECTION]->(s:Section)
            OPTIONAL MATCH (c)-[:HAS_SUBSIDIARY]->(sub:Subsidiary)
            OPTIONAL MATCH (c)-[:HAS_BOARD_MEMBER]->(bm:BoardMember)
            OPTIONAL MATCH (c)-[:HAS_PATENT]->(p:Patent)
            OPTIONAL MATCH (p)-[:BELONGS_TO_DOMAIN]->(td:TechnologyDomain)
            RETURN count(DISTINCT f) AS filings,
                   count(DISTINCT s) AS sections,
                   count(DISTINCT sub) AS subsidiaries,
                   count(DISTINCT bm) AS board_members,
                   count(DISTINCT p) AS patents,
                   count(DISTINCT td) AS technology_domains
            """,
            ticker=ticker.upper(),
        ).single()
    logger.info("Graph summary for %s: %s", ticker.upper(), dict(result or {}))


def run_filings_kg_build(
    ticker: str, data_dir: str | Path, driver=None
) -> dict[str, Any]:
    ticker_value = ticker.upper()
    if ticker_value != "AAPL":
        raise ValueError(
            "This standalone filings KG builder currently supports only AAPL."
        )

    data_path = Path(data_dir)
    own_driver = driver is None

    if own_driver:
        if GraphDatabase is None:
            raise ImportError("neo4j is required when no driver is injected")
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    docling_paths = [
        data_path / "aapl_10-k_docling.json",
        data_path / "aapl_10-q_docling.json",
        data_path / "aapl_8-k_docling.json",
    ]
    subsidiaries_path = data_path / "aapl_10-k_subsidiaries.json"
    board_members_path = data_path / "aapl_def14a_board_members.json"
    raw_patents_path = data_path / "patents.csv"
    filtered_patents_path = data_path / "patents_2021_2025.csv"

    try:
        create_schema(driver)
        upsert_company(
            driver,
            ticker_value,
            "Apple Inc." if ticker_value == "AAPL" else ticker_value,
        )
        for docling_path in docling_paths:
            ingest_docling_filings(driver, docling_path, ticker_value)
        ingest_subsidiaries(driver, subsidiaries_path, ticker_value)
        ingest_board_members(driver, board_members_path, ticker_value)
        filter_patents_csv(raw_patents_path, filtered_patents_path)
        ingest_patents(driver, filtered_patents_path, ticker_value)
        log_graph_summary(driver, ticker_value)
    finally:
        if own_driver and driver is not None:
            driver.close()

    return {
        "ticker": ticker_value,
        "data_dir": str(data_path),
        "docling_files": len(docling_paths),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Build the standalone filings KG.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    run_filings_kg_build(ticker=args.ticker, data_dir=args.data_dir)
