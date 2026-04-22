"""
Litigation pipeline orchestrator for M&A Oracle.

run_pipeline:
    Fetches court opinions from CourtListener for a given company and date range,
    loads cases and opinions into PostgreSQL, and returns counts.

main:
    CLI entry point. Run the pipeline then optionally upsert to Pinecone.

CLI:
    python -m src.litigation.main --company "Apple Inc"
    python -m src.litigation.main --company "Microsoft" --filed-after 2018-01-01
    python -m src.litigation.main --company "Apple Inc" --upsert-pinecone
"""

from __future__ import annotations

import argparse
import time

from src.litigation.fetcher import fetch_cases, fetch_opinion_text
from src.litigation.loader import load_litigation_data
from src.utils.logger import get_logger

logger = get_logger(__name__)

_KNOWN_COMPANIES: dict[str, str] = {
    "apple":     "Apple Inc",
    "microsoft": "Microsoft Corporation",
}

# Polite delay between individual opinion text fetches
_OPINION_FETCH_DELAY = 0.3


def run_pipeline(
    company_name: str,
    filed_after: str | None = None,
    filed_before: str | None = None,
    max_cases: int = 100,
    fetch_full_text: bool = True,
) -> dict:
    """
    Fetch litigation data from CourtListener and load into PostgreSQL.

    Step 1: Search for cases by company name (paginated).
    Step 2: Optionally fetch full opinion text for each opinion ID.
    Step 3: Load cases and opinions into PostgreSQL via COPY.

    Args:
        company_name:    Company to search for (e.g. "Apple Inc").
        filed_after:     ISO date lower bound on date_filed.
        filed_before:    ISO date upper bound on date_filed.
        max_cases:       Maximum number of cases to fetch.
        fetch_full_text: If True, fetch full opinion plain_text via API.
                         If False, store only the snippet (faster, less content).

    Returns:
        Dict with keys: cases_fetched, opinions_fetched, cases_inserted, opinions_inserted.
    """
    logger.info(
        "Litigation pipeline start",
        extra={
            "company": company_name,
            "filed_after": filed_after,
            "filed_before": filed_before,
            "max_cases": max_cases,
            "fetch_full_text": fetch_full_text,
        },
    )

    cases = fetch_cases(
        company_name=company_name,
        filed_after=filed_after,
        filed_before=filed_before,
        max_results=max_cases,
    )

    if not cases:
        logger.info("No cases found", extra={"company": company_name})
        return {"cases_fetched": 0, "opinions_fetched": 0, "cases_inserted": 0, "opinions_inserted": 0}

    # Separate case records (strip the opinions list) and build opinion records
    case_records: list[dict] = []
    opinion_records: list[dict] = []

    for case in cases:
        case_records.append({k: v for k, v in case.items() if k != "opinions"})

        for op in case.get("opinions", []):
            opinion_id = str(op.get("id", ""))
            if not opinion_id:
                continue

            plain_text = ""
            if fetch_full_text:
                try:
                    plain_text = fetch_opinion_text(opinion_id)
                    time.sleep(_OPINION_FETCH_DELAY)
                except Exception as exc:
                    logger.warning(
                        "Opinion text fetch failed — using snippet",
                        extra={"opinion_id": opinion_id, "error": str(exc)},
                    )

            opinion_records.append({
                "opinion_id":    opinion_id,
                "case_id":       case["case_id"],
                "opinion_type":  op.get("type", ""),
                "plain_text":    plain_text,
                "snippet":       op.get("snippet", ""),
            })

    cases_inserted, opinions_inserted = load_litigation_data(case_records, opinion_records)

    result = {
        "cases_fetched":     len(cases),
        "opinions_fetched":  len(opinion_records),
        "cases_inserted":    cases_inserted,
        "opinions_inserted": opinions_inserted,
    }

    logger.info("Litigation pipeline complete", extra=result)
    return result


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description="Fetch CourtListener litigation data into PostgreSQL."
    )
    p.add_argument(
        "--company", required=True,
        help="Company name to search for (e.g. 'Apple Inc').",
    )
    p.add_argument(
        "--filed-after", default=None,
        help="ISO date lower bound on date_filed (e.g. '2018-01-01').",
    )
    p.add_argument(
        "--filed-before", default=None,
        help="ISO date upper bound on date_filed (e.g. '2024-12-31').",
    )
    p.add_argument(
        "--max-cases", type=int, default=100,
        help="Maximum number of cases to fetch (default: 100).",
    )
    p.add_argument(
        "--snippets-only", action="store_true",
        help="Skip fetching full opinion text; store snippet only (faster).",
    )
    p.add_argument(
        "--upsert-pinecone", action="store_true",
        help="After loading PostgreSQL, upsert to Pinecone.",
    )
    args = p.parse_args()

    result = run_pipeline(
        company_name=args.company,
        filed_after=args.filed_after,
        filed_before=args.filed_before,
        max_cases=args.max_cases,
        fetch_full_text=not args.snippets_only,
    )
    print(result)

    if args.upsert_pinecone:
        from src.litigation.pinecone_loader import upsert_litigation
        total = upsert_litigation(company=args.company)
        logger.info("Pinecone upsert complete", extra={"total_vectors": total})


if __name__ == "__main__":
    main()
