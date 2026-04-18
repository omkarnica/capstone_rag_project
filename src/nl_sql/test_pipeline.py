"""
Manual smoke-test for the NL-to-SQL pipeline against Microsoft data.

Run with:
    python -m src.nl_sql.test_pipeline

Requires GCP application-default credentials and a live PostgreSQL connection.
Not a pytest file — prints question, generated SQL, and results to stdout.
"""

from __future__ import annotations

from src.nl_sql.pipeline import ask
from src.utils.logger import get_logger

logger = get_logger(__name__)

_QUESTIONS = [
    "What is Microsoft's total revenue for fiscal year 2024?",
    "What are all the financial metrics available for Microsoft?",
    "What was Microsoft's net income in the most recent annual filing?",
    "Show me Microsoft's total assets over time",
]


def _print_result(result: dict) -> None:
    """Print question, SQL, and synthesized answer."""
    print(f"\n{'='*70}")
    print(f"Question : {result['question']}")
    print(f"SQL      :\n{result['sql']}")
    print(f"Answer   : {result['answer']}")


def main() -> None:
    for question in _QUESTIONS:
        logger.info("Running test question", extra={"question": question})
        try:
            result = ask(question)
            _print_result(result)
        except Exception as exc:
            print(f"\n{'='*70}")
            print(f"Question : {question}")
            print(f"ERROR    : {exc}")


if __name__ == "__main__":
    main()
