"""
Tests for the error handling layer: exception hierarchy, context manager
boundaries, PipelineResult, and psycopg2 → DatabaseError translation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg2
import pytest

from src.utils.exceptions import (
    DatabaseError,
    DownloadError,
    MAOracleError,
    NormalizationError,
    ParseError,
    db_error_boundary,
    download_error_boundary,
    parse_error_boundary,
)


# ── Exception hierarchy ───────────────────────────────────────────────────────


def test_all_exceptions_inherit_from_base() -> None:
    assert issubclass(ParseError, MAOracleError)
    assert issubclass(DownloadError, MAOracleError)
    assert issubclass(DatabaseError, MAOracleError)
    assert issubclass(NormalizationError, MAOracleError)


def test_parse_error_str_includes_path() -> None:
    exc = ParseError("File not found", path="/data/sub.txt")
    assert "/data/sub.txt" in str(exc)


def test_download_error_str_includes_url_and_status() -> None:
    exc = DownloadError("HTTP error", url="https://sec.gov/foo.zip", status_code=429)
    assert "https://sec.gov/foo.zip" in str(exc)
    assert "429" in str(exc)


def test_database_error_str_includes_operation_and_table() -> None:
    exc = DatabaseError("COPY failed", operation="COPY", table="filings_staging")
    assert "COPY" in str(exc)
    assert "filings_staging" in str(exc)


def test_base_str_no_context_omits_brackets() -> None:
    # When no context fields are set, __str__ should not append " []"
    exc = MAOracleError("bare error")
    assert str(exc) == "bare error"


# ── parse_error_boundary ──────────────────────────────────────────────────────


def test_bad_tsv_path_raises_parse_error_not_file_not_found(tmp_path: Path) -> None:
    """
    The most important contract: callers never see a raw FileNotFoundError.
    """
    missing = tmp_path / "does_not_exist.txt"

    with pytest.raises(ParseError) as exc_info:
        with parse_error_boundary(str(missing)):
            import pandas as pd
            pd.read_csv(missing, sep="\t")

    # Must NOT be the raw stdlib exception
    assert not isinstance(exc_info.value.__cause__, type(None))
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)
    # path kwarg must be threaded through
    assert exc_info.value.path == str(missing)


def test_parse_error_boundary_does_not_swallow_other_exceptions() -> None:
    """Exceptions outside the catch list must propagate unchanged."""
    with pytest.raises(ValueError):
        with parse_error_boundary("/irrelevant"):
            raise ValueError("unrelated")


# ── db_error_boundary ─────────────────────────────────────────────────────────


def test_psycopg2_operational_error_raises_database_error() -> None:
    """
    A psycopg2.OperationalError (e.g. connection refused) must surface as
    DatabaseError, never leak as a raw psycopg2 type.
    """
    with pytest.raises(DatabaseError) as exc_info:
        with db_error_boundary("connect"):
            raise psycopg2.OperationalError("connection refused")

    assert isinstance(exc_info.value.__cause__, psycopg2.OperationalError)
    assert exc_info.value.operation == "connect"


def test_db_error_boundary_carries_table_name() -> None:
    with pytest.raises(DatabaseError) as exc_info:
        with db_error_boundary("COPY", table="filings_staging"):
            raise psycopg2.ProgrammingError("relation does not exist")

    assert exc_info.value.table == "filings_staging"
    assert "filings_staging" in str(exc_info.value)


# ── PipelineResult + run_all_quarters ────────────────────────────────────────


def test_pipeline_result_captures_failed_quarters() -> None:
    """
    When download_and_extract raises a MAOracleError, run_all_quarters must
    record a failed PipelineResult for that quarter and continue with the rest.
    """
    from src.xbrl.main import QUARTERS, run_all_quarters

    failing_quarter = QUARTERS[0]

    def fake_download(quarter: str, extract_dir: Path):
        if quarter == failing_quarter:
            raise DownloadError(
                "Simulated HTTP failure",
                url="https://sec.gov/fake.zip",
                status_code=503,
            )
        # For all other quarters, also fail quickly so the test stays fast
        raise DownloadError("Fast-fail", url="https://sec.gov/fake.zip")

    with patch("src.xbrl.main.download_and_extract", side_effect=fake_download):
        results = run_all_quarters(extract_dir=Path("/tmp/test_quarters"))

    # Every quarter should have a result
    assert len(results) == len(QUARTERS)

    # The first quarter must be recorded as failed
    first = next(r for r in results if r.quarter == failing_quarter)
    assert first.success is False
    assert first.filings_loaded == 0
    assert first.facts_loaded == 0
    assert first.error is not None
    assert "503" in first.error  # status_code threaded into str(exc)


def test_pipeline_result_captures_success() -> None:
    """
    A quarter that completes successfully must record filings and facts counts.
    """
    from src.xbrl.main import QUARTERS, run_all_quarters

    target = QUARTERS[0]

    def fake_download(quarter: str, extract_dir: Path):
        if quarter == target:
            sub = MagicMock(spec=Path)
            num = MagicMock(spec=Path)
            return sub, num
        raise DownloadError("Fast-fail", url="https://sec.gov/fake.zip")

    def fake_pipeline(sub_path, num_path):
        return 42, 1000  # filings, facts

    with (
        patch("src.xbrl.main.download_and_extract", side_effect=fake_download),
        patch("src.xbrl.main.run_pipeline", side_effect=fake_pipeline),
    ):
        results = run_all_quarters(extract_dir=Path("/tmp/test_quarters"))

    success = next(r for r in results if r.quarter == target)
    assert success.success is True
    assert success.filings_loaded == 42
    assert success.facts_loaded == 1000
    assert success.error is None
