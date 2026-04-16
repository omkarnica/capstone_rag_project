"""
Domain exception hierarchy for M&A Oracle.

All pipeline exceptions inherit from MAOracleError so callers can catch
broadly at the pipeline level or narrowly at the subsystem level.

Usage:
    from src.utils.exceptions import ParseError, DatabaseError, DownloadError

    raise ParseError("File not found", path="/data/raw/2024q1/sub.txt")
    raise DownloadError("HTTP 429", url=url, status_code=429)
    raise DatabaseError("COPY failed", operation="COPY", table="filings_staging")

Logging:
    str(exc) includes all context fields inline, so logger.error(str(e))
    gives the full picture without accessing individual attributes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import pandas as pd
import psycopg2
import requests


class MAOracleError(Exception):
    """Base exception for all M&A Oracle pipeline errors."""

    def _context_str(self) -> str:
        """Return formatted context fields. Overridden by each subclass."""
        return ""

    def __str__(self) -> str:
        base = super().__str__()
        ctx = self._context_str()
        return f"{base} [{ctx}]" if ctx else base


class DownloadError(MAOracleError):
    """Raised when an HTTP download from an external source fails."""

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code

    def _context_str(self) -> str:
        parts = []
        if self.url is not None:
            parts.append(f"url={self.url}")
        if self.status_code is not None:
            parts.append(f"status_code={self.status_code}")
        return ", ".join(parts)


class ParseError(MAOracleError):
    """Raised when a file cannot be read or parsed."""

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path

    def _context_str(self) -> str:
        return f"path={self.path}" if self.path is not None else ""


class DatabaseError(MAOracleError):
    """Raised when a database operation fails."""

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        table: str | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.table = table

    def _context_str(self) -> str:
        parts = []
        if self.operation is not None:
            parts.append(f"operation={self.operation}")
        if self.table is not None:
            parts.append(f"table={self.table}")
        return ", ".join(parts)


class NormalizationError(MAOracleError):
    """Raised when tag normalization fails."""

    def __init__(self, message: str, *, tag: str | None = None) -> None:
        super().__init__(message)
        self.tag = tag

    def _context_str(self) -> str:
        return f"tag={self.tag}" if self.tag is not None else ""


# ── Context manager error boundaries ─────────────────────────────────────────


@contextmanager
def parse_error_boundary(path: str) -> Generator[None, None, None]:
    """
    Translate file-read exceptions into ParseError.

    Catches:
        FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError,
        UnicodeDecodeError, OSError

    Usage:
        with parse_error_boundary(str(path)):
            df = pd.read_csv(path, ...)
    """
    try:
        yield
    except (
        FileNotFoundError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        OSError,
    ) as exc:
        raise ParseError(str(exc), path=path) from exc


@contextmanager
def db_error_boundary(
    operation: str, table: str | None = None
) -> Generator[None, None, None]:
    """
    Translate psycopg2 exceptions into DatabaseError.

    Catches:
        psycopg2.Error (covers OperationalError, ProgrammingError, etc.)

    Usage:
        with db_error_boundary("COPY", table="filings_staging"):
            cur.copy_expert(sql, buffer)
    """
    try:
        yield
    except psycopg2.Error as exc:
        raise DatabaseError(str(exc), operation=operation, table=table) from exc


@contextmanager
def download_error_boundary(url: str) -> Generator[None, None, None]:
    """
    Translate requests exceptions into DownloadError.

    Catches:
        requests.HTTPError (non-2xx), requests.ConnectionError,
        requests.Timeout, requests.RequestException

    Usage:
        with download_error_boundary(url):
            response = requests.get(url, ...)
            response.raise_for_status()
    """
    try:
        yield
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        raise DownloadError(str(exc), url=url, status_code=status) from exc
    except requests.RequestException as exc:
        raise DownloadError(str(exc), url=url) from exc
