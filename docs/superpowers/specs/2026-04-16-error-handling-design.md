# Error Handling Design — M&A Oracle

**Date:** 2026-04-16  
**Branch:** karnica/error-handling  
**Status:** Implemented and tested

---

## Problem

Before this feature, all pipeline exceptions were raw Python/library types: `FileNotFoundError`, `psycopg2.Error`, `requests.HTTPError`. Callers had no stable contract — they either caught bare `Exception` or let errors propagate unchecked. HTTP downloads had no retry logic, so a single transient SEC EDGAR blip would abort an entire quarter load.

---

## Design

### Exception Hierarchy (`src/utils/exceptions.py`)

All domain exceptions inherit from `MAOracleError` so callers can catch broadly at the pipeline level or narrowly by subsystem:

```
MAOracleError (base)
├── DownloadError(url, status_code)      ← HTTP failures
├── ParseError(path)                     ← file read / TSV parse failures
├── DatabaseError(operation, table)      ← psycopg2 failures
└── NormalizationError(tag)              ← tag mapping failures (future)
```

Each subclass stores typed kwargs (not a generic dict). `MAOracleError.__str__` formats context fields inline so `str(exc)` gives the full picture for logging without attribute access.

### Context Manager Boundaries (also in `src/utils/exceptions.py`)

Three context managers translate raw library exceptions into domain exceptions at the point of the risky operation:

| Boundary | Catches | Raises |
|----------|---------|--------|
| `parse_error_boundary(path)` | `FileNotFoundError`, `pd.errors.ParserError`, `EmptyDataError`, `UnicodeDecodeError`, `OSError` | `ParseError(path=path)` |
| `db_error_boundary(operation, table)` | `psycopg2.Error` (all subclasses) | `DatabaseError(operation=..., table=...)` |
| `download_error_boundary(url)` | `requests.HTTPError`, `requests.RequestException` | `DownloadError(url=..., status_code=...)` |

This pattern was chosen over inline try/except because the translation logic lives once in `exceptions.py` — future pipelines (transcripts, patents, litigation) reuse the same boundaries.

### Tenacity Retry (`xbrl/main.py`)

`_download_zip(url)` is a private helper decorated with `@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)`.

**Critical ordering:** `download_error_boundary` wraps the *call* to `_download_zip`, not its body. This means tenacity exhausts all 3 attempts re-raising raw `requests` exceptions, then the boundary translates the survivor exactly once into `DownloadError`. If the boundary were inside each attempt it would convert on attempt 1 and tenacity would never retry.

### PipelineResult (`xbrl/main.py`)

```python
@dataclass
class PipelineResult:
    quarter: str
    success: bool
    filings_loaded: int = 0
    facts_loaded: int = 0
    error: str | None = None
```

`run_all_quarters` now returns `list[PipelineResult]` instead of `None`. Failed quarters are caught as `MAOracleError` (not bare `Exception`), recorded in a `PipelineResult`, and the loop continues. The final log line includes `failed_quarters` as a structured field.

---

## Files Changed

| File | Change |
|------|--------|
| `src/utils/exceptions.py` | New — exception hierarchy + 3 context manager boundaries |
| `xbrl/parser.py` | `_read_tsv` body wrapped in `parse_error_boundary` |
| `xbrl/loader.py` | `get_connection` and `_copy_dataframe` COPY op wrapped in `db_error_boundary` |
| `xbrl/main.py` | `_download_zip` private helper with `@retry`; `download_and_extract` uses boundary; `PipelineResult` dataclass; `run_all_quarters` returns `list[PipelineResult]` |
| `tests/test_error_handling.py` | New — 9 tests covering hierarchy, boundaries, and PipelineResult |
| `pyproject.toml` | Added `[tool.pytest.ini_options] pythonpath = ["."]` |

---

## Test Coverage

- Exception subclass relationships
- `__str__` format for each exception type (context fields inline, no brackets when empty)
- `parse_error_boundary`: `FileNotFoundError` → `ParseError` with correct `path` kwarg; non-caught exceptions pass through
- `db_error_boundary`: `OperationalError` → `DatabaseError` with `operation`; `ProgrammingError` → `DatabaseError` with `table` in str
- `PipelineResult`: failed quarter recorded with `success=False` and `error` populated; successful quarter recorded with counts
