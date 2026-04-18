from __future__ import annotations

import json
from pathlib import Path

from src.utils.logger import get_logger


def test_get_logger_returns_logger_without_error() -> None:
    logger = get_logger("tests.logger.basic")
    assert logger is not None
    assert logger.name == "tests.logger.basic"


def test_log_line_written_and_json_parseable() -> None:
    logger = get_logger("tests.logger.file")
    log_file = Path("logs/app.log")

    if log_file.exists():
        before_size = log_file.stat().st_size
    else:
        before_size = 0

    logger.info("logger test line", extra={"test_case": "file_write_and_json"})
    for handler in logger.handlers:
        handler.flush()

    assert log_file.exists()
    assert log_file.stat().st_size >= before_size

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "Expected at least one log line in logs/app.log"

    parsed = json.loads(lines[-1])
    assert parsed["message"] == "logger test line"
    assert parsed["logger_name"] == "tests.logger.file"
    assert isinstance(parsed["kwargs"], dict)
