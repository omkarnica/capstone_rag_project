from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_STANDARD_LOG_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        kwargs_context: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_ATTRS and key != "kwargs":
                kwargs_context[key] = value

        if isinstance(getattr(record, "kwargs", None), dict):
            kwargs_context.update(record.kwargs)

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
            "kwargs": kwargs_context,
        }
        return json.dumps(payload, default=str)


def _resolve_log_level() -> int:
    return logging.DEBUG if os.getenv("LOG_LEVEL", "").upper() == "DEBUG" else logging.INFO


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    level = _resolve_log_level()
    logger.setLevel(level)

    if logger.handlers:
        return logger

    logger.propagate = False
    logs_dir = _project_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=logs_dir / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
