"""
Shared YAML configuration loader for src.filings modules.

Dependency diagram:

    src.filings.config.yaml
      |
      v
    load_config_yaml()
      |
      |-- ingestion.py
      |-- chunking.py
      |-- raptor.py
      |-- raptor_retrieval.py
      `-- raptor_verification.py

The default config path is this package's config.yaml. Callers may pass an
explicit path for tests or alternate runtime configs.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config_yaml(config_path: str | Path | None = None) -> Dict[str, Any]:
    base_dir = Path(__file__).resolve().parent
    resolved_path = Path(config_path) if config_path else (base_dir / "config.yaml")

    if not resolved_path.exists():
        return {}

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}
