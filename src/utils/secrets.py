from __future__ import annotations

import os

from dotenv import load_dotenv

from src.utils.logger import get_logger

_PROJECT_ID = "codelab-2-485215"
_cache: dict[str, str] = {}
logger = get_logger(__name__)

# All secrets the app uses — fetched eagerly at startup
_KNOWN_SECRETS = [
    "PINECONE_API_KEY",
    "COURTLISTENER_API_KEY",
    "LangSmith_key",
]


def _fetch_from_gcp(secret_id: str, project_id: str) -> str:
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8").strip()


def _fetch_from_env(secret_id: str) -> str:
    load_dotenv()
    value = os.getenv(secret_id)
    if not value:
        raise ValueError(f"Secret '{secret_id}' not found in GCP Secret Manager or .env")
    return value


def get_secret(secret_id: str, project_id: str = _PROJECT_ID) -> str:
    """Return secret value, using in-memory cache to avoid repeated GCP calls."""
    if secret_id in _cache:
        return _cache[secret_id]
    try:
        value = _fetch_from_gcp(secret_id, project_id)
        logger.info("Loaded secret from GCP Secret Manager", extra={"secret_id": secret_id})
    except Exception:
        value = _fetch_from_env(secret_id)
        logger.info("Loaded secret from .env", extra={"secret_id": secret_id})
    _cache[secret_id] = value
    return value


def preload_secrets(project_id: str = _PROJECT_ID) -> None:
    """Fetch all known secrets from GCP at startup so runtime calls are instant.
    Secrets unavailable in GCP fall back to .env; missing from both are warned, not raised.
    """
    for secret_id in _KNOWN_SECRETS:
        if secret_id in _cache:
            continue
        try:
            _cache[secret_id] = _fetch_from_gcp(secret_id, project_id)
            logger.info("Preloaded secret from GCP", extra={"secret_id": secret_id})
        except Exception as gcp_exc:
            try:
                _cache[secret_id] = _fetch_from_env(secret_id)
                logger.info("Preloaded secret from .env fallback", extra={"secret_id": secret_id})
            except ValueError:
                logger.warning(
                    "Secret unavailable at startup — will fail if used at runtime",
                    extra={"secret_id": secret_id, "gcp_error": str(gcp_exc)},
                )
