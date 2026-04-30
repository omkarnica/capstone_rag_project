from __future__ import annotations

from unittest.mock import patch


def test_get_secret_fetches_courtlistener_key_from_gcp_and_caches_it() -> None:
    from src.utils import secrets

    secrets._cache.clear()

    with patch("src.utils.secrets._fetch_from_gcp", return_value="court-secret") as mock_gcp:
        with patch("src.utils.secrets._fetch_from_env") as mock_env:
            first = secrets.get_secret("COURTLISTNER_API_KEY")
            second = secrets.get_secret("COURTLISTNER_API_KEY")

    assert first == "court-secret"
    assert second == "court-secret"
    mock_gcp.assert_called_once_with("COURTLISTNER_API_KEY", "codelab-2-485215")
    mock_env.assert_not_called()


def test_preload_secrets_requests_courtlistener_api_key() -> None:
    from src.utils import secrets

    secrets._cache.clear()

    with patch("src.utils.secrets._fetch_from_gcp", return_value="court-secret") as mock_gcp:
        secrets.preload_secrets()

    requested_secret_ids = [call.args[0] for call in mock_gcp.call_args_list]
    assert "COURTLISTNER_API_KEY" in requested_secret_ids
    assert "NEO4J_URI" in requested_secret_ids
    assert "NEO4J_USER" in requested_secret_ids
    assert "NEO4J_PASSWORD" in requested_secret_ids


def test_courtlistener_headers_use_secret_manager_key_name() -> None:
    from src.litigation.fetcher import _headers

    with patch("src.litigation.fetcher.get_secret", return_value="court-secret") as mock_get_secret:
        headers = _headers()

    assert headers == {"Authorization": "Token court-secret"}
    mock_get_secret.assert_called_once_with("COURTLISTNER_API_KEY")
