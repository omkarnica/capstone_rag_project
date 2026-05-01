from __future__ import annotations

from unittest.mock import patch


def test_get_connection_prefers_database_url_secret_when_no_explicit_params() -> None:
    from src.xbrl.loader import get_connection

    with patch("src.xbrl.loader.get_secret", return_value="postgresql://user:pass@db.example.io:5432/postgres") as mock_get_secret:
        with patch("src.xbrl.loader.psycopg2.connect", return_value=object()) as mock_connect:
            get_connection()

    mock_get_secret.assert_called_once_with("DATABASE_URL")
    mock_connect.assert_called_once_with("postgresql://user:pass@db.example.io:5432/postgres")


def test_get_connection_falls_back_to_pg_env_when_database_url_unavailable() -> None:
    from src.xbrl.loader import get_connection

    with patch("src.xbrl.loader.get_secret", side_effect=ValueError("missing")) as mock_get_secret:
        with patch.dict(
            "os.environ",
            {
                "PGHOST": "localhost",
                "PGPORT": "5433",
                "PGDATABASE": "ma_oracle",
                "PGUSER": "postgres",
                "PGPASSWORD": "secret",
            },
            clear=False,
        ):
            with patch("src.xbrl.loader.psycopg2.connect", return_value=object()) as mock_connect:
                get_connection()

    mock_get_secret.assert_called_once_with("DATABASE_URL")
    mock_connect.assert_called_once_with(
        host="localhost",
        port=5433,
        dbname="ma_oracle",
        user="postgres",
        password="secret",
    )


def test_app_startup_does_not_open_sql_connection() -> None:
    from fastapi.testclient import TestClient

    from src.app import app

    with (
        patch("src.app.preload_secrets"),
        patch("src.app.configure_langsmith_environment"),
        patch("src.xbrl.loader.psycopg2.connect", side_effect=AssertionError("SQL connection should be lazy")),
    ):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
