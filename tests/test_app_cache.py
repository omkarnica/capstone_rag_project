from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_cache_stats_returns_backend_stats() -> None:
    from src.app import app

    with patch("src.app.preload_secrets"), patch("src.app.configure_langsmith_environment"):
        client = TestClient(app)

        with patch(
            "src.app.get_cache_backend",
            return_value=type(
                "Backend",
                (),
                {
                    "get_stats": lambda self: {
                        "backend": "redis",
                        "doc_version": 3,
                        "exact": {"entries": 1, "total_hits": 2},
                        "semantic": {"entries": 4, "total_hits": 5},
                        "retrieval": {"entries": 6, "total_hits": 7},
                    }
                },
            )(),
        ):
            response = client.get("/cache/stats")

    assert response.status_code == 200
    assert response.json()["backend"] == "redis"
    assert response.json()["semantic"]["entries"] == 4


def test_cache_clear_returns_cleared_counts() -> None:
    from src.app import app

    with patch("src.app.preload_secrets"), patch("src.app.configure_langsmith_environment"):
        client = TestClient(app)

        with patch(
            "src.app.get_cache_backend",
            return_value=type(
                "Backend",
                (),
                {
                    "clear_all": lambda self: {
                        "exact": 1,
                        "semantic": 2,
                        "retrieval": 3,
                    }
                },
            )(),
        ):
            response = client.post("/cache/clear")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "cleared": {
            "exact": 1,
            "semantic": 2,
            "retrieval": 3,
        },
    }


def test_cache_endpoints_return_503_when_backend_unavailable() -> None:
    from src.app import app

    with patch("src.app.preload_secrets"), patch("src.app.configure_langsmith_environment"):
        client = TestClient(app)

        with patch("src.app.get_cache_backend", return_value=None):
            stats_response = client.get("/cache/stats")
            clear_response = client.post("/cache/clear")

    assert stats_response.status_code == 503
    assert clear_response.status_code == 503
