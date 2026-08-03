"""Tests for FastAPI endpoints — mocked engine for CI (no Ollama needed)."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api.main as api_mod
from retrieval.searcher import SearchResult


@pytest.fixture(autouse=True)
def setup():
    from api.main import reset_rate_limit_store

    reset_rate_limit_store()
    mock_searcher = MagicMock()
    mock_searcher.search.return_value = [
        SearchResult(text="chunk text about transformers...", source="transformer-architecture.md", score=0.95),
        SearchResult(text="chunk text about attention...", source="attention-is-all-you-need.md", score=0.87),
        SearchResult(text="chunk text about embeddings...", source="vector-databases.md", score=0.72),
    ]
    mock_engine = MagicMock()
    mock_engine.answer.return_value = {
        "question": "What is RAG?",
        "answer": "RAG stands for Retrieval-Augmented Generation...",
        "model": "Mock Model",
        "sources": [{"document": "rag-explained.md", "score": 0.92}],
    }
    mock_engine.answer_stream.return_value = iter([
        "RAG stands for ",
        "Retrieval-Augmented Generation.",
        {
            "_done": True,
            "sources": [{"document": "rag-explained.md", "score": 0.92}],
            "model": "Mock Model",
        },
    ])
    api_mod.searcher = mock_searcher
    api_mod.engine = mock_engine

@pytest.fixture
def client():
    from api.main import app
    # raise_server_exceptions=False so the 500 path is testable (handler returns a response).
    return TestClient(app, raise_server_exceptions=False)

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_search(client):
    r = client.get("/search?q=transformer&top_k=3")
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 3
    assert all("source" in r for r in data["results"])

def test_ask(client):
    r = client.post("/ask", json={"question": "What is RAG?", "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert len(data["answer"]) > 10
    assert len(data["sources"]) > 0

def test_ask_unknown_handled(client):
    r = client.post("/ask", json={"question": "What is the capital of Mars?", "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert len(data["answer"]) > 5

def test_ask_empty_question_422(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422

def test_ask_top_k_out_of_range_422(client):
    r = client.post("/ask", json={"question": "hi", "top_k": 0})
    assert r.status_code == 422
    r = client.post("/ask", json={"question": "hi", "top_k": 100})
    assert r.status_code == 422

def test_search_empty_query_422(client):
    r = client.get("/search?q=")
    assert r.status_code == 422


def test_ask_stream_sse(client):
    r = client.post("/ask/stream", json={"question": "What is RAG?", "top_k": 3})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert "data: " in body
    parsed = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
    assert parsed and parsed[-1] == "[DONE]"
    tokens = [json.loads(p) for p in parsed[:-1]]
    assert any("token" in t for t in tokens)
    # name of the sources/meta frame must not collide with [DONE]
    meta = [t for t in tokens if "sources" in t]
    assert meta and meta[0]["model"] == "Mock Model"


def test_ask_stream_empty_question_422(client):
    r = client.post("/ask/stream", json={"question": ""})
    assert r.status_code == 422


def test_ask_stream_engine_none_503(client, monkeypatch):
    monkeypatch.setattr(api_mod, "engine", None)
    r = client.post("/ask/stream", json={"question": "hi", "top_k": 3})
    assert r.status_code == 503


def test_ask_engine_none_503(client, monkeypatch):
    monkeypatch.setattr(api_mod, "engine", None)
    r = client.post("/ask", json={"question": "hi", "top_k": 3})
    assert r.status_code == 503


def test_search_engine_none_503(client, monkeypatch):
    monkeypatch.setattr(api_mod, "searcher", None)
    r = client.get("/search?q=hi")
    assert r.status_code == 503


def test_value_error_returns_generic(client, monkeypatch):
    """Internal ValueError -> 400 generic, no internal detail leaked."""
    def boom(q="x", top_k=5):
        raise ValueError("secret internal detail: db table users corrupt")
    monkeypatch.setattr(api_mod.engine, "answer", boom)
    r = client.post("/ask", json={"question": "x", "top_k": 5})
    assert r.status_code == 400
    assert "secret internal" not in str(r.json())


def test_unhandled_error_returns_generic(client, monkeypatch):
    """Unhandled exception -> 500 generic + trace_id, no internal detail leaked."""
    def boom(q="x", top_k=5):
        raise RuntimeError("secret internal detail: leaked password hash")
    monkeypatch.setattr(api_mod.engine, "answer", boom)
    r = client.post("/ask", json={"question": "x", "top_k": 5})
    assert r.status_code == 500
    body = r.json()
    assert "secret internal" not in str(body)
    assert body["error"] == "Internal server error"
    assert "trace_id" in body


def test_with_retry_decorator_retries(tmp_path, monkeypatch):
    """Tenacity: transient ConnectionError retried up to 3 attempts."""
    import resilience

    attempts = {"n": 0}

    @resilience.with_retry
    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("simulated transient failure")
        return "ok"

    assert flaky() == "ok"
    assert attempts["n"] == 3


def test_with_retry_gives_up_after_max_attempts(monkeypatch):
    import resilience

    attempts = {"n": 0}

    @resilience.with_retry
    def always_fails():
        attempts["n"] += 1
        raise TimeoutError("simulated timeout")

    with pytest.raises(TimeoutError):
        always_fails()
    assert attempts["n"] == 3


def test_with_retry_does_not_retry_non_transient(monkeypatch):
    """RuntimeError is not retried — one attempt only."""
    import resilience

    attempts = {"n": 0}

    @resilience.with_retry
    def nope():
        attempts["n"] += 1
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError):
        nope()
    assert attempts["n"] == 1


# ── Auth (RAG_API_KEYS) ─────────────────────────────────────────────

AUTH_KEYS = {"test-key-1", "test-key-2"}


def test_auth_disabled_without_key_allows_all(client):
    """Env empty (auth off) -> endpoints open, like the demo path."""
    assert not api_mod._authorized
    r = client.get("/search?q=transformer")
    assert r.status_code == 200


def test_auth_missing_key_401(client, monkeypatch):
    monkeypatch.setattr(api_mod, "_authorized", AUTH_KEYS)
    for method, url, body in [
        ("get", "/search?q=transformer", None),
        ("post", "/ask", {"question": "What is RAG?", "top_k": 3}),
        ("post", "/ask/stream", {"question": "What is RAG?", "top_k": 3}),
        ("get", "/metrics", None),
    ]:
        r = getattr(client, method)(url, json=body) if body else getattr(client, method)(url)
        assert r.status_code == 401, (method, url)


def test_auth_wrong_key_401(client, monkeypatch):
    monkeypatch.setattr(api_mod, "_authorized", AUTH_KEYS)
    r = client.get("/search?q=transformer", headers={"Authorization": "Bearer wrong-key"})
    assert r.status_code == 401
    assert "unauthorized" in r.json().get("error", r.json().get("detail", ""))


def test_auth_x_api_key_header_works(client, monkeypatch):
    monkeypatch.setattr(api_mod, "_authorized", AUTH_KEYS)
    r = client.get("/search?q=transformer", headers={"X-API-Key": "test-key-1"})
    assert r.status_code == 200


def test_auth_bearer_key_works(client, monkeypatch):
    monkeypatch.setattr(api_mod, "_authorized", AUTH_KEYS)
    r = client.get("/search?q=transformer", headers={"Authorization": "Bearer test-key-2"})
    assert r.status_code == 200


def test_auth_health_public_when_auth_active(client, monkeypatch):
    monkeypatch.setattr(api_mod, "_authorized", AUTH_KEYS)
    r = client.get("/health")
    assert r.status_code == 200


def test_auth_ask_with_valid_key_200(client, monkeypatch):
    monkeypatch.setattr(api_mod, "_authorized", AUTH_KEYS)
    r = client.post(
        "/ask",
        json={"question": "What is RAG?", "top_k": 3},
        headers={"Authorization": "Bearer test-key-1"},
    )
    assert r.status_code == 200


def test_auth_metrics_requires_key(client, monkeypatch):
    """/metrics is internal-only: 401 without key even when auth enabled."""
    monkeypatch.setattr(api_mod, "_authorized", AUTH_KEYS)
    r = client.get("/metrics")
    assert r.status_code == 401


# ── Rate limiting (RAG_RATE_LIMIT) ──────────────────────────────────

def test_rate_limit_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(api_mod, "settings", type("S", (), {"rag_rate_limit": 3})())
    monkeypatch.setattr(api_mod, "_authorized", set())
    for _ in range(3):
        r = client.get("/search?q=transformer")
        assert r.status_code == 200
    r = client.get("/search?q=transformer")
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1


def test_rate_limit_resets_between_tests(client, monkeypatch):
    """reset_rate_limit_store() in the autouse fixture clears state."""
    monkeypatch.setattr(api_mod, "settings", type("S", (), {"rag_rate_limit": 2})())
    monkeypatch.setattr(api_mod, "_authorized", set())
    r = client.get("/search?q=transformer")
    assert r.status_code == 200  # fresh store, not throttled by prior test
