"""Tests for FastAPI endpoints — mocked engine for CI (no Ollama needed)."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api.main as api_mod
from retrieval.searcher import SearchResult


@pytest.fixture(autouse=True)
def setup():
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
    api_mod.searcher = mock_searcher
    api_mod.engine = mock_engine

@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)

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
