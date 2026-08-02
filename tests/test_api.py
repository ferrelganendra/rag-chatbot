"""Tests for FastAPI endpoints (local model, no API key needed)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from fastapi.testclient import TestClient
import api.main as api_mod
from retrieval.searcher import Searcher
from retrieval.qa_engine import QAEngine


@pytest.fixture(autouse=True)
def setup():
    api_mod.searcher = Searcher()
    api_mod.engine = QAEngine(searcher=api_mod.searcher, model_key="llama3")


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
