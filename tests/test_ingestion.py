"""Tests for ingestion pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tempfile
from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents, ChunkConfig


def test_load_documents():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "test.md").write_text("# Hello\nWorld")
        docs = load_documents(d)
        assert len(docs) == 1
        assert docs[0]["source"] == "test.md"
        assert "Hello" in docs[0]["text"]


def test_load_empty_skips():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "blank.md").write_text("")
        (Path(d) / "real.md").write_text("content")
        docs = load_documents(d)
        assert len(docs) == 1


def test_chunker():
    docs = [{"text": "This is a test document. It has multiple sentences. This is the third.", "source": "test.md"}]
    config = ChunkConfig(chunk_size=30, chunk_overlap=5)
    chunks = chunk_documents(docs, config)
    assert len(chunks) >= 1
    for c in chunks:
        assert c["source"] == "test.md"
        assert len(c["text"]) > 0


def test_chunker_preserves_source():
    docs = [
        {"text": "Doc A content. More content here.", "source": "a.md"},
        {"text": "Doc B content. Different content.", "source": "b.md"},
    ]
    config = ChunkConfig(chunk_size=20, chunk_overlap=3)
    chunks = chunk_documents(docs, config)
    sources = {c["source"] for c in chunks}
    assert "a.md" in sources
    assert "b.md" in sources
