"""Tests for ingestion pipeline."""

import tempfile
from pathlib import Path

from ingestion.chunker import ChunkConfig, chunk_documents
from ingestion.loader import load_documents


def test_load_markdown():
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


def test_load_pdf():
    """PDF text extraction via pymupdf."""
    import fitz
    with tempfile.TemporaryDirectory() as d:
        pdf_path = Path(d) / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello from PDF", fontsize=12, fontname="helv")
        doc.save(str(pdf_path))
        doc.close()

        docs = load_documents(d)
        assert len(docs) == 1
        assert docs[0]["source"] == "test.pdf"
        assert "Hello from PDF" in docs[0]["text"]


def test_load_pdf_empty_skips():
    """Empty PDFs should be skipped."""
    import fitz
    with tempfile.TemporaryDirectory() as d:
        pdf_path = Path(d) / "blank.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()

        docs = load_documents(d)
        assert len(docs) == 0


def test_load_mixed_formats():
    """Load .md, .txt, .pdf together."""
    import fitz
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.md").write_text("markdown content")
        (Path(d) / "b.txt").write_text("text content")
        pdf_path = Path(d) / "c.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "pdf content", fontsize=12, fontname="helv")
        doc.save(str(pdf_path))
        doc.close()

        docs = load_documents(d)
        assert len(docs) == 3
        sources = {d["source"] for d in docs}
        assert sources == {"a.md", "b.txt", "c.pdf"}


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
