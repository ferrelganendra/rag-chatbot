"""Load documents from various formats."""

from pathlib import Path
import fitz  # pymupdf


def _load_pdf(filepath: Path) -> str | None:
    """Extract text from a PDF using pymupdf. Returns None if empty."""
    doc = fitz.open(filepath)
    try:
        pages: list[str] = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text)
        return "\n".join(pages) if pages else None
    finally:
        doc.close()


def load_documents(doc_dir: str) -> list[dict[str, str]]:
    """Load .md, .txt, and .pdf files from directory. Returns list of {text, source}."""
    path = Path(doc_dir)
    documents: list[dict[str, str]] = []

    for filepath in path.glob("*.md"):
        text = filepath.read_text(encoding="utf-8")
        if text.strip():
            documents.append({"text": text, "source": filepath.name})

    for filepath in path.glob("*.txt"):
        text = filepath.read_text(encoding="utf-8")
        if text.strip():
            documents.append({"text": text, "source": filepath.name})

    for filepath in path.glob("*.pdf"):
        text = _load_pdf(filepath)
        if text and text.strip():
            documents.append({"text": text, "source": filepath.name})

    return documents
