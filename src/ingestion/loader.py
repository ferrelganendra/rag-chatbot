"""Load documents from various formats."""

from pathlib import Path


def load_documents(doc_dir: str) -> list[dict[str, str]]:
    """Load .md and .txt files from directory. Returns list of {text, source}."""
    path = Path(doc_dir)
    documents = []

    for ext in ("*.md", "*.txt"):
        for filepath in path.glob(ext):
            text = filepath.read_text(encoding="utf-8")
            if text.strip():
                documents.append({"text": text, "source": filepath.name})

    return documents
