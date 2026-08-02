"""Document chunking with configurable strategy."""

from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: tuple = ("\n\n", "\n", ". ", " ", "")


def chunk_documents(
    documents: list[dict[str, str]],
    config: ChunkConfig | None = None,
) -> list[dict[str, str]]:
    """Split documents into overlapping chunks. Returns list of {text, source}."""
    cfg = config or ChunkConfig()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=list(cfg.separators),
    )

    chunks = []
    for doc in documents:
        texts = splitter.split_text(doc["text"])
        for text in texts:
            chunks.append({"text": text, "source": doc["source"]})
    return chunks
