"""Document chunking with configurable strategy."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


@dataclass
class ChunkConfig:
    chunk_size: int = settings.chunk_size
    chunk_overlap: int = settings.chunk_overlap
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
