"""End-to-end ingestion pipeline: load → chunk → embed → index."""

import logging
import chromadb
from pathlib import Path
from config import settings
from .loader import load_documents
from .chunker import chunk_documents, ChunkConfig
from .embedder import Embedder

logger = logging.getLogger(__name__)


def run_ingestion(
    doc_dir: str | None = None,
    chroma_path: str | None = None,
    collection_name: str | None = None,
) -> tuple[int, int]:
    """
    Run full ingestion pipeline.
    Returns (num_docs, num_chunks).
    """
    doc_dir = doc_dir or settings.doc_dir
    chroma_path_val = chroma_path or settings.chroma_path
    collection_name_val = collection_name or settings.collection_name

    logger.info(f"Loading documents from {doc_dir}...")
    docs = load_documents(doc_dir)
    if not docs:
        raise ValueError(f"No documents found in {doc_dir}")
    logger.info(f"  Loaded {len(docs)} documents")

    logger.info("Chunking...")
    chunks = chunk_documents(docs)
    logger.info(f"  Created {len(chunks)} chunks")

    logger.info("Embedding...")
    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts)

    logger.info(f"Indexing into ChromaDB ({chroma_path_val})...")
    client = chromadb.PersistentClient(path=chroma_path_val)

    # Recreate collection for fresh index
    try:
        client.delete_collection(collection_name_val)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name_val,
        metadata={"embedding_model": settings.embedding_model, "embedding_dim": embedder.dim},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": c["source"]} for c in chunks]

    for i in range(0, len(chunks), settings.chroma_batch_size):
        end = min(i + settings.chroma_batch_size, len(chunks))
        collection.add(
            ids=ids[i:end],
            documents=texts[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )

    logger.info(f"  Indexed {collection.count()} chunks")
    return len(docs), len(chunks)
