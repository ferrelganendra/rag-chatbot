"""End-to-end ingestion pipeline: load → chunk → embed → index."""

import chromadb
from pathlib import Path
from .loader import load_documents
from .chunker import chunk_documents, ChunkConfig
from .embedder import Embedder


def run_ingestion(
    doc_dir: str = "data/documents",
    chroma_path: str = "data/chroma",
    collection_name: str = "rag_docs",
) -> tuple[int, int]:
    """
    Run full ingestion pipeline.
    Returns (num_docs, num_chunks).
    """
    print(f"Loading documents from {doc_dir}...")
    docs = load_documents(doc_dir)
    if not docs:
        raise ValueError(f"No documents found in {doc_dir}")
    print(f"  Loaded {len(docs)} documents")

    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"  Created {len(chunks)} chunks")

    print("Embedding...")
    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts)

    print(f"Indexing into ChromaDB ({chroma_path})...")
    client = chromadb.PersistentClient(path=chroma_path)

    # Recreate collection for fresh index
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"embedding_model": "all-MiniLM-L6-v2", "embedding_dim": embedder.dim},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": c["source"]} for c in chunks]

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:end],
            documents=texts[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )

    print(f"  Indexed {collection.count()} chunks")
    return len(docs), len(chunks)
