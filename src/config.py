"""Centralized configuration for DocQ RAG engine."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    # Paths
    doc_dir: str = "data/documents"
    chroma_path: str = "data/chroma"
    collection_name: str = "rag_docs"
    
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Retrieval
    default_top_k: int = 5
    
    # LLM
    default_model_key: str = "groq-70b"
    temperature: float = 0.3
    max_tokens: int = 800
    
    # Judge (for LLM-as-judge eval)
    judge_model: str = "llama3.2:3b"
    judge_provider: str = "ollama"

    # Batch size for ChromaDB indexing
    chroma_batch_size: int = 100

    @property
    def project_root(self) -> Path:
        """Find project root (where src/ lives)."""
        return Path(__file__).resolve().parent.parent

# Singleton
settings = Settings()
