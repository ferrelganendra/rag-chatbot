"""Centralized configuration for DocQ RAG engine."""

import os
from dataclasses import dataclass, field
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
    
    # LLM — default is Groq 70B (cloud, needs GROQ_API_KEY); falls back to
    # local Ollama llama3.2:3b via QAEngine auto-detection when no key is set.
    default_model_key: str = "groq-70b"
    temperature: float = 0.3
    max_tokens: int = 800
    
    # Judge (for LLM-as-judge eval)
    judge_model: str = "llama3.2:3b"
    judge_provider: str = "ollama"

    # Batch size for ChromaDB indexing
    chroma_batch_size: int = 100

    # API security
    # RAG_API_KEYS: comma-separated API keys. Empty/None -> auth disabled (local demo).
    # RAG_RATE_LIMIT: requests per minute per IP for /ask, /ask/stream, /search.
    rag_api_keys: list[str] = field(default_factory=list)
    rag_rate_limit: int = 60

    def __post_init__(self):
        raw = os.environ.get("RAG_API_KEYS", "").strip()
        self.rag_api_keys = [k.strip() for k in raw.split(",") if k.strip()]
        try:
            self.rag_rate_limit = int(os.environ.get("RAG_RATE_LIMIT", "60"))
        except ValueError:
            self.rag_rate_limit = 60
        if self.rag_rate_limit < 1:
            self.rag_rate_limit = 60

    @property
    def project_root(self) -> Path:
        """Find project root (where src/ lives)."""
        return Path(__file__).resolve().parent.parent

# Singleton
settings = Settings()
