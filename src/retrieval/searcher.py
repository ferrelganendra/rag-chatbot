"""Semantic search over ChromaDB with configurable top-k."""

import chromadb
from dataclasses import dataclass
from ingestion.embedder import Embedder
from config import settings
from resilience import with_retry

@dataclass
class SearchResult:
    text: str
    source: str
    score: float

class Searcher:
    def __init__(
        self,
        chroma_path: str | None = None,
        collection_name: str | None = None,
        model_name: str | None = None,
    ):
        self.client = chromadb.PersistentClient(
            path=chroma_path or settings.chroma_path
        )
        try:
            self.collection = self.client.get_collection(
                collection_name or settings.collection_name
            )
        except Exception:
            raise ValueError(
                f"Collection '{collection_name or settings.collection_name}' not found. "
                f"Run ingestion first."
            )
        self.embedder = Embedder(model_name=model_name or settings.embedding_model)

    @with_retry
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retrieve top-k most relevant chunks for a query."""
        query_embedding = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        search_results = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance -> similarity: 1 - dist
            search_results.append(
                SearchResult(text=doc, source=meta["source"], score=1.0 - dist)
            )
        return search_results
