"""Semantic search over ChromaDB with configurable top-k."""

import chromadb
from dataclasses import dataclass
from ingestion.embedder import Embedder


@dataclass
class SearchResult:
    text: str
    source: str
    score: float


class Searcher:
    def __init__(
        self,
        chroma_path: str = "data/chroma",
        collection_name: str = "rag_docs",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(collection_name)
        self.embedder = Embedder(model_name=model_name)

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
            # ChromaDB cosine distance → similarity: 2 - dist (approx), or just use dist
            search_results.append(
                SearchResult(text=doc, source=meta["source"], score=1.0 - dist)
            )
        return search_results
