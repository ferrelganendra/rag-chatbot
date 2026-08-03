"""Tests for retrieval pipeline — hermetic (uses tmp_path-backed ChromaDB, no global index)."""

import chromadb
import pytest

from retrieval.searcher import Searcher, SearchResult


@pytest.fixture
def searcher(tmp_path):
    """Build a Searcher over a tmp_path-backed ChromaDB collection with seeded docs."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.create_collection(name="rag_docs")
    # 3 docs with distinct embeddings so retrieval ordering is deterministic.
    docs = [
        "RAG pipeline with vector database retrieval",
        "cosine similarity distance metric for vectors",
        "async Python coroutines and event loops",
    ]
    metadatas = [
        {"source": "rag-explained.md"},
        {"source": "vector-databases.md"},
        {"source": "python-async.md"},
    ]
    from ingestion.embedder import Embedder

    embeddings = Embedder().embed(docs)
    collection.add(
        ids=["0", "1", "2"],
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return Searcher(
        collection_name="rag_docs",
        model_name="all-MiniLM-L6-v2",
        client=client,
    )


def test_search_returns_results(searcher):
    results = searcher.search("RAG", top_k=3)
    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_relevance(searcher):
    results = searcher.search("cosine similarity vector database", top_k=3)
    # Query vector [0.9, 0.1, 0.0] should rank the vector-databases doc first.
    assert results[0].source == "vector-databases.md"


def test_search_unknown_query_returns_results(searcher):
    """Even unknown queries should return *something* (nearest neighbors)."""
    results = searcher.search("xyzqqq abcdef", top_k=3)
    assert len(results) == 3


def test_search_missing_collection_raises(tmp_path):
    """Fresh client with no collection -> clear ValueError, not a crash."""
    client = chromadb.PersistentClient(path=str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="not found"):
        Searcher(collection_name="rag_docs", client=client)