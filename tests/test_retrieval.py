"""Tests for retrieval pipeline."""

from retrieval.searcher import Searcher, SearchResult


def test_search_returns_results():
    s = Searcher()
    results = s.search("RAG", top_k=3)
    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_relevance():
    s = Searcher()
    results = s.search("cosine similarity vector database", top_k=3)
    top_sources = {r.source for r in results}
    assert "vector-databases.md" in top_sources


def test_search_unknown_query_returns_results():
    """Even unknown queries should return *something* (nearest neighbors)."""
    s = Searcher()
    results = s.search("xyzqqq abcdef", top_k=3)
    assert len(results) == 3
