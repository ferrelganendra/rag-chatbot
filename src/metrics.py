"""Prometheus metrics for RAG QA Engine observability."""

from prometheus_client import Counter, Gauge, Histogram, generate_latest

# Request metrics
ask_requests_total = Counter(
    "rag_ask_requests_total",
    "Total /ask requests",
    ["model", "status"],
)
search_requests_total = Counter(
    "rag_search_requests_total",
    "Total /search requests",
)

# Latency metrics
ask_latency_seconds = Histogram(
    "rag_ask_latency_seconds",
    "Ask endpoint latency",
    ["model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)
retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval-only latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# Quality metrics
answer_length_chars = Histogram(
    "rag_answer_length_chars",
    "Answer length distribution",
    buckets=(50, 100, 250, 500, 1000, 2000),
)

# System metrics
index_chunks_total = Gauge(
    "rag_index_chunks_total",
    "Total chunks in ChromaDB",
)
index_documents_total = Gauge(
    "rag_index_documents_total",
    "Total documents indexed",
)


def get_metrics() -> bytes:
    """Return Prometheus text format metrics."""
    return generate_latest()
