# Vector Databases for Semantic Search

## What is a Vector Database?

A vector database stores data as high-dimensional vectors (embeddings) and enables similarity search. Unlike traditional databases that match exact values, vector DBs find "semantically similar" items by measuring distance in the embedding space.

## Core Concepts

### Embeddings
Embeddings are dense numerical representations of data (text, images, audio) in a continuous vector space. Models like `all-MiniLM-L6-v2` produce 384-dimensional embeddings where semantically similar content maps to nearby vectors.

### Similarity Metrics

Three common distance metrics:

**Cosine Similarity** (most common for text):
```
cos(a,b) = (a·b) / (||a|| × ||b||)
```
Range: -1 to 1. Higher = more similar. Normalized vectors: cosine similarity equals dot product.

**Euclidean Distance (L2)**:
```
dist(a,b) = sqrt(Σ(ai - bi)²)
```
Range: 0 to ∞. Lower = more similar.

**Dot Product**:
For normalized vectors, dot product equals cosine similarity. Useful when vectors are already normalized.

### Approximate Nearest Neighbors (ANN)

Exact search is O(n×d) where n = vectors and d = dimensions. ANN algorithms trade small accuracy losses for massive speed gains:

- **HNSW** (Hierarchical Navigable Small World): graph-based index, fast queries, higher memory
- **IVF** (Inverted File): clusters vectors, searches nearest clusters only
- **PQ** (Product Quantization): compresses vectors, trades accuracy for memory

## ChromaDB

ChromaDB is an open-source embedding database designed for LLM applications. Features:

- **Collections**: logical groupings of embeddings with metadata
- **Multiple backends**: in-memory, persistent (DuckDB), client-server
- **Metadata filtering**: filter results by metadata fields
- **Distance functions**: cosine, L2, inner product
- **Native LangChain integration**

### Usage Pattern
```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.create_collection("docs")
collection.add(
    documents=["document text here"],
    embeddings=[embedding_vector],
    metadatas=[{"source": "file.txt"}],
    ids=["doc_1"],
)
results = collection.query(query_embeddings=[query_vector], n_results=5)
```

## Choosing a Vector Database

| Database | Best For |
|----------|----------|
| ChromaDB | Simple, local-first RAG apps |
| Pinecone | Managed, serverless, production scale |
| Weaviate | Hybrid search (vector + keyword) |
| Milvus | Billion-scale, distributed deployments |
| Qdrant | High-performance, filtering-heavy workloads |
| FAISS | Offline indexing, maximum speed |
