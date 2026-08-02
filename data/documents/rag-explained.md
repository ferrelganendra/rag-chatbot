# Retrieval-Augmented Generation (RAG)

## Overview

RAG combines retrieval systems with generative AI to produce grounded, factual responses. Instead of relying solely on the model's training data, RAG retrieves relevant documents and provides them as context to the LLM.

## Architecture

The standard RAG pipeline has three stages:

1. **Ingestion**: Documents → Chunking → Embedding → Vector Store
2. **Retrieval**: Query → Embed → Similarity Search → Top-K Chunks
3. **Generation**: Prompt(Query + Retrieved Chunks) → LLM → Answer

## Chunking Strategies

Chunking directly impacts retrieval quality:

- **Fixed-size**: Simple, predictable. Works for uniform documents.
- **Semantic**: Split by paragraphs/sections using NLP. Better context preservation.
- **Recursive**: Hierarchical splitting using separators (paragraph → sentence → word). Most flexible.
- **Sliding window**: Overlapping chunks. Reduces missed context at boundaries.

Trade-off: larger chunks = more context but less precise retrieval. Smaller chunks = precise but may miss surrounding context.

## Retrieval Optimization

Beyond basic vector search:

### Hybrid Search
Combine dense (semantic) and sparse (keyword/BM25) retrieval. Catches both conceptual matches and exact keyword matches.

### Re-ranking
Two-stage retrieval: fast ANN search for candidates, then a slower cross-encoder model scores and re-sorts top results for precision.

### Query Transformations
- **HyDE**: Generate hypothetical answer, use it as search query
- **Multi-query**: Generate multiple query variations, merge results
- **Step-back**: Abstract then search. "What are the components of X?"

## Evaluation

Key metrics for RAG systems:

**Retrieval Quality**:
- **Hit Rate**: % of queries where at least one relevant doc is in top-K
- **MRR** (Mean Reciprocal Rank): average of 1/rank of first relevant document
- **NDCG** (Normalized Discounted Cumulative Gain): position-weighted relevance

**Answer Quality**:
- **Faithfulness**: Does the answer match the retrieved context?
- **Answer Relevance**: Does the answer address the question?
- **Context Relevance**: Is the retrieved context actually relevant?

## Common Pitfalls

- **Hallucination**: LLM fabricates when context is insufficient
- **Context overflow**: Too many/long chunks exceed context window
- **Stale embeddings**: Re-index when documents change
- **Poor chunk boundaries**: Splitting mid-sentence breaks context
- **Embedding mismatch**: Inference model different from indexing model
