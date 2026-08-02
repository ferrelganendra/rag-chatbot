# ⚡ DocQ — Production-Grade RAG QA Engine

A production-quality Retrieval-Augmented Generation system with streaming responses, chat UI, and built-in evaluation. Upload documents, ask questions, get cited answers — all running locally with no API keys needed.

## 🎯 Why This Project Matters (For Recruiters)

This isn't a notebook tutorial. It's a **complete AI engineering pipeline** demonstrating:

- ✅ **End-to-end RAG**: ingestion → retrieval → generation → evaluation → serving
- ✅ **Measurable quality**: numeric metrics (Hit Rate, MRR, LLM-as-judge), not "looks good"
- ✅ **Production patterns**: streaming responses, source citations, graceful degradation
- ✅ **Engineering discipline**: unit tests, modular architecture, professional documentation

## ✨ Features

### Core Pipeline
- **Smart Ingestion**: recursive chunking with configurable size/overlap
- **Semantic Search**: cosine similarity via ChromaDB with metadata filtering
- **Streaming Generation**: Llama 3.2 3B answers with word-by-word streaming
- **Source Citations**: every answer links back to source documents with relevance scores

### Quality & Evaluation
- **Retrieval Metrics**: 100% Hit Rate@5, 0.920 MRR on 10 test queries
- **LLM-as-Judge**: automated answer quality scoring (relevance, groundedness, completeness)
- **Anti-Hallucination**: engineered prompts that say "I don't know" instead of making things up

### Interface
- **Chat UI**: conversation history, streaming tokens, markdown rendering
- **Suggested Questions**: one-click example queries to get started
- **Source Cards**: color-coded by relevance (green/yellow/red)
- **Live Metrics**: real-time document and chunk counts in sidebar

### Engineering
- **FastAPI Backend**: `/ask`, `/search`, `/health` endpoints with OpenAPI docs
- **Test Suite**: 18 pytest tests across ingestion, retrieval, API, and evaluation
- **Modular Architecture**: separate packages for each pipeline stage

## 🏗️ Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌────────────────┐
│  Documents   │────▶│  Ingestion      │────▶│  ChromaDB      │
│  (MD/TXT)    │     │  Load→Chunk     │     │  Cosine Search │
└──────────────┘     │  →Embed→Index   │     └───────┬────────┘
                     └─────────────────┘             │
                                                    │ top-k chunks
                     ┌─────────────────┐             │
   User Question ───▶│  Retrieval      │◀───────────┘
                     │  Embed→Search   │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │  QA Engine      │
                     │  RAG Prompt     │  Streaming
                     │  Llama 3.2 3B   │────▶ Answer + Citations
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        ┌─────▼─────┐  ┌──────▼──────┐  ┌─────▼─────┐
        │  FastAPI  │  │  Streamlit  │  │  Eval     │
        │  Endpoints│  │  Chat UI    │  │  Metrics  │
        └───────────┘  └─────────────┘  └───────────┘
```

## 📊 Evaluation Results (Llama 3.2 3B)

| Metric | Score | What It Means |
|--------|-------|---------------|
| **Hit Rate@5** | **1.000** | Every question finds relevant docs in top-5 results |
| **MRR** | **0.920** | First relevant document appears very early in results |
| **Avg Relevance** | **5.0/5** | Answers directly address the question asked |
| **Avg Groundedness** | **5.0/5** | Answers are based on context, not hallucination |
| **Avg Completeness** | **5.0/5** | Answers fully cover what was asked |

*Evaluated on 10 curated queries covering Python async, transformers, RAG, vector DBs, and prompt engineering.*

## 🛠️ Tech Stack & Decisions

| Layer | Technology | Why This Choice | Trade-off |
|-------|-----------|----------------|-----------|
| **LLM** | Llama 3.2 3B (Ollama) | Strong enough for RAG, free, local, private | Weaker than 7B/70B on complex reasoning |
| **Embeddings** | all-MiniLM-L6-v2 (384d) | Fast, efficient, runs on CPU, proven in production | Less nuanced than 1536-dim models |
| **Vector DB** | ChromaDB | Embedded, zero-config, native LangChain integration | Not distributed, single-machine |
| **Chunking** | RecursiveCharacterTextSplitter | Handles varied document structures gracefully | May split mid-paragraph occasionally |
| **API** | FastAPI | Fast, async, auto-generated OpenAPI docs | Not as battle-tested as Flask |
| **UI** | Streamlit | Rapid development, Python-native, no frontend code | Not suitable for public-facing SaaS |
| **Testing** | pytest | Standard, fixtures, parametrization | — |
| **Eval** | Custom metrics + LLM-as-judge | Reproducible, no external dependencies | LLM judge can be overconfident |

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.com) installed

```bash
# 1. Pull the LLM
ollama pull llama3.2:3b

# 2. Clone and setup
git clone https://github.com/ferrelganendra/rag-chatbot
cd rag-chatbot

# 3. Install dependencies (uv recommended)
uv pip install -r requirements.txt

# 4. Run ingestion (indexes sample documents)
python src/ingestion/run.py
```

### Launch Options

**Streamlit Chat UI** (recommended for demo):
```bash
streamlit run src/ui/app.py
# Open http://localhost:8501
```

**FastAPI Server** (for integration):
```bash
uvicorn src.api.main:app --reload
# Open http://localhost:8000/docs for interactive API docs
```

### API Usage

```python
import requests

# Search
r = requests.get("http://localhost:8000/search", params={"q": "What is RAG?", "top_k": 5})
print(r.json())

# Ask
r = requests.post("http://localhost:8000/ask", json={"question": "Explain cosine similarity"})
print(r.json())
```

### Run Tests
```bash
pytest tests/ -v  # 18 tests pass
```

### Run Evaluation
```bash
python -c "
import sys; sys.path.insert(0, 'src')
from eval import evaluate_retrieval, TEST_QUERIES
from retrieval.searcher import Searcher
r = evaluate_retrieval(TEST_QUERIES, Searcher())
print(f'Hit Rate@5: {r[\"hit_rate\"]}, MRR: {r[\"mrr\"]}')
"
```

## 📁 Project Structure

```
rag-chatbot/
├── src/
│   ├── ingestion/       # loader.py, chunker.py, embedder.py, pipeline.py, run.py
│   ├── retrieval/       # searcher.py, qa_engine.py (streaming)
│   ├── eval/            # metrics.py (Hit Rate, MRR, LLM-judge), test_queries.py
│   ├── api/             # main.py (FastAPI: /ask, /search, /health)
│   └── ui/              # app.py (Streamlit chat interface)
├── data/
│   ├── documents/       # 5 sample technical documents
│   └── chroma/          # Persisted ChromaDB index
├── tests/               # test_ingestion.py, test_retrieval.py, test_api.py, test_eval.py
├── requirements.txt
└── README.md
```

## 🔮 Roadmap — If I Were to Productionize Further

- [ ] **Hybrid Search**: BM25 keyword search + dense retrieval for better recall
- [ ] **Re-ranking**: Cross-encoder to refine top-10 into top-3
- [ ] **PDF Support**: PyMuPDF integration for PDF ingestion
- [ ] **Observability**: LangSmith tracing or custom event logging
- [ ] **Multi-Model**: Config toggle between local (llama3.2) and cloud (GPT-4, Claude)
- [ ] **Docker**: Containerized with docker-compose (Ollama + ChromaDB + FastAPI + Streamlit)
- [ ] **Rate Limiting**: API protection with token bucket algorithm
- [ ] **Auth**: API key authentication for production endpoints

## 📄 License

MIT
