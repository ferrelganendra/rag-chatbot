# ⚡ DocQ — Production-Grade RAG QA Engine

[![CI](https://github.com/ferrelganendra/rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/ferrelganendra/rag-chatbot/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


Upload documents, ask questions, get cited answers — runs locally with Ollama, or cloud via Groq/Gemini.

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
- **Streaming Generation**: word-by-word streaming answers (Groq 70B by default, local Ollama fallback)
- **Source Citations**: every answer links back to source documents with relevance scores

### Quality & Evaluation
- **Retrieval Metrics**: 90% Hit Rate@5, 0.775 MRR on 10 test queries
- **LLM-as-Judge**: automated answer quality scoring (relevance, groundedness, completeness)
- **Anti-Hallucination**: engineered prompts that say "I don't know" instead of making things up

### Interface
- **Chat UI**: conversation history, streaming tokens, markdown rendering
- **Suggested Questions**: one-click example queries to get started
- **Source Cards**: color-coded by relevance (green/yellow/red)
- **Live Metrics**: real-time document and chunk counts in sidebar

### Engineering
- **FastAPI Backend**: `/ask`, `/ask/stream`, `/search`, `/health` endpoints with OpenAPI docs
- **Test Suite**: hermetic pytest suite across ingestion, retrieval, API, and evaluation
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
                      │  Groq 70B       │────▶ Answer + Citations
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        ┌─────▼─────┐  ┌──────▼──────┐  ┌─────▼─────┐
        │  FastAPI  │  │  Streamlit  │  │  Eval     │
        │  Endpoints│  │  Chat UI    │  │  Metrics  │
        └───────────┘  └─────────────┘  └───────────┘
```

## 📊 Evaluation Results (all-MiniLM-L6-v2 retrieval)

| Metric | Score | What It Means |
|--------|-------|---------------|
| **Hit Rate@5** | **0.9** | 90% of questions find a relevant doc in top-5 results |
| **MRR** | **0.775** | First relevant document appears early in results (avg. rank ~1.3) |

*Evaluated on 10 curated queries covering Python async, transformers, RAG, vector DBs, and prompt engineering. LLM-as-judge scores (relevance/groundedness/completeness) are not listed because no reproducible harness artifact exists for them.*

## 🛠️ Tech Stack & Decisions

| Layer | Technology | Why This Choice | Trade-off |
|-------|-----------|----------------|-----------|
| **LLM** | Groq Llama 3.3 70B (default) / Llama 3.2 3B (local Ollama) | 70B via free Groq tier for quality; local 3B fallback for private/offline | Cloud needs API key |
| **Embeddings** | all-MiniLM-L6-v2 (384d) | Fast, efficient, runs on CPU, proven in production | Less nuanced than 1536-dim models |
| **Vector DB** | ChromaDB | Embedded, zero-config, native LangChain integration | Not distributed, single-machine |
| **Chunking** | RecursiveCharacterTextSplitter | Handles varied document structures gracefully | May split mid-paragraph occasionally |
| **API** | FastAPI | Fast, async, auto-generated OpenAPI docs | Not as battle-tested as Flask |
| **UI** | Streamlit | Rapid development, Python-native, no frontend code | Not suitable for public-facing SaaS |
| **Testing** | pytest | Standard, fixtures, parametrization | — |
| **Eval** | Custom metrics + LLM-as-judge | Reproducible, no external dependencies | LLM judge can be overconfident |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- **Cloud (default)**: a [Groq](https://console.groq.com) API key (free tier) → `export GROQ_API_KEY=...`, or `GOOGLE_API_KEY` for Gemini.
- **Local (fallback)**: [Ollama](https://ollama.com) installed with `llama3.2:3b`.

> If you run a cloud model, set `GROQ_API_KEY` (and/or `GOOGLE_API_KEY`) **before** starting the app. Without any key, the app auto-detects Ollama locally (and errors clearly if nothing is available). Gemini is selected explicitly in the UI if `GOOGLE_API_KEY` is set. The default model is Groq 70B.

```bash
# 1. (Cloud) set your API key
export GROQ_API_KEY="gsk_..."

# 2. (Local fallback) pull the LLM
ollama pull llama3.2:3b

# 3. Clone and setup
git clone https://github.com/ferrelganendra/rag-chatbot
cd rag-chatbot

# 4. Install dependencies
uv pip install -r requirements.txt   # or: pip install -r requirements.txt

# 5. Run ingestion (indexes sample documents)
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

# Ask (full response)
r = requests.post("http://localhost:8000/ask", json={"question": "Explain cosine similarity"})
print(r.json())

# Streaming ask (SSE)
import requests, json

with requests.post("http://localhost:8000/ask/stream", json={"question": "Explain cosine similarity"}, stream=True) as r:
    for line in r.iter_lines():
        if line:
            line = line.decode()
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                print(json.loads(data).get("token", ""), end="", flush=True)
```

### Run Tests
```bash
pytest tests/ -v  # hermetic — no Ollama/Chroma index needed
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

### Development

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="gsk_..."   # cloud default; or: ollama pull llama3.2:3b

# Run pipeline
python src/ingestion/run.py

# Tests + watch
pytest tests/ -v
# OR: ptw tests/ -- -v   (needs: pip install pytest-watch)
```

## 📁 Project Structure

```
rag-chatbot/
├── src/
│   ├── ingestion/       # loader.py, chunker.py, embedder.py, pipeline.py, run.py
│   ├── retrieval/       # searcher.py, qa_engine.py (streaming)
│   ├── eval/            # metrics.py (Hit Rate, MRR, LLM-judge), test_queries.py
│   ├── api/             # main.py (FastAPI: /ask, /ask/stream, /search, /health)
│   └── ui/              # app.py (Streamlit chat interface)
├── data/
│   ├── documents/       # 7 sample technical documents (5 md + 2 pdf)
│   └── chroma/          # Persisted ChromaDB index
├── tests/               # test_ingestion.py, test_retrieval.py, test_api.py, test_eval.py
├── requirements.txt
└── README.md
```

## 🔮 Roadmap — If I Were to Productionize Further

- [ ] **Hybrid Search**: BM25 keyword search + dense retrieval for better recall
- [ ] **Re-ranking**: Cross-encoder to refine top-10 into top-3
- [x] **PDF Support**: PyMuPDF integration for PDF ingestion
- [ ] **Observability**: LangSmith tracing or custom event logging
- [ ] **Multi-Model**: Config toggle between local (llama3.2) and cloud (GPT-4, Claude)
- [x] **Docker**: Containerized with docker-compose (FastAPI + Streamlit)
- [ ] **Rate Limiting**: API protection with token bucket algorithm (see Security Notes)
- [ ] **Auth**: API key authentication for production endpoints

## 🔒 Security Notes

- The FastAPI server (`/ask`, `/ask/stream`, `/search`, `/metrics`) has **no auth and no rate limiting** — fine for a local demo, not for public exposure.
- CORS defaults to `localhost:8501`/`localhost:8000`; override with `RAG_CORS_ORIGINS` (comma-separated).
- Before any public deployment: add API-key auth (or OAuth), rate limiting, and TLS termination.
- Error handlers log full exceptions server-side and return generic messages + a `trace_id` to clients — no internal details leaked.
- `/metrics` exposes internal Prometheus counters — keep it behind the same auth as the rest when going public.

## 📄 License

MIT
