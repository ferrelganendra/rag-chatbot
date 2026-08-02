# 🚀 RAG QA Engine — Production-Grade Document Q&A

A production-quality Retrieval-Augmented Generation system: upload documents, ask questions, get grounded answers with source citations. Built end-to-end with evaluation metrics.

## ✨ Features

- **Document Ingestion**: load markdown/text → chunk → embed → ChromaDB
- **Semantic Search**: cosine similarity retrieval with configurable top-k
- **LLM-Powered Answers**: qwen3:0.6b generates answers grounded in retrieved context
- **Evaluation Built-In**: Hit Rate, MRR, LLM-as-judge quality scoring
- **FastAPI Backend**: REST API with `/ask` and `/search` endpoints
- **Streamlit UI**: interactive web interface for document Q&A
- **Test Suite**: 18 pytest tests covering ingestion, retrieval, API, evaluation

## 🏗️ Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Documents   │────▶│  Ingestion    │────▶│  ChromaDB    │
│  (MD/TXT)    │     │  Load→Chunk   │     │  (Vector DB) │
└──────────────┘     │  →Embed→Index │     └──────┬───────┘
                     └───────────────┘            │
                                                  │ query
                     ┌───────────────┐            │
   User Question ───▶│  Retrieval    │◀───────────┘
                     │  Embed→Search │
                     └───────┬───────┘
                             │ top-k chunks
                     ┌───────▼───────┐
                     │  QA Engine    │
                     │  Prompt + LLM │
                     └───────┬───────┘
                             │
                     ┌───────▼───────┐     ┌──────────────┐
                     │  Answer +     │────▶│  Evaluation  │
                     │  Sources      │     │  Metrics      │
                     └───────────────┘     └──────────────┘
```

## 📊 Evaluation Results

| Metric | Score | Description |
|--------|-------|-------------|
| **Hit Rate@5** | **100%** | All 10 queries found at least 1 relevant doc in top-5 |
| **MRR** | **0.920** | Mean Reciprocal Rank — first relevant doc appears early |
| **Avg Relevance** | **5.0/5** | LLM-as-judge: answers are relevant to questions |
| **Avg Groundedness** | **5.0/5** | Answers are based on context, not hallucination |
| **Avg Completeness** | **5.0/5** | Answers fully address the questions |

*Note: Scores of 5.0/5 reflect a small curated dataset. Production use would show more variance.*

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **LLM** | qwen3:0.6b (Ollama) | Free, local, no API key needed |
| **Embeddings** | all-MiniLM-L6-v2 (384d) | Efficient, production-proven, runs on CPU |
| **Vector DB** | ChromaDB | Simple, embedded, LangChain-native |
| **Chunking** | RecursiveCharacterTextSplitter | Handles varied doc structures |
| **API** | FastAPI | Fast, auto-docs, async |
| **UI** | Streamlit | Rapid data apps, no frontend code |
| **Tests** | pytest | Standard, fixture support |
| **Evaluation** | Custom metrics + LLM-as-judge | Reproducible, no external services |

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) with qwen3 model

```bash
# Install model
ollama pull qwen3:0.6b

# Clone and setup
git clone <this-repo>
cd rag-chatbot

# Create venv and install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Or with uv (recommended)
uv pip install -r requirements.txt
```

### Run Ingestion

```bash
python src/ingestion/pipeline.py
```

### Start API Server

```bash
uvicorn src.api.main:app --reload
# Open http://localhost:8000/docs for Swagger UI
```

### Start Streamlit UI

```bash
streamlit run src/ui/app.py
```

### Run Tests

```bash
pytest tests/ -v
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
│   ├── ingestion/       # Document loading, chunking, embedding, pipeline
│   ├── retrieval/       # Semantic search, QA engine with LLM
│   ├── eval/            # Hit Rate, MRR, LLM-as-judge evaluation
│   ├── api/             # FastAPI REST endpoints
│   └── ui/              # Streamlit web interface
├── data/
│   ├── documents/       # Source documents (markdown)
│   └── chroma/          # Persisted ChromaDB index
├── tests/               # pytest test suite
└── requirements.txt
```

## 🧠 Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **qwen3:0.6b** over cloud API | Free, private, works offline | Smaller model = weaker reasoning |
| **all-MiniLM-L6-v2** over Ada-002 | Free, 384-dim fast search | Less nuanced than 1536-dim models |
| **Recursive chunking** over semantic | Simple, predictable, handles mixed formats | May split mid-paragraph sometimes |
| **Cosine distance** over dot product | Standard for text, ChromaDB default | Requires normalized embeddings |
| **ChromaDB** over Pinecone | Embedded, zero-config, no cloud cost | Not distributed, single-machine only |
| **LLM-as-judge** over human eval | Reproducible, scalable, instant | Model bias, overconfidence on small models |

## 🔮 Future Improvements

- **Hybrid search**: BM25 + dense retrieval for better keyword matching
- **Re-ranking**: Cross-encoder to refine top results
- **Streaming**: Server-sent events for real-time answer display
- **Observability**: LangSmith or custom tracing for debugging
- **Multi-model support**: Easy toggle between local and cloud LLMs
- **PDF support**: Add PyMuPDF for PDF ingestion
- **Production deployment**: Docker, rate limiting, API auth

## 📄 License

MIT
