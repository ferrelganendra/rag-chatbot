"""FastAPI server for RAG QA endpoints."""

import sys
import traceback
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ensure src/ is on path, run from project root
SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from retrieval.qa_engine import QAEngine
from retrieval.searcher import Searcher
from config import settings

engine: QAEngine | None = None
searcher: Searcher | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, searcher
    searcher = Searcher(
        chroma_path=settings.chroma_path,
        collection_name=settings.collection_name,
        model_name=settings.embedding_model,
    )
    engine = QAEngine(
        searcher=searcher,
        model_key=settings.default_model_key,
        top_k=settings.default_top_k,
        temperature=settings.temperature,
    )
    yield

app = FastAPI(
    title="RAG QA Engine",
    description="Retrieval-Augmented Generation API — ask questions over your documents.",
    version="0.1.0",
    lifespan=lifespan,
)

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5

class SourceInfo(BaseModel):
    document: str
    score: float

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)},
    )

@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "engine_ready": engine is not None and searcher is not None}


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Service not initialized. Check that ChromaDB index exists."},
        )
    result = engine.answer(req.question)
    return AnswerResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
    )

@app.get("/search")
async def search(q: str, top_k: int = 5):
    if searcher is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Service not initialized. Check that ChromaDB index exists."},
        )
    results = searcher.search(q, top_k=top_k)
    return {
        "query": q,
        "results": [
            {"text": r.text[:300], "source": r.source, "score": round(r.score, 4)}
            for r in results
        ],
    }
