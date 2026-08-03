"""FastAPI server for RAG QA endpoints."""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from metrics import (
    answer_length_chars,
    ask_latency_seconds,
    ask_requests_total,
    get_metrics,
    search_requests_total,
)
from retrieval.qa_engine import QAEngine
from retrieval.searcher import Searcher

logger = logging.getLogger(__name__)

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

# CORS: default allow localhost; override via RAG_CORS_ORIGINS env (comma-separated).
# NOTE: open for local demo by default. For public deployment, add auth + rate limiting.
_origins = [o.strip() for o in os.environ.get("RAG_CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["http://localhost:8501", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


class AskStreamRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


class SourceInfo(BaseModel):
    document: str
    score: float


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.exception("ValueError: %s", exc)
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid request. Please check your input and try again."},
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    trace_id = uuid.uuid4().hex[:8]
    logger.exception("Unhandled error (trace_id=%s): %s", trace_id, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "trace_id": trace_id,
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine_ready": engine is not None and searcher is not None,
    }


@app.get("/metrics")
async def metrics():
    # NOTE: no auth + no rate limiting on /ask, /search, /metrics.
    # Open is fine for a local demo (README "Security Notes"); for any public
    # deployment, add API-key auth and rate limiting before exposing.
    return Response(content=get_metrics(), media_type="text/plain")


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service not initialized. Check that ChromaDB index exists."
            },
        )
    t0 = time.perf_counter()
    try:
        result = engine.answer(req.question)
        ask_latency_seconds.labels(model=engine.model_name).observe(
            time.perf_counter() - t0
        )
        answer_length_chars.observe(len(result["answer"]))
        ask_requests_total.labels(
            model=engine.model_name, status="success"
        ).inc()
        return AnswerResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result["sources"]],
        )
    except Exception:
        ask_requests_total.labels(
            model=getattr(engine, "model_name", "unknown"), status="error"
        ).inc()
        raise


@app.post("/ask/stream")
async def ask_stream(req: AskStreamRequest):
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Service not initialized."},
        )

    async def generate():
        for item in engine.answer_stream(req.question):
            if isinstance(item, dict) and item.get("_done"):
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "sources": item["sources"],
                            "model": item["model"],
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"
            else:
                yield "data: " + json.dumps({"token": item}) + "\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1, max_length=2000),
    top_k: int = Query(5, ge=1, le=20),
):
    if searcher is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service not initialized. Check that ChromaDB index exists."
            },
        )
    search_requests_total.inc()
    results = searcher.search(q, top_k=top_k)
    return {
        "query": q,
        "results": [
            {
                "text": r.text[:300],
                "source": r.source,
                "score": round(r.score, 4),
            }
            for r in results
        ],
    }
