"""FastAPI server for RAG QA endpoints."""

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request, Response
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
    return {
        "status": "ok",
        "engine_ready": engine is not None and searcher is not None,
    }


@app.get("/metrics")
async def metrics():
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
