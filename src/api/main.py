"""FastAPI server for RAG QA endpoints."""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

# Ensure src/ is on path, run from project root
SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from retrieval.qa_engine import QAEngine
from retrieval.searcher import Searcher

engine: QAEngine | None = None
searcher: Searcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, searcher
    searcher = Searcher()
    engine = QAEngine(searcher=searcher)
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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    result = engine.answer(req.question)
    return AnswerResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
    )


@app.get("/search")
async def search(q: str, top_k: int = 5):
    results = searcher.search(q, top_k=top_k)
    return {
        "query": q,
        "results": [
            {"text": r.text[:300], "source": r.source, "score": round(r.score, 4)}
            for r in results
        ],
    }
