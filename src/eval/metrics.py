"""RAG evaluation: retrieval quality + answer quality."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from config import settings

if TYPE_CHECKING:
    from retrieval.qa_engine import QAEngine
    from retrieval.searcher import Searcher


def hit_rate(ground_truth: list[str], retrieved_sources: list[str], k: int = 5) -> float:
    """Fraction of queries where at least one relevant doc in top-k."""
    hits = sum(1 for gt in ground_truth if gt in retrieved_sources[:k])
    return hits / len(ground_truth) if ground_truth else 0.0

def mrr(ground_truth: list[str], retrieved_sources: list[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant document."""
    for i, src in enumerate(retrieved_sources):
        if src in ground_truth:
            return 1.0 / (i + 1)
    return 0.0

def evaluate_retrieval(
    test_cases: list[dict[str, Any]],
    searcher: Searcher,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    test_cases: [{"query": "...", "relevant_docs": ["doc1.md", "doc2.md"]}, ...]
    Returns hit_rate, mrr, and per-query details.
    """
    all_hit_rates = []
    all_mrr = []
    per_query = []

    for tc in test_cases:
        results = searcher.search(tc["query"], top_k=top_k)
        sources = [r.source for r in results]

        hr = hit_rate(tc["relevant_docs"], sources, k=top_k)
        mr = mrr(tc["relevant_docs"], sources)

        all_hit_rates.append(hr)
        all_mrr.append(mr)

        per_query.append({
            "query": tc["query"],
            "hit": hr > 0,
            "mrr": round(mr, 4),
            "retrieved": sources[:3],
            "relevant": tc["relevant_docs"],
        })

    return {
        "hit_rate": round(np.mean(all_hit_rates), 4),
        "mrr": round(np.mean(all_mrr), 4),
        "num_queries": len(test_cases),
        "top_k": top_k,
        "per_query": per_query,
    }

def _get_judge_llm():
    """Dynamic judge LLM based on settings."""
    if settings.judge_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=settings.judge_model, temperature=0.0)
    elif settings.judge_provider == "groq":
        import os

        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.judge_model,
            temperature=0.0,
            api_key=os.environ.get("GROQ_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown judge_provider: {settings.judge_provider}")

def evaluate_answer_quality(
    engine: QAEngine,
    test_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Use the LLM itself to judge answer quality (LLM-as-judge).
    Scores: relevance, groundedness, completeness.
    """
    judge_prompt = (
        "You are evaluating a RAG system's answer. Score from 1-5 on:\n"
        "1. Relevance: Does the answer address the question? (1=no, 5=perfect)\n"
        "2. Groundedness: Is the answer based on context, not hallucination? (1=hallucinated, 5=fully grounded)\n"
        "3. Completeness: Does it fully answer the question? (1=incomplete, 5=complete)\n\n"
        "Return ONLY a JSON: {{\"relevance\": X, \"groundedness\": X, \"completeness\": X}}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer: {answer}"
    )

    scores = []
    for tc in test_cases:
        result = engine.answer(tc["query"])
        prompt = judge_prompt.format(
            context=str(result["context_chunks"][:3]),
            question=tc["query"],
            answer=result["answer"],
        )

        try:
            judge_llm = _get_judge_llm()
            from langchain_core.messages import HumanMessage
            response = judge_llm.invoke([HumanMessage(content=prompt)])

            # Extract JSON from response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            parsed = json.loads(content)
            scores.append({
                "query": tc["query"],
                **parsed,
                "answer_preview": result["answer"][:150],
            })
        except Exception as e:
            scores.append({
                "query": tc["query"],
                "relevance": -1,
                "groundedness": -1,
                "completeness": -1,
                "error": str(e),
            })

    if scores:
        valid = [s for s in scores if isinstance(s.get("relevance"), (int, float)) and s["relevance"] > 0]
        def avg(key):
            return round(np.mean([s[key] for s in valid]), 2) if valid else -1
        return {
            "avg_relevance": avg("relevance"),
            "avg_groundedness": avg("groundedness"),
            "avg_completeness": avg("completeness"),
            "num_evaluated": len(valid),
            "per_question": scores,
        }
    return {}

def evaluate_latency(
    test_cases: list[dict[str, Any]],
    engine: QAEngine,
) -> dict[str, Any]:
    """Measure retrieval + generation latency across test queries.

    Times each phase separately using time.perf_counter() and returns
    average, p50, p95, p99 for retrieval, generation, and total.
    """
    from langchain_core.messages import HumanMessage

    retrieval_times: list[float] = []
    generation_times: list[float] = []
    total_times: list[float] = []
    per_query: list[dict[str, Any]] = []

    for tc in test_cases:
        question = tc["query"] if isinstance(tc, dict) else tc

        t0 = time.perf_counter()

        # Retrieval (search)
        results = engine.searcher.search(question, top_k=engine.top_k)
        t1 = time.perf_counter()

        # Generation (LLM call)
        context = engine._format_context(results)
        prompt_text = (
            f"Context documents:\n{context}\n\n"
            f"Question: {question}"
        )
        engine.llm.invoke([HumanMessage(content=prompt_text)])
        t2 = time.perf_counter()

        retrieval_ms = (t1 - t0) * 1000
        generation_ms = (t2 - t1) * 1000
        total_ms = (t2 - t0) * 1000

        retrieval_times.append(retrieval_ms)
        generation_times.append(generation_ms)
        total_times.append(total_ms)

        per_query.append({
            "query": question,
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round(total_ms, 2),
        })

    def _pct(data: list[float], p: float) -> float:
        return float(np.percentile(data, p))

    return {
        "num_queries": len(test_cases),
        "avg_retrieval_ms": round(float(np.mean(retrieval_times)), 2),
        "avg_generation_ms": round(float(np.mean(generation_times)), 2),
        "avg_total_ms": round(float(np.mean(total_times)), 2),
        "p50_retrieval_ms": round(_pct(retrieval_times, 50), 2),
        "p50_generation_ms": round(_pct(generation_times, 50), 2),
        "p50_total_ms": round(_pct(total_times, 50), 2),
        "p95_retrieval_ms": round(_pct(retrieval_times, 95), 2),
        "p95_generation_ms": round(_pct(generation_times, 95), 2),
        "p95_total_ms": round(_pct(total_times, 95), 2),
        "p99_retrieval_ms": round(_pct(retrieval_times, 99), 2),
        "p99_generation_ms": round(_pct(generation_times, 99), 2),
        "p99_total_ms": round(_pct(total_times, 99), 2),
        "per_query": per_query,
    }
