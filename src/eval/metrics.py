"""RAG evaluation: retrieval quality + answer quality."""

from typing import Any
import numpy as np
from retrieval.searcher import Searcher, SearchResult
from retrieval.qa_engine import QAEngine


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

    import json
    scores = []
    for tc in test_cases:
        result = engine.answer(tc["query"])
        prompt = judge_prompt.format(
            context=result["context_chunks"][:3],
            question=tc["query"],
            answer=result["answer"],
        )

        try:
            from langchain_ollama import ChatOllama
            judge_llm = ChatOllama(model="qwen3:0.6b", temperature=0.0)
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
        avg = lambda key: round(np.mean([s[key] for s in valid]), 2) if valid else -1
        return {
            "avg_relevance": avg("relevance"),
            "avg_groundedness": avg("groundedness"),
            "avg_completeness": avg("completeness"),
            "num_evaluated": len(valid),
            "per_question": scores,
        }
    return {}
