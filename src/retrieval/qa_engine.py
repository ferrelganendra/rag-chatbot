"""QA engine: combine retrieved context with LLM to generate answers."""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from .searcher import Searcher, SearchResult


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a precise technical assistant. Answer questions based ONLY on the provided context. "
        "If the context does not contain the answer, say \"I don't have enough information to answer that.\" "
        "Cite which document the information came from when possible. "
        "Keep answers concise and factual."
    )),
    ("user", (
        "Context documents:\n"
        "{context}\n\n"
        "Question: {question}"
    )),
])


class QAEngine:
    def __init__(
        self,
        searcher: Searcher | None = None,
        model_name: str = "qwen3:0.6b",
        top_k: int = 5,
    ):
        self.searcher = searcher or Searcher()
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        self.top_k = top_k

    def _format_context(self, results: list[SearchResult]) -> str:
        parts = []
        for i, r in enumerate(results):
            parts.append(f"[Document: {r.source}]\n{r.text}")
        return "\n\n---\n\n".join(parts)

    def answer(self, question: str) -> dict:
        """Retrieve context → generate answer. Returns dict with answer, sources, context."""
        results = self.searcher.search(question, top_k=self.top_k)
        context = self._format_context(results)

        chain = RAG_PROMPT | self.llm
        response = chain.invoke({"context": context, "question": question})

        return {
            "question": question,
            "answer": response.content,
            "sources": [{"document": r.source, "score": round(r.score, 4)} for r in results],
            "context_chunks": [r.text[:200] + "..." for r in results],
        }
