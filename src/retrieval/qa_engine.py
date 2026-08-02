"""QA engine: combine retrieved context with LLM to generate answers — with streaming."""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from .searcher import Searcher, SearchResult

SYSTEM_PROMPT = (
    "You are a precise technical assistant. Answer the user's question using the provided context documents. "
    "RULES:\n"
    "1. Use information FROM the context. Cite documents like [Source: filename].\n"
    "2. Format answers cleanly with markdown: use bullets, code blocks, bold for key terms.\n"
    "3. If multiple relevant points exist, list them clearly.\n"
    "4. If the context has no relevant information, say: 'This topic is not covered in the indexed documents.'\n"
    "5. Be concise but complete — don't leave the user needing to ask again."
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", (
        "Context documents:\n{context}\n\nQuestion: {question}"
    )),
])


class QAEngine:
    def __init__(
        self,
        searcher: Searcher | None = None,
        model_name: str = "llama3.2:3b",
        top_k: int = 5,
        temperature: float = 0.3,
    ):
        self.searcher = searcher or Searcher()
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=500,
        )
        self.top_k = top_k

    def _format_context(self, results: list[SearchResult]) -> str:
        parts = []
        for i, r in enumerate(results):
            parts.append(f"[{i+1}. {r.source}]\n{r.text}")
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
            "context_chunks": [r.text[:300] + "..." for r in results],
        }

    def answer_stream(self, question: str):
        """Retrieve context → stream answer tokens. Yields tokens and finally sources."""
        results = self.searcher.search(question, top_k=self.top_k)
        context = self._format_context(results)

        chain = RAG_PROMPT | self.llm
        tokens = []
        for chunk in chain.stream({"context": context, "question": question}):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            tokens.append(token)
            yield token

        sources = [{"document": r.source, "score": round(r.score, 4)} for r in results]
        chunks = [r.text[:300] + "..." for r in results]
        yield {"_done": True, "sources": sources, "context_chunks": chunks, "full_answer": "".join(tokens)}
