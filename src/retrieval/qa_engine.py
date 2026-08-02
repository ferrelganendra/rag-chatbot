"""QA engine: multi-LLM support (Local Ollama + Cloud Groq + Cloud Gemini) with streaming."""

import os
from typing import Generator, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from .searcher import Searcher, SearchResult
from ..config import settings

SYSTEM_PROMPT = (
    "You are a precise technical assistant. Answer the user's question using the provided context documents. "
    "RULES:\n"
    "1. Use information FROM the context. Cite documents like [Source: filename].\n"
    "2. Format answers cleanly with markdown: use bullets, code blocks, bold for key terms.\n"
    "3. If multiple relevant points exist, list them clearly.\n"
    "4. If the context has no relevant information, say: 'This topic is not covered in the indexed documents.'\n"
    "5. Be concise but complete — don't leave the user needing to ask again."
)

MODELS = {
    "groq-70b": {
        "name": "Groq Llama 3.3 70B (Cloud)",
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
    },
    "groq-8b": {
        "name": "Groq Llama 3.1 8B (Cloud)",
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
    },
    "gemini-flash": {
        "name": "Gemini 2.0 Flash (Cloud)",
        "provider": "google",
        "model": "gemini-2.0-flash",
    },
    "llama3": {
        "name": "Llama 3.2 3B (Local)",
        "provider": "ollama",
        "model": "llama3.2:3b",
    },
}

class QAEngine:
    def __init__(
        self,
        searcher: Searcher | None = None,
        model_key: str | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        groq_api_key: str | None = None,
        google_api_key: str | None = None,
    ):
        self.searcher = searcher or Searcher()
        self.model_key = model_key or settings.default_model_key
        self.top_k = top_k if top_k is not None else settings.default_top_k
        self.temperature = temperature if temperature is not None else settings.temperature
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        self._llm = None
        self._init_llm()

    def _init_llm(self):
        cfg = MODELS[self.model_key]
        if cfg["provider"] == "ollama":
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(
                model=cfg["model"],
                temperature=self.temperature,
                num_predict=settings.max_tokens,
            )
        elif cfg["provider"] == "groq":
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(
                model=cfg["model"],
                temperature=self.temperature,
                api_key=self.groq_api_key,
                max_tokens=settings.max_tokens,
            )
        elif cfg["provider"] == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model=cfg["model"],
                temperature=self.temperature,
                google_api_key=self.google_api_key,
                max_output_tokens=settings.max_tokens,
            )

    def switch_model(self, model_key: str):
        self.model_key = model_key
        self._init_llm()

    @property
    def llm(self):
        return self._llm

    @property
    def model_name(self) -> str:
        return MODELS[self.model_key]["name"]

    def _format_context(self, results: list[SearchResult]) -> str:
        parts = []
        for i, r in enumerate(results):
            parts.append(f"[{i+1}. {r.source}]\n{r.text}")
        return "\n\n---\n\n".join(parts)

    def answer(self, question: str) -> dict:
        results = self.searcher.search(question, top_k=self.top_k)
        context = self._format_context(results)

        prompt_text = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context documents:\n{context}\n\n"
            f"Question: {question}"
        )
        response = self.llm.invoke([HumanMessage(content=prompt_text)])

        return {
            "question": question,
            "answer": response.content,
            "model": self.model_name,
            "sources": [{"document": r.source, "score": round(r.score, 4)} for r in results],
            "context_chunks": [r.text[:300] + "..." for r in results],
        }

    def answer_stream(self, question: str) -> Generator[Any, None, None]:
        results = self.searcher.search(question, top_k=self.top_k)
        context = self._format_context(results)

        prompt_text = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context documents:\n{context}\n\n"
            f"Question: {question}"
        )

        tokens = []
        for chunk in self.llm.stream([HumanMessage(content=prompt_text)]):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            tokens.append(token)
            yield token

        sources = [{"document": r.source, "score": round(r.score, 4)} for r in results]
        chunks = [r.text[:300] + "..." for r in results]
        yield {
            "_done": True,
            "sources": sources,
            "context_chunks": chunks,
            "full_answer": "".join(tokens),
            "model": self.model_name,
        }
