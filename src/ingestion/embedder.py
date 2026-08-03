"""Embedding wrapper using sentence-transformers (local, no API key needed)."""

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    @property
    def dim(self) -> int:
        # sentence-transformers is pinned (==3.4.1) so this API name is stable.
        return self.model.get_sentence_embedding_dimension()
