import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    """Wrapper around Sentence-Transformers for embedding text."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Embed single text chunk."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: list) -> np.ndarray:
        """Embed multiple texts efficiently."""
        return self.model.encode(texts, convert_to_numpy=True, batch_size=32)
