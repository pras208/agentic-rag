import json
import numpy as np
import faiss
from pathlib import Path

class VectorStore:
    """FAISS-backed vector store for document chunks."""

    def __init__(self, dimension: int, index_path: str = "indexes/faiss.index",
                 metadata_path: str = "indexes/metadata.json"):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index = None
        self.metadata = []
        self._load_or_create()

    def _load_or_create(self):
        """Load index from disk or create new one."""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def add_documents(self, embeddings: np.ndarray, metadatas: list):
        """Add embeddings and their metadata to the index."""
        if embeddings.shape[0] == 0:
            return

        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadatas)
        self.save()

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list:
        """Search for top-k similar documents."""
        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append({
                "distance": float(distances[0][i]),
                "metadata": self.metadata[idx],
                "text": self.metadata[idx].get("text", "")
            })
        return results

    def save(self):
        """Persist index and metadata to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def get_size(self) -> int:
        """Return number of indexed documents."""
        return self.index.ntotal
