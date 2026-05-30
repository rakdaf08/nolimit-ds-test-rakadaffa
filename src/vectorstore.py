import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np


class VectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        assert embeddings.shape[0] == len(
            chunks
        ), "Number of embeddings must match number of chunks"
        self.index.add(embeddings)
        self.metadata.extend(chunks)
        print(f"VectorStore: {self.index.ntotal} vector(s) indexed")

    def search(
        self, query_vec: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))

        return results

    def save(self, save_dir: str) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        index_path = save_dir / "faiss_index.bin"
        meta_path = save_dir / "metadata.pkl"

        faiss.write_index(self.index, str(index_path))

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"VectorStore saved → {save_dir}")

    @classmethod
    def load(cls, save_dir: str, embedding_dim: int) -> "VectorStore":
        save_dir = Path(save_dir)
        index_path = save_dir / "faiss_index.bin"
        meta_path = save_dir / "metadata.pkl"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Index or metadata file not found in '{save_dir}'. "
                "Run the indexing pipeline first."
            )

        store = cls(embedding_dim)
        store.index = faiss.read_index(str(index_path))

        with open(meta_path, "rb") as f:
            store.metadata = pickle.load(f)

        print(
            f"VectorStore loaded from '{save_dir}' " f"({store.index.ntotal} vectors)"
        )
        return store
