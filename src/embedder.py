from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Default model
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.device = device
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")

    def encode_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        texts = [c["text"] for c in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        print(
            f"Generated {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}"
        )
        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec.astype("float32")
