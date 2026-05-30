from typing import List, Dict, Any, Tuple

from embedder import Embedder
from vectorstore import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        query_vec = self.embedder.encode_query(query)
        results = self.vector_store.search(query_vec, top_k=top_k)
        return results

    def format_context(self, results: List[Tuple[Dict[str, Any], float]]) -> str:
        parts = []
        for i, (chunk, score) in enumerate(results, start=1):
            header = (
                f"[Source {i}] {chunk['filename']} "
                f"(page {chunk['page_number']}, score={score:.3f})"
            )
            parts.append(f"{header}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def format_citations(
        self, results: List[Tuple[Dict[str, Any], float]]
    ) -> List[Dict[str, Any]]:
        citations = []
        for i, (chunk, score) in enumerate(results, start=1):
            citations.append(
                {
                    "source_num": i,
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "score": round(score, 4),
                    "snippet": chunk["text"][:300]
                    + ("..." if len(chunk["text"]) > 300 else ""),
                }
            )
        return citations
