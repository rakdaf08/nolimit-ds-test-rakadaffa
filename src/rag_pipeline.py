import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

sys.path.insert(0, str(Path(__file__).parent))

from loader import load_pdfs
from chunker import chunk_documents
from embedder import Embedder
from vectorstore import VectorStore
from retriever import Retriever

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FAISS_INDEX_DIR = "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5

FALLBACK_ANSWER = "I could not find relevant information in the document collection."

PROMPT_TEMPLATE = """\
<|system|>
You are a helpful assistant that answers questions strictly based on the provided context documents.
If the answer is not contained in the context, say exactly: "{fallback}"
Do not make up information.
</s>
<|user|>
Context:
{context}

Question: {question}
</s>
<|assistant|>
"""

# RAGPipeline class


class RAGPipeline:
    def __init__(
        self,
        pdf_dir: str = "data/pdfs",
        index_dir: str = FAISS_INDEX_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
        device: Optional[str] = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        top_k: int = TOP_K,
        load_in_4bit: bool = False,
    ):
        self.pdf_dir = pdf_dir
        self.index_dir = index_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Embedding model
        self.embedder = Embedder(model_name=embedding_model, device=self.device)

        # VectorStore (will be populated on index() or loaded on load_index())
        self.vector_store: Optional[VectorStore] = None
        self.retriever: Optional[Retriever] = None

        # LLM
        self._load_llm(llm_model, load_in_4bit)

    def _load_llm(self, model_name: str, load_in_4bit: bool) -> None:
        print(f"Loading LLM: {model_name}")

        quant_kwargs = {}
        if load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            **quant_kwargs,
        )

        if self.device == "cpu":
            self.llm = self.llm.to("cpu")

        self.generator = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            device=None if self.device == "cuda" else -1,
        )
        print("LLM loaded.")

    def index(self, force_reindex: bool = False) -> None:
        index_path = Path(self.index_dir) / "faiss_index.bin"

        if not force_reindex and index_path.exists():
            print("Existing index found. Loading from disk...")
            self.load_index()
            return

        print("\n=== INDEXING PIPELINE ===")

        # Load PDFs
        documents = load_pdfs(self.pdf_dir)

        # Chunk
        chunks = chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Embed
        embeddings = self.embedder.encode_chunks(chunks)

        # Build FAISS index
        self.vector_store = VectorStore(self.embedder.embedding_dim)
        self.vector_store.add(embeddings, chunks)
        self.vector_store.save(self.index_dir)

        # Build retriever
        self.retriever = Retriever(self.embedder, self.vector_store)
        print("=== INDEXING COMPLETE ===\n")

    def load_index(self) -> None:
        """Load a previously saved FAISS index from disk."""
        self.vector_store = VectorStore.load(
            self.index_dir, self.embedder.embedding_dim
        )
        self.retriever = Retriever(self.embedder, self.vector_store)

    def query(self, question: str) -> Dict[str, Any]:
        if self.retriever is None:
            raise RuntimeError("Index not built. Call pipeline.index() first.")

        # Retrieve
        raw_results = self.retriever.retrieve(question, top_k=self.top_k)
        context = self.retriever.format_context(raw_results)
        citations = self.retriever.format_citations(raw_results)

        # Build prompt
        prompt = PROMPT_TEMPLATE.format(
            fallback=FALLBACK_ANSWER,
            context=context,
            question=question,
        )

        # Generate
        output = self.generator(
            prompt,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Strip the prompt from the output
        generated = output[0]["generated_text"]
        answer = generated[len(prompt) :].strip()

        return {
            "question": question,
            "answer": answer if answer else FALLBACK_ANSWER,
            "sources": citations,
            "raw_results": raw_results,
        }
