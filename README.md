# 📄 Multi-Document PDF RAG Assistant

Chatbot berbasis **Retrieval-Augmented Generation (RAG)** yang menjawab pertanyaan berdasarkan kumpulan dokumen PDF.

---

## Tech Stack

| Component | Library |
|---|---|
| PDF Extraction | pypdf |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| Language Model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| UI (Bonus) | Streamlit |

---

## System Architecture

```
PDF Collection
    │
    ▼
PDF Loader (loader.py)
    │   • Read all .pdf files
    │   • Extract text per page
    │
    ▼
Text Chunking (chunker.py)
    │   • chunk_size = 500 chars
    │   • chunk_overlap = 100 chars
    │
    ▼
Embedding Generation (embedder.py)
    │   • all-MiniLM-L6-v2
    │   • L2-normalised vectors (dim=384)
    │
    ▼
FAISS Vector DB (vectorstore.py)
    │   • IndexFlatIP (cosine similarity)
    │   • faiss_index/faiss_index.bin
    │   • faiss_index/metadata.pkl
    │
    ▼
User Question ──► Semantic Retrieval (retriever.py)
                    │   • Encode query
                    │   • top-k=5 chunks
                    │
                    ▼
              Prompt Construction
                    │   • TinyLlama chat template
                    │   • Injected context + question
                    │
                    ▼
              LLM Generation (rag_pipeline.py)
                    │   • TinyLlama-1.1B-Chat
                    │   • max_new_tokens=512
                    │
                    ▼
              Answer + Source Citations
```

---

## Repository Structure

```
nolimit-ds-test-raka/
├── data/
│   └── pdfs/               ← letakkan file PDF di sini
│
├── notebooks/
│   └── rag_pipeline.ipynb  ← notebook utama (jalankan di Colab)
│
├── src/
│   ├── loader.py           ← PDF loading & metadata extraction
│   ├── chunker.py          ← sliding window text chunking
│   ├── embedder.py         ← SentenceTransformer encoding
│   ├── vectorstore.py      ← FAISS index build/save/load/search
│   ├── retriever.py        ← query → top-k chunks
│   └── rag_pipeline.py     ← orchestrates full pipeline + LLM
│
├── app/
│   └── streamlit_app.py    ← Streamlit chatbot UI (bonus)
│
├── faiss_index/            ← generated after indexing
│   ├── faiss_index.bin
│   └── metadata.pkl
│
├── outputs/
│   ├── sample_results.md
│   └── evaluation_results.md
│
├── flowchart/
│   └── rag_pipeline.png    ← architecture diagram
│
├── requirements.txt
├── README.md
└── PLAN.md
```

---

## Quick Start — Google Colab

1. Buka `notebooks/rag_pipeline.ipynb` di Google Colab
2. Set runtime ke **T4 GPU**: Runtime → Change runtime type → T4
3. Jalankan cell **0** untuk cek GPU
4. Jalankan cell **1** untuk install dependencies
5. Upload PDF ke `data/pdfs/` (cell 4)
6. Jalankan cell **7** untuk indexing pipeline
7. Jalankan cell **8** untuk load LLM
8. Jalankan cell **10** untuk test query

---

## Quick Start — Local

```bash
# Install dependencies
pip install -r requirements.txt

# Letakkan PDF di data/pdfs/

# Jalankan RAG pipeline
cd src
python -c "
from rag_pipeline import RAGPipeline
pipe = RAGPipeline()
pipe.index()
result = pipe.query('What is the main topic of the documents?')
print(result['answer'])
for src in result['sources']:
    print(src['filename'], 'page', src['page_number'], 'score', src['score'])
"
```

## Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 500 | Characters per chunk |
| `chunk_overlap` | 100 | Overlap between chunks |
| `top_k` | 5 | Retrieved chunks per query |
| Embedding model | all-MiniLM-L6-v2 | 384-dim, fast |
| LLM | TinyLlama-1.1B-Chat | ~2.2GB, runs on T4 |

---

## Features

- ✅ Multi-PDF indexing — baca seluruh direktori secara otomatis
- ✅ Semantic search — cosine similarity via FAISS IndexFlatIP
- ✅ Question answering — TinyLlama dengan chat template
- ✅ Source citation — filename, page number, similarity score, snippet
- ✅ Persistent index — simpan & load FAISS tanpa re-indexing
- ✅ Streamlit UI — chatbot interface dengan source expander
