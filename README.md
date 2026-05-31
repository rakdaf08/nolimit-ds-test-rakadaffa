# 📄 Multi-Document PDF RAG Assistant

Chatbot berbasis **Retrieval-Augmented Generation (RAG)** yang menjawab pertanyaan berdasarkan kumpulan dokumen PDF.

---

## Tech Stack

| Component      | Library / Model                        |
| -------------- | -------------------------------------- |
| PDF Extraction | pypdf                                  |
| Embeddings     | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB      | FAISS (IndexFlatIP)                    |
| Language Model | TinyLlama/TinyLlama-1.1B-Chat-v1.0     |
| UI (Bonus)     | Streamlit                              |

---

## Chosen Models

### 1. Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`

- **Dimensionality**: 384 dimensions.
- **Rationale**: Merupakan model embedding yang sangat ringan, efisien, dan cepat. Memiliki performa luar biasa dalam pemetaan semantik dokumen teks (mapping text to dense vectors) dan sangat cocok untuk dijalankan pada perangkat lokal dengan memory footprint (RAM/VRAM) yang minimal.
- **Similarity Metric**: Menggunakan metrik **Cosine Similarity** (diimplementasikan via FAISS `IndexFlatIP` dengan vektor hasil L2-normalisasi) untuk mencari chunk teks yang paling relevan dengan query pengguna.

### 2. Large Language Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

- **Size**: 1.1 Miliar parameters (~2.2 GB VRAM/RAM required).
- **Rationale**: Model bahasa kausal (Causal LM) berukuran mini yang dioptimalkan untuk percakapan (chat). Pilihan terbaik untuk eksekusi lokal pada CPU maupun GPU kelas konsumen. Mendukung template instruksi percakapan standar secara konsisten sehingga andal dalam memproses prompt RAG.

---

## Dataset Source & Annotations

- **Source**: Dataset ini diperoleh dari public source di Kaggle: [Dataset of PDF Files by Manisha717](https://www.kaggle.com/datasets/manisha717/dataset-of-pdf-files). Dokumen PDF di dalamnya bersumber dari **Library of Congress Web Archives (LCWA)** untuk dokumen berekstensi `.pdf` pada domain `.gov`.
- **License**: Berdasarkan metadata Kaggle, lisensi dataset ini adalah **Unknown / Creative Work** (merupakan data publik terbuka yang ditujukan untuk keperluan testing, evaluasi, dan riset RAG).
- **Annotations & Metadata**: Metadata dataset lengkap disediakan di dalam file [lcwa_gov_pdf_metadata.csv](file:///C:/Users/rakad/OneDrive/Dokumen/Projects/nolimit-ds-test-rakadaffa/data/pdfs/lcwa_gov_pdf_metadata.csv) dengan rincian kolom annotations sebagai berikut:
  - `urlkey` / `original`: URL asal dari dokumen PDF pemerintah yang diarsipkan.
  - `timestamp`: Waktu saat dokumen diarsipkan.
  - `pdf_version`, `creator_tool`, `producer`: Metadata metadata penyusun file PDF.
  - `pages`, `page_width`, `page_height`, `file_size`: Dimensi halaman dan ukuran file fisik PDF.
  - `sha256` / `sha512` / `digest`: Hash kriptografis unik yang digunakan untuk penamaan file PDF fisik di dalam folder `data/pdfs/`.

---

## Repository Structure

```
nolimit-ds-test-raka/
├── data/
│   └── pdfs/               ← Test dataset (30 File)
│
├── notebook/
│   ├── rag_pipeline_kaggle.ipynb   ← Notebook untuk kaggle
│   └── rag_pipeline.ipynb          ← Notebook lokal
│
├── src/
│   ├── loader.py           ← PDF loading & metadata extraction
│   ├── chunker.py          ← sliding window text chunking
│   ├── embedder.py         ← SentenceTransformer encoding
│   ├── vectorstore.py      ← FAISS index build/save/load/search
│   ├── retriever.py        ← query → top-k chunks
│   └── rag_pipeline.py     ← orchestrator pipeline utama + LLM
│
├── faiss_index/            ← Dibuat otomatis setelah indexing
│   ├── faiss_index.bin
│   └── metadata.pkl
│
├── outputs/                ← Hasil running evaluasi & sample
│   ├── sample_results.md
│   └── evaluation_results.md
│
├── flowchart/
│   └── rag_pipeline.png    ← Diagram arsitektur sistem
│
├── requirements.txt
├── README.md
└── PLAN.md
```

---

## Setup & Installation Instructions

### 1. Setup Lokal (Local Environment Setup)

Ikuti langkah-langkah di bawah ini untuk memasang dan menjalankan aplikasi secara lokal:

#### **Prasyarat (Prerequisites)**:

- Python versi 3.9 ke atas terinstall di sistem Anda.

#### **Langkah-langkah Instalasi**:

1.  **Clone Repository** ke komputer Anda.
2.  **Buat Virtual Environment** di folder proyek:
    ```bash
    python -m venv .venv
    ```
3.  **Aktifkan Virtual Environment**:
    - **Windows (PowerShell)**:
      ```powershell
      .venv\Scripts\activate
      ```
    - **Linux / macOS**:
      ```bash
      source .venv/bin/activate
      ```
4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    _Catatan: File `requirements.txt` telah dikonfigurasi untuk Windows lokal dengan menggunakan `faiss-cpu`. Jika berjalan di Google Colab dengan GPU, gunakan `faiss-gpu`._

#### **Menjalankan Pipeline RAG via Python CLI**:

Masuk ke folder `src` dan jalankan perintah Python berikut:

```bash
cd src
python -c "
from rag_pipeline import RAGPipeline
pipe = RAGPipeline()
pipe.index()
result = pipe.query('What is the main topic of the documents?')
print('ANSWER:', result['answer'])
"
```

#### **Menjalankan Streamlit Chatbot UI**:

Untuk meluncurkan antarmuka web lokal (Streamlit):

```bash
streamlit run src/streamlit_app.py
```

Aplikasi akan otomatis terbuka pada browser Anda di tautan: **`http://localhost:8501`**

---

### 2. Setup Kaggle (Cloud Execution)

Jika Anda ingin menjalankan notebook di Kaggle:

1.  Buat **Notebook** baru di Kaggle.
2.  Unggah berkas notebook khusus kaggle ke Kaggle Notebook Anda.
3.  Ubah pengaturan **Accelerator** di panel kanan menjadi **GPU T4** untuk mengaktifkan pemrosesan GPU secara gratis.
4.  Tambahkan input dataset ke notebook Anda dengan mengeklik tombol **Add Input** di kanan atas, lalu cari dataset: **`manisha717/dataset-of-pdf-files`**.
5.  Jalankan cell **0** (Runtime Check) untuk memverifikasi GPU aktif.
6.  Jalankan cell **1** (Install Dependencies) untuk menginstal semua pustaka pendukung (termasuk `faiss-gpu`).
7.  Jalankan cell **2** untuk memuat pustaka pembantu secara otomatis.
8.  Jalankan cell hingga selesai untuk memproses indeks FAISS, memuat TinyLlama, menguji kueri RAG, dan melakukan evaluasi.

---

## Configuration

| Parameter       | Default             | Description                                          |
| --------------- | ------------------- | ---------------------------------------------------- |
| `chunk_size`    | 500                 | Ukuran karakter per chunk teks                       |
| `chunk_overlap` | 100                 | Jumlah karakter tumpang tindih (overlap) antar chunk |
| `top_k`         | 5                   | Jumlah dokumen teks relevan yang diambil per kueri   |
| Embedding model | all-MiniLM-L6-v2    | Model representasi semantik (384 dimensi)            |
| LLM             | TinyLlama-1.1B-Chat | Model generator teks lokal (~2.2GB VRAM/RAM)         |
