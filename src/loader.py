import os
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
from tqdm import tqdm


def load_pdfs(pdf_dir: str) -> List[Dict[str, Any]]:
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {pdf_dir}")

    print(f"Found {len(pdf_files)} PDF file(s) in '{pdf_dir}'")

    documents = []

    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            reader = PdfReader(str(pdf_path))
            num_pages = len(reader.pages)

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    documents.append({
                        "filename": pdf_path.name,
                        "page_number": page_num,
                        "total_pages": num_pages,
                        "text": text.strip(),
                    })

        except Exception as e:
            print(f"  [WARN] Failed to read '{pdf_path.name}': {e}")

    print(f"Extracted text from {len(documents)} page(s) across {len(pdf_files)} PDF(s)")
    return documents


def get_dataset_summary(pdf_dir: str) -> Dict[str, Any]:
    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    summary = {
        "total_files": len(pdf_files),
        "files": [],
    }

    total_pages = 0
    total_size_kb = 0

    for pdf_path in pdf_files:
        size_kb = pdf_path.stat().st_size / 1024
        total_size_kb += size_kb

        try:
            reader = PdfReader(str(pdf_path))
            pages = len(reader.pages)
            total_pages += pages
            readable = True
        except Exception:
            pages = 0
            readable = False

        summary["files"].append({
            "filename": pdf_path.name,
            "size_kb": round(size_kb, 2),
            "pages": pages,
            "readable": readable,
        })

    summary["total_pages"] = total_pages
    summary["total_size_kb"] = round(total_size_kb, 2)
    return summary
