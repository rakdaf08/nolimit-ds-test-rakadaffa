from typing import List, Dict, Any


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    chunk_id = 0

    for doc in documents:
        text = doc["text"]
        filename = doc["filename"]
        page_number = doc["page_number"]

        # Slide a window over the text
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "filename": filename,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "char_start": start,
                        "char_end": min(end, len(text)),
                    }
                )
                chunk_id += 1
                chunk_index += 1

            # Move window forward
            start += chunk_size - chunk_overlap

    print(
        f"Generated {len(chunks)} chunk(s) from {len(documents)} page(s) "
        f"[size={chunk_size}, overlap={chunk_overlap}]"
    )
    return chunks
