import re
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "animals_sample_book.pdf"


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    all_pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text is not None and text.strip():
            all_pages_text.append(text)
        else:
            print(f"Warning: page {page_number} returned empty text.")

    return "\n".join(all_pages_text)


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)

        if end == text_length:
            break

        start = end - overlap

    return chunks


def cosine_similarity(query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)
    doc_norms = np.linalg.norm(doc_vectors, axis=1)

    similarities = np.dot(doc_vectors, query_vector) / (doc_norms * query_norm + 1e-12)
    return similarities


def retrieve_top_k(query: str, chunks: list[str], model: SentenceTransformer, k: int = 3):
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    query_embedding = model.encode(query, convert_to_numpy=True)

    similarities = cosine_similarity(query_embedding, chunk_embeddings)
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    results = []
    for idx in top_k_indices:
        results.append(
            {
                "chunk_index": int(idx),
                "score": float(similarities[idx]),
                "text": chunks[idx]
            }
        )

    return results


def main():
    if not PDF_PATH.exists():
        print(f"File not found: {PDF_PATH}")
        return

    raw_text = extract_text_from_pdf(PDF_PATH)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text, chunk_size=1000, overlap=200)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    query = "What is a mammal?"
    results = retrieve_top_k(query=query, chunks=chunks, model=model, k=3)

    print(f"\nQuery: {query}\n")
    print("--- TOP 3 RETRIEVED CHUNKS ---\n")

    for rank, result in enumerate(results, start=1):
        print(f"Rank {rank}")
        print(f"Chunk index: {result['chunk_index']}")
        print(f"Similarity score: {result['score']:.4f}")
        print(result["text"][:700])
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()