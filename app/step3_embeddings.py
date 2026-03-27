import re
from pathlib import Path

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


def generate_embeddings(chunks: list[str], model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


def main():
    if not PDF_PATH.exists():
        print(f"File not found: {PDF_PATH}")
        return

    raw_text = extract_text_from_pdf(PDF_PATH)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text, chunk_size=1000, overlap=200)

    print(f"Number of chunks: {len(chunks)}")

    embeddings = generate_embeddings(chunks)

    print("\n--- EMBEDDING INFO ---")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Single embedding dimension: {embeddings.shape[1]}")

    print("\n--- FIRST CHUNK PREVIEW ---\n")
    print(chunks[0][:500])

    print("\n--- FIRST 10 VALUES OF FIRST EMBEDDING ---")
    print(embeddings[0][:10])


if __name__ == "__main__":
    main()