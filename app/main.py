from app.chunking import chunk_text
from app.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    PDF_PATH,
)
from app.embeddings import load_embedding_model
from app.ingest import clean_text, extract_text_from_pdf
from app.retrieval import retrieve_top_k


def main() -> None:
    if not PDF_PATH.exists():
        print(f"File not found: {PDF_PATH}")
        return

    raw_text = extract_text_from_pdf(PDF_PATH)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(
        cleaned_text,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_OVERLAP,
    )

    model = load_embedding_model(EMBEDDING_MODEL_NAME)

    query = "What is a mammal?"
    results = retrieve_top_k(query=query, chunks=chunks, model=model, k=DEFAULT_TOP_K)

    print(f"\nQuery: {query}\n")
    print(f"--- TOP {DEFAULT_TOP_K} RETRIEVED CHUNKS ---\n")

    for rank, result in enumerate(results, start=1):
        print(f"Rank {rank}")
        print(f"Chunk index: {result['chunk_index']}")
        print(f"Similarity score: {result['score']:.4f}")
        print(result["text"][:700])
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()