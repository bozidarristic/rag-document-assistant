from app.core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_PDF_PATH,
    EMBEDDING_MODEL_NAME,
    TOP_K,
)
from app.core.logging import get_logger
from app.indexing.chunking import chunk_text
from app.indexing.embeddings import TextEmbedder
from app.ingestion.ingest import load_pdf
from app.ingestion.text_cleaner import clean_text
from app.retrieval.retrieval import retrieve_top_k

logger = get_logger(__name__)


def main() -> None:
    query = "What is a mammal?"

    logger.info("Loading PDF...")
    document = load_pdf(DEFAULT_PDF_PATH)

    logger.info("Cleaning text...")
    document.content = clean_text(document.content)

    logger.info("Chunking document...")
    chunks = chunk_text(
        document=document,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    logger.info("Encoding chunks...")
    embedder = TextEmbedder(EMBEDDING_MODEL_NAME)
    chunk_embeddings = embedder.encode_chunks(chunks)

    logger.info("Encoding query...")
    query_vector = embedder.encode_query(query)

    logger.info("Retrieving top-k chunks...")
    results = retrieve_top_k(
        query_vector=query_vector,
        chunk_embeddings=chunk_embeddings,
        chunks=chunks,
        top_k=TOP_K,
    )

    print(f"\nQuery: {query}\n")
    for rank, result in enumerate(results, start=1):
        print(f"[{rank}] score={result.score:.4f}")
        print(result.chunk.text[:400])
        print("-" * 80)


if __name__ == "__main__":
    main()