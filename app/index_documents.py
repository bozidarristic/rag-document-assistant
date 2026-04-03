from app.core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_PDF_PATH,
    EMBEDDING_MODEL_NAME,
)
from app.core.logging import get_logger
from app.indexing.chunking import chunk_text
from app.indexing.embeddings import TextEmbedder
from app.ingestion.ingest import load_pdf
from app.ingestion.text_cleaner import clean_text
from app.vectorstore.qdrant_store import QdrantVectorStore

logger = get_logger(__name__)


def main() -> None:
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
    logger.info("Generated %s chunks.", len(chunks))

    logger.info("Encoding chunks...")
    embedder = TextEmbedder(EMBEDDING_MODEL_NAME)
    chunk_embeddings = embedder.encode_chunks(chunks)

    logger.info("Initializing Qdrant...")
    vector_store = QdrantVectorStore()

    if vector_store.document_exists(document.source_id):
        logger.info(
            "Document already exists in index. Replacing: %s",
            document.source_id,
        )
        vector_store.delete_document(document.source_id)

    logger.info("Upserting chunks into Qdrant...")
    vector_store.upsert_chunks(
        chunks=chunks,
        embeddings=chunk_embeddings.tolist(),
    )

    logger.info("Indexed documents:")
    for doc_info in vector_store.list_documents():
        logger.info(
            "- %s | %s | chunks=%s",
            doc_info["document_id"],
            doc_info.get("source_name"),
            doc_info["chunk_count"],
        )

    logger.info("Total chunks in collection: %s", vector_store.count_chunks())
    logger.info("Indexing completed successfully.")


if __name__ == "__main__":
    main()