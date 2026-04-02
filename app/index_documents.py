from app.core.config import (
    CHROMA_PERSIST_DIR,
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
from app.vectorstore.chroma_store import ChromaVectorStore

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

    logger.info("Encoding chunks...")
    embedder = TextEmbedder(EMBEDDING_MODEL_NAME)
    chunk_embeddings = embedder.encode_chunks(chunks)

    logger.info("Initializing ChromaDB...")
    vector_store = ChromaVectorStore(persist_directory=CHROMA_PERSIST_DIR)

    logger.info("Adding chunks to ChromaDB...")
    vector_store.add_chunks(
        chunks=chunks,
        embeddings=chunk_embeddings.tolist(),
    )

    logger.info("Indexing completed successfully.")


if __name__ == "__main__":
    main()