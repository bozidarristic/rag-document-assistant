from app.core.config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    QUERY_TEXT,
)
from app.core.logging import get_logger
from app.indexing.embeddings import TextEmbedder
from app.vectorstore.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


def main() -> None:
    logger.info("Initializing ChromaDB...")
    vector_store = ChromaVectorStore(persist_directory=CHROMA_PERSIST_DIR)

    logger.info("Encoding query...")
    embedder = TextEmbedder(EMBEDDING_MODEL_NAME)
    query_vector = embedder.encode_query(QUERY_TEXT)

    logger.info("Querying ChromaDB...")
    results = vector_store.query(query_embedding=query_vector.tolist())

    print(f"\nQuery: {QUERY_TEXT}\n")

    documents = results["documents"][0]
    distances = results["distances"][0]

    for rank, (document_text, distance) in enumerate(
        zip(documents, distances),
        start=1,
    ):
        similarity = 1 - distance
        print(f"[{rank}] distance={distance:.4f} | similarity={similarity:.4f}")
        print(document_text[:400])
        print("-" * 80)


if __name__ == "__main__":
    main()