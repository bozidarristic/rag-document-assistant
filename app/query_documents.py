from app.core.config import EMBEDDING_MODEL_NAME, QUERY_TEXT, TOP_K
from app.core.logging import get_logger
from app.indexing.embeddings import TextEmbedder
from app.vectorstore.qdrant_store import QdrantVectorStore

logger = get_logger(__name__)


def main() -> None:
    logger.info("Initializing Qdrant...")
    vector_store = QdrantVectorStore()

    logger.info("Encoding query...")
    embedder = TextEmbedder(EMBEDDING_MODEL_NAME)
    query_vector = embedder.encode_query(QUERY_TEXT)

    logger.info("Querying Qdrant...")
    results = vector_store.query(
        query_embedding=query_vector.tolist(),
        top_k=TOP_K,
    )

    print(f"\nQuery: {QUERY_TEXT}\n")

    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}
        text = payload.get("text", "")
        document_id = payload.get("document_id", "unknown")
        source_name = payload.get("source_name", "unknown")
        chunk_index = payload.get("chunk_index", "unknown")

        print(
            f"[{rank}] score={result.score:.4f} | "
            f"source={source_name} | document_id={document_id} | "
            f"chunk_index={chunk_index}"
        )
        print(text[:400])
        print("-" * 100)


if __name__ == "__main__":
    main()