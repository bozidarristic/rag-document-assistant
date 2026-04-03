from app.core.logging import get_logger
from app.vectorstore.qdrant_store import QdrantVectorStore

logger = get_logger(__name__)


def main() -> None:
    logger.info("Initializing Qdrant...")
    vector_store = QdrantVectorStore()

    documents = vector_store.list_documents()

    if not documents:
        print("No indexed documents found.")
        return

    print("\nIndexed documents:\n")
    for index, doc_info in enumerate(documents, start=1):
        print(
            f"[{index}] "
            f"document_id={doc_info['document_id']} | "
            f"source_type={doc_info.get('source_type')} | "
            f"source_name={doc_info.get('source_name')} | "
            f"chunks={doc_info['chunk_count']}"
        )


if __name__ == "__main__":
    main()