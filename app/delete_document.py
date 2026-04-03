DOCUMENT_ID_TO_DELETE = "pdf::animals_sample_book::436e12b8e6c1"


from app.core.logging import get_logger
from app.vectorstore.qdrant_store import QdrantVectorStore

logger = get_logger(__name__)


def main() -> None:
    if not DOCUMENT_ID_TO_DELETE.strip():
        raise ValueError(
            "Set DOCUMENT_ID_TO_DELETE in app/delete_document.py before running."
        )

    logger.info("Initializing Qdrant...")
    vector_store = QdrantVectorStore()

    if not vector_store.document_exists(DOCUMENT_ID_TO_DELETE):
        print(f"Document not found: {DOCUMENT_ID_TO_DELETE}")
        return

    vector_store.delete_document(DOCUMENT_ID_TO_DELETE)
    print(f"Deleted document: {DOCUMENT_ID_TO_DELETE}")


if __name__ == "__main__":
    main()