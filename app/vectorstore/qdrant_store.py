import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import QDRANT_COLLECTION_NAME, QDRANT_PATH, TOP_K
from app.core.models import Chunk


class QdrantVectorStore:
    def __init__(
        self,
        qdrant_path: str = QDRANT_PATH,
        collection_name: str = QDRANT_COLLECTION_NAME,
    ):
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name

    def _collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(
            collection.name == self.collection_name for collection in collections
        )

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_exists():
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            on_disk_payload=True,
        )

    @staticmethod
    def _to_qdrant_point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        vector_size = len(embeddings[0])
        self._ensure_collection(vector_size=vector_size)

        points: list[models.PointStruct] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            payload = {
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }

            points.append(
                models.PointStruct(
                    id=self._to_qdrant_point_id(chunk.chunk_id),
                    vector=embedding,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K,
        document_ids: list[str] | None = None,
        source_types: list[str] | None = None,
    ):
        query_filter = self._build_filter(
            document_ids=document_ids,
            source_types=source_types,
        )

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return result.points

    def delete_document(self, document_id: str) -> None:
        if not self._collection_exists():
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def document_exists(self, document_id: str) -> bool:
        if not self._collection_exists():
            return False

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(records) > 0

    def count_chunks(self) -> int:
        if not self._collection_exists():
            return 0

        return self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count

    def list_documents(self) -> list[dict[str, Any]]:
        if not self._collection_exists():
            return []

        documents: dict[str, dict[str, Any]] = {}
        offset = None

        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for record in records:
                payload = record.payload or {}
                document_id = payload.get("document_id")
                if not document_id:
                    continue

                if document_id not in documents:
                    documents[document_id] = {
                        "document_id": document_id,
                        "source_type": payload.get("source_type"),
                        "source_name": payload.get("source_name"),
                        "file_name": payload.get("file_name"),
                        "file_path": payload.get("file_path"),
                        "chunk_count": 0,
                    }

                documents[document_id]["chunk_count"] += 1

            if offset is None:
                break

        return sorted(documents.values(), key=lambda item: item["document_id"])

    def rebuild_collection(self, vector_size: int) -> None:
        if self._collection_exists():
            self.client.delete_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            on_disk_payload=True,
        )

    @staticmethod
    def _build_filter(
        document_ids: list[str] | None = None,
        source_types: list[str] | None = None,
    ) -> models.Filter | None:
        conditions: list[models.FieldCondition] = []

        if document_ids:
            if len(document_ids) == 1:
                conditions.append(
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_ids[0]),
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=document_ids),
                    )
                )

        if source_types:
            if len(source_types) == 1:
                conditions.append(
                    models.FieldCondition(
                        key="source_type",
                        match=models.MatchValue(value=source_types[0]),
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key="source_type",
                        match=models.MatchAny(any=source_types),
                    )
                )

        if not conditions:
            return None

        return models.Filter(must=conditions)