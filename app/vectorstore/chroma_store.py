from typing import List

from app.core.config import TOP_K, CHROMA_PERSIST_DIR

import chromadb
from chromadb.config import Settings

from app.core.models import Chunk


class ChromaVectorStore:
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        self.client = chromadb.Client(
            Settings(persist_directory=persist_directory, is_persistent=True)
        )

        self.collection = self.client.get_or_create_collection(
            name="documents"
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk.text for chunk in chunks]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
        )

    def query(self, query_embedding: List[float], top_k: int = TOP_K):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        return results