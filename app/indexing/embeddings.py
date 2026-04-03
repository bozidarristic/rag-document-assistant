from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.models import Chunk


class TextEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )