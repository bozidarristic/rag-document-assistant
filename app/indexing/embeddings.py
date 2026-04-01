import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.models import Chunk


class TextEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def encode_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        return self.model.encode(texts, convert_to_numpy=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True)[0]