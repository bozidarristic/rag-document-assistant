import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    return model.encode(query, convert_to_numpy=True)