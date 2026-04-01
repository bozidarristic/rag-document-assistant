import numpy as np

from app.core.models import Chunk, RetrievedChunk


def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    denominator = (matrix_norms * query_norm) + 1e-12
    return np.dot(matrix, query_vector) / denominator


def retrieve_top_k(
    query_vector: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunks: list[Chunk],
    top_k: int,
) -> list[RetrievedChunk]:
    scores = cosine_similarity(query_vector, chunk_embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        RetrievedChunk(chunk=chunks[i], score=float(scores[i]))
        for i in top_indices
    ]