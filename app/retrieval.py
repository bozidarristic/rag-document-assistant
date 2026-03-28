import numpy as np
from sentence_transformers import SentenceTransformer

from app.embeddings import encode_query, encode_texts


def cosine_similarity(query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)
    doc_norms = np.linalg.norm(doc_vectors, axis=1)
    similarities = np.dot(doc_vectors, query_vector) / (doc_norms * query_norm + 1e-12)
    return similarities


def retrieve_top_k(
    query: str,
    chunks: list[str],
    model: SentenceTransformer,
    k: int = 3,
) -> list[dict]:
    chunk_embeddings = encode_texts(model, chunks)
    query_embedding = encode_query(model, query)

    similarities = cosine_similarity(query_embedding, chunk_embeddings)
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    results: list[dict] = []
    for idx in top_k_indices:
        results.append(
            {
                "chunk_index": int(idx),
                "score": float(similarities[idx]),
                "text": chunks[idx],
            }
        )

    return results