from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    return pairwise_distances(embeddings, metric="cosine")


def euclidean_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    return pairwise_distances(embeddings, metric="euclidean")


def pairwise_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    return cosine_similarity(embeddings)


def compute_semantic_distance_matrix(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    if metric == "cosine":
        distances = cosine_distance_matrix(embeddings)
    elif metric == "euclidean":
        distances = euclidean_distance_matrix(embeddings)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")

    np.fill_diagonal(distances, 0.0)
    return distances
