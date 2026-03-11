from __future__ import annotations

import math

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors


def neighborhood_preservation(
    high_dimensional: np.ndarray,
    low_dimensional: np.ndarray,
    k: int = 5,
    metric_high: str = "cosine",
    metric_low: str = "euclidean",
) -> float:
    if len(high_dimensional) <= 1:
        return 1.0

    k = min(k, len(high_dimensional) - 1)
    nn_high = NearestNeighbors(n_neighbors=k + 1, metric=metric_high).fit(high_dimensional)
    nn_low = NearestNeighbors(n_neighbors=k + 1, metric=metric_low).fit(low_dimensional)

    high_neighbors = nn_high.kneighbors(return_distance=False)[:, 1:]
    low_neighbors = nn_low.kneighbors(return_distance=False)[:, 1:]

    overlaps = []
    for idx in range(len(high_dimensional)):
        overlaps.append(len(set(high_neighbors[idx]).intersection(low_neighbors[idx])) / k)
    return float(np.mean(overlaps))


def trustworthiness_score(
    high_dimensional: np.ndarray,
    low_dimensional: np.ndarray,
    k: int = 5,
    metric: str = "cosine",
) -> float:
    if len(high_dimensional) <= 2:
        return 1.0
    k = min(k, len(high_dimensional) - 1)
    return float(trustworthiness(high_dimensional, low_dimensional, n_neighbors=k, metric=metric))


def stress_score(
    original_distances: np.ndarray | None = None,
    mapped_coordinates: np.ndarray | None = None,
    original_coordinates: np.ndarray | None = None,
    metric: str = "euclidean",
) -> float:
    if original_distances is None:
        if original_coordinates is None:
            raise ValueError("Provide either original_distances or original_coordinates.")
        original_distances = pairwise_distances(original_coordinates, metric=metric)

    if mapped_coordinates is None:
        raise ValueError("mapped_coordinates is required.")

    mapped_distances = pairwise_distances(mapped_coordinates, metric="euclidean")
    numerator = np.sum((original_distances - mapped_distances) ** 2)
    denominator = np.sum(original_distances**2)
    if denominator == 0:
        return 0.0
    return float(math.sqrt(numerator / denominator))


def silhouette_on_map(mapped_coordinates: np.ndarray, labels: list[str] | np.ndarray) -> float | None:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None
    return float(silhouette_score(mapped_coordinates, labels))


def distance_correlation(
    original_distances: np.ndarray,
    mapped_coordinates: np.ndarray,
) -> float:
    mapped_distances = pairwise_distances(mapped_coordinates, metric="euclidean")
    original_flat = original_distances[np.triu_indices_from(original_distances, k=1)]
    mapped_flat = mapped_distances[np.triu_indices_from(mapped_distances, k=1)]
    if np.std(original_flat) == 0 or np.std(mapped_flat) == 0:
        return 0.0
    return float(np.corrcoef(original_flat, mapped_flat)[0, 1])
