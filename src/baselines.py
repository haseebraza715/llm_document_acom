from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def run_pca(embeddings: np.ndarray, random_seed: int = 42) -> np.ndarray:
    model = PCA(n_components=2, random_state=random_seed)
    return model.fit_transform(embeddings)


def run_tsne(embeddings: np.ndarray, perplexity: float = 30.0, random_seed: int = 42) -> np.ndarray:
    adjusted_perplexity = min(perplexity, max(5.0, (len(embeddings) - 1.0) / 3.0))
    model = TSNE(
        n_components=2,
        perplexity=adjusted_perplexity,
        init="pca",
        learning_rate="auto",
        metric="cosine",
        random_state=random_seed,
    )
    return model.fit_transform(embeddings)


def run_umap(embeddings: np.ndarray, random_seed: int = 42) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP is not installed. Add `umap-learn` to the environment.") from exc

    model = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(2, len(embeddings) - 1)),
        metric="cosine",
        random_state=random_seed,
    )
    return model.fit_transform(embeddings)
