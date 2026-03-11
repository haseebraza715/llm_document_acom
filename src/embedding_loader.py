from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_embeddings(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {file_path}")

    embeddings = np.load(file_path)
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D array. Received shape: {embeddings.shape}")
    return embeddings


def save_embeddings(path: str | Path, embeddings: np.ndarray) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, embeddings)


def load_embedding_metadata(path: str | Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Embedding metadata file not found: {file_path}")

    frame = pd.read_csv(file_path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns in {file_path}: {missing}")

    frame = frame[list(required_columns)].copy()
    frame["doc_id"] = frame["doc_id"].astype(str)
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["category_name"] = frame["category_name"].astype(str)
    frame["subset"] = frame["subset"].astype(str)

    if frame["doc_id"].duplicated().any():
        duplicates = frame.loc[frame["doc_id"].duplicated(), "doc_id"].tolist()
        raise ValueError(f"Duplicate doc_id values found in {file_path}: {duplicates}")

    if (frame["text"].str.strip() == "").any():
        missing_text_ids = frame.loc[frame["text"].str.strip() == "", "doc_id"].tolist()
        raise ValueError(f"Metadata contains empty texts in {file_path}: {missing_text_ids}")

    return frame.reset_index(drop=True)


def validate_embedding_alignment(metadata: pd.DataFrame, embeddings: np.ndarray, name: str) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"{name} embeddings must be 2D. Received shape: {embeddings.shape}")
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"{name} metadata/embedding row mismatch. "
            f"metadata={len(metadata)}, embeddings={embeddings.shape[0]}"
        )
    if embeddings.shape[1] == 0:
        raise ValueError(f"{name} embeddings have zero dimension.")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{name} embeddings contain non-finite values.")


def validate_split_consistency(
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    all_metadata: pd.DataFrame,
    train_embeddings: np.ndarray | None = None,
    test_embeddings: np.ndarray | None = None,
    all_embeddings: np.ndarray | None = None,
) -> None:
    combined_metadata = pd.concat([train_metadata, test_metadata], ignore_index=True)
    if combined_metadata["doc_id"].tolist() != all_metadata["doc_id"].tolist():
        raise ValueError("all_metadata.csv is not aligned with train_metadata.csv + test_metadata.csv order.")

    if train_embeddings is not None and test_embeddings is not None and all_embeddings is not None:
        combined_embeddings = np.vstack([train_embeddings, test_embeddings])
        if combined_embeddings.shape != all_embeddings.shape:
            raise ValueError(
                "all_embeddings.npy shape is not aligned with train/test embedding concatenation. "
                f"combined={combined_embeddings.shape}, all={all_embeddings.shape}"
            )
        if not np.allclose(combined_embeddings, all_embeddings):
            raise ValueError("all_embeddings.npy does not match the concatenation of train/test embeddings.")


def build_tfidf_embeddings(documents: pd.DataFrame, max_features: int = 512) -> np.ndarray:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    matrix = vectorizer.fit_transform(documents["text"].tolist())
    return matrix.toarray().astype(np.float64)


def load_or_create_embeddings(
    path: str | Path,
    documents: pd.DataFrame,
    source_path: str | Path | None = None,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, bool]:
    file_path = Path(path)
    should_rebuild = force_rebuild

    if source_path is not None and file_path.exists():
        source_mtime = Path(source_path).stat().st_mtime
        embedding_mtime = file_path.stat().st_mtime
        should_rebuild = should_rebuild or source_mtime > embedding_mtime

    if file_path.exists() and not should_rebuild:
        embeddings = load_embeddings(file_path)
        created = False
    else:
        embeddings = build_tfidf_embeddings(documents)
        save_embeddings(file_path, embeddings)
        created = True

    if embeddings.shape[0] != len(documents):
        raise ValueError(
            "Embedding count does not match document count. "
            f"Embeddings: {embeddings.shape[0]}, documents: {len(documents)}"
        )

    return embeddings, created
