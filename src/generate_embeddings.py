from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EMBEDDING_DIR = BASE_DIR / "data" / "embeddings"
ARCHIVE_EMBEDDING_DIR = BASE_DIR / "archive" / "embeddings"

TRAIN_INPUT_PATH = PROCESSED_DIR / "embedding_input_train.jsonl"
TEST_INPUT_PATH = PROCESSED_DIR / "embedding_input_test.jsonl"
ALL_INPUT_PATH = PROCESSED_DIR / "embedding_input_all.jsonl"

TRAIN_EMBEDDING_PATH = EMBEDDING_DIR / "train_embeddings.npy"
TEST_EMBEDDING_PATH = EMBEDDING_DIR / "test_embeddings.npy"
ALL_EMBEDDING_PATH = EMBEDDING_DIR / "all_embeddings.npy"

TRAIN_METADATA_PATH = EMBEDDING_DIR / "train_metadata.csv"
TEST_METADATA_PATH = EMBEDDING_DIR / "test_metadata.csv"
ALL_METADATA_PATH = EMBEDDING_DIR / "all_metadata.csv"

EMBEDDING_REPORT_PATH = ARCHIVE_EMBEDDING_DIR / "embedding_report.json"

REQUIRED_COLUMNS = ("doc_id", "text", "category_name", "subset")


@dataclass(slots=True)
class EmbeddingArtifacts:
    embeddings: dict[str, np.ndarray]
    model_used: str
    backend_used: str
    embedding_dimension: int
    runtime_seconds: float
    failed_records: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings for the prepared 20 Newsgroups dataset.")
    parser.add_argument(
        "--backend",
        choices=["auto", "sentence-transformers", "tfidf"],
        default="auto",
        help="Embedding backend. 'auto' prefers sentence-transformers and falls back to TF-IDF.",
    )
    parser.add_argument(
        "--model-name",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name to load when using that backend.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for sentence-transformers encoding.",
    )
    parser.add_argument(
        "--tfidf-max-features",
        type=int,
        default=2048,
        help="Maximum TF-IDF feature count for fallback embeddings.",
    )
    return parser.parse_args()


def ensure_directories() -> None:
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)


def load_embedding_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Embedding input file not found: {path}")

    frame = pd.read_json(path, lines=True)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    frame = frame[list(REQUIRED_COLUMNS)].copy()
    frame["doc_id"] = frame["doc_id"].astype(str)
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["category_name"] = frame["category_name"].astype(str)
    frame["subset"] = frame["subset"].astype(str)

    if frame["doc_id"].duplicated().any():
        duplicates = frame.loc[frame["doc_id"].duplicated(), "doc_id"].tolist()
        raise ValueError(f"Duplicate doc_id values found in {path}: {duplicates}")

    if (frame["text"].str.strip() == "").any():
        missing_text_ids = frame.loc[frame["text"].str.strip() == "", "doc_id"].tolist()
        raise ValueError(f"Missing or empty texts found in {path}: {missing_text_ids}")

    return frame.reset_index(drop=True)


def validate_split_relationships(train_frame: pd.DataFrame, test_frame: pd.DataFrame, all_frame: pd.DataFrame) -> None:
    combined_frame = pd.concat([train_frame, test_frame], ignore_index=True)
    if len(all_frame) != len(combined_frame):
        raise ValueError(
            "Combined input size mismatch. "
            f"all={len(all_frame)}, train+test={len(combined_frame)}"
        )

    combined_ids = combined_frame["doc_id"].tolist()
    all_ids = all_frame["doc_id"].tolist()
    if combined_ids != all_ids:
        raise ValueError("The row order in embedding_input_all.jsonl does not match train+test concatenation.")


def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def generate_sentence_transformer_embeddings(
    all_texts: list[str],
    train_count: int,
    test_count: int,
    model_name: str,
    batch_size: int,
) -> EmbeddingArtifacts:
    start_time = time.perf_counter()
    model = _load_sentence_transformer(model_name)
    all_embeddings = model.encode(
        all_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ).astype(np.float32)
    runtime_seconds = time.perf_counter() - start_time

    return EmbeddingArtifacts(
        embeddings={
            "train": all_embeddings[:train_count],
            "test": all_embeddings[train_count : train_count + test_count],
            "all": all_embeddings,
        },
        model_used=f"sentence-transformers:{model_name}",
        backend_used="sentence-transformers",
        embedding_dimension=int(all_embeddings.shape[1]),
        runtime_seconds=runtime_seconds,
        failed_records=[],
    )


def generate_tfidf_embeddings(
    train_texts: list[str],
    test_texts: list[str],
    all_texts: list[str],
    max_features: int,
) -> EmbeddingArtifacts:
    start_time = time.perf_counter()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    all_embeddings = vectorizer.fit_transform(all_texts).toarray().astype(np.float32)
    train_embeddings = vectorizer.transform(train_texts).toarray().astype(np.float32)
    test_embeddings = vectorizer.transform(test_texts).toarray().astype(np.float32)
    runtime_seconds = time.perf_counter() - start_time

    return EmbeddingArtifacts(
        embeddings={
            "train": train_embeddings,
            "test": test_embeddings,
            "all": all_embeddings,
        },
        model_used=f"tfidf:max_features={max_features}",
        backend_used="tfidf",
        embedding_dimension=int(all_embeddings.shape[1]),
        runtime_seconds=runtime_seconds,
        failed_records=[],
    )


def generate_embeddings(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    all_frame: pd.DataFrame,
    backend: str,
    model_name: str,
    batch_size: int,
    tfidf_max_features: int,
) -> EmbeddingArtifacts:
    all_texts = all_frame["text"].tolist()
    train_texts = train_frame["text"].tolist()
    test_texts = test_frame["text"].tolist()

    if backend == "sentence-transformers":
        return generate_sentence_transformer_embeddings(
            all_texts=all_texts,
            train_count=len(train_frame),
            test_count=len(test_frame),
            model_name=model_name,
            batch_size=batch_size,
        )

    if backend == "tfidf":
        return generate_tfidf_embeddings(
            train_texts=train_texts,
            test_texts=test_texts,
            all_texts=all_texts,
            max_features=tfidf_max_features,
        )

    try:
        return generate_sentence_transformer_embeddings(
            all_texts=all_texts,
            train_count=len(train_frame),
            test_count=len(test_frame),
            model_name=model_name,
            batch_size=batch_size,
        )
    except Exception as exc:
        fallback_artifacts = generate_tfidf_embeddings(
            train_texts=train_texts,
            test_texts=test_texts,
            all_texts=all_texts,
            max_features=tfidf_max_features,
        )
        fallback_artifacts.failed_records.append(
            {
                "stage": "backend_selection",
                "backend": "sentence-transformers",
                "reason": str(exc),
            }
        )
        return fallback_artifacts


def generate_embeddings_for_frame(
    frame: pd.DataFrame,
    backend: str,
    model_name: str,
    batch_size: int,
    tfidf_max_features: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if backend == "sentence-transformers":
        start_time = time.perf_counter()
        model = _load_sentence_transformer(model_name)
        embeddings = model.encode(
            frame["text"].tolist(),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)
        runtime_seconds = time.perf_counter() - start_time
        return embeddings, {
            "backend_used": "sentence-transformers",
            "embedding_model_used": f"sentence-transformers:{model_name}",
            "runtime_seconds": round(float(runtime_seconds), 4),
            "failed_records": [],
        }

    if backend == "tfidf":
        start_time = time.perf_counter()
        vectorizer = TfidfVectorizer(stop_words="english", max_features=tfidf_max_features)
        embeddings = vectorizer.fit_transform(frame["text"].tolist()).toarray().astype(np.float32)
        runtime_seconds = time.perf_counter() - start_time
        return embeddings, {
            "backend_used": "tfidf",
            "embedding_model_used": f"tfidf:max_features={tfidf_max_features}",
            "runtime_seconds": round(float(runtime_seconds), 4),
            "failed_records": [],
        }

    try:
        return generate_embeddings_for_frame(
            frame=frame,
            backend="sentence-transformers",
            model_name=model_name,
            batch_size=batch_size,
            tfidf_max_features=tfidf_max_features,
        )
    except Exception as exc:
        embeddings, report = generate_embeddings_for_frame(
            frame=frame,
            backend="tfidf",
            model_name=model_name,
            batch_size=batch_size,
            tfidf_max_features=tfidf_max_features,
        )
        report["failed_records"] = [
            {
                "stage": "backend_selection",
                "backend": "sentence-transformers",
                "reason": str(exc),
            }
        ]
        return embeddings, report


def validate_embeddings(frame: pd.DataFrame, embeddings: np.ndarray, split_name: str) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"{split_name} embeddings must be 2D. Received shape {embeddings.shape}.")

    if len(frame) != embeddings.shape[0]:
        raise ValueError(
            f"{split_name} metadata/embedding row count mismatch. "
            f"metadata={len(frame)}, embeddings={embeddings.shape[0]}"
        )

    if embeddings.shape[1] == 0:
        raise ValueError(f"{split_name} embeddings have zero dimension.")

    if not np.isfinite(embeddings).all():
        raise ValueError(f"{split_name} embeddings contain non-finite values.")

    row_norms = np.linalg.norm(embeddings, axis=1)
    if np.any(row_norms == 0):
        zero_docs = frame.loc[row_norms == 0, "doc_id"].tolist()
        raise ValueError(f"{split_name} embeddings contain empty vectors for doc_ids: {zero_docs}")

    if frame["doc_id"].isnull().any():
        raise ValueError(f"{split_name} metadata contains null doc_id values.")


def save_split_outputs(frame: pd.DataFrame, embeddings: np.ndarray, embedding_path: Path, metadata_path: Path) -> None:
    np.save(embedding_path, embeddings)
    frame.to_csv(metadata_path, index=False)


def build_report(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    all_frame: pd.DataFrame,
    artifacts: EmbeddingArtifacts,
) -> dict:
    return {
        "inputs": {
            "train": str(TRAIN_INPUT_PATH.relative_to(BASE_DIR)),
            "test": str(TEST_INPUT_PATH.relative_to(BASE_DIR)),
            "all": str(ALL_INPUT_PATH.relative_to(BASE_DIR)),
        },
        "outputs": {
            "train_embeddings": str(TRAIN_EMBEDDING_PATH.relative_to(BASE_DIR)),
            "test_embeddings": str(TEST_EMBEDDING_PATH.relative_to(BASE_DIR)),
            "all_embeddings": str(ALL_EMBEDDING_PATH.relative_to(BASE_DIR)),
            "train_metadata": str(TRAIN_METADATA_PATH.relative_to(BASE_DIR)),
            "test_metadata": str(TEST_METADATA_PATH.relative_to(BASE_DIR)),
            "all_metadata": str(ALL_METADATA_PATH.relative_to(BASE_DIR)),
            "report": str(EMBEDDING_REPORT_PATH.relative_to(BASE_DIR)),
        },
        "backend_used": artifacts.backend_used,
        "embedding_model_used": artifacts.model_used,
        "embedding_dimension": artifacts.embedding_dimension,
        "documents_embedded": {
            "train": int(len(train_frame)),
            "test": int(len(test_frame)),
            "all": int(len(all_frame)),
        },
        "embedding_shapes": {
            "train": list(artifacts.embeddings["train"].shape),
            "test": list(artifacts.embeddings["test"].shape),
            "all": list(artifacts.embeddings["all"].shape),
        },
        "runtime_summary": {
            "total_seconds": round(float(artifacts.runtime_seconds), 4),
        },
        "failed_records": artifacts.failed_records,
        "skipped_records": [],
    }


def main() -> None:
    args = parse_args()
    ensure_directories()

    train_frame = load_embedding_input(TRAIN_INPUT_PATH)
    test_frame = load_embedding_input(TEST_INPUT_PATH)
    all_frame = load_embedding_input(ALL_INPUT_PATH)
    validate_split_relationships(train_frame, test_frame, all_frame)

    artifacts = generate_embeddings(
        train_frame=train_frame,
        test_frame=test_frame,
        all_frame=all_frame,
        backend=args.backend,
        model_name=args.model_name,
        batch_size=args.batch_size,
        tfidf_max_features=args.tfidf_max_features,
    )

    validate_embeddings(train_frame, artifacts.embeddings["train"], "train")
    validate_embeddings(test_frame, artifacts.embeddings["test"], "test")
    validate_embeddings(all_frame, artifacts.embeddings["all"], "all")

    save_split_outputs(train_frame, artifacts.embeddings["train"], TRAIN_EMBEDDING_PATH, TRAIN_METADATA_PATH)
    save_split_outputs(test_frame, artifacts.embeddings["test"], TEST_EMBEDDING_PATH, TEST_METADATA_PATH)
    save_split_outputs(all_frame, artifacts.embeddings["all"], ALL_EMBEDDING_PATH, ALL_METADATA_PATH)

    report = build_report(train_frame, test_frame, all_frame, artifacts)
    EMBEDDING_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
