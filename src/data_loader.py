from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_documents(path: str | Path, required_columns: tuple[str, ...] = ("id", "text")) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        frame = pd.read_csv(file_path)
    elif file_path.suffix.lower() == ".json":
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        frame = pd.DataFrame(payload)
    else:
        raise ValueError(f"Unsupported document format: {file_path.suffix}")

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required document columns: {missing}")

    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    frame["text"] = frame["text"].fillna("").astype(str)

    if "label" not in frame.columns:
        frame["label"] = "unlabeled"
    else:
        frame["label"] = frame["label"].fillna("unlabeled").astype(str)

    if frame["id"].duplicated().any():
        duplicates = frame.loc[frame["id"].duplicated(), "id"].tolist()
        raise ValueError(f"Document ids must be unique. Duplicate ids: {duplicates}")

    return frame.reset_index(drop=True)


def summarize_documents(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "num_documents": int(len(frame)),
        "num_labels": int(frame["label"].nunique()),
    }
