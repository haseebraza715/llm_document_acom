from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from text_cleaning import is_valid_document, light_clean_text


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
SKLEARN_CACHE_DIR = RAW_DIR / "sklearn_cache"

CATEGORIES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "sci.space",
    "talk.politics.misc",
]

SAMPLES_PER_CATEGORY = 10
RANDOM_SEED = 42
REMOVE_PARTS = ("headers", "footers", "quotes")


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    subset: str
    category_name: str
    category_id: int
    text: str


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR, SKLEARN_CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def fetch_subset(subset: str):
    return fetch_20newsgroups(
        subset=subset,
        categories=CATEGORIES,
        remove=REMOVE_PARTS,
        data_home=str(SKLEARN_CACHE_DIR),
        shuffle=True,
        random_state=RANDOM_SEED,
    )


def iter_subset_records(subset: str) -> Iterable[tuple[DocumentRecord, DocumentRecord]]:
    dataset = fetch_subset(subset)
    counters = {category_name: 0 for category_name in dataset.target_names}

    for raw_text, target in zip(dataset.data, dataset.target, strict=True):
        category_name = dataset.target_names[int(target)]
        counters[category_name] += 1
        doc_id = f"{subset}_{category_name.replace('.', '_')}_{counters[category_name]:04d}"
        raw_record = DocumentRecord(
            doc_id=doc_id,
            subset=subset,
            category_name=category_name,
            category_id=int(target),
            text=raw_text,
        )
        cleaned_record = DocumentRecord(
            doc_id=doc_id,
            subset=subset,
            category_name=category_name,
            category_id=int(target),
            text=light_clean_text(raw_text),
        )
        yield raw_record, cleaned_record


def select_balanced_records(cleaned_records: list[DocumentRecord], sample_size: int, seed: int) -> list[DocumentRecord]:
    selected_frames: list[pd.DataFrame] = []
    frame = pd.DataFrame(asdict(record) for record in cleaned_records)

    for category_name in CATEGORIES:
        category_frame = frame[frame["category_name"] == category_name].copy()
        valid_frame = category_frame[category_frame["text"].apply(is_valid_document)].copy()
        if len(valid_frame) < sample_size:
            raise ValueError(
                f"Not enough valid documents for {category_name}. "
                f"Required {sample_size}, found {len(valid_frame)}."
            )

        sampled = valid_frame.sample(n=sample_size, random_state=seed).sort_values("doc_id")
        selected_frames.append(sampled)

    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.sort_values(["category_name", "doc_id"]).reset_index(drop=True)
    return [DocumentRecord(**record) for record in selected.to_dict(orient="records")]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def records_to_frame(records: list[DocumentRecord]) -> pd.DataFrame:
    return pd.DataFrame(asdict(record) for record in records)


def embedding_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["doc_id", "text", "category_name", "subset"]].copy()


def build_dataset_report(raw_train: list[DocumentRecord], raw_test: list[DocumentRecord], train: pd.DataFrame, test: pd.DataFrame) -> dict:
    def subset_stats(selected_frame: pd.DataFrame, raw_records: list[DocumentRecord]) -> dict:
        counts = selected_frame["category_name"].value_counts().sort_index().to_dict()
        text_lengths = selected_frame["text"].str.len()
        word_lengths = selected_frame["text"].str.split().str.len()
        return {
            "raw_download_count": len(raw_records),
            "selected_count": int(len(selected_frame)),
            "per_category_selected": counts,
            "avg_chars": round(float(text_lengths.mean()), 2),
            "avg_words": round(float(word_lengths.mean()), 2),
            "min_chars": int(text_lengths.min()),
            "min_words": int(word_lengths.min()),
        }

    return {
        "dataset": "20 Newsgroups",
        "categories": CATEGORIES,
        "remove": list(REMOVE_PARTS),
        "random_seed": RANDOM_SEED,
        "samples_per_category_per_subset": SAMPLES_PER_CATEGORY,
        "schema": ["doc_id", "subset", "category_name", "category_id", "text"],
        "train": subset_stats(train, raw_train),
        "test": subset_stats(test, raw_test),
        "combined_selected_count": int(len(train) + len(test)),
    }


def main() -> None:
    ensure_directories()

    raw_train_records: list[DocumentRecord] = []
    cleaned_train_records: list[DocumentRecord] = []
    for raw_record, cleaned_record in iter_subset_records("train"):
        raw_train_records.append(raw_record)
        cleaned_train_records.append(cleaned_record)

    raw_test_records: list[DocumentRecord] = []
    cleaned_test_records: list[DocumentRecord] = []
    for raw_record, cleaned_record in iter_subset_records("test"):
        raw_test_records.append(raw_record)
        cleaned_test_records.append(cleaned_record)

    selected_train_records = select_balanced_records(cleaned_train_records, SAMPLES_PER_CATEGORY, RANDOM_SEED)
    selected_test_records = select_balanced_records(cleaned_test_records, SAMPLES_PER_CATEGORY, RANDOM_SEED)

    raw_train_path = RAW_DIR / "raw_train.jsonl"
    raw_test_path = RAW_DIR / "raw_test.jsonl"
    cleaned_train_path = PROCESSED_DIR / "cleaned_train.jsonl"
    cleaned_test_path = PROCESSED_DIR / "cleaned_test.jsonl"
    train_csv_path = SPLITS_DIR / "train.csv"
    test_csv_path = SPLITS_DIR / "test.csv"
    combined_csv_path = SPLITS_DIR / "combined.csv"
    embedding_train_path = PROCESSED_DIR / "embedding_input_train.jsonl"
    embedding_test_path = PROCESSED_DIR / "embedding_input_test.jsonl"
    embedding_all_path = PROCESSED_DIR / "embedding_input_all.jsonl"
    report_path = PROCESSED_DIR / "dataset_report.json"

    write_jsonl(raw_train_path, (asdict(record) for record in raw_train_records))
    write_jsonl(raw_test_path, (asdict(record) for record in raw_test_records))
    write_jsonl(cleaned_train_path, (asdict(record) for record in selected_train_records))
    write_jsonl(cleaned_test_path, (asdict(record) for record in selected_test_records))

    train_frame = records_to_frame(selected_train_records)
    test_frame = records_to_frame(selected_test_records)
    combined_frame = pd.concat([train_frame, test_frame], ignore_index=True)

    train_frame.to_csv(train_csv_path, index=False)
    test_frame.to_csv(test_csv_path, index=False)
    combined_frame.to_csv(combined_csv_path, index=False)

    write_jsonl(embedding_train_path, embedding_input_frame(train_frame).to_dict(orient="records"))
    write_jsonl(embedding_test_path, embedding_input_frame(test_frame).to_dict(orient="records"))
    write_jsonl(embedding_all_path, embedding_input_frame(combined_frame).to_dict(orient="records"))

    report = build_dataset_report(raw_train_records, raw_test_records, train_frame, test_frame)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = {
        "created_files": [
            str(raw_train_path.relative_to(BASE_DIR)),
            str(raw_test_path.relative_to(BASE_DIR)),
            str(cleaned_train_path.relative_to(BASE_DIR)),
            str(cleaned_test_path.relative_to(BASE_DIR)),
            str(train_csv_path.relative_to(BASE_DIR)),
            str(test_csv_path.relative_to(BASE_DIR)),
            str(combined_csv_path.relative_to(BASE_DIR)),
            str(embedding_train_path.relative_to(BASE_DIR)),
            str(embedding_test_path.relative_to(BASE_DIR)),
            str(embedding_all_path.relative_to(BASE_DIR)),
            str(report_path.relative_to(BASE_DIR)),
        ],
        "train_selected": len(train_frame),
        "test_selected": len(test_frame),
        "combined_selected": len(combined_frame),
        "raw_train_downloaded": len(raw_train_records),
        "raw_test_downloaded": len(raw_test_records),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
