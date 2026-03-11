from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARCHIVE_DIR = BASE_DIR / "archive"
OUTPUTS_DIR = BASE_DIR / "outputs"


@dataclass(slots=True)
class ExperimentConfig:
    acom_variant_name: str = "acom_v1_baseline"
    metadata_path: Path = DATA_DIR / "embeddings" / "all_metadata.csv"
    embedding_path: Path = DATA_DIR / "embeddings" / "all_embeddings.npy"

    train_metadata_path: Path = DATA_DIR / "embeddings" / "train_metadata.csv"
    test_metadata_path: Path = DATA_DIR / "embeddings" / "test_metadata.csv"
    train_embedding_path: Path = DATA_DIR / "embeddings" / "train_embeddings.npy"
    test_embedding_path: Path = DATA_DIR / "embeddings" / "test_embeddings.npy"

    grid_rows: int = 10
    grid_cols: int = 10

    num_ants: int = 30
    max_iter: int = 120
    neighbor_radius: int = 1
    distance_metric: str = "cosine"
    random_seed: int = 42

    neighborhood_k: int = 10
    acom_semantic_k: int = 8
    acom_attraction_weight: float = 1.0
    acom_repulsion_weight: float = 0.35
    acom_swap_candidates: int = 12
    acom_acceptance_rule: str = "greedy"
    acom_temperature_start: float = 0.05
    acom_temperature_decay: float = 0.97
    acom_early_stopping_rounds: int = 3
    tsne_perplexity: float = 20.0

    archive_embedding_dir: Path = ARCHIVE_DIR / "embeddings"
    archive_mapping_dir: Path = ARCHIVE_DIR / "mappings"
    archive_metrics_dir: Path = ARCHIVE_DIR / "metrics"
    archive_text_dir: Path = ARCHIVE_DIR / "extracted_text"
    archive_runs_dir: Path = ARCHIVE_DIR / "runs"

    figure_dir: Path = OUTPUTS_DIR / "figures"
    map_dir: Path = OUTPUTS_DIR / "maps"
    report_dir: Path = OUTPUTS_DIR / "reports"

    required_metadata_columns: tuple[str, ...] = field(default=("doc_id", "text", "category_name", "subset"))


DEFAULT_CONFIG = ExperimentConfig()
