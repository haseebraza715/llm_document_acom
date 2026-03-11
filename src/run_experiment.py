from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parents[1] / "outputs" / "reports" / ".mplconfig").resolve()),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str((Path(__file__).resolve().parents[1] / "outputs" / "reports" / ".cache").resolve()),
)
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str((Path(__file__).resolve().parents[1] / "outputs" / "reports" / ".numba_cache").resolve()),
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

from acom import ACOMMapper
from baselines import run_pca, run_tsne, run_umap
from config import DEFAULT_CONFIG, ExperimentConfig
from distance_utils import compute_semantic_distance_matrix
from embedding_loader import (
    load_embedding_metadata,
    load_embeddings,
    validate_embedding_alignment,
    validate_split_consistency,
)
from grid import GridMap
from metrics import (
    distance_correlation,
    neighborhood_preservation,
    silhouette_on_map,
    stress_score,
    trustworthiness_score,
)
from visualization import (
    plot_acom_cost_history,
    plot_acom_grid,
    plot_2d_scatter,
    plot_distance_correlation,
    plot_metric_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mapping experiment on prepared embeddings.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_CONFIG.metadata_path)
    parser.add_argument("--embedding-path", type=Path, default=DEFAULT_CONFIG.embedding_path)
    parser.add_argument("--train-metadata-path", type=Path, default=DEFAULT_CONFIG.train_metadata_path)
    parser.add_argument("--test-metadata-path", type=Path, default=DEFAULT_CONFIG.test_metadata_path)
    parser.add_argument("--train-embedding-path", type=Path, default=DEFAULT_CONFIG.train_embedding_path)
    parser.add_argument("--test-embedding-path", type=Path, default=DEFAULT_CONFIG.test_embedding_path)
    parser.add_argument("--grid-rows", type=int, default=DEFAULT_CONFIG.grid_rows)
    parser.add_argument("--grid-cols", type=int, default=DEFAULT_CONFIG.grid_cols)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_CONFIG.max_iter)
    parser.add_argument("--num-ants", type=int, default=DEFAULT_CONFIG.num_ants)
    parser.add_argument("--neighbor-radius", type=int, default=DEFAULT_CONFIG.neighbor_radius)
    parser.add_argument("--distance-metric", type=str, default=DEFAULT_CONFIG.distance_metric)
    parser.add_argument("--semantic-k", type=int, default=DEFAULT_CONFIG.acom_semantic_k)
    parser.add_argument("--repulsion-weight", type=float, default=DEFAULT_CONFIG.acom_repulsion_weight)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed)
    parser.add_argument("--enable-umap", action="store_true", help="Run the optional UMAP baseline.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig()
    config.metadata_path = args.metadata_path
    config.embedding_path = args.embedding_path
    config.train_metadata_path = args.train_metadata_path
    config.test_metadata_path = args.test_metadata_path
    config.train_embedding_path = args.train_embedding_path
    config.test_embedding_path = args.test_embedding_path
    config.grid_rows = args.grid_rows
    config.grid_cols = args.grid_cols
    config.max_iter = args.max_iter
    config.num_ants = args.num_ants
    config.neighbor_radius = args.neighbor_radius
    config.distance_metric = args.distance_metric
    config.acom_semantic_k = args.semantic_k
    config.acom_repulsion_weight = args.repulsion_weight
    config.random_seed = args.seed
    return config


def ensure_output_directories(config: ExperimentConfig) -> None:
    for path in [config.figure_dir, config.map_dir, config.report_dir, config.archive_runs_dir]:
        path.mkdir(parents=True, exist_ok=True)


def create_run_context(config: ExperimentConfig) -> tuple[str, str, Path]:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"run_{timestamp}"
    run_dir = config.archive_runs_dir / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"run_{timestamp}_{suffix:02d}"
        run_dir = config.archive_runs_dir / run_id
        suffix += 1
    return run_id, timestamp, run_dir


def make_json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {key: make_json_safe(val) for key, val in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def archive_run_outputs(
    run_dir: Path,
    file_registry: dict[str, list[Path] | Path],
    config: ExperimentConfig,
) -> None:
    maps_dir = run_dir / "maps"
    figures_dir = run_dir / "figures"
    reports_dir = run_dir / "reports"
    config_dir = run_dir / "config"
    for path in [maps_dir, figures_dir, reports_dir, config_dir]:
        path.mkdir(parents=True, exist_ok=True)

    for source_path in file_registry["maps"]:
        shutil.copy2(source_path, maps_dir / source_path.name)
    for source_path in file_registry["figures"]:
        shutil.copy2(source_path, figures_dir / source_path.name)
    for source_path in file_registry["reports"]:
        shutil.copy2(source_path, reports_dir / source_path.name)

    config_snapshot_path = config_dir / "config_snapshot.json"
    config_snapshot_path.write_text(json.dumps(make_json_safe(config), indent=2), encoding="utf-8")
    shutil.copy2(config.metadata_path, config_dir / config.metadata_path.name)


def update_run_index(index_path: Path, entry: dict[str, object]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    entry_frame = pd.DataFrame([entry])
    if index_path.exists():
        existing = pd.read_csv(index_path)
        updated = pd.concat([existing, entry_frame], ignore_index=True)
    else:
        updated = entry_frame
    updated.to_csv(index_path, index=False)


def load_experiment_inputs(config: ExperimentConfig) -> tuple[pd.DataFrame, np.ndarray]:
    all_metadata = load_embedding_metadata(config.metadata_path, config.required_metadata_columns)
    all_embeddings = load_embeddings(config.embedding_path)
    validate_embedding_alignment(all_metadata, all_embeddings, "all")

    if (
        config.train_metadata_path.exists()
        and config.test_metadata_path.exists()
        and config.train_embedding_path.exists()
        and config.test_embedding_path.exists()
    ):
        train_metadata = load_embedding_metadata(config.train_metadata_path, config.required_metadata_columns)
        test_metadata = load_embedding_metadata(config.test_metadata_path, config.required_metadata_columns)
        train_embeddings = load_embeddings(config.train_embedding_path)
        test_embeddings = load_embeddings(config.test_embedding_path)
        validate_embedding_alignment(train_metadata, train_embeddings, "train")
        validate_embedding_alignment(test_metadata, test_embeddings, "test")
        validate_split_consistency(
            train_metadata=train_metadata,
            test_metadata=test_metadata,
            all_metadata=all_metadata,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            all_embeddings=all_embeddings,
        )

    return all_metadata, all_embeddings


def acom_positions_to_frame(positions: dict[str, tuple[int, int]], metadata: pd.DataFrame) -> pd.DataFrame:
    positions_frame = pd.DataFrame(
        [{"doc_id": doc_id, "grid_row": row, "grid_col": col} for doc_id, (row, col) in positions.items()]
    )
    merged = metadata[["doc_id", "category_name", "subset"]].merge(positions_frame, on="doc_id", how="left")
    if merged[["grid_row", "grid_col"]].isnull().any().any():
        raise ValueError("ACOM positions could not be aligned back to metadata.")
    return merged[["doc_id", "grid_row", "grid_col", "category_name", "subset"]].copy()


def coordinates_to_frame(coordinates: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    frame = metadata[["doc_id", "category_name", "subset"]].copy()
    frame["x"] = coordinates[:, 0]
    frame["y"] = coordinates[:, 1]
    return frame[["doc_id", "x", "y", "category_name", "subset"]]


def evaluate_method(
    method_name: str,
    high_dimensional: np.ndarray,
    low_dimensional: np.ndarray,
    semantic_distances: np.ndarray,
    labels: list[str],
    k: int,
) -> dict[str, float | str | None]:
    return {
        "method": method_name,
        "neighborhood_preservation": neighborhood_preservation(high_dimensional, low_dimensional, k=k),
        "trustworthiness": trustworthiness_score(high_dimensional, low_dimensional, k=k),
        "stress": stress_score(original_distances=semantic_distances, mapped_coordinates=low_dimensional),
        "distance_correlation": distance_correlation(semantic_distances, low_dimensional),
        "silhouette": silhouette_on_map(low_dimensional, labels),
    }


def main() -> None:
    args = parse_args()
    config = build_config(args)
    ensure_output_directories(config)
    run_id, timestamp, run_dir = create_run_context(config)

    metadata, embeddings = load_experiment_inputs(config)
    if len(metadata) != config.grid_rows * config.grid_cols:
        raise ValueError(
            "Grid dimensions must match document count for a fully occupied discrete map. "
            f"documents={len(metadata)}, grid={config.grid_rows}x{config.grid_cols}"
        )

    semantic_distances = compute_semantic_distance_matrix(embeddings, metric=config.distance_metric)

    grid = GridMap(
        rows=config.grid_rows,
        cols=config.grid_cols,
        doc_ids=metadata["doc_id"].tolist(),
        random_seed=config.random_seed,
    )
    grid.initialize_random()

    acom_mapper = ACOMMapper(
        grid=grid,
        semantic_distances=semantic_distances,
        num_ants=config.num_ants,
        max_iter=config.max_iter,
        radius=config.neighbor_radius,
        semantic_k=config.acom_semantic_k,
        repulsion_weight=config.acom_repulsion_weight,
        random_seed=config.random_seed,
    )
    acom_result = acom_mapper.run()

    acom_positions = acom_positions_to_frame(acom_result.positions, metadata)
    acom_coordinates = acom_positions[["grid_col", "grid_row"]].to_numpy(dtype=float)
    labels = metadata["category_name"].tolist()

    pca_coords = run_pca(embeddings, random_seed=config.random_seed)
    tsne_coords = run_tsne(embeddings, perplexity=config.tsne_perplexity, random_seed=config.random_seed)

    method_frames: dict[str, pd.DataFrame] = {
        "PCA": coordinates_to_frame(pca_coords, metadata),
        "t-SNE": coordinates_to_frame(tsne_coords, metadata),
    }
    method_coordinates: dict[str, np.ndarray] = {
        "ACOM": acom_coordinates,
        "PCA": pca_coords,
        "t-SNE": tsne_coords,
    }
    method_statuses: dict[str, str] = {
        "ACOM": "ok",
        "PCA": "ok",
        "t-SNE": "ok",
    }

    if args.enable_umap:
        try:
            umap_coords = run_umap(embeddings, random_seed=config.random_seed)
            method_frames["UMAP"] = coordinates_to_frame(umap_coords, metadata)
            method_coordinates["UMAP"] = umap_coords
            method_statuses["UMAP"] = "ok"
        except Exception as exc:
            method_statuses["UMAP"] = f"skipped: {exc}"
    else:
        method_statuses["UMAP"] = "skipped: disabled (pass --enable-umap to run)"

    metrics_rows = [
        evaluate_method(
            method_name=method_name,
            high_dimensional=embeddings,
            low_dimensional=coordinates,
            semantic_distances=semantic_distances,
            labels=labels,
            k=config.neighborhood_k,
        )
        for method_name, coordinates in method_coordinates.items()
    ]
    metrics_frame = pd.DataFrame(metrics_rows)

    acom_map_path = config.map_dir / "acom_positions.csv"
    pca_map_path = config.map_dir / "pca_positions.csv"
    tsne_map_path = config.map_dir / "tsne_positions.csv"
    umap_map_path = config.map_dir / "umap_positions.csv"
    metrics_csv_path = config.report_dir / "metrics_summary.csv"
    metrics_json_path = config.report_dir / "metrics_summary.json"
    acom_summary_path = config.report_dir / "acom_run_summary.json"
    acom_history_path = config.report_dir / "acom_cost_history.csv"

    acom_positions.to_csv(acom_map_path, index=False)
    method_frames["PCA"].to_csv(pca_map_path, index=False)
    method_frames["t-SNE"].to_csv(tsne_map_path, index=False)
    if "UMAP" in method_frames:
        method_frames["UMAP"].to_csv(umap_map_path, index=False)

    metrics_frame.to_csv(metrics_csv_path, index=False)
    metrics_json_path.write_text(metrics_frame.to_json(orient="records", indent=2), encoding="utf-8")

    cost_history_frame = pd.DataFrame({"iteration": range(len(acom_result.history)), "cost": acom_result.history})
    cost_history_frame.to_csv(acom_history_path, index=False)

    acom_summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "input_embedding_path": str(config.embedding_path),
        "input_metadata_path": str(config.metadata_path),
        "documents_processed": int(len(metadata)),
        "grid_rows": config.grid_rows,
        "grid_cols": config.grid_cols,
        "distance_metric": config.distance_metric,
        "semantic_k": config.acom_semantic_k,
        "neighbor_radius": config.neighbor_radius,
        "repulsion_weight": config.acom_repulsion_weight,
        "num_ants": config.num_ants,
        "max_iter": config.max_iter,
        "accepted_swaps": acom_result.accepted_swaps,
        "total_attempts": acom_result.total_attempts,
        "initial_cost": round(float(acom_result.initial_cost), 6),
        "final_cost": round(float(acom_result.final_cost), 6),
        "improved": bool(acom_result.final_cost < acom_result.initial_cost),
        "iterations_recorded": len(acom_result.history) - 1,
        "method_statuses": method_statuses,
    }
    acom_summary_path.write_text(json.dumps(acom_summary, indent=2), encoding="utf-8")

    plot_acom_grid(acom_positions, config.figure_dir / "acom_grid.png")
    plot_2d_scatter(method_frames["PCA"], "PCA", config.figure_dir / "pca_scatter.png")
    plot_2d_scatter(method_frames["t-SNE"], "t-SNE", config.figure_dir / "tsne_scatter.png")
    if "UMAP" in method_frames:
        plot_2d_scatter(method_frames["UMAP"], "UMAP", config.figure_dir / "umap_scatter.png")

    plot_metric_comparison(metrics_frame, config.figure_dir / "metric_comparison.png")
    plot_acom_cost_history(cost_history_frame, config.figure_dir / "acom_cost_history.png")
    for method_name, coordinates in method_coordinates.items():
        plot_distance_correlation(
            semantic_distances,
            coordinates,
            method_name,
            config.figure_dir / f"distance_correlation_{method_name.lower().replace('-', '_')}.png",
        )

    figure_paths = [
        config.figure_dir / "acom_grid.png",
        config.figure_dir / "pca_scatter.png",
        config.figure_dir / "tsne_scatter.png",
        config.figure_dir / "metric_comparison.png",
        config.figure_dir / "acom_cost_history.png",
        config.figure_dir / "distance_correlation_acom.png",
        config.figure_dir / "distance_correlation_pca.png",
        config.figure_dir / "distance_correlation_t_sne.png",
    ]
    if "UMAP" in method_frames:
        figure_paths.extend(
            [
                config.figure_dir / "umap_scatter.png",
                config.figure_dir / "distance_correlation_umap.png",
            ]
        )

    map_paths = [acom_map_path, pca_map_path, tsne_map_path]
    if "UMAP" in method_frames:
        map_paths.append(umap_map_path)

    report_paths = [metrics_csv_path, metrics_json_path, acom_summary_path, acom_history_path]

    file_registry = {
        "maps": map_paths,
        "figures": figure_paths,
        "reports": report_paths,
    }

    warnings = [status for status in method_statuses.values() if status != "ok"]
    command_used = " ".join(["python3", "src/run_experiment.py", *sys.argv[1:]])
    run_manifest = {
        "run_id": run_id,
        "timestamp": timestamp,
        "command_used": command_used,
        "embedding_file_used": str(config.embedding_path),
        "metadata_file_used": str(config.metadata_path),
        "methods_run_successfully": [method for method, status in method_statuses.items() if status == "ok"],
        "method_statuses": method_statuses,
        "umap_enabled": bool(args.enable_umap),
        "documents_processed": int(len(metadata)),
        "grid_size": [config.grid_rows, config.grid_cols],
        "random_seed": config.random_seed,
        "acom_initial_cost": round(float(acom_result.initial_cost), 6),
        "acom_final_cost": round(float(acom_result.final_cost), 6),
        "acom_improved": bool(acom_result.final_cost < acom_result.initial_cost),
        "notes_or_warnings": warnings,
        "latest_output_paths": make_json_safe(file_registry),
    }

    archive_run_outputs(run_dir, file_registry, config)
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    run_index_entry = {
        "run_id": run_id,
        "timestamp": timestamp,
        "methods_completed": ";".join(run_manifest["methods_run_successfully"]),
        "docs_count": int(len(metadata)),
        "initial_cost": round(float(acom_result.initial_cost), 6),
        "final_cost": round(float(acom_result.final_cost), 6),
        "improvement": bool(acom_result.final_cost < acom_result.initial_cost),
        "run_path": str(run_dir),
    }
    update_run_index(config.archive_runs_dir / "run_index.csv", run_index_entry)

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "documents_processed": int(len(metadata)),
        "embedding_shape": list(embeddings.shape),
        "methods_succeeded": [method for method, status in method_statuses.items() if status == "ok"],
        "methods_skipped": {method: status for method, status in method_statuses.items() if status != "ok"},
        "acom_initial_cost": round(float(acom_result.initial_cost), 6),
        "acom_final_cost": round(float(acom_result.final_cost), 6),
        "acom_improved": bool(acom_result.final_cost < acom_result.initial_cost),
        "metrics_path": str(metrics_csv_path),
        "acom_summary_path": str(acom_summary_path),
        "run_index_path": str(config.archive_runs_dir / "run_index.csv"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
