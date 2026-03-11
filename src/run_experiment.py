from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
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

BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class BaselineResults:
    frames: dict[str, pd.DataFrame]
    coordinates: dict[str, np.ndarray]
    method_statuses: dict[str, str]
    metrics_frame: pd.DataFrame


@dataclass(slots=True)
class ExperimentResult:
    run_id: str
    timestamp: str
    run_dir: Path
    variant_name: str
    metrics_frame: pd.DataFrame
    acom_summary: dict[str, object]
    summary: dict[str, object]
    comparison_row: dict[str, object]
    file_registry: dict[str, list[Path]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mapping experiment on prepared embeddings.")
    parser.add_argument("--variant-name", type=str, default=DEFAULT_CONFIG.acom_variant_name)
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
    parser.add_argument("--attraction-weight", type=float, default=DEFAULT_CONFIG.acom_attraction_weight)
    parser.add_argument("--repulsion-weight", type=float, default=DEFAULT_CONFIG.acom_repulsion_weight)
    parser.add_argument("--swap-candidates", type=int, default=DEFAULT_CONFIG.acom_swap_candidates)
    parser.add_argument(
        "--acceptance-rule",
        type=str,
        choices=["greedy", "annealed"],
        default=DEFAULT_CONFIG.acom_acceptance_rule,
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_CONFIG.acom_early_stopping_rounds,
    )
    parser.add_argument("--temperature-start", type=float, default=DEFAULT_CONFIG.acom_temperature_start)
    parser.add_argument("--temperature-decay", type=float, default=DEFAULT_CONFIG.acom_temperature_decay)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed)
    parser.add_argument("--enable-umap", action="store_true", help="Run the optional UMAP baseline.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig()
    config.acom_variant_name = args.variant_name
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
    config.acom_attraction_weight = args.attraction_weight
    config.acom_repulsion_weight = args.repulsion_weight
    config.acom_swap_candidates = args.swap_candidates
    config.acom_acceptance_rule = args.acceptance_rule
    config.acom_temperature_start = args.temperature_start
    config.acom_temperature_decay = args.temperature_decay
    config.acom_early_stopping_rounds = args.early_stopping_rounds
    config.random_seed = args.seed
    return config


def ensure_output_directories(config: ExperimentConfig) -> None:
    for path in [config.figure_dir, config.map_dir, config.report_dir, config.archive_runs_dir]:
        path.mkdir(parents=True, exist_ok=True)


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", label.strip())
    return cleaned.strip("-") or "experiment"


def create_run_context(config: ExperimentConfig, variant_name: str | None = None) -> tuple[str, str, Path]:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{sanitize_label(variant_name)}" if variant_name else ""
    run_id = f"run_{timestamp}{suffix}"
    run_dir = config.archive_runs_dir / run_id
    counter = 1
    while run_dir.exists():
        run_id = f"run_{timestamp}{suffix}_{counter:02d}"
        run_dir = config.archive_runs_dir / run_id
        counter += 1
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


def relative_repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(BASE_DIR))


def archive_run_outputs(run_dir: Path, file_registry: dict[str, list[Path]], config: ExperimentConfig) -> None:
    maps_dir = run_dir / "maps"
    figures_dir = run_dir / "figures"
    reports_dir = run_dir / "reports"
    config_dir = run_dir / "config"
    for path in [maps_dir, figures_dir, reports_dir, config_dir]:
        path.mkdir(parents=True, exist_ok=True)

    for group_name, target_dir in [("maps", maps_dir), ("figures", figures_dir), ("reports", reports_dir)]:
        for source_path in file_registry[group_name]:
            if source_path.exists():
                shutil.copy2(source_path, target_dir / source_path.name)

    (config_dir / "config_snapshot.json").write_text(json.dumps(make_json_safe(config), indent=2), encoding="utf-8")
    if config.metadata_path.exists():
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


def validate_config(config: ExperimentConfig) -> None:
    if config.grid_rows <= 0 or config.grid_cols <= 0:
        raise ValueError("Grid dimensions must be positive.")
    if config.max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if config.num_ants <= 0:
        raise ValueError("num_ants must be positive.")
    if config.neighborhood_k <= 0:
        raise ValueError("neighborhood_k must be positive.")
    if config.acom_semantic_k <= 0:
        raise ValueError("acom_semantic_k must be positive.")
    if config.acom_attraction_weight <= 0:
        raise ValueError("acom_attraction_weight must be positive.")
    if config.acom_repulsion_weight < 0:
        raise ValueError("acom_repulsion_weight must be non-negative.")
    if config.acom_swap_candidates <= 0:
        raise ValueError("acom_swap_candidates must be positive.")
    if config.acom_acceptance_rule not in {"greedy", "annealed"}:
        raise ValueError("acom_acceptance_rule must be either 'greedy' or 'annealed'.")
    if config.acom_temperature_start <= 0:
        raise ValueError("acom_temperature_start must be positive.")
    if not 0 < config.acom_temperature_decay <= 1:
        raise ValueError("acom_temperature_decay must be in the interval (0, 1].")
    if config.acom_early_stopping_rounds <= 0:
        raise ValueError("acom_early_stopping_rounds must be positive.")


def load_experiment_inputs(config: ExperimentConfig) -> tuple[pd.DataFrame, np.ndarray]:
    validate_config(config)
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

    if len(all_metadata) != config.grid_rows * config.grid_cols:
        raise ValueError(
            "Grid dimensions must match document count for a fully occupied discrete map. "
            f"documents={len(all_metadata)}, grid={config.grid_rows}x{config.grid_cols}"
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


def compute_baseline_results(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    config: ExperimentConfig,
    enable_umap: bool,
) -> BaselineResults:
    labels = metadata["category_name"].tolist()
    semantic_distances = compute_semantic_distance_matrix(embeddings, metric=config.distance_metric)

    pca_coords = run_pca(embeddings, random_seed=config.random_seed)
    tsne_coords = run_tsne(embeddings, perplexity=config.tsne_perplexity, random_seed=config.random_seed)

    frames: dict[str, pd.DataFrame] = {
        "PCA": coordinates_to_frame(pca_coords, metadata),
        "t-SNE": coordinates_to_frame(tsne_coords, metadata),
    }
    coordinates: dict[str, np.ndarray] = {
        "PCA": pca_coords,
        "t-SNE": tsne_coords,
    }
    method_statuses: dict[str, str] = {
        "PCA": "ok",
        "t-SNE": "ok",
    }

    if enable_umap:
        try:
            umap_coords = run_umap(embeddings, random_seed=config.random_seed)
            frames["UMAP"] = coordinates_to_frame(umap_coords, metadata)
            coordinates["UMAP"] = umap_coords
            method_statuses["UMAP"] = "ok"
        except Exception as exc:
            method_statuses["UMAP"] = f"skipped: {exc}"
    else:
        method_statuses["UMAP"] = "skipped: disabled (pass --enable-umap to run)"

    metrics_rows = [
        evaluate_method(method, embeddings, coords, semantic_distances, labels, config.neighborhood_k)
        for method, coords in coordinates.items()
    ]
    return BaselineResults(
        frames=frames,
        coordinates=coordinates,
        method_statuses=method_statuses,
        metrics_frame=pd.DataFrame(metrics_rows),
    )


def write_baseline_assets(config: ExperimentConfig, baseline_results: BaselineResults, semantic_distances: np.ndarray) -> tuple[list[Path], list[Path]]:
    map_paths: list[Path] = []
    figure_paths: list[Path] = []

    baseline_map_specs = {
        "PCA": config.map_dir / "pca_positions.csv",
        "t-SNE": config.map_dir / "tsne_positions.csv",
        "UMAP": config.map_dir / "umap_positions.csv",
    }
    baseline_figure_specs = {
        "PCA": config.figure_dir / "pca_scatter.png",
        "t-SNE": config.figure_dir / "tsne_scatter.png",
        "UMAP": config.figure_dir / "umap_scatter.png",
    }

    for method_name, frame in baseline_results.frames.items():
        map_path = baseline_map_specs[method_name]
        frame.to_csv(map_path, index=False)
        map_paths.append(map_path)

        figure_path = baseline_figure_specs[method_name]
        plot_2d_scatter(frame, method_name, figure_path)
        figure_paths.append(figure_path)

        correlation_path = config.figure_dir / f"distance_correlation_{method_name.lower().replace('-', '_')}.png"
        plot_distance_correlation(semantic_distances, baseline_results.coordinates[method_name], method_name, correlation_path)
        figure_paths.append(correlation_path)

    return map_paths, figure_paths


def run_single_experiment(
    config: ExperimentConfig,
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    baseline_results: BaselineResults | None = None,
    enable_umap: bool = False,
    variant_name: str | None = None,
    notes: str | None = None,
    command_used: str | None = None,
    write_baseline_outputs: bool = True,
    archive_baseline_assets: bool = True,
) -> ExperimentResult:
    ensure_output_directories(config)
    variant = variant_name or config.acom_variant_name
    config.acom_variant_name = variant
    run_id, timestamp, run_dir = create_run_context(config, variant_name=variant)

    semantic_distances = compute_semantic_distance_matrix(embeddings, metric=config.distance_metric)
    labels = metadata["category_name"].tolist()

    baseline_cache = baseline_results or compute_baseline_results(metadata, embeddings, config, enable_umap)
    baseline_statuses = dict(baseline_cache.method_statuses)

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
        attraction_weight=config.acom_attraction_weight,
        repulsion_weight=config.acom_repulsion_weight,
        swap_candidates_per_step=config.acom_swap_candidates,
        acceptance_rule=config.acom_acceptance_rule,
        temperature_start=config.acom_temperature_start,
        temperature_decay=config.acom_temperature_decay,
        early_stopping_rounds=config.acom_early_stopping_rounds,
        random_seed=config.random_seed,
    )
    acom_result = acom_mapper.run()

    acom_positions = acom_positions_to_frame(acom_result.positions, metadata)
    acom_coordinates = acom_positions[["grid_col", "grid_row"]].to_numpy(dtype=float)
    acom_metrics = evaluate_method("ACOM", embeddings, acom_coordinates, semantic_distances, labels, config.neighborhood_k)

    metrics_frames = [pd.DataFrame([acom_metrics]), baseline_cache.metrics_frame.copy()]
    metrics_frame = pd.concat(metrics_frames, ignore_index=True)

    acom_map_path = config.map_dir / "acom_positions.csv"
    metrics_csv_path = config.report_dir / "metrics_summary.csv"
    metrics_json_path = config.report_dir / "metrics_summary.json"
    acom_summary_path = config.report_dir / "acom_run_summary.json"
    acom_history_path = config.report_dir / "acom_cost_history.csv"

    acom_positions.to_csv(acom_map_path, index=False)
    metrics_frame.to_csv(metrics_csv_path, index=False)
    metrics_json_path.write_text(metrics_frame.to_json(orient="records", indent=2), encoding="utf-8")

    cost_history_frame = pd.DataFrame({"iteration": range(len(acom_result.history)), "cost": acom_result.history})
    cost_history_frame.to_csv(acom_history_path, index=False)

    acom_summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "variant_name": variant,
        "input_embedding_path": str(config.embedding_path),
        "input_metadata_path": str(config.metadata_path),
        "documents_processed": int(len(metadata)),
        "grid_rows": config.grid_rows,
        "grid_cols": config.grid_cols,
        "distance_metric": config.distance_metric,
        "semantic_k": config.acom_semantic_k,
        "neighbor_radius": config.neighbor_radius,
        "attraction_weight": config.acom_attraction_weight,
        "repulsion_weight": config.acom_repulsion_weight,
        "swap_candidates_per_step": config.acom_swap_candidates,
        "acceptance_rule": config.acom_acceptance_rule,
        "temperature_start": config.acom_temperature_start,
        "temperature_decay": config.acom_temperature_decay,
        "early_stopping_rounds": config.acom_early_stopping_rounds,
        "num_ants": config.num_ants,
        "max_iter": config.max_iter,
        "accepted_swaps": acom_result.accepted_swaps,
        "total_attempts": acom_result.total_attempts,
        "initial_cost": round(float(acom_result.initial_cost), 6),
        "final_cost": round(float(acom_result.final_cost), 6),
        "cost_improvement": round(float(acom_result.initial_cost - acom_result.final_cost), 6),
        "improved": bool(acom_result.final_cost < acom_result.initial_cost),
        "iterations_recorded": len(acom_result.history) - 1,
        "method_statuses": {"ACOM": "ok", **baseline_statuses},
        "notes": notes,
    }
    acom_summary_path.write_text(json.dumps(acom_summary, indent=2), encoding="utf-8")

    plot_acom_grid(acom_positions, config.figure_dir / "acom_grid.png")
    plot_acom_cost_history(cost_history_frame, config.figure_dir / "acom_cost_history.png")
    plot_distance_correlation(
        semantic_distances,
        acom_coordinates,
        "ACOM",
        config.figure_dir / "distance_correlation_acom.png",
    )
    plot_metric_comparison(metrics_frame, config.figure_dir / "metric_comparison.png")

    map_paths = [acom_map_path]
    figure_paths = [
        config.figure_dir / "acom_grid.png",
        config.figure_dir / "acom_cost_history.png",
        config.figure_dir / "distance_correlation_acom.png",
        config.figure_dir / "metric_comparison.png",
    ]
    if write_baseline_outputs:
        baseline_map_paths, baseline_figure_paths = write_baseline_assets(config, baseline_cache, semantic_distances)
        if archive_baseline_assets:
            map_paths.extend(baseline_map_paths)
            figure_paths.extend(baseline_figure_paths)

    report_paths = [metrics_csv_path, metrics_json_path, acom_summary_path, acom_history_path]
    file_registry = {
        "maps": map_paths,
        "figures": figure_paths,
        "reports": report_paths,
    }

    warnings = [status for status in baseline_statuses.values() if status != "ok"]
    run_manifest = {
        "run_id": run_id,
        "timestamp": timestamp,
        "variant_name": variant,
        "command_used": command_used or " ".join(["python3", "src/run_experiment.py", *sys.argv[1:]]),
        "embedding_file_used": relative_repo_path(config.embedding_path),
        "metadata_file_used": relative_repo_path(config.metadata_path),
        "methods_run_successfully": ["ACOM"] + [method for method, status in baseline_statuses.items() if status == "ok"],
        "method_statuses": {"ACOM": "ok", **baseline_statuses},
        "umap_enabled": bool(enable_umap),
        "documents_processed": int(len(metadata)),
        "grid_size": [config.grid_rows, config.grid_cols],
        "random_seed": config.random_seed,
        "acom_initial_cost": round(float(acom_result.initial_cost), 6),
        "acom_final_cost": round(float(acom_result.final_cost), 6),
        "acom_improved": bool(acom_result.final_cost < acom_result.initial_cost),
        "notes_or_warnings": ([notes] if notes else []) + warnings,
        "latest_output_paths": {
            key: [relative_repo_path(path) for path in paths]
            for key, paths in file_registry.items()
        },
    }

    archive_run_outputs(run_dir, file_registry, config)
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    run_index_entry = {
        "run_id": run_id,
        "timestamp": timestamp,
        "variant_name": variant,
        "methods_completed": ";".join(run_manifest["methods_run_successfully"]),
        "docs_count": int(len(metadata)),
        "initial_cost": round(float(acom_result.initial_cost), 6),
        "final_cost": round(float(acom_result.final_cost), 6),
        "improvement": bool(acom_result.final_cost < acom_result.initial_cost),
        "run_path": relative_repo_path(run_dir),
    }
    update_run_index(config.archive_runs_dir / "run_index.csv", run_index_entry)

    acom_row = metrics_frame.loc[metrics_frame["method"] == "ACOM"].iloc[0]
    comparison_row = {
        "run_id": run_id,
        "variant_name": variant,
        "timestamp": timestamp,
        "grid_size": f"{config.grid_rows}x{config.grid_cols}",
        "top_k_semantic_neighbors": config.acom_semantic_k,
        "neighborhood_radius": config.neighbor_radius,
        "max_iterations": config.max_iter,
        "attraction_weight": config.acom_attraction_weight,
        "repulsion_weight": config.acom_repulsion_weight,
        "swap_candidates_per_step": config.acom_swap_candidates,
        "acceptance_rule": config.acom_acceptance_rule,
        "temperature_start": config.acom_temperature_start,
        "temperature_decay": config.acom_temperature_decay,
        "early_stopping_rounds": config.acom_early_stopping_rounds,
        "initial_cost": round(float(acom_result.initial_cost), 6),
        "final_cost": round(float(acom_result.final_cost), 6),
        "cost_improvement": round(float(acom_result.initial_cost - acom_result.final_cost), 6),
        "neighborhood_preservation": float(acom_row["neighborhood_preservation"]),
        "trustworthiness": float(acom_row["trustworthiness"]),
        "stress": float(acom_row["stress"]),
        "silhouette": None if pd.isna(acom_row["silhouette"]) else float(acom_row["silhouette"]),
        "notes": notes or "",
    }

    summary = {
        "run_id": run_id,
        "variant_name": variant,
        "run_dir": str(run_dir),
        "documents_processed": int(len(metadata)),
        "embedding_shape": list(embeddings.shape),
        "methods_succeeded": run_manifest["methods_run_successfully"],
        "methods_skipped": {method: status for method, status in run_manifest["method_statuses"].items() if status != "ok"},
        "acom_initial_cost": round(float(acom_result.initial_cost), 6),
        "acom_final_cost": round(float(acom_result.final_cost), 6),
        "acom_improved": bool(acom_result.final_cost < acom_result.initial_cost),
        "metrics_path": str(metrics_csv_path),
        "acom_summary_path": str(acom_summary_path),
        "run_index_path": str(config.archive_runs_dir / "run_index.csv"),
    }

    return ExperimentResult(
        run_id=run_id,
        timestamp=timestamp,
        run_dir=run_dir,
        variant_name=variant,
        metrics_frame=metrics_frame,
        acom_summary=acom_summary,
        summary=summary,
        comparison_row=comparison_row,
        file_registry=file_registry,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    metadata, embeddings = load_experiment_inputs(config)
    result = run_single_experiment(
        config=config,
        metadata=metadata,
        embeddings=embeddings,
        baseline_results=None,
        enable_umap=args.enable_umap,
        variant_name=config.acom_variant_name,
        notes=None,
        command_used=" ".join(["python3", "src/run_experiment.py", *sys.argv[1:]]),
        write_baseline_outputs=True,
        archive_baseline_assets=True,
    )
    print(json.dumps(result.summary, indent=2))


if __name__ == "__main__":
    main()
