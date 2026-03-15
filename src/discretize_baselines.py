"""Discretize continuous PCA, t-SNE, and UMAP positions onto a 10x10 grid
and recompute evaluation metrics for a like-for-like comparison with ACOM.

Usage:
    python src/discretize_baselines.py
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import ExperimentConfig
from distance_utils import compute_semantic_distance_matrix
from metrics import (
    distance_correlation,
    neighborhood_preservation,
    silhouette_on_map,
    stress_score,
    trustworthiness_score,
)

GRID_SIZE = 10

METHODS = {
    "PCA": ("tuned_pca_positions.csv", "pca_positions.csv"),
    "t-SNE": ("tuned_tsne_positions.csv", "tsne_positions.csv"),
    "UMAP": ("tuned_umap_positions.csv", "umap_positions.csv"),
}


def load_positions(map_dir: Path, method: str) -> pd.DataFrame | None:
    """Load position CSV for a method, preferring tuned version."""
    tuned_name, fallback_name = METHODS[method]
    tuned_path = map_dir / tuned_name
    fallback_path = map_dir / fallback_name
    if tuned_path.exists():
        print(f"  [{method}] Loaded {tuned_path.name}")
        return pd.read_csv(tuned_path)
    if fallback_path.exists():
        print(f"  [{method}] Loaded {fallback_path.name} (fallback)")
        return pd.read_csv(fallback_path)
    print(f"  [{method}] SKIPPED — no position file found")
    return None


def discretize_coordinates(
    x: np.ndarray, y: np.ndarray, grid_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin continuous (x, y) into grid cells and return cell center coordinates.

    Returns (col_idx, row_idx, center_x, center_y).
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range == 0:
        x_range = 1.0
    if y_range == 0:
        y_range = 1.0

    col_idx = np.floor((x - x_min) / x_range * grid_size).astype(int)
    row_idx = np.floor((y - y_min) / y_range * grid_size).astype(int)

    # Clamp boundary values into last bin
    col_idx = np.clip(col_idx, 0, grid_size - 1)
    row_idx = np.clip(row_idx, 0, grid_size - 1)

    # Cell center coordinates
    cell_width = x_range / grid_size
    cell_height = y_range / grid_size
    center_x = x_min + (col_idx + 0.5) * cell_width
    center_y = y_min + (row_idx + 0.5) * cell_height

    return col_idx, row_idx, center_x, center_y


def compute_collisions(
    col_idx: np.ndarray, row_idx: np.ndarray, grid_size: int
) -> dict:
    """Compute collision statistics for discretized positions."""
    cell_counts: dict[tuple[int, int], int] = {}
    for c, r in zip(col_idx, row_idx):
        key = (int(c), int(r))
        cell_counts[key] = cell_counts.get(key, 0) + 1

    collision_cells = {k: v for k, v in cell_counts.items() if v > 1}
    total_occupied = len(cell_counts)
    max_per_cell = max(cell_counts.values()) if cell_counts else 0

    return {
        "total_documents": int(len(col_idx)),
        "total_occupied_cells": total_occupied,
        "total_possible_cells": grid_size * grid_size,
        "collision_cell_count": len(collision_cells),
        "max_documents_per_cell": max_per_cell,
        "collision_details": {
            f"({k[0]},{k[1]})": v for k, v in sorted(collision_cells.items())
        },
    }


def main() -> None:
    cfg = ExperimentConfig()
    map_dir = cfg.map_dir
    report_dir = cfg.report_dir

    # Load embeddings and metadata
    print("Loading embeddings and metadata...")
    embeddings = np.load(cfg.embedding_path)
    metadata = pd.read_csv(cfg.metadata_path)
    labels = metadata["category_name"].values

    # Compute reference distance matrix
    print("Computing semantic distance matrix...")
    sem_dist = compute_semantic_distance_matrix(embeddings, metric="cosine")

    # Process each method
    print(f"\nDiscretizing onto {GRID_SIZE}x{GRID_SIZE} grid (uniform bins)...\n")
    results_rows: list[dict] = []
    collision_report: dict[str, dict] = {}
    skipped: list[str] = []

    for method in METHODS:
        pos_df = load_positions(map_dir, method)
        if pos_df is None:
            skipped.append(method)
            continue

        # Align to metadata order by doc_id
        pos_df = pos_df.set_index("doc_id").reindex(metadata["doc_id"]).reset_index()
        if pos_df["x"].isna().any():
            print(f"  [{method}] WARNING: some doc_ids missing from positions, skipping")
            skipped.append(method)
            continue

        x = pos_df["x"].values.astype(float)
        y = pos_df["y"].values.astype(float)

        col_idx, row_idx, center_x, center_y = discretize_coordinates(
            x, y, GRID_SIZE
        )
        mapped = np.column_stack([center_x, center_y])

        # Collision report
        collisions = compute_collisions(col_idx, row_idx, GRID_SIZE)
        collision_report[method] = collisions
        print(
            f"  [{method}] Occupied cells: {collisions['total_occupied_cells']}/{GRID_SIZE * GRID_SIZE}, "
            f"collision cells: {collisions['collision_cell_count']}, "
            f"max per cell: {collisions['max_documents_per_cell']}"
        )

        # Compute metrics
        np_val = neighborhood_preservation(
            embeddings, mapped, k=cfg.neighborhood_k, metric_high="cosine", metric_low="euclidean"
        )
        tw_val = trustworthiness_score(
            embeddings, mapped, k=cfg.neighborhood_k, metric="cosine"
        )
        st_val = stress_score(original_distances=sem_dist, mapped_coordinates=mapped)
        dc_val = distance_correlation(sem_dist, mapped)
        sil_val = silhouette_on_map(mapped, labels)

        row = {
            "method": f"{method} (Discretized)",
            "neighborhood_preservation": np_val,
            "trustworthiness": tw_val,
            "stress": st_val,
            "distance_correlation": dc_val,
            "silhouette": sil_val if sil_val is not None else float("nan"),
            "collision_cells": collisions["collision_cell_count"],
            "max_docs_per_cell": collisions["max_documents_per_cell"],
            "occupied_cells": collisions["total_occupied_cells"],
        }
        results_rows.append(row)
        print(
            f"  [{method}] NP={np_val:.4f}  Trust={tw_val:.4f}  "
            f"Stress={st_val:.4f}  DistCorr={dc_val:.4f}  Sil={sil_val:.4f}\n"
        )

    if skipped:
        print(f"Skipped methods: {', '.join(skipped)}\n")

    # Save discretized metrics
    disc_df = pd.DataFrame(results_rows)
    disc_path = report_dir / "discretized_baselines_metrics.csv"
    disc_df.to_csv(disc_path, index=False)
    print(f"Saved: {disc_path}")

    # Save collision report
    coll_path = report_dir / "discretized_baselines_collisions.json"
    coll_path.write_text(json.dumps(collision_report, indent=2))
    print(f"Saved: {coll_path}")

    # Build full comparison table
    existing_path = report_dir / "tuned_acom_metrics_summary.csv"
    core_cols = ["method", "neighborhood_preservation", "trustworthiness", "stress", "silhouette"]

    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)[core_cols]
    else:
        print(f"WARNING: {existing_path} not found, full comparison will only contain discretized rows")
        existing_df = pd.DataFrame(columns=core_cols)

    disc_core = disc_df[core_cols]
    full_df = pd.concat([existing_df, disc_core], ignore_index=True)
    full_path = report_dir / "full_comparison_with_discretized.csv"
    full_df.to_csv(full_path, index=False)
    print(f"Saved: {full_path}")

    # Archive the run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"run_{timestamp}_discretize_baselines"
    run_dir = cfg.archive_runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    reports_archive = run_dir / "reports"
    reports_archive.mkdir(exist_ok=True)
    shutil.copy2(disc_path, reports_archive / disc_path.name)
    shutil.copy2(coll_path, reports_archive / coll_path.name)
    shutil.copy2(full_path, reports_archive / full_path.name)

    manifest = {
        "run_id": run_id,
        "timestamp": timestamp,
        "description": "Discretized PCA/t-SNE/UMAP onto 10x10 grid for like-for-like comparison with ACOM",
        "command_used": "python src/discretize_baselines.py",
        "grid_size": GRID_SIZE,
        "neighborhood_k": cfg.neighborhood_k,
        "methods_processed": [r["method"] for r in results_rows],
        "methods_skipped": skipped,
        "output_files": [
            str(disc_path.relative_to(cfg.report_dir.parents[1])),
            str(coll_path.relative_to(cfg.report_dir.parents[1])),
            str(full_path.relative_to(cfg.report_dir.parents[1])),
        ],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Archived: {run_dir}")

    # Print full comparison table
    print("\n" + "=" * 90)
    print("FULL COMPARISON TABLE (Continuous + Discretized)")
    print("=" * 90)
    print(
        full_df.to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}",
        )
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
