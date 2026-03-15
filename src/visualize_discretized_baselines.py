"""Generate grid visualizations for discretized PCA, t-SNE, and UMAP baselines.

Usage:
    python src/visualize_discretized_baselines.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ExperimentConfig
from discretize_baselines import GRID_SIZE, METHODS, discretize_coordinates, load_positions
from visualization import _build_label_palette, plot_discretized_grid


def _load_collision_report(report_dir: Path) -> dict:
    path = report_dir / "discretized_baselines_collisions.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run discretize_baselines.py first"
        )
    return json.loads(path.read_text())


def _make_title(method: str, collisions: dict) -> str:
    info = collisions[method]
    return (
        f"{method} Discretized — "
        f"{info['collision_cell_count']} collision cells, "
        f"max {info['max_documents_per_cell']} docs/cell"
    )


def main() -> None:
    cfg = ExperimentConfig()
    fig_dir = cfg.figure_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(cfg.metadata_path)
    labels = metadata["category_name"].values.tolist()
    collisions = _load_collision_report(cfg.report_dir)

    # Discretize each method and collect results
    grids: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    generated: list[str] = []

    for method in METHODS:
        pos_df = load_positions(cfg.map_dir, method)
        if pos_df is None:
            continue

        pos_df = pos_df.set_index("doc_id").reindex(metadata["doc_id"]).reset_index()
        if pos_df["x"].isna().any():
            print(f"  [{method}] WARNING: doc_id mismatch, skipping")
            continue

        col_idx, row_idx, _, _ = discretize_coordinates(
            pos_df["x"].values.astype(float),
            pos_df["y"].values.astype(float),
            GRID_SIZE,
        )
        grids[method] = (col_idx, row_idx)

        # Individual figure
        title = _make_title(method, collisions)
        suffix = method.lower().replace("-", "").replace(" ", "")
        out_path = fig_dir / "discretized" / f"discretized_{suffix}_grid.png"
        plot_discretized_grid(
            col_idx, row_idx, labels, title,
            output_path=out_path, grid_size=GRID_SIZE,
        )
        print(f"Saved: {out_path}")
        generated.append(str(out_path))

    # --- Three-panel comparison figure ---
    if grids:
        n_methods = len(grids)
        fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 7))
        if n_methods == 1:
            axes = [axes]

        for ax, method in zip(axes, grids):
            col_idx, row_idx = grids[method]
            title = _make_title(method, collisions)
            plot_discretized_grid(
                col_idx, row_idx, labels, title,
                ax=ax, grid_size=GRID_SIZE,
            )

        # Shared legend
        palette = _build_label_palette(labels)
        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=palette[lab],
                markeredgecolor="black", markersize=10, label=lab,
            )
            for lab in palette
        ]
        fig.legend(
            handles=handles, loc="lower center",
            ncol=len(palette), fontsize=9, frameon=True,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        comp_path = fig_dir / "discretized" / "discretized_comparison_grid.png"
        fig.savefig(comp_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {comp_path}")
        generated.append(str(comp_path))

    # --- Four-panel: ACOM vs discretized baselines ---
    acom_path = cfg.map_dir / "tuned_acom_positions.csv"
    if not acom_path.exists():
        acom_path = cfg.map_dir / "acom_positions.csv"
    if acom_path.exists() and grids:
        acom_df = pd.read_csv(acom_path)
        acom_df = acom_df.set_index("doc_id").reindex(metadata["doc_id"]).reset_index()
        acom_col = acom_df["grid_col"].values.astype(int)
        acom_row = acom_df["grid_row"].values.astype(int)

        methods_ordered = ["ACOM"] + list(grids.keys())
        panel_titles = ["ACOM (Tuned) — 0 collisions"] + [
            _make_title(m, collisions) for m in grids
        ]
        panel_grids = [(acom_col, acom_row)] + [grids[m] for m in grids]

        nrows, ncols = 2, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 14))
        axes_flat = axes.flatten()

        for idx, (ax, title, (ci, ri)) in enumerate(
            zip(axes_flat, panel_titles, panel_grids)
        ):
            plot_discretized_grid(
                ci, ri, labels, title,
                ax=ax, grid_size=GRID_SIZE,
            )

        # Hide unused axes if fewer than 4 panels
        for idx in range(len(panel_grids), nrows * ncols):
            axes_flat[idx].set_visible(False)

        # Shared legend
        palette = _build_label_palette(labels)
        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=palette[lab],
                markeredgecolor="black", markersize=10, label=lab,
            )
            for lab in palette
        ]
        fig.legend(
            handles=handles, loc="lower center",
            ncol=len(palette), fontsize=10, frameon=True,
            bbox_to_anchor=(0.5, -0.01),
        )
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        quad_path = fig_dir / "discretized" / "discretized_vs_acom_grid.png"
        fig.savefig(quad_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {quad_path}")
        generated.append(str(quad_path))
    else:
        print("Skipping 4-panel figure — ACOM positions not found or no grids")

    # Summary
    print(f"\nGenerated {len(generated)} figures:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
