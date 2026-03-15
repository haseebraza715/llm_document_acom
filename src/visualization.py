from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

try:
    import seaborn as sns
except ImportError:
    sns = None

if sns is not None:
    sns.set_theme(style="whitegrid")
else:
    plt.style.use("ggplot")


def _build_label_palette(labels: list[str]) -> dict[str, tuple[float, float, float]]:
    unique_labels = list(dict.fromkeys(labels))
    if sns is not None:
        colors = sns.color_palette("tab10", n_colors=max(3, len(unique_labels)))
    else:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(index) for index in range(max(3, len(unique_labels)))]
    return {label: colors[index] for index, label in enumerate(unique_labels)}


def plot_acom_grid(
    positions: pd.DataFrame,
    output_path: str | Path,
    show_doc_labels: bool = False,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = positions["category_name"].astype(str).tolist()
    palette = _build_label_palette(labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel("Grid Column", fontsize=12)
    ax.set_ylabel("Grid Row", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)

    for _, row in positions.iterrows():
        ax.scatter(
            row["grid_col"],
            row["grid_row"],
            s=380,
            c=[palette[row["category_name"]]],
            edgecolors="black",
            linewidths=0.8,
        )
        if show_doc_labels:
            ax.text(row["grid_col"], row["grid_row"], str(row["doc_id"]).split("_")[-1], ha="center", va="center", fontsize=6)

    ax.set_xticks(sorted(positions["grid_col"].unique()))
    ax.set_yticks(sorted(positions["grid_row"].unique()))
    ax.invert_yaxis()
    ax.set_aspect("equal")

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=palette[lab],
            markeredgecolor="black", markersize=10, label=lab,
        )
        for lab in palette
    ]
    ax.legend(handles=handles, fontsize=10, frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_2d_scatter(coordinates: pd.DataFrame, method_name: str, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = coordinates["category_name"].astype(str).tolist()
    palette = _build_label_palette(labels)

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, group in coordinates.groupby("category_name"):
        ax.scatter(group["x"], group["y"], label=label, s=120, alpha=0.85, color=palette[str(label)])

    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_metric_comparison(metrics_frame: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    melted = metrics_frame.melt(id_vars="method", var_name="metric", value_name="value").dropna()

    if sns is not None:
        sns.barplot(data=melted, x="metric", y="value", hue="method", ax=ax)
    else:
        pivoted = melted.pivot(index="metric", columns="method", values="value")
        pivoted.plot(kind="bar", ax=ax)

    ax.set_title("Mapping Quality Comparison", fontsize=13, fontweight="bold")
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(title="Method", fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_distance_correlation(
    original_distances: np.ndarray,
    mapped_coordinates: np.ndarray,
    method_name: str,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    mapped_distances = pairwise_distances(mapped_coordinates, metric="euclidean")
    original_flat = original_distances[np.triu_indices_from(original_distances, k=1)]
    mapped_flat = mapped_distances[np.triu_indices_from(mapped_distances, k=1)]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(original_flat, mapped_flat, alpha=0.65, s=28)
    ax.set_title(f"Distance Preservation: {method_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Original Semantic Distance", fontsize=12)
    ax.set_ylabel("Mapped 2D Distance", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_acom_cost_history(cost_history: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost_history["iteration"], cost_history["cost"], marker="o", linewidth=2, markersize=4)
    ax.set_title("ACOM Optimization Cost Over Iterations", fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cost", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_acom_variant_comparison(comparison_frame: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    variant_order = [
        "acom_v1_baseline",
        "acom_v1_k10",
        "acom_v1_more_iters",
        "acom_v1_radius2",
        "acom_v1_stronger_repulsion",
        "acom_v1_wider_swap_search",
        "acom_v1_wider_swap_annealed",
    ]
    ordered = comparison_frame.set_index("variant_name").loc[variant_order].reset_index()
    roman_labels = {
        "acom_v1_baseline": "I",
        "acom_v1_k10": "II",
        "acom_v1_more_iters": "III",
        "acom_v1_radius2": "IV",
        "acom_v1_stronger_repulsion": "V",
        "acom_v1_wider_swap_search": "VI",
        "acom_v1_wider_swap_annealed": "VII",
    }
    ordered["variant_label"] = ordered["variant_name"].map(roman_labels).fillna(ordered["variant_name"])

    figure, axes = plt.subplots(1, 3, figsize=(16, 6))
    metrics = [
        ("cost_improvement", "Cost Improvement"),
        ("neighborhood_preservation", "Neighborhood Preservation"),
        ("trustworthiness", "Trustworthiness"),
    ]

    bar_colors = [
        "#888888" if lab == "I" else "#2196F3" if lab == "VII" else "#e53935"
        for lab in ordered["variant_label"]
    ]

    for axis, (column, title) in zip(axes, metrics, strict=True):
        axis.bar(ordered["variant_label"], ordered[column], color=bar_colors)
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.set_xlabel("Variant", fontsize=12)
        axis.set_ylabel(title, fontsize=12)
        axis.tick_params(axis="both", labelsize=11)

    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_discretized_grid(
    col_idx: np.ndarray,
    row_idx: np.ndarray,
    labels: list[str] | np.ndarray,
    title: str,
    output_path: str | Path | None = None,
    ax: plt.Axes | None = None,
    grid_size: int = 10,
) -> plt.Figure | None:
    """Plot a discretized 10x10 grid showing document placements and collisions.

    Parameters
    ----------
    col_idx, row_idx : array of int, cell indices for each document.
    labels : category label per document.
    title : plot title.
    output_path : if given (and *ax* is None), save a standalone figure.
    ax : if given, draw into this axes and return None.
    grid_size : number of rows/columns.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = None

    labels_list = list(labels) if not isinstance(labels, list) else labels
    palette = _build_label_palette(labels_list)

    # Build cell -> list of labels mapping
    cell_docs: dict[tuple[int, int], list[str]] = {}
    for c, r, lab in zip(col_idx, row_idx, labels_list):
        cell_docs.setdefault((int(c), int(r)), []).append(lab)

    # Draw grid cells
    for col in range(grid_size):
        for row in range(grid_size):
            key = (col, row)
            docs = cell_docs.get(key, [])
            if len(docs) == 0:
                # Empty cell — light gray
                ax.add_patch(plt.Rectangle(
                    (col - 0.45, row - 0.45), 0.9, 0.9,
                    facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5,
                ))
            elif len(docs) == 1:
                # Single document — colored dot
                ax.scatter(
                    col, row, s=320, c=[palette[docs[0]]],
                    edgecolors="black", linewidths=0.8, zorder=3,
                )
            else:
                # Collision cell — shaded background + count
                ax.add_patch(plt.Rectangle(
                    (col - 0.45, row - 0.45), 0.9, 0.9,
                    facecolor="#ffcccc", edgecolor="#cc4444", linewidth=1.2,
                ))
                ax.text(
                    col, row, str(len(docs)),
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="#990000", zorder=4,
                )

    ax.set_xlim(-0.6, grid_size - 0.4)
    ax.set_ylim(-0.6, grid_size - 0.4)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xlabel("Grid Column", fontsize=12)
    ax.set_ylabel("Grid Row", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, fontweight="bold")

    if standalone and output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output, dpi=200)
        plt.close(fig)
    return fig


def plot_scaling_metric(
    results_frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ordered = results_frame.sort_values(x_column)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ordered[x_column], ordered[y_column], marker="o", linewidth=2)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Dataset Size", fontsize=12)
    ax.set_ylabel(y_column.replace("_", " ").title(), fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
