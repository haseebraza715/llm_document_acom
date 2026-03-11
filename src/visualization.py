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


def plot_acom_grid(positions: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = positions["category_name"].astype(str).tolist()
    palette = _build_label_palette(labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("ACOM Discrete Document Map")
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")

    for _, row in positions.iterrows():
        ax.scatter(
            row["grid_col"],
            row["grid_row"],
            s=380,
            c=[palette[row["category_name"]]],
            edgecolors="black",
            linewidths=0.8,
        )
        ax.text(row["grid_col"], row["grid_row"], str(row["doc_id"]).split("_")[-1], ha="center", va="center", fontsize=6)

    ax.set_xticks(sorted(positions["grid_col"].unique()))
    ax.set_yticks(sorted(positions["grid_row"].unique()))
    ax.invert_yaxis()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_2d_scatter(coordinates: pd.DataFrame, method_name: str, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = coordinates["category_name"].astype(str).tolist()
    palette = _build_label_palette(labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"{method_name} Document Layout")

    for label, group in coordinates.groupby("category_name"):
        ax.scatter(group["x"], group["y"], label=label, s=120, alpha=0.85, color=palette[str(label)])

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(frameon=True)
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

    ax.set_title("Mapping Quality Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(title="Method")
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
    ax.set_title(f"Distance Preservation: {method_name}")
    ax.set_xlabel("Original Semantic Distance")
    ax.set_ylabel("Mapped 2D Distance")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_acom_cost_history(cost_history: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost_history["iteration"], cost_history["cost"], marker="o", linewidth=2, markersize=4)
    ax.set_title("ACOM Optimization Cost Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
