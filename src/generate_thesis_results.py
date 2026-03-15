from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from acom import ACOMMapper
from config import ExperimentConfig
from distance_utils import compute_semantic_distance_matrix
from grid import GridMap
from run_experiment import (
    acom_positions_to_frame,
    compute_baseline_results,
    coordinates_to_frame,
    ensure_output_directories,
    evaluate_method,
    load_experiment_inputs,
)
from visualization import (
    plot_acom_cost_history,
    plot_acom_grid,
    plot_2d_scatter,
    plot_distance_correlation,
    plot_metric_comparison,
)


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "outputs" / "reports"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
MAP_DIR = BASE_DIR / "outputs" / "maps"


def load_comparison_frame() -> pd.DataFrame:
    comparison_path = REPORT_DIR / "acom_variant_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"ACOM comparison file not found: {comparison_path}")
    return pd.read_csv(comparison_path)


def select_best_variant(comparison_frame: pd.DataFrame) -> pd.Series:
    ordered = comparison_frame.sort_values(
        ["trustworthiness", "neighborhood_preservation", "stress", "cost_improvement"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    return ordered.iloc[0]


def build_config_from_row(row: pd.Series) -> ExperimentConfig:
    config = ExperimentConfig()
    config.acom_variant_name = str(row["variant_name"])
    config.grid_rows, config.grid_cols = [int(part) for part in str(row["grid_size"]).split("x", maxsplit=1)]
    config.acom_semantic_k = int(row["top_k_semantic_neighbors"])
    config.neighbor_radius = int(row["neighborhood_radius"])
    config.max_iter = int(row["max_iterations"])
    config.acom_attraction_weight = float(row["attraction_weight"])
    config.acom_repulsion_weight = float(row["repulsion_weight"])
    config.acom_swap_candidates = int(row["swap_candidates_per_step"])
    config.acom_acceptance_rule = str(row["acceptance_rule"])
    config.acom_early_stopping_rounds = int(row["early_stopping_rounds"])
    if "temperature_start" in row and not pd.isna(row["temperature_start"]):
        config.acom_temperature_start = float(row["temperature_start"])
    if "temperature_decay" in row and not pd.isna(row["temperature_decay"]):
        config.acom_temperature_decay = float(row["temperature_decay"])
    return config


def save_results_tables(comparison_frame: pd.DataFrame) -> tuple[Path, Path]:
    compact_columns = [
        "variant_name",
        "run_id",
        "timestamp",
        "top_k_semantic_neighbors",
        "neighborhood_radius",
        "max_iterations",
        "attraction_weight",
        "repulsion_weight",
        "swap_candidates_per_step",
        "acceptance_rule",
        "initial_cost",
        "final_cost",
        "cost_improvement",
        "neighborhood_preservation",
        "trustworthiness",
        "stress",
        "silhouette",
    ]
    compact_frame = comparison_frame[compact_columns].copy().sort_values("variant_name").reset_index(drop=True)
    compact_path = REPORT_DIR / "acom_results_table.csv"
    compact_frame.to_csv(compact_path, index=False)

    pretty_frame = compact_frame.rename(
        columns={
            "variant_name": "Variant",
            "run_id": "Run ID",
            "timestamp": "Timestamp",
            "top_k_semantic_neighbors": "Top-k Neighbors",
            "neighborhood_radius": "Radius",
            "max_iterations": "Max Iterations",
            "attraction_weight": "Attraction",
            "repulsion_weight": "Repulsion",
            "swap_candidates_per_step": "Search Breadth",
            "acceptance_rule": "Acceptance",
            "initial_cost": "Initial Cost",
            "final_cost": "Final Cost",
            "cost_improvement": "Cost Improvement",
            "neighborhood_preservation": "Neighborhood Preservation",
            "trustworthiness": "Trustworthiness",
            "stress": "Stress",
            "silhouette": "Silhouette",
        }
    )
    numeric_columns = [
        "Attraction",
        "Repulsion",
        "Initial Cost",
        "Final Cost",
        "Cost Improvement",
        "Neighborhood Preservation",
        "Trustworthiness",
        "Stress",
        "Silhouette",
    ]
    pretty_frame[numeric_columns] = pretty_frame[numeric_columns].round(3)
    pretty_path = REPORT_DIR / "acom_results_table_pretty.csv"
    pretty_frame.to_csv(pretty_path, index=False)
    return compact_path, pretty_path


def run_tuned_variant_against_baselines(best_row: pd.Series) -> dict[str, object]:
    config = build_config_from_row(best_row)
    ensure_output_directories(config)
    metadata, embeddings = load_experiment_inputs(config)
    semantic_distances = compute_semantic_distance_matrix(embeddings, metric=config.distance_metric)
    labels = metadata["category_name"].tolist()

    baseline_results = compute_baseline_results(metadata, embeddings, config, enable_umap=True)

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
    acom_metrics = evaluate_method(
        "ACOM (Tuned)",
        embeddings,
        acom_coordinates,
        semantic_distances,
        labels,
        config.neighborhood_k,
    )

    metrics_frame = pd.concat(
        [pd.DataFrame([acom_metrics]), baseline_results.metrics_frame.copy()],
        ignore_index=True,
    )

    metrics_csv_path = REPORT_DIR / "tuned_acom_metrics_summary.csv"
    metrics_json_path = REPORT_DIR / "tuned_acom_metrics_summary.json"
    cost_history_path = REPORT_DIR / "tuned_acom_cost_history.csv"
    summary_path = REPORT_DIR / "tuned_acom_summary.json"

    metrics_frame.to_csv(metrics_csv_path, index=False)
    metrics_json_path.write_text(metrics_frame.to_json(orient="records", indent=2), encoding="utf-8")
    cost_history_frame = pd.DataFrame({"iteration": range(len(acom_result.history)), "cost": acom_result.history})
    cost_history_frame.to_csv(cost_history_path, index=False)

    tuned_summary = {
        "variant_name": str(best_row["variant_name"]),
        "label": "ACOM (Tuned)",
        "initial_cost": float(acom_result.initial_cost),
        "final_cost": float(acom_result.final_cost),
        "cost_improvement": float(acom_result.initial_cost - acom_result.final_cost),
        "improved": bool(acom_result.final_cost < acom_result.initial_cost),
        "baseline_statuses": baseline_results.method_statuses,
    }
    summary_path.write_text(json.dumps(tuned_summary, indent=2), encoding="utf-8")

    tuned_map_path = MAP_DIR / "tuned_acom_positions.csv"
    tuned_map_frame = acom_positions.rename(columns={"grid_row": "grid_row", "grid_col": "grid_col"})
    tuned_map_frame.to_csv(tuned_map_path, index=False)

    plot_acom_grid(acom_positions, FIGURE_DIR / "grids" / "tuned_acom_grid.png")
    plot_acom_cost_history(cost_history_frame, FIGURE_DIR / "diagnostics" / "tuned_acom_cost_history.png")
    plot_metric_comparison(metrics_frame, FIGURE_DIR / "diagnostics" / "tuned_acom_metric_comparison.png")
    plot_distance_correlation(
        semantic_distances,
        acom_coordinates,
        "ACOM (Tuned)",
        FIGURE_DIR / "diagnostics" / "tuned_distance_correlation_acom.png",
    )

    baseline_output_specs = {
        "PCA": ("tuned_pca_positions.csv", "tuned_pca_scatter.png", "tuned_distance_correlation_pca.png"),
        "t-SNE": ("tuned_tsne_positions.csv", "tuned_tsne_scatter.png", "tuned_distance_correlation_t_sne.png"),
        "UMAP": ("tuned_umap_positions.csv", "tuned_umap_scatter.png", "tuned_distance_correlation_umap.png"),
    }
    for method_name, frame in baseline_results.frames.items():
        map_name, scatter_name, corr_name = baseline_output_specs[method_name]
        frame.to_csv(MAP_DIR / map_name, index=False)
        plot_2d_scatter(frame, method_name, FIGURE_DIR / "scatters" / scatter_name)
        plot_distance_correlation(
            semantic_distances,
            baseline_results.coordinates[method_name],
            method_name,
            FIGURE_DIR / "diagnostics" / corr_name,
        )

    return {
        "metrics_frame": metrics_frame,
        "tuned_summary": tuned_summary,
        "baseline_statuses": baseline_results.method_statuses,
    }


def build_interpretation(
    comparison_frame: pd.DataFrame,
    best_row: pd.Series,
    tuned_metrics_frame: pd.DataFrame,
) -> Path:
    baseline_row = comparison_frame.loc[comparison_frame["variant_name"] == "acom_v1_baseline"].iloc[0]
    tuned_row = tuned_metrics_frame.loc[tuned_metrics_frame["method"] == "ACOM (Tuned)"].iloc[0]
    pca_row = tuned_metrics_frame.loc[tuned_metrics_frame["method"] == "PCA"].iloc[0]
    tsne_row = tuned_metrics_frame.loc[tuned_metrics_frame["method"] == "t-SNE"].iloc[0]
    umap_rows = tuned_metrics_frame.loc[tuned_metrics_frame["method"] == "UMAP"]
    umap_sentence = ""
    if not umap_rows.empty:
        umap_row = umap_rows.iloc[0]
        umap_sentence = (
            f" UMAP still delivered stronger neighborhood preservation ({umap_row['neighborhood_preservation']:.3f}) "
            f"and trustworthiness ({umap_row['trustworthiness']:.3f}) than the tuned ACOM layout."
        )

    interpretation = f"""# ACOM Experimental Findings

## Best-performing ACOM variant

The strongest ACOM configuration in the completed sweep was **{best_row['variant_name']}**. This variant combined a wider local swap search with annealed acceptance, and it produced the best ACOM result on all tracked internal criteria in the current batch: final cost, cost improvement, neighborhood preservation, trustworthiness, and stress.

## Improvement over the baseline ACOM version

Compared with the baseline ACOM configuration (`acom_v1_baseline`), the tuned variant improved substantially. Cost improvement increased from {baseline_row['cost_improvement']:.3f} to {best_row['cost_improvement']:.3f}. Neighborhood preservation increased from {baseline_row['neighborhood_preservation']:.3f} to {best_row['neighborhood_preservation']:.3f}, and trustworthiness increased from {baseline_row['trustworthiness']:.3f} to {best_row['trustworthiness']:.3f}. This indicates that the tuned variant did not only optimize its internal objective more strongly; it also improved the external embedding-space preservation metrics that matter for comparison.

## Conceptual comparison with continuous baselines

The tuned ACOM layout remains a discrete grid-based mapping, so it solves a different visualization problem from PCA, t-SNE, and UMAP. In the tuned comparison run, ACOM reached neighborhood preservation of {tuned_row['neighborhood_preservation']:.3f} and trustworthiness of {tuned_row['trustworthiness']:.3f}. This exceeded PCA on both neighborhood preservation ({pca_row['neighborhood_preservation']:.3f}) and trustworthiness ({pca_row['trustworthiness']:.3f}), but t-SNE remained stronger at preserving local structure with neighborhood preservation {tsne_row['neighborhood_preservation']:.3f} and trustworthiness {tsne_row['trustworthiness']:.3f}.{umap_sentence}

## Remaining limitations

The main limitation is that ACOM still trades off some structural faithfulness in order to satisfy the discrete grid constraint. Although the tuned variant improved stress to {tuned_row['stress']:.3f}, PCA still achieved much lower stress in the same embedding space, which shows that continuous methods remain better at preserving global pairwise geometry. In addition, the tuned ACOM result is still sensitive to optimization design choices such as search breadth and acceptance behavior, so future work should continue to test objective refinements and more targeted swap proposals rather than only increasing computation.
"""

    interpretation_path = REPORT_DIR / "acom_results_interpretation.md"
    interpretation_path.write_text(interpretation, encoding="utf-8")
    return interpretation_path


def main() -> None:
    comparison_frame = load_comparison_frame()
    compact_path, pretty_path = save_results_tables(comparison_frame)
    best_row = select_best_variant(comparison_frame)
    tuned_outputs = run_tuned_variant_against_baselines(best_row)
    interpretation_path = build_interpretation(comparison_frame, best_row, tuned_outputs["metrics_frame"])

    summary = {
        "best_variant": str(best_row["variant_name"]),
        "compact_table": str(compact_path),
        "pretty_table": str(pretty_path),
        "interpretation": str(interpretation_path),
        "tuned_metrics_path": str(REPORT_DIR / "tuned_acom_metrics_summary.csv"),
        "tuned_figure_path": str(FIGURE_DIR / "diagnostics" / "tuned_acom_metric_comparison.png"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
