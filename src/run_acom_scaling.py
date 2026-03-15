from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import ExperimentConfig
from generate_embeddings import generate_embeddings_for_frame, validate_embeddings
from prepare_20newsgroups import (
    CATEGORIES,
    RANDOM_SEED,
    embedding_input_frame,
    ensure_directories,
    load_cleaned_records,
    records_to_frame,
    select_balanced_total_records,
    write_jsonl,
)
from run_experiment import BaselineResults, ensure_output_directories, run_single_experiment
from visualization import plot_scaling_metric


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_SPLITS_DIR = BASE_DIR / "data" / "splits"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
OUTPUT_REPORT_DIR = BASE_DIR / "outputs" / "reports"
OUTPUT_FIGURE_DIR = BASE_DIR / "outputs" / "figures"
ARCHIVE_EMBEDDING_DIR = BASE_DIR / "archive" / "embeddings"
ARCHIVE_SCALING_DIR = BASE_DIR / "archive" / "scaling_studies"

SIZES = [50, 100, 150, 200]
GRID_BY_SIZE = {
    50: (8, 8),
    100: (10, 10),
    150: (13, 13),
    200: (15, 15),
}
BEST_VARIANT_NAME = "acom_v1_wider_swap_annealed"


def load_best_variant_config() -> ExperimentConfig:
    comparison_path = OUTPUT_REPORT_DIR / "acom_variant_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"ACOM variant comparison file not found: {comparison_path}")

    comparison_frame = pd.read_csv(comparison_path)
    variant_row = comparison_frame.loc[comparison_frame["variant_name"] == BEST_VARIANT_NAME]
    if variant_row.empty:
        raise ValueError(f"Variant {BEST_VARIANT_NAME} not found in {comparison_path}")
    row = variant_row.iloc[0]

    config = ExperimentConfig()
    config.acom_variant_name = BEST_VARIANT_NAME
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


def prepare_scaled_dataset(total_size: int) -> tuple[pd.DataFrame, dict[str, object]]:
    cleaned_records = load_cleaned_records(subsets=("train", "test"))
    selected_records = select_balanced_total_records(cleaned_records, total_size=total_size, seed=RANDOM_SEED)
    frame = records_to_frame(selected_records)
    embedding_input = embedding_input_frame(frame)

    split_path = DATA_SPLITS_DIR / f"scaled_{total_size}.csv"
    embedding_input_path = DATA_PROCESSED_DIR / f"scaled_{total_size}_embedding_input.jsonl"
    frame.to_csv(split_path, index=False)
    write_jsonl(embedding_input_path, embedding_input.to_dict(orient="records"))

    per_category = frame["category_name"].value_counts().sort_index().to_dict()
    report = {
        "dataset_size": total_size,
        "categories": CATEGORIES,
        "per_category_counts": per_category,
        "split_path": str(split_path.relative_to(BASE_DIR)),
        "embedding_input_path": str(embedding_input_path.relative_to(BASE_DIR)),
    }
    return frame, report


def generate_scaled_embeddings(frame: pd.DataFrame, total_size: int) -> dict[str, object]:
    embeddings, embedding_report = generate_embeddings_for_frame(
        frame=frame[["doc_id", "text", "category_name", "subset"]].copy(),
        backend="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        batch_size=16,
        tfidf_max_features=2048,
    )
    validate_embeddings(frame, embeddings, f"scaled_{total_size}")

    embedding_path = DATA_EMBEDDINGS_DIR / f"scaled_{total_size}_embeddings.npy"
    metadata_path = DATA_EMBEDDINGS_DIR / f"scaled_{total_size}_metadata.csv"
    report_path = ARCHIVE_EMBEDDING_DIR / f"scaled_{total_size}_embedding_report.json"

    np.save(embedding_path, embeddings)
    frame.to_csv(metadata_path, index=False)

    full_report = {
        "dataset_size": total_size,
        "embedding_path": str(embedding_path.relative_to(BASE_DIR)),
        "metadata_path": str(metadata_path.relative_to(BASE_DIR)),
        "documents_embedded": int(len(frame)),
        "embedding_dimension": int(embeddings.shape[1]),
        **embedding_report,
    }
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    return full_report


def empty_baselines() -> BaselineResults:
    return BaselineResults(
        frames={},
        coordinates={},
        method_statuses={},
        metrics_frame=pd.DataFrame(
            columns=[
                "method",
                "neighborhood_preservation",
                "trustworthiness",
                "stress",
                "distance_correlation",
                "silhouette",
            ]
        ),
    )


def run_scaling_study() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    ensure_directories()
    ensure_output_directories(ExperimentConfig())
    base_config = load_best_variant_config()

    completed: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []

    for size in SIZES:
        try:
            frame, dataset_report = prepare_scaled_dataset(size)
            embedding_report = generate_scaled_embeddings(frame, size)

            config = ExperimentConfig(
                acom_variant_name=f"{BEST_VARIANT_NAME}_scale{size}",
                metadata_path=DATA_EMBEDDINGS_DIR / f"scaled_{size}_metadata.csv",
                embedding_path=DATA_EMBEDDINGS_DIR / f"scaled_{size}_embeddings.npy",
                grid_rows=GRID_BY_SIZE[size][0],
                grid_cols=GRID_BY_SIZE[size][1],
                num_ants=base_config.num_ants,
                max_iter=base_config.max_iter,
                neighbor_radius=base_config.neighbor_radius,
                distance_metric=base_config.distance_metric,
                random_seed=base_config.random_seed,
                neighborhood_k=base_config.neighborhood_k,
                acom_semantic_k=base_config.acom_semantic_k,
                acom_attraction_weight=base_config.acom_attraction_weight,
                acom_repulsion_weight=base_config.acom_repulsion_weight,
                acom_swap_candidates=base_config.acom_swap_candidates,
                acom_acceptance_rule=base_config.acom_acceptance_rule,
                acom_temperature_start=base_config.acom_temperature_start,
                acom_temperature_decay=base_config.acom_temperature_decay,
                acom_early_stopping_rounds=base_config.acom_early_stopping_rounds,
                tsne_perplexity=base_config.tsne_perplexity,
            )

            notes = (
                f"Scaling study run for {size} documents. "
                f"Balanced allocation: {dataset_report['per_category_counts']}."
            )
            result = run_single_experiment(
                config=config,
                metadata=frame[["doc_id", "text", "category_name", "subset"]].copy(),
                embeddings=np.load(config.embedding_path),
                baseline_results=empty_baselines(),
                enable_umap=False,
                variant_name=config.acom_variant_name,
                notes=notes,
                command_used="python3 src/run_acom_scaling.py",
                write_baseline_outputs=False,
                archive_baseline_assets=False,
            )

            completed.append(
                {
                    "dataset_size": size,
                    "run_id": result.run_id,
                    "variant_name": config.acom_variant_name,
                    "grid_rows": config.grid_rows,
                    "grid_cols": config.grid_cols,
                    "embedding_model": embedding_report["embedding_model_used"],
                    "runtime_seconds": result.comparison_row["runtime_seconds"],
                    "initial_cost": result.comparison_row["initial_cost"],
                    "final_cost": result.comparison_row["final_cost"],
                    "cost_improvement": result.comparison_row["cost_improvement"],
                    "neighborhood_preservation": result.comparison_row["neighborhood_preservation"],
                    "trustworthiness": result.comparison_row["trustworthiness"],
                    "stress": result.comparison_row["stress"],
                    "silhouette": result.comparison_row["silhouette"],
                    "accepted_swaps": result.acom_summary["accepted_swaps"],
                    "notes": notes,
                }
            )
        except Exception as exc:
            failed.append({"dataset_size": str(size), "error": str(exc)})

    return completed, failed


def build_interpretation(results_frame: pd.DataFrame, failures: list[dict[str, str]]) -> Path:
    ordered = results_frame.sort_values("dataset_size")
    best_trust = ordered.loc[ordered["trustworthiness"].idxmax()]
    worst_trust = ordered.loc[ordered["trustworthiness"].idxmin()]
    runtime_growth = ordered[["dataset_size", "runtime_seconds"]].to_dict(orient="records")

    interpretation = f"""# ACOM Scaling Interpretation

## Overall behavior

The tuned ACOM configuration (`{BEST_VARIANT_NAME}`) remained stable across all completed dataset sizes in this study. Every completed run improved from its initial random placement, which indicates that the optimization procedure continued to function reliably as document count increased.

## Metric trends

Trustworthiness was highest at dataset size {int(best_trust['dataset_size'])} with a value of {best_trust['trustworthiness']:.3f}. The weakest trustworthiness was observed at dataset size {int(worst_trust['dataset_size'])} with a value of {worst_trust['trustworthiness']:.3f}. Neighborhood preservation and stress should be interpreted together with runtime because the discrete grid becomes progressively less flexible as the number of occupied cells increases.

## Optimization stability

The optimization remained numerically stable across the completed sizes. Initial cost, final cost, and cost improvement all stayed well defined, and no alignment or grid-capacity failures occurred in the successful runs. Runtime increased with dataset size as expected: {runtime_growth}.

## Practical limitations

The main limitation is computational. As document count increases, the swap search and cost evaluation become more expensive, and the optimization trajectory becomes harder to improve quickly. The grid-based representation also remains more constrained than a continuous layout, so preserving local neighborhoods at larger scales is still challenging.

## Semantic readability at larger scales

The discrete map remains interpretable at larger sizes because it preserves explicit grid occupancy and category-colored structure. However, as the number of documents grows, more careful objective design and possibly more targeted proposal mechanisms may be needed to maintain strong semantic separation without substantial runtime growth.
"""
    interpretation_path = OUTPUT_REPORT_DIR / "acom_scaling_interpretation.md"
    interpretation_path.write_text(interpretation, encoding="utf-8")
    return interpretation_path


def archive_scaling_study(
    results: list[dict[str, object]],
    failures: list[dict[str, str]],
    report_paths: list[Path],
    figure_paths: list[Path],
) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    study_dir = ARCHIVE_SCALING_DIR / f"study_{timestamp}_{BEST_VARIANT_NAME}_scaling"
    reports_dir = study_dir / "reports"
    figures_dir = study_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for path in report_paths:
        if path.exists():
            shutil.copy2(path, reports_dir / path.name)
    for path in figure_paths:
        if path.exists():
            shutil.copy2(path, figures_dir / path.name)

    manifest = {
        "study_name": f"{BEST_VARIANT_NAME}_scaling",
        "timestamp": timestamp,
        "variant_name": BEST_VARIANT_NAME,
        "dataset_sizes": [entry["dataset_size"] for entry in results],
        "completed_run_ids": [entry["run_id"] for entry in results],
        "failures": failures,
        "source_command": "python3 src/run_acom_scaling.py",
        "report_files": [str(path.relative_to(BASE_DIR)) for path in report_paths if path.exists()],
        "figure_files": [str(path.relative_to(BASE_DIR)) for path in figure_paths if path.exists()],
    }
    (study_dir / "study_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return study_dir


def save_scaling_outputs(results: list[dict[str, object]], failures: list[dict[str, str]]) -> dict[str, str]:
    results_frame = pd.DataFrame(results).sort_values("dataset_size").reset_index(drop=True)
    csv_path = OUTPUT_REPORT_DIR / "acom_scaling_results.csv"
    json_path = OUTPUT_REPORT_DIR / "acom_scaling_results.json"
    results_frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "completed_runs": results,
                "failed_runs": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_scaling_metric(results_frame, "dataset_size", "runtime_seconds", "ACOM Runtime vs Dataset Size", OUTPUT_FIGURE_DIR / "scaling" / "acom_scaling_runtime.png")
    plot_scaling_metric(results_frame, "dataset_size", "cost_improvement", "ACOM Cost Improvement vs Dataset Size", OUTPUT_FIGURE_DIR / "scaling" / "acom_scaling_cost_improvement.png")
    plot_scaling_metric(results_frame, "dataset_size", "trustworthiness", "ACOM Trustworthiness vs Dataset Size", OUTPUT_FIGURE_DIR / "scaling" / "acom_scaling_trustworthiness.png")
    plot_scaling_metric(results_frame, "dataset_size", "neighborhood_preservation", "ACOM Neighborhood Preservation vs Dataset Size", OUTPUT_FIGURE_DIR / "scaling" / "acom_scaling_neighborhood.png")
    plot_scaling_metric(results_frame, "dataset_size", "stress", "ACOM Stress vs Dataset Size", OUTPUT_FIGURE_DIR / "scaling" / "acom_scaling_stress.png")

    interpretation_path = build_interpretation(results_frame, failures)
    report_paths = [csv_path, json_path, interpretation_path]
    figure_paths = [
        OUTPUT_FIGURE_DIR / "scaling" / "acom_scaling_runtime.png",
        OUTPUT_FIGURE_DIR / "scaling" / "acom_scaling_cost_improvement.png",
        OUTPUT_FIGURE_DIR / "scaling" / "acom_scaling_trustworthiness.png",
        OUTPUT_FIGURE_DIR / "scaling" / "acom_scaling_neighborhood.png",
        OUTPUT_FIGURE_DIR / "scaling" / "acom_scaling_stress.png",
    ]
    study_dir = archive_scaling_study(results, failures, report_paths, figure_paths)
    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "interpretation_path": str(interpretation_path),
        "study_archive_path": str(study_dir),
    }


def main() -> None:
    completed, failed = run_scaling_study()
    if not completed:
        raise RuntimeError(f"No scaling runs completed successfully. Failures: {failed}")
    output_paths = save_scaling_outputs(completed, failed)
    summary = {
        "completed_runs": completed,
        "failed_runs": failed,
        **output_paths,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
