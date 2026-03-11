from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import DEFAULT_CONFIG, ExperimentConfig
from run_experiment import (
    compute_baseline_results,
    ensure_output_directories,
    load_experiment_inputs,
    run_single_experiment,
)
from visualization import plot_acom_variant_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled batch of ACOM variant experiments.")
    parser.add_argument("--enable-umap", action="store_true", help="Include UMAP in the shared baseline reference.")
    return parser.parse_args()


def build_variant_configs(base_config: ExperimentConfig) -> list[tuple[str, str, ExperimentConfig]]:
    return [
        (
            "acom_v1_baseline",
            "Baseline Version 1 configuration.",
            replace(base_config, acom_variant_name="acom_v1_baseline"),
        ),
        (
            "acom_v1_k10",
            "Increase semantic neighbor coverage from 8 to 10.",
            replace(base_config, acom_variant_name="acom_v1_k10", acom_semantic_k=10),
        ),
        (
            "acom_v1_more_iters",
            "Allow longer optimization with the same objective.",
            replace(
                base_config,
                acom_variant_name="acom_v1_more_iters",
                max_iter=180,
                acom_early_stopping_rounds=5,
            ),
        ),
        (
            "acom_v1_stronger_repulsion",
            "Increase local repulsion against semantically distant neighbors.",
            replace(
                base_config,
                acom_variant_name="acom_v1_stronger_repulsion",
                acom_repulsion_weight=0.5,
            ),
        ),
        (
            "acom_v1_wider_swap_search",
            "Broaden the local candidate search during swap proposals.",
            replace(
                base_config,
                acom_variant_name="acom_v1_wider_swap_search",
                acom_swap_candidates=20,
            ),
        ),
        (
            "acom_v1_radius2",
            "Expand the local neighborhood radius to two cells.",
            replace(
                base_config,
                acom_variant_name="acom_v1_radius2",
                neighbor_radius=2,
            ),
        ),
    ]


def append_comparison_row(base_config: ExperimentConfig, comparison_row: dict[str, object]) -> pd.DataFrame:
    comparison_csv_path = base_config.report_dir / "acom_variant_comparison.csv"
    comparison_json_path = base_config.report_dir / "acom_variant_comparison.json"
    comparison_plot_path = base_config.figure_dir / "acom_variant_comparison.png"

    if comparison_csv_path.exists():
        comparison_frame = pd.read_csv(comparison_csv_path)
        comparison_frame = comparison_frame[comparison_frame["variant_name"] != comparison_row["variant_name"]]
        comparison_frame = pd.concat([comparison_frame, pd.DataFrame([comparison_row])], ignore_index=True)
    else:
        comparison_frame = pd.DataFrame([comparison_row])

    comparison_frame = comparison_frame.sort_values("variant_name").reset_index(drop=True)
    comparison_frame.to_csv(comparison_csv_path, index=False)
    comparison_json_path.write_text(comparison_frame.to_json(orient="records", indent=2), encoding="utf-8")
    plot_acom_variant_comparison(comparison_frame, comparison_plot_path)
    return comparison_frame


def main() -> None:
    args = parse_args()
    base_config = ExperimentConfig()
    ensure_output_directories(base_config)
    metadata, embeddings = load_experiment_inputs(base_config)

    baseline_results = compute_baseline_results(metadata, embeddings, base_config, enable_umap=args.enable_umap)
    baseline_metrics_path = base_config.report_dir / "baseline_reference_metrics.csv"
    baseline_metrics_json_path = base_config.report_dir / "baseline_reference_metrics.json"
    baseline_results.metrics_frame.to_csv(baseline_metrics_path, index=False)
    baseline_metrics_json_path.write_text(
        baseline_results.metrics_frame.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    completed_runs: list[dict[str, object]] = []
    failed_runs: list[dict[str, str]] = []

    for index, (variant_name, notes, variant_config) in enumerate(build_variant_configs(base_config)):
        try:
            result = run_single_experiment(
                config=variant_config,
                metadata=metadata,
                embeddings=embeddings,
                baseline_results=baseline_results,
                enable_umap=args.enable_umap,
                variant_name=variant_name,
                notes=notes,
                command_used="python3 src/run_acom_sweep.py" + (" --enable-umap" if args.enable_umap else ""),
                write_baseline_outputs=(index == 0),
                archive_baseline_assets=False,
            )
            completed_runs.append(result.comparison_row)
        except Exception as exc:
            failed_runs.append({"variant_name": variant_name, "error": str(exc)})

    if not completed_runs:
        raise RuntimeError(f"No ACOM variants completed successfully. Failures: {failed_runs}")

    comparison_frame = pd.DataFrame(completed_runs).sort_values("variant_name").reset_index(drop=True)
    comparison_csv_path = base_config.report_dir / "acom_variant_comparison.csv"
    comparison_json_path = base_config.report_dir / "acom_variant_comparison.json"
    comparison_plot_path = base_config.figure_dir / "acom_variant_comparison.png"
    comparison_frame.to_csv(comparison_csv_path, index=False)
    comparison_json_path.write_text(comparison_frame.to_json(orient="records", indent=2), encoding="utf-8")
    plot_acom_variant_comparison(comparison_frame, comparison_plot_path)

    summary = {
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "baseline_metrics_path": str(baseline_metrics_path),
        "comparison_csv_path": str(comparison_csv_path),
        "comparison_json_path": str(comparison_json_path),
        "comparison_plot_path": str(comparison_plot_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
