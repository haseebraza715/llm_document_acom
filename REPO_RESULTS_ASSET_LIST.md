# Figures

| File path | What it is | Label |
|---|---|---|
| `outputs/figures/tuned_acom_grid.png` | final discrete ACOM map; main qualitative evidence | must include |
| `outputs/figures/tuned_pca_scatter.png` | PCA comparator for a 2x2 qualitative panel | must include |
| `outputs/figures/tuned_tsne_scatter.png` | t-SNE comparator for a 2x2 qualitative panel | must include |
| `outputs/figures/tuned_umap_scatter.png` | UMAP comparator for a 2x2 qualitative panel | must include |
| `outputs/figures/acom_variant_comparison.png` | strongest ablation / method-selection figure | must include |
| `outputs/figures/acom_scaling_neighborhood.png` | strongest scaling-quality figure | must include |
| `outputs/figures/acom_scaling_trustworthiness.png` | second-best scaling-quality figure | must include |
| `outputs/figures/acom_v1_baseline_cost_history.png` | baseline optimizer curve; useful for tuning narrative | maybe include |
| `outputs/figures/tuned_acom_cost_history.png` | tuned optimizer curve; useful for tuning narrative | maybe include |
| `outputs/figures/acom_scaling_runtime.png` | runtime scaling curve; useful if computational cost is discussed | maybe include |
| `outputs/figures/acom_scaling_stress.png` | distortion scaling curve; useful if space allows | maybe include |
| `outputs/figures/tuned_acom_metric_comparison.png` | drafting figure only; should be redrawn if used | maybe include |
| `outputs/figures/tuned_distance_correlation_acom.png` | diagnostic scatter | exclude |
| `outputs/figures/tuned_distance_correlation_pca.png` | diagnostic scatter | exclude |
| `outputs/figures/tuned_distance_correlation_t_sne.png` | diagnostic scatter | exclude |
| `outputs/figures/tuned_distance_correlation_umap.png` | diagnostic scatter | exclude |

# Tables

| File path | What it is | Label |
|---|---|---|
| `outputs/reports/tuned_acom_metrics_summary.csv` | main comparison table source; strongest support for the headline claim | must include |
| `outputs/reports/acom_results_table_pretty.csv` | ACOM ablation / tuning table source | must include |
| `outputs/reports/acom_scaling_results.csv` | scaling table source | must include |
| `data/processed/dataset_report.json` | compressed Methods/setup table source | maybe include |
| `archive/embeddings/embedding_report.json` | compressed embedding/setup table source | maybe include |
| `outputs/reports/acom_results_table.csv` | compact sweep table | maybe include |
| `outputs/reports/acom_variant_comparison.json` | machine-readable duplicate | exclude |
| `outputs/reports/tuned_acom_metrics_summary.json` | machine-readable duplicate | exclude |

# Metrics

| Metric | Best source file | Label |
|---|---|---|
| Neighborhood preservation | `outputs/reports/tuned_acom_metrics_summary.csv` and `outputs/reports/acom_variant_comparison.csv` | must include |
| Trustworthiness | `outputs/reports/tuned_acom_metrics_summary.csv` and `outputs/reports/acom_variant_comparison.csv` | must include |
| Stress | `outputs/reports/tuned_acom_metrics_summary.csv` and `outputs/reports/acom_scaling_results.csv` | must include |
| Cost improvement | `outputs/reports/acom_variant_comparison.csv` and `outputs/reports/acom_scaling_results.csv` | must include |
| Runtime | `outputs/reports/acom_scaling_results.csv` | maybe include |
| Silhouette | `outputs/reports/tuned_acom_metrics_summary.csv` | maybe include |
| Distance correlation | `outputs/reports/tuned_acom_metrics_summary.csv` | maybe include |
| Accepted swaps | archived `acom_run_summary.json` files | exclude |

# Include Labels

- Must include: tuned ACOM vs baseline ACOM improvement, tuned ACOM vs PCA/t-SNE/UMAP table, tuned ACOM grid plus baseline layout panel, and scaling quality evidence.
- Maybe include: optimizer curves, runtime scaling, dataset/embedding setup table, and silhouette as secondary evidence.
- Exclude: duplicate `docs/` copies, generic mixed-provenance `outputs/` files, and per-method distance-correlation scatters unless extra appendix space exists.
