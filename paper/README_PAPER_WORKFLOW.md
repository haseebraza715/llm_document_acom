# Paper Workflow

This `paper/` directory now uses the ACM conference template files already present in the repository, but the manuscript content is project-specific and currently organized as a results-first ACM draft. The current Results section is structured as qualitative comparison, baseline-vs-tuned ACOM improvement, variant sweep and model selection, final benchmark comparison, and scaling behavior.

Unused ACM sample sources were moved to `paper/notes/template_examples/` so the working manuscript files remain easy to find.

## Figures

- Copied from `outputs/figures/` into `paper/figures/`
- Combined panels generated for the paper:
  - `paper/figures/qualitative_comparison_compact.png`
  - `paper/figures/baseline_vs_tuned_panel.png`
  - `paper/figures/scaling_panel_compact.png`

Primary source figures:

- `outputs/figures/tuned_acom_grid.png`
- `outputs/figures/acom_v1_baseline_grid.png`
- `outputs/figures/tuned_pca_scatter.png`
- `outputs/figures/tuned_tsne_scatter.png`
- `outputs/figures/tuned_umap_scatter.png`
- `outputs/figures/acom_variant_comparison.png`
- `outputs/figures/acom_scaling_neighborhood.png`
- `outputs/figures/acom_scaling_trustworthiness.png`
- `outputs/figures/acom_scaling_runtime.png`

## Tables

Generated manually from these source result files:

- `outputs/reports/tuned_acom_metrics_summary.csv`
- `outputs/reports/acom_results_table_pretty.csv`
- `outputs/reports/acom_scaling_results.csv`

LaTeX table files live in `paper/tables/`.

## Build

Run:

```bash
cd paper
./build.sh
```

The script prefers `tectonic` if available, otherwise falls back to `pdflatex`/`bibtex`.

## Manual Writing Still Needed

- Replace the title if desired
- Replace conference metadata
- Add real CCS concepts if required
- Add bibliography entries and any related-work discussion
- Expand Introduction / Method / Discussion only after the results layout is finalized
- Decide whether the ablation/scaling tables stay in the main paper or move to an appendix
