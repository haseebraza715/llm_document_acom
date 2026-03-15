# 1. Project at a Glance

- Short explanation of the project: map document embeddings onto a discrete 2D grid with an ACOM-style swap optimizer, then compare that discrete map against continuous 2D baselines.
- Main research question: can a discrete semantic grid preserve enough embedding-space structure to be competitive with standard 2D projections while remaining more interpretable as an explicit placement map?
- Dataset used: balanced 5-class subset of 20 Newsgroups with 100 documents by default (`comp.graphics`, `rec.sport.baseball`, `sci.med`, `sci.space`, `talk.politics.misc`), plus scaled variants at 50/100/150/200 docs. Evidence: `data/processed/dataset_report.json`, `src/prepare_20newsgroups.py`.
- Embedding model used: `sentence-transformers:all-MiniLM-L6-v2`, 384 dimensions, 100-doc embedding runtime `5.7658 s`. Evidence: `archive/embeddings/embedding_report.json`, `src/generate_embeddings.py`.
- Algorithms compared: ACOM, PCA, t-SNE, optional UMAP. Evidence: `src/run_experiment.py`, `src/baselines.py`, `src/generate_thesis_results.py`.
- Main claimed achievement: the tuned ACOM variant `acom_v1_wider_swap_annealed` turns an initially weak discrete baseline into a publishable comparison point: it improves baseline ACOM from neighborhood preservation `0.134` to `0.367` and trustworthiness `0.567` to `0.787`, and it surpasses PCA on those two local-structure metrics while retaining an explicit grid output. Evidence: `outputs/reports/acom_variant_comparison.csv`, `outputs/reports/tuned_acom_metrics_summary.csv`.

# 2. Repository Structure Relevant to the Paper

| Path | What it contains | Paper relevance |
|---|---|---|
| `outputs/reports/` | working comparison tables and paper-facing summaries | highest-priority paper asset directory, but mixed provenance |
| `outputs/figures/` | working figures for final comparison, tuning, and scaling | highest-priority figure directory, but mixed provenance |
| `archive/runs/` | timestamped per-run evidence bundles | primary source for reproducible benchmark and ablation claims |
| `archive/scaling_studies/` | archived scaling-study bundle | primary source for scaling claims |
| `archive/embeddings/` | embedding configuration and runtime reports | Methods/setup evidence |
| `data/processed/dataset_report.json` | dataset composition statistics | Methods/setup evidence |
| `src/generate_thesis_results.py` | script that generates the final comparison assets | important for tracing what the repo treated as paper-ready |
| `src/run_acom_sweep.py` | script that generates the ACOM ablation/tuning evidence | important for tracing the method-selection story |
| `src/run_acom_scaling.py` | script that generates the scaling study | important for tracing scalability claims |
| `src/acom.py`, `src/baselines.py`, `src/metrics.py` | method and metric definitions | Methods section support only |
| `docs/PROJECT_RESEARCH_REPORT.md` | curated narrative synthesis | secondary drafting aid, not primary evidence |
| `docs/figures/`, `docs/results/` | duplicated selected assets | low priority; mostly mirrors of `outputs/` |

Where figures are located:

- Working figures: `outputs/figures/`
- Archived run figures: `archive/runs/<run_id>/figures/`
- Archived scaling figures: `archive/scaling_studies/<study_id>/figures/`
- Curated duplicates: `docs/figures/`

Where tables/reports are located:

- Working reports: `outputs/reports/`
- Archived per-run reports: `archive/runs/<run_id>/reports/`
- Archived study reports: `archive/scaling_studies/<study_id>/reports/`
- Curated duplicates: `docs/results/`

# 3. Experimental Pipeline

1. Prepare a balanced five-category 20 Newsgroups subset with light cleaning and deterministic sampling (`src/prepare_20newsgroups.py`).
2. Write cleaned splits and embedding-input JSONL/CSV files under `data/processed/` and `data/splits/`.
3. Generate embeddings with `sentence-transformers:all-MiniLM-L6-v2` and save aligned `.npy` arrays plus metadata CSVs under `data/embeddings/` (`src/generate_embeddings.py`).
4. Load embeddings and metadata, validate alignment, and compute cosine semantic distances (`src/run_experiment.py`, `src/distance_utils.py`, `src/embedding_loader.py`).
5. Initialize a grid and run ACOM as a swap-based stochastic optimizer (`src/grid.py`, `src/acom.py`).
6. Run continuous baselines on the same embeddings: PCA, t-SNE, and optionally UMAP (`src/baselines.py`).
7. Evaluate all methods with shared metrics: neighborhood preservation, trustworthiness, stress, distance correlation, silhouette (`src/metrics.py`).
8. Write CSV/JSON summaries, plots, coordinate files, and archived run bundles; later synthesize a best-variant comparison and scaling study (`src/run_acom_sweep.py`, `src/generate_thesis_results.py`, `src/run_acom_scaling.py`).

# 4. Experiments Found in the Repo

| Experiment / family | Purpose | Script / file used | Input data | Methods compared | Main outputs produced | Status |
|---|---|---|---|---|---|---|
| Final tuned comparison vs PCA/t-SNE/UMAP | main publishable benchmark comparison | `src/generate_thesis_results.py` | default 100-doc benchmark | `ACOM (Tuned)`, PCA, t-SNE, UMAP | `outputs/reports/tuned_acom_metrics_summary.csv`, tuned scatter/grid figures, tuned maps | Complete, high-value, but not archived as its own timestamped run |
| ACOM variant sweep | ablation / tuning evidence that justifies the chosen ACOM variant | `src/run_acom_sweep.py` | default 100-doc benchmark | named ACOM variants; PCA/t-SNE reference cached once | `outputs/reports/acom_variant_comparison.csv`, `outputs/reports/acom_results_table_pretty.csv`, `outputs/figures/acom_variant_comparison.png`, archived variant runs | Complete and paper-relevant |
| Scaling study | main scalability evidence for the tuned ACOM variant | `src/run_acom_scaling.py` | regenerated balanced datasets and embeddings: `scaled_50`, `scaled_100`, `scaled_150`, `scaled_200` | tuned ACOM only | `outputs/reports/acom_scaling_results.csv`, scaling figures, archived scale runs, archived study bundle | Complete and paper-relevant |
| Baseline ACOM comparison run | baseline anchor for the tuning story | `src/run_acom_sweep.py` calling `run_single_experiment(...)` | default 100-doc benchmark | ACOM baseline plus fixed PCA/t-SNE reference | `archive/runs/run_2026-03-11_02-44-06_acom_v1_baseline/*` | Complete |
| Runtime-related evidence | limited computational-cost evidence | `src/generate_embeddings.py`, `src/run_acom_scaling.py` | default benchmark and scaled datasets | no cross-method runtime comparison | `archive/embeddings/embedding_report.json`, `outputs/reports/acom_scaling_results.csv` | Partial; useful but not publication-grade benchmarking |
| Early benchmark run `run_2026-03-11_02-26-49` | older comparison run that predates the organized sweep | `src/run_experiment.py` | `data/embeddings/all_embeddings.npy`, `data/embeddings/all_metadata.csv` | ACOM, PCA, t-SNE | `archive/runs/run_2026-03-11_02-26-49/*` | Obsolete / ambiguous; should not anchor paper claims |
| Paper/report synthesis | converts raw results into tables and narrative | `src/generate_thesis_results.py`, `docs/PROJECT_RESEARCH_REPORT.md` | sweep outputs and tuned comparison outputs | not a new experiment | curated tables, copied figures, narrative report | Drafting support only |

Named ACOM variants found:

- `acom_v1_baseline`
- `acom_v1_k10`
- `acom_v1_more_iters`
- `acom_v1_stronger_repulsion`
- `acom_v1_wider_swap_search`
- `acom_v1_radius2`
- `acom_v1_wider_swap_annealed`

# 5. Results Inventory

## Final comparison results

| File path | Artifact type | What it shows | Paper importance | Suggested paper use |
|---|---|---|---|---|
| `outputs/reports/tuned_acom_metrics_summary.csv` | table / metrics file | final ACOM (Tuned) vs PCA/t-SNE/UMAP metrics | highest-value numeric summary; likely the main comparison table | must include |
| `outputs/reports/tuned_acom_metrics_summary.json` | json summary | same final comparison in JSON | backup / machine-readable | probably exclude |
| `outputs/reports/tuned_acom_summary.json` | json summary | tuned ACOM cost summary and baseline availability | useful provenance | maybe include |
| `outputs/reports/tuned_acom_cost_history.csv` | metric time series | tuned ACOM optimization trajectory | supports optimizer story | maybe include |
| `outputs/figures/tuned_acom_grid.png` | figure | final discrete ACOM grid on benchmark | strongest qualitative ACOM result | must include |
| `outputs/figures/tuned_acom_cost_history.png` | figure | tuned ACOM cost curve | useful for optimization story | maybe include |
| `outputs/figures/tuned_acom_metric_comparison.png` | figure | bar chart of final metrics across ACOM/PCA/t-SNE/UMAP | useful as an internal drafting figure; should be redrawn or replaced for a paper | maybe include |
| `outputs/figures/tuned_pca_scatter.png` | figure | final PCA layout | strong qualitative comparator | must include |
| `outputs/figures/tuned_tsne_scatter.png` | figure | final t-SNE layout | strong qualitative comparator | must include |
| `outputs/figures/tuned_umap_scatter.png` | figure | final UMAP layout | strong qualitative comparator | must include |
| `outputs/maps/tuned_acom_positions.csv` | map csv | tuned ACOM grid assignments | useful for figure regeneration | maybe include |
| `outputs/maps/tuned_pca_positions.csv` | map csv | final PCA coordinates | useful for figure regeneration | maybe include |
| `outputs/maps/tuned_tsne_positions.csv` | map csv | final t-SNE coordinates | useful for figure regeneration | maybe include |
| `outputs/maps/tuned_umap_positions.csv` | map csv | final UMAP coordinates | useful for figure regeneration | maybe include |
| `outputs/figures/tuned_distance_correlation_acom.png` | figure | pairwise semantic distance vs mapped distance for tuned ACOM | secondary diagnostic | probably exclude |
| `outputs/figures/tuned_distance_correlation_pca.png` | figure | same diagnostic for PCA | secondary diagnostic | probably exclude |
| `outputs/figures/tuned_distance_correlation_t_sne.png` | figure | same diagnostic for t-SNE | secondary diagnostic | probably exclude |
| `outputs/figures/tuned_distance_correlation_umap.png` | figure | same diagnostic for UMAP | secondary diagnostic | probably exclude |

## ACOM variant / tuning results

| File path | Artifact type | What it shows | Paper importance | Suggested paper use |
|---|---|---|---|---|
| `outputs/reports/acom_variant_comparison.csv` | table | sweep-wide ACOM variant comparison with exact metrics | strongest tuning / ablation table; essential for method-selection justification | must include |
| `outputs/reports/acom_variant_comparison.json` | json summary | same variant comparison in JSON | backup / machine-readable | probably exclude |
| `outputs/reports/acom_results_table.csv` | compact table | sweep results in a cleaner tabular format | useful for appendix | maybe include |
| `outputs/reports/acom_results_table_pretty.csv` | curated table | rounded sweep table ready for thesis/paper drafting | high-value drafting asset | must include |
| `outputs/reports/acom_results_interpretation.md` | narrative report | text interpretation of sweep plus final comparison | useful for framing, not primary evidence | maybe include |
| `outputs/figures/acom_variant_comparison.png` | figure | three-panel variant comparison: cost improvement, neighborhood preservation, trustworthiness | strongest tuning figure; one of the most publishable figures in the repo | must include |
| `archive/runs/run_2026-03-11_02-44-06_acom_v1_baseline/reports/acom_run_summary.json` | archived run summary | baseline ACOM parameters, cost statistics, accepted swaps | key baseline provenance | maybe include |
| `archive/runs/run_2026-03-11_02-44-06_acom_v1_baseline/reports/metrics_summary.csv` | archived metrics file | baseline ACOM + PCA + t-SNE metrics | baseline evidence | must include |
| `archive/runs/run_2026-03-11_02-44-07_acom_v1_k10/reports/acom_run_summary.json` | archived run summary | k=10 variant provenance | useful ablation evidence | maybe include |
| `archive/runs/run_2026-03-11_02-44-09_acom_v1_more_iters/reports/acom_run_summary.json` | archived run summary | longer-iteration variant provenance | useful ablation evidence | maybe include |
| `archive/runs/run_2026-03-11_02-44-11_acom_v1_stronger_repulsion/reports/acom_run_summary.json` | archived run summary | stronger-repulsion variant provenance | notable negative result | maybe include |
| `archive/runs/run_2026-03-11_02-44-11_acom_v1_wider_swap_search/reports/acom_run_summary.json` | archived run summary | wider-search variant provenance | key tuning step before annealing | must include |
| `archive/runs/run_2026-03-11_02-44-15_acom_v1_radius2/reports/acom_run_summary.json` | archived run summary | radius-2 variant provenance | useful negative / mixed result | maybe include |
| `archive/runs/run_2026-03-11_02-59-54_acom_v1_wider_swap_annealed/reports/acom_run_summary.json` | archived run summary | best ACOM variant provenance | core method result | must include |
| `archive/runs/run_2026-03-11_02-59-54_acom_v1_wider_swap_annealed/reports/metrics_summary.csv` | archived metrics file | best ACOM vs PCA/t-SNE metrics before UMAP was added | core comparison evidence | must include |
| `outputs/figures/acom_v1_baseline_grid.png` | figure | manually surfaced baseline grid figure | good for baseline-vs-tuned qualitative contrast | maybe include |
| `outputs/figures/acom_v1_baseline_cost_history.png` | figure | manually surfaced baseline cost curve | strongest companion to tuned cost curve | must include |

## Scaling results

| File path | Artifact type | What it shows | Paper importance | Suggested paper use |
|---|---|---|---|---|
| `outputs/reports/acom_scaling_results.csv` | table | tuned ACOM metrics and runtime at 50/100/150/200 docs | strongest scaling table; likely the paper's main scalability table | must include |
| `outputs/reports/acom_scaling_results.json` | json summary | same scaling results in JSON | backup / machine-readable | probably exclude |
| `outputs/reports/acom_scaling_interpretation.md` | narrative report | concise auto-generated scaling interpretation | useful text support | maybe include |
| `outputs/reports/acom_scaling_discussion.md` | narrative report | longer manually written scaling discussion | useful synthesis, not a primary result file | maybe include |
| `outputs/figures/acom_scaling_runtime.png` | figure | runtime vs dataset size | publishable if runtime is part of the evaluation story | must include |
| `outputs/figures/acom_scaling_cost_improvement.png` | figure | cost improvement vs dataset size | optimizer-internal evidence, weaker than quality plots | maybe include |
| `outputs/figures/acom_scaling_neighborhood.png` | figure | neighborhood preservation vs size | strongest quality-scaling figure | must include |
| `outputs/figures/acom_scaling_trustworthiness.png` | figure | trustworthiness vs size | strong quality-scaling figure | must include |
| `outputs/figures/acom_scaling_stress.png` | figure | stress vs size | strong distortion-scaling figure | maybe include |
| `archive/scaling_studies/study_2026-03-11_03-26-48_acom_v1_wider_swap_annealed_scaling/study_manifest.json` | study manifest | archived bundle index for the scaling study | reproducibility aid | maybe include |
| `archive/scaling_studies/study_2026-03-11_03-26-48_acom_v1_wider_swap_annealed_scaling/reports/acom_scaling_results.csv` | archived table | archived copy of scaling results | source-of-truth duplicate | must include |
| `archive/runs/run_2026-03-11_03-23-20_acom_v1_wider_swap_annealed_scale50/reports/acom_run_summary.json` | archived run summary | 50-doc run provenance | useful when discussing runtime and accepted swaps | maybe include |
| `archive/runs/run_2026-03-11_03-23-38_acom_v1_wider_swap_annealed_scale100/reports/acom_run_summary.json` | archived run summary | 100-doc scaling run provenance | useful; also reveals slight mismatch vs final tuned benchmark values | maybe include |
| `archive/runs/run_2026-03-11_03-23-57_acom_v1_wider_swap_annealed_scale150/reports/acom_run_summary.json` | archived run summary | 150-doc run provenance | useful negative scaling evidence | maybe include |
| `archive/runs/run_2026-03-11_03-24-42_acom_v1_wider_swap_annealed_scale200/reports/acom_run_summary.json` | archived run summary | 200-doc run provenance | useful negative scaling evidence | maybe include |

## Qualitative visualizations

| File path | Artifact type | What it shows | Paper importance | Suggested paper use |
|---|---|---|---|---|
| `archive/runs/run_2026-03-11_02-59-54_acom_v1_wider_swap_annealed/figures/acom_grid.png` | archived figure | archived tuned ACOM grid from sweep run | source-of-truth duplicate of tuned qualitative result | maybe include |
| `archive/runs/run_2026-03-11_02-44-06_acom_v1_baseline/figures/acom_grid.png` | archived figure | archived baseline grid | useful baseline contrast | maybe include |
| `archive/runs/run_2026-03-11_02-26-49/figures/pca_scatter.png` | archived figure | early PCA scatter | older comparator figure | probably exclude |
| `archive/runs/run_2026-03-11_02-26-49/figures/tsne_scatter.png` | archived figure | early t-SNE scatter | older comparator figure | probably exclude |
| `docs/figures/tuned_acom_grid.png` | duplicate figure | curated copy of tuned ACOM grid | duplicate of output; no added paper value | probably exclude |
| `docs/figures/tuned_acom_metric_comparison.png` | duplicate figure | curated copy of tuned metric chart | duplicate of output | probably exclude |
| `docs/figures/acom_variant_comparison.png` | duplicate figure | curated copy of variant comparison | duplicate of output | probably exclude |

## Runtime results

| File path | Artifact type | What it shows | Paper importance | Suggested paper use |
|---|---|---|---|---|
| `archive/embeddings/embedding_report.json` | json summary | embedding backend, dimension, shapes, total runtime | Methods/setup evidence, not a headline result | maybe include |
| `archive/embeddings/scaled_50_embedding_report.json` | json summary | embedding runtime/model for 50-doc scaled set | secondary runtime provenance | maybe include |
| `archive/embeddings/scaled_100_embedding_report.json` | json summary | same for 100 docs | secondary runtime provenance | maybe include |
| `archive/embeddings/scaled_150_embedding_report.json` | json summary | same for 150 docs | secondary runtime provenance | maybe include |
| `archive/embeddings/scaled_200_embedding_report.json` | json summary | same for 200 docs | secondary runtime provenance | maybe include |
| `archive/runs/run_index.csv` | run index table | timeline of archived runs; runtime filled only for scaling runs | useful audit trail, incomplete as runtime table | maybe include |

## Auxiliary / debug / provenance outputs

| File path | Artifact type | What it shows | Paper importance | Suggested paper use |
|---|---|---|---|---|
| `outputs/reports/metrics_summary.csv` | metrics file | currently only ACOM metrics for the latest scale-200 run | low standalone value; generic filename is misleading and not publishable as-is | probably exclude |
| `outputs/reports/acom_run_summary.json` | run summary | currently scale-200 ACOM-only run summary | provenance only; generic filename is misleading | probably exclude |
| `outputs/reports/acom_cost_history.csv` | metric time series | currently latest generic ACOM run cost history | provenance only | probably exclude |
| `outputs/figures/acom_grid.png` | figure | currently latest generic ACOM grid, corresponding to scale-200 run | useful only if discussing scale-200 qualitative map | maybe include |
| `outputs/figures/metric_comparison.png` | figure | generic metric chart from the latest run; likely ACOM-only or mixed provenance | not paper-ready | probably exclude |
| `outputs/maps/acom_positions.csv` | map csv | current generic ACOM positions, corresponding to latest run | provenance only | probably exclude |
| `outputs/maps/pca_positions.csv` | map csv | older generic PCA positions from a different run | mixed-provenance working artifact | probably exclude |
| `outputs/maps/tsne_positions.csv` | map csv | older generic t-SNE positions from a different run | mixed-provenance working artifact | probably exclude |
| `docs/results/acom_results_interpretation.md` | duplicate report | curated copy of output interpretation | duplicate | probably exclude |
| `docs/results/acom_results_table_pretty.csv` | duplicate table | curated copy of output sweep table | duplicate | probably exclude |
| `docs/results/tuned_acom_metrics_summary.csv` | duplicate table | curated copy of final tuned metrics | duplicate | probably exclude |
| `archive/runs/run_2026-03-11_02-26-49/reports/metrics_summary.csv` | archived metrics file | early undocumented run with better-than-baseline ACOM but older code snapshot | important ambiguity to acknowledge, not final evidence | maybe include |

# 6. Key Metrics Collected

| Metric | Where it appears | Reported in which experiments | Higher / lower better | Central to paper? |
|---|---|---|---|---|
| Neighborhood preservation | `outputs/reports/tuned_acom_metrics_summary.csv`, `outputs/reports/acom_variant_comparison.csv`, `outputs/reports/acom_scaling_results.csv`, per-run `metrics_summary.csv` | early benchmark, variant sweep, final tuned comparison, scaling | higher | yes; this is the strongest local-structure metric in the repo |
| Trustworthiness | same files as neighborhood preservation | early benchmark, variant sweep, final tuned comparison, scaling | higher | yes; paired with neighborhood preservation throughout |
| Stress | same files as neighborhood preservation | early benchmark, variant sweep, final tuned comparison, scaling | lower | yes; main global-distortion metric |
| Cost improvement | `outputs/reports/acom_variant_comparison.csv`, `outputs/reports/acom_results_table*.csv`, per-run `acom_run_summary.json`, `outputs/reports/acom_scaling_results.csv` | variant sweep, scaling, tuned ACOM summary | higher | yes for internal ACOM tuning, secondary for cross-method comparison |
| Initial cost | per-run `acom_run_summary.json`, `outputs/reports/acom_variant_comparison.csv`, `outputs/reports/acom_scaling_results.csv` | variant sweep, scaling, tuned ACOM summary | context only; lower is not directly the goal because starts from random layout | secondary |
| Final cost | per-run `acom_run_summary.json`, `outputs/reports/acom_variant_comparison.csv`, `outputs/reports/acom_scaling_results.csv` | variant sweep, scaling, tuned ACOM summary | lower | important for ACOM internal optimization story |
| Runtime | `archive/embeddings/embedding_report.json`, `outputs/reports/acom_scaling_results.csv`, scale-run `acom_run_summary.json`, partially `archive/runs/run_index.csv` | embedding generation, scaling study | lower | secondary; useful for feasibility/scaling, weak for baseline comparison because PCA/t-SNE/UMAP runtimes are missing |
| Distance correlation | per-run `metrics_summary.csv`, `outputs/reports/tuned_acom_metrics_summary.csv`, `outputs/reports/baseline_reference_metrics.csv` | early benchmark, baseline reference, final tuned comparison | higher | secondary; present, but not emphasized in the narrative |
| Silhouette | per-run `metrics_summary.csv`, `outputs/reports/tuned_acom_metrics_summary.csv`, `outputs/reports/acom_variant_comparison.csv`, `outputs/reports/acom_scaling_results.csv` | benchmark, variant sweep, final tuned comparison, scaling | higher | secondary; useful for class separation, especially to flag degradation at 150/200 docs |
| Accepted swaps | per-run `acom_run_summary.json` | all ACOM runs | ambiguous; more is not automatically better | secondary diagnostic only |
| Total attempts / iterations recorded | per-run `acom_run_summary.json` | all ACOM runs | ambiguous | secondary diagnostic only |

Important quantitative values directly supported by the repo:

- Baseline ACOM: neighborhood preservation `0.134`, trustworthiness `0.567`, stress `5.120`, final cost `275.767`. Source: `outputs/reports/acom_variant_comparison.csv`.
- Best ACOM (`acom_v1_wider_swap_annealed`): neighborhood preservation `0.367`, trustworthiness `0.787`, stress `5.107`, final cost `213.015`, cost improvement `87.815`. Source: `outputs/reports/acom_variant_comparison.csv`.
- Final tuned comparison: PCA `0.329 / 0.758 / 0.612`; t-SNE `0.523 / 0.893 / 13.242`; UMAP `0.505 / 0.897 / 2.850` for neighborhood preservation / trustworthiness / stress. Source: `outputs/reports/tuned_acom_metrics_summary.csv`.
- Scaling study: best trustworthiness at 100 docs (`0.780`), worst at 200 docs (`0.705`); stress rises from `3.989` at 50 docs to `8.180` at 200 docs. Source: `outputs/reports/acom_scaling_results.csv`.

# 7. Best Results Identified

- Best ACOM variant: `acom_v1_wider_swap_annealed`. Evidence: `outputs/reports/acom_variant_comparison.csv`. It is best on all tracked ACOM sweep criteria shown there: final cost `213.015`, cost improvement `87.815`, neighborhood preservation `0.367`, trustworthiness `0.787`, stress `5.107`, silhouette `0.097`.
- Best comparison point vs PCA: tuned ACOM beats PCA on the two local-structure metrics that are most defensible as the paper's main claim, neighborhood preservation (`0.367` vs `0.329`) and trustworthiness (`0.787` vs `0.758`). Evidence: `outputs/reports/tuned_acom_metrics_summary.csv`. Negative side: PCA remains far better on stress (`0.612` vs `5.107`), distance correlation (`0.411` vs `0.330`), and silhouette (`0.243` vs `0.097`).
- Best comparison point vs t-SNE: there is no quantitative win for ACOM in the final comparison. t-SNE is better on neighborhood preservation (`0.523`), trustworthiness (`0.893`), distance correlation (`0.545`), and silhouette (`0.406`). ACOM’s only defensible advantage is solving a discrete layout problem rather than a continuous one.
- Best comparison point vs UMAP: there is also no quantitative win for ACOM. UMAP combines stronger trustworthiness (`0.897`), strong neighborhood preservation (`0.505`), much lower stress (`2.850`), and the highest silhouette (`0.534`). Again, ACOM’s distinct value is discreteness, not raw metric dominance.
- Most convincing scaling result: the tuned ACOM variant remains operational and improves cost at every tested size from 50 to 200 documents, with a quality peak around 100 documents (`neighborhood_preservation=0.351`, `trustworthiness=0.780`) before declining at 150 and 200. Evidence: `outputs/reports/acom_scaling_results.csv`, `outputs/figures/acom_scaling_neighborhood.png`, `outputs/figures/acom_scaling_trustworthiness.png`.
- Most convincing tuning result: wider swap search is the first strong gain (`0.134 -> 0.239` neighborhood preservation; `0.567 -> 0.683` trustworthiness), and annealed acceptance produces the final jump (`0.239 -> 0.367`; `0.683 -> 0.787`). Evidence: `outputs/reports/acom_variant_comparison.csv`.
- Notable negative result: stronger repulsion and radius-2 neighborhood do not improve the core external metrics meaningfully despite larger internal cost magnitudes. Evidence: `outputs/reports/acom_variant_comparison.csv`.
- Notable limitation: scaled 150-doc and 200-doc runs have negative silhouette (`-0.009`, `-0.068`), indicating worsening class separation at larger scales. Evidence: `outputs/reports/acom_scaling_results.csv`.
- Important disagreement to note: the 100-doc scaling run for the same tuned variant reports slightly worse values (`0.351`, `0.780`) than the final tuned benchmark (`0.367`, `0.787`). Evidence: `outputs/reports/acom_scaling_results.csv` vs `outputs/reports/tuned_acom_metrics_summary.csv`. Likely explanation: separate regenerated `scaled_100` dataset/embedding artifacts, but the repo does not state this explicitly enough.

# 8. Candidate Figures for the Conference Paper

| File path | Tentative figure title | Message supported | Why it deserves space | 2x2 panel candidate? |
|---|---|---|---|---|
| `outputs/figures/tuned_acom_grid.png` | Tuned ACOM discrete semantic grid | shows the core output type and why the method is different | strongest qualitative evidence for the paper’s central idea | yes |
| `outputs/figures/tuned_pca_scatter.png` | PCA baseline on the benchmark embeddings | shows linear baseline structure | useful in a layout comparison panel | yes |
| `outputs/figures/tuned_tsne_scatter.png` | t-SNE baseline on the benchmark embeddings | shows strongest local separation among baselines | strong panel comparator | yes |
| `outputs/figures/tuned_umap_scatter.png` | UMAP baseline on the benchmark embeddings | shows nonlinear baseline with strong separation and moderate readability | strong panel comparator | yes |
| `outputs/figures/acom_variant_comparison.png` | ACOM variant sweep: cost and local-structure gains | supports the tuning/ablation story | best single figure for defending the proposed ACOM design choices | yes |
| `outputs/figures/acom_v1_baseline_cost_history.png` | Baseline ACOM optimization plateaus early | supports the local-trapping diagnosis | pairs well with tuned cost history | yes |
| `outputs/figures/tuned_acom_cost_history.png` | Annealed wider-search ACOM continues descending | supports the optimizer improvement story | pairs well with baseline cost history | yes |
| `outputs/figures/acom_scaling_neighborhood.png` | Neighborhood preservation degrades past 100 documents | supports the main scaling limitation | strongest single scaling-quality figure | yes |
| `outputs/figures/acom_scaling_trustworthiness.png` | Trustworthiness peaks near 100 documents then declines | reinforces the scaling-quality story | strong companion to scaling neighborhood | yes |
| `outputs/figures/acom_scaling_runtime.png` | Runtime growth with dataset size | supports computational feasibility and cost discussion | useful if runtime is discussed explicitly | yes |
| `outputs/figures/acom_scaling_stress.png` | Distortion rises with dataset size | supports the discrete-distortion limitation | useful if space allows | yes |
| `outputs/figures/tuned_acom_metric_comparison.png` | Final metrics across ACOM/PCA/t-SNE/UMAP | supports overall method comparison | useful as a drafting reference, but should be replaced with a cleaner paper figure or table | maybe |

## Suggested 2x2 Figure Groupings

1. Qualitative method comparison:
   - `outputs/figures/tuned_acom_grid.png`
   - `outputs/figures/tuned_pca_scatter.png`
   - `outputs/figures/tuned_tsne_scatter.png`
   - `outputs/figures/tuned_umap_scatter.png`

2. ACOM optimization story:
   - `outputs/figures/acom_variant_comparison.png`
   - `outputs/figures/acom_v1_baseline_cost_history.png`
   - `outputs/figures/tuned_acom_cost_history.png`
   - `outputs/figures/acom_scaling_neighborhood.png`

3. Scaling panel:
   - `outputs/figures/acom_scaling_runtime.png`
   - `outputs/figures/acom_scaling_neighborhood.png`
   - `outputs/figures/acom_scaling_trustworthiness.png`
   - `outputs/figures/acom_scaling_stress.png`

# 9. Candidate Tables for the Conference Paper

| Source file / path | Title suggestion | Message supported | Essential or optional | Can it be compressed? |
|---|---|---|---|---|
| `outputs/reports/tuned_acom_metrics_summary.csv` | Final benchmark comparison: tuned ACOM vs PCA/t-SNE/UMAP | main quantitative comparison and the clearest support for the headline claim | essential | yes; likely 4 methods x 3 core metrics, with secondary metrics in appendix |
| `outputs/reports/acom_results_table_pretty.csv` | ACOM variant ablation / tuning results | explains why the reported ACOM configuration was selected | essential | yes; keep only 4-5 most informative variants if space is tight |
| `outputs/reports/acom_scaling_results.csv` | Scaling behavior of tuned ACOM from 50 to 200 documents | shows stability plus degradation with scale | essential | yes; likely keep size, runtime, neighborhood preservation, trustworthiness, stress |
| `data/processed/dataset_report.json` | Benchmark dataset composition | documents per split/category and text statistics | optional | yes; compress into a small Methods table |
| `archive/embeddings/embedding_report.json` | Embedding configuration and runtime | embedding backend, dimension, runtime | optional | yes; better as a Methods footnote or short setup table |

# 10. Likely Core Story of the Paper

- Document mapping to a discrete semantic grid is a different problem from standard continuous dimensionality reduction and deserves its own evaluation.
- A simple ACOM-style swap optimizer provides a reproducible baseline for this discrete placement problem.
- The initial greedy baseline works in the narrow sense of reducing cost, but it preserves local semantic structure poorly.
- Controlled tuning shows that search strategy matters more than simply changing local penalties: wider candidate search helps substantially, and annealed acceptance helps even more.
- The tuned ACOM variant `acom_v1_wider_swap_annealed` is the first version that is clearly competitive on local-structure metrics within the repo’s evidence.
- On the 100-document benchmark, tuned ACOM exceeds PCA on neighborhood preservation and trustworthiness while remaining a discrete layout.
- Continuous nonlinear baselines still win the overall metric contest: t-SNE has the highest neighborhood preservation, UMAP has the highest trustworthiness and silhouette, and PCA has the lowest stress.
- The main value proposition for ACOM is therefore not metric dominance, but a discrete, interpretable semantic map with respectable preservation quality.
- Scaling results show that the tuned optimizer remains stable up to 200 documents, but quality peaks around 100 documents and then degrades.
- The clearest limitation is that discrete layout constraints and current search efficiency are not yet strong enough to match t-SNE/UMAP on local structure or PCA on global geometry.

# 11. What Seems Strong Enough to Include

Strongest and most paper-worthy:

- The ACOM sweep result that wider search plus annealing raises neighborhood preservation from `0.134` to `0.367` and trustworthiness from `0.567` to `0.787`.
- The final comparison result that tuned ACOM beats PCA on neighborhood preservation (`0.367` vs `0.329`) and trustworthiness (`0.787` vs `0.758`).
- The qualitative 4-panel comparison of tuned ACOM vs PCA/t-SNE/UMAP layouts.
- The scaling result that ACOM stays operational and improves cost at all tested sizes but quality drops after 100 documents.

Useful but secondary:

- Baseline vs tuned cost-history comparison showing reduced plateauing.
- UMAP’s role as the strongest overall continuous baseline in this repo (`trustworthiness=0.897`, `stress=2.850`, `silhouette=0.534`).
- The negative / mixed ACOM ablations (`stronger_repulsion`, `radius2`) as evidence that not every complexity increase helps.
- Embedding setup/runtime details (`all-MiniLM-L6-v2`, 384D, `5.7658 s` for 100 docs).

Probably not worth including unless space allows:

- Distance-correlation scatter plots for every method.
- Generic `outputs/` artifacts with overloaded names such as `metric_comparison.png` and `acom_positions.csv`.
- Duplicate copies under `docs/figures/` and `docs/results/`.
- The ambiguous early run `run_2026-03-11_02-26-49`, except possibly in a limitations or reproducibility note.

# 12. Missing or Unclear Information

- The final tuned ACOM vs PCA/t-SNE/UMAP comparison is not archived as a timestamped run bundle; it lives only in `outputs/`.
- `outputs/` contains mixed-provenance artifacts. For example, `outputs/reports/acom_run_summary.json` and `outputs/reports/metrics_summary.csv` currently correspond to the scale-200 ACOM-only run, while `outputs/maps/pca_positions.csv` and `outputs/maps/tsne_positions.csv` are older benchmark outputs.
- The generic filenames in `outputs/` make it easy to misread which run a figure belongs to.
- Variant-sweep runtime is effectively missing. `run_index.csv` has blank runtime values for non-scaling runs, and `acom_variant_comparison.csv` does not include runtime even though the current code suggests it should.
- The early archived run `run_2026-03-11_02-26-49` is not directly comparable to later named variants: its config snapshot lacks fields such as attraction weight, swap breadth, and acceptance rule, yet its ACOM metrics are better than the later named baseline.
- No repeated-seed experiments or variance estimates are present; all reported results appear single-seed and deterministic.
- No statistical significance analysis is present.
- Runtime comparisons for PCA, t-SNE, and UMAP are absent.
- The final metric comparison figure mixes metrics with different directions and very different scales; it is informative for inspection but weak as a final paper figure.
- The 100-document scaling run and the 100-document final tuned benchmark disagree slightly; the repo likely uses a separately regenerated `scaled_100` dataset, but this should be stated explicitly.
- `outputs/reports/acom_scaling_discussion.md` appears useful, but it is not produced by the scaling script and is not mirrored in the archived study manifest.
- UMAP is present in final tuned outputs, but not in archived run bundles, because the benchmark comparison runs were generally executed without `--enable-umap`.

# 13. Recommended Next Step for Paper Preparation

- Review first: `outputs/reports/tuned_acom_metrics_summary.csv`, `outputs/reports/acom_results_table_pretty.csv`, and `outputs/reports/acom_scaling_results.csv`. These are the three files most likely to survive directly into a paper draft.
- Turn into final figures/tables next: one headline comparison table, one ACOM ablation table, one scaling table, and a compact 2x2 qualitative panel. Those are the highest-yield paper assets in this repo.
- Verify before writing: which run is the benchmark source of truth, whether the tuned comparison should be archived explicitly, whether the `scaled_100` run is meant to be directly compared with the benchmark 100-doc run, and whether missing runtime data for the variant sweep needs to be regenerated.

# Appendix: One-Page Paper Asset Checklist

- Final candidate figures:
  - `outputs/figures/tuned_acom_grid.png`
  - `outputs/figures/tuned_pca_scatter.png`
  - `outputs/figures/tuned_tsne_scatter.png`
  - `outputs/figures/tuned_umap_scatter.png`
  - `outputs/figures/acom_variant_comparison.png`
  - `outputs/figures/acom_v1_baseline_cost_history.png`
  - `outputs/figures/tuned_acom_cost_history.png`
  - `outputs/figures/acom_scaling_neighborhood.png`
  - `outputs/figures/acom_scaling_trustworthiness.png`
  - `outputs/figures/acom_scaling_runtime.png`
- Final candidate tables:
  - `outputs/reports/tuned_acom_metrics_summary.csv`
  - `outputs/reports/acom_results_table_pretty.csv`
  - `outputs/reports/acom_scaling_results.csv`
  - optional setup table derived from `data/processed/dataset_report.json` and `archive/embeddings/embedding_report.json`
- Best metrics to report:
  - neighborhood preservation
  - trustworthiness
  - stress
  - cost improvement
  - runtime for scaling
  - silhouette as secondary evidence
- Strongest comparisons:
  - tuned ACOM vs baseline ACOM
  - tuned ACOM vs PCA
  - tuned ACOM scaling at 50/100/150/200 docs
  - t-SNE and UMAP as honest upper baselines for local structure
- Biggest limitation to acknowledge:
  - tuned ACOM is competitive with PCA on local structure but still clearly behind t-SNE and UMAP, and its quality degrades beyond about 100 documents in the current study.
