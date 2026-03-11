# Project Brief

This repository implements a research pipeline for document mapping:

`documents -> embeddings -> ACOM mapping to discrete 2D space -> evaluation -> comparison with PCA / t-SNE / UMAP`

The expected output of the project is not only source code. The repository must produce:

1. A working Python implementation of ACOM.
2. A discrete 2D semantic layout of documents on a grid.
3. Quantitative metrics that evaluate mapping quality.
4. A comparison between ACOM and standard dimensionality reduction baselines.

# Repository Structure

The project should follow this structure:

```text
llm_document_acom/
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
├── archive/
│   ├── extracted_text/
│   ├── embeddings/
│   ├── mappings/
│   └── metrics/
├── outputs/
│   ├── figures/
│   ├── maps/
│   └── reports/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── embedding_loader.py
│   ├── distance_utils.py
│   ├── grid.py
│   ├── acom.py
│   ├── baselines.py
│   ├── metrics.py
│   ├── visualization.py
│   └── run_experiment.py
├── requirements.txt
└── README.md
```

# Module Responsibilities

## `src/config.py`

Centralize experiment settings such as:

- input paths for documents and embeddings
- grid shape
- ACOM hyperparameters
- metric settings
- random seed
- output locations

Keep experiment parameters here rather than hardcoding values throughout the codebase.

## `src/data_loader.py`

Load document data and metadata from CSV or JSON. At minimum, each document should expose:

- `id`
- `text`
- optional `label`

The loader should validate required fields and return a clean tabular structure for downstream steps.

## `src/embedding_loader.py`

Load archived embeddings from disk and validate shape consistency against the document collection.

If embeddings do not exist in a fresh workspace, the pipeline may bootstrap a sample embedding set for demonstration, but the main interface should still support loading saved embeddings from disk.

## `src/distance_utils.py`

Compute semantic relationships from embeddings, including:

- cosine distance matrix
- euclidean distance matrix
- similarity matrix

This semantic distance matrix is the reference structure ACOM should preserve.

## `src/grid.py`

Implement the discrete 2D space used by ACOM.

Responsibilities:

- initialize an empty grid
- place documents on cells
- track document-to-position mappings
- swap document positions
- return neighbors
- compute grid-space distances

This module should stay independent and reusable because several parts of the pipeline depend on it.

## `src/acom.py`

Implement the core ACOM algorithm.

The initial version should be a simple stochastic swap optimizer:

1. start from a random placement
2. select candidate swaps
3. evaluate whether a swap improves the arrangement
4. keep beneficial swaps and revert harmful ones
5. repeat over many iterations

Version 1 should prioritize correctness, clarity, and measurable improvement over sophistication.

## `src/baselines.py`

Provide comparison methods:

- PCA
- t-SNE
- UMAP

These baselines output continuous 2D coordinates and are evaluated with the same preservation-oriented metrics where appropriate.

## `src/metrics.py`

Implement the evaluation layer. Core metrics should include:

- neighborhood preservation
- trustworthiness
- stress / distortion

These metrics are necessary to justify the quality of the ACOM mapping scientifically.

## `src/visualization.py`

Generate figures for analysis and reporting, including:

- ACOM grid visualization
- baseline scatter plots
- metric comparison chart
- semantic-distance vs mapped-distance scatter

## `src/run_experiment.py`

Run the pipeline end to end:

1. load documents
2. load or create embeddings
3. compute semantic distances
4. initialize the grid
5. run ACOM
6. run PCA, t-SNE, and UMAP
7. compute metrics
8. save maps, metrics, and reports
9. generate figures
10. print a concise summary

# Implementation Order

Build the project in the following order:

1. Data and embeddings
   Create `config.py`, `data_loader.py`, and `embedding_loader.py`.

2. Semantic distance layer
   Create `distance_utils.py` and verify the pairwise distance matrix.

3. Grid system
   Create `grid.py` and verify placement, lookup, swapping, and neighbor retrieval.

4. ACOM Version 1
   Create `acom.py` as a swap-based stochastic optimizer and verify that mapping cost decreases over time.

5. Baselines
   Create `baselines.py` for PCA, t-SNE, and UMAP projections.

6. Metrics
   Create `metrics.py` to evaluate preservation quality.

7. Visualizations
   Create `visualization.py` for figures suitable for reports or presentations.

8. Final pipeline
   Create `run_experiment.py` so the whole experiment runs with one command.

# Practical Standard

This repository should behave like a research artifact, not just a code demo. That means:

- deterministic configuration when possible
- archived outputs
- clear separation between inputs, mappings, metrics, and plots
- reproducible experiment settings
- documentation that makes the workflow easy to explain in an academic context
