# LLM Document ACOM

This repository contains a research pipeline for mapping document embeddings into a discrete two-dimensional layout with ACOM and comparing that layout against standard continuous dimensionality reduction methods. The current implementation targets a balanced five-class subset of the 20 Newsgroups corpus and is structured to support repeatable experiments, quantitative evaluation, and archived run tracking.

## Motivation

The project studies whether a discrete semantic map can preserve meaningful document neighborhoods while remaining easy to inspect visually. ACOM is used as the discrete mapping method, while PCA, t-SNE, and optional UMAP provide continuous baselines for comparison against the same embedding space.

## Pipeline

```text
20 Newsgroups subset
-> light text cleaning
-> embedding generation
-> semantic distance matrix
-> ACOM / PCA / t-SNE / optional UMAP
-> evaluation metrics
-> archived experiment run
```

## Methods

- `ACOM`: discrete grid-based mapping
- `PCA`: linear 2D baseline
- `t-SNE`: nonlinear neighborhood-preserving baseline
- `UMAP`: optional nonlinear baseline with graceful failure handling

## Evaluation

The shared evaluation pipeline compares mappings against the original embedding space using:

- neighborhood preservation
- trustworthiness
- stress
- silhouette score by category when labels are available

## Repository Overview

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── embeddings/
├── archive/
│   ├── runs/
│   ├── embeddings/
│   ├── mappings/
│   └── metrics/
├── outputs/
├── src/
├── requirements.txt
├── README.md
└── AGENTS.md
```

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Download and prepare the balanced 20 Newsgroups subset:

```bash
python3 src/prepare_20newsgroups.py
```

This step:

- uses `subset='train'` and `subset='test'`
- removes headers, footers, and quotes
- keeps these categories only:
  - `comp.graphics`
  - `rec.sport.baseball`
  - `sci.med`
  - `sci.space`
  - `talk.politics.misc`
- creates balanced train and test splits with 10 documents per category

## Embedding Generation

Generate embeddings from the prepared JSONL inputs:

```bash
python3 src/generate_embeddings.py
```

Embedding backends are configurable:

- preferred local backend: `sentence-transformers`
- fallback backend: TF-IDF via scikit-learn

The script writes aligned embedding arrays and metadata tables under `data/embeddings/`.

## Running Experiments

Run the mapping and evaluation pipeline:

```bash
python3 src/run_experiment.py
```

Enable UMAP explicitly if the local environment supports it:

```bash
python3 src/run_experiment.py --enable-umap
```

The experiment runner:

- validates metadata and embedding alignment
- runs ACOM on the discrete grid
- runs PCA and t-SNE on the same embedding matrix
- runs UMAP optionally
- computes shared evaluation metrics
- saves latest outputs under `outputs/`
- archives the full run under `archive/runs/`

## Outputs and Run Archiving

The repository keeps two output layers:

- `outputs/`: latest run artifacts for quick inspection
- `archive/runs/`: timestamped experiment records for comparison across runs

Each archived run contains:

- `maps/`
- `figures/`
- `reports/`
- `config/config_snapshot.json`
- `run_manifest.json`

The run index is stored in:

- `archive/runs/run_index.csv`

This index records run IDs, timestamps, completed methods, document counts, ACOM cost before and after optimization, and the path to each archived run.

## Current Status

The repository currently includes:

- a complete data-preparation pipeline for the selected 20 Newsgroups subset
- configurable embedding generation
- a strengthened Version 1 ACOM implementation
- PCA, t-SNE, and optional UMAP baselines
- shared metric computation and figure generation
- timestamped experiment archiving for comparative tracking

## Example Commands

```bash
python3 src/prepare_20newsgroups.py
python3 src/generate_embeddings.py
python3 src/run_experiment.py
python3 src/run_experiment.py --enable-umap
```
