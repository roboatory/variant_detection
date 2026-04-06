# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A structural variant (SV) detection pipeline inspired by SVHUNTER. It extracts alignment-derived features from BAM/VCF files using MAMNET, encodes them as 2D numeric matrices, and classifies genomic windows using CNNs & transformers.

## Commands

```bash
uv sync                  # install all dependencies from lockfile
uv run <script>          # run any script in the project environment

# featurization (example; start/end are 0-based; end is exclusive)
uv run python src/featurizers/extract_mamnet_features.py --bam data/HG002_chr21.bam --contig chr21 --start 1000000 --end 1002000

# training (labels.txt is resolved per split: <split>/labels.txt)
uv run python src/models/train.py \
  --train_directory data/features/training/matrices \
  --validation_directory data/features/validation/matrices \
  --test_directory data/features/test/matrices \
  --output_directory output

# inference
uv run python src/models/inference.py \
  --checkpoint_file_path output/best_model.pt \
  --split_directory data/features/test/matrices \
  --output_file_path output/predictions.tsv

# linting
uv run ruff check .
uv run ruff format .
```

Pre-commit hooks run ruff lint + format on Python and Jupyter files.

## Architecture

### Pipeline stages

1. **Feature extraction** (`src/featurizers/`) — reads BAM+VCF, produces per-window `.npy` matrices
2. **Model training** (`src/models/train.py`) — trains the SVHUNTER model, logs to W&B
3. **Inference** (`src/models/inference.py`) — loads a checkpoint, outputs a TSV of predictions

### Feature matrix contract

Each example is a `.npy` file of shape `(2000, 9)` representing a 2 KB genomic window with 9 MAMNET-style feature channels (mismatch count, deletion count, soft/hard count, insertion count/mean/max, deletion mean/max, depth). Column order matters — changing it breaks model semantics.

### Featurizers

- `extract_mamnet_features.py` — produces the `(2000, 9)` matrices consumed by the model; divides the window into 200 bp sliding subwindows
- `extract_cutesv_indels.py` — extracts intra/inter-alignment SV signatures from CIGAR and split reads; outputs BED files, visualizations, and encoded matrices
- `parse_sample_specific_strings.py` — analyzes sample-specific string (SFS) signatures overlapping variants

### Model (`src/models/architecture.py`)

SVHUNTER: splits a `(batch, 2000, 9)` input into 10 subwindows of 200 bp, encodes each with a shared CNN, projects to an embedding space with positional embeddings, runs through transformer blocks, and produces 10 binary logits (one per subwindow). Trained with `BCEWithLogitsLoss`.

### Label format

`labels.txt` is a TSV: `<npy_basename>\t<v0,v1,...,v9>` where each `vi` is 0 or 1 indicating SV presence in the corresponding 200 bp subwindow.

### Data layout

```
data/
  HG002_chr21.bam / .bai          # alignment data
  variants/                        # VCF files (train/validate/test splits)
  features/{training,validation,test}/
    labels.txt                     # label file
    matrices/                      # .npy feature matrices
```
