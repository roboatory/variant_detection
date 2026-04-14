# DIP-SV-FILTER

A post-calling structural variant (SV) filtering tool for long-read sequencing data. DIP-SV-FILTER reduces false positives in SV callsets by jointly modeling clustered SV candidates as alternative diploid sequence hypotheses, using a time-distributed CNN-Transformer classifier to score residual alignment signal after local realignment.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync            # install dependencies from lockfile
uv run pre-commit install   # set up linting hooks
```

## Usage

### Feature extraction

Extract MAMNET-style alignment features from a BAM file into per-window `.npy` matrices (shape `2000 x 9`):

```bash
uv run python src/featurizers/extract_mamnet_features.py \
  --bam data/HG002_chr21.bam \
  --contig chr21 \
  --start 1000000 \
  --end 1002000
```

### Training

Train the CNN-Transformer classifier on pre-extracted feature matrices:

```bash
uv run python src/models/train.py \
  --train_directory data/features/training/matrices \
  --validation_directory data/features/validation/matrices \
  --test_directory data/features/test/matrices \
  --output_directory output
```

Training logs to [Weights & Biases](https://wandb.ai) by default (`--wandb_mode disabled` to turn off).

### Inference

Run a trained model on new feature matrices:

```bash
uv run python src/models/inference.py \
  --checkpoint_file_path output/best_model.pt \
  --split_directory data/features/test/matrices \
  --output_file_path output/predictions.tsv
```

## Project structure

```
src/
  featurizers/
    extract_mamnet_features.py      # BAM -> (2000, 9) .npy feature matrices
    extract_cutesv_indels.py        # intra/inter-alignment SV signature extraction
    parse_sample_specific_strings.py # sample-specific string analysis
  models/
    architecture.py                 # CNN-Transformer model definition
    train.py                        # training loop, metrics, checkpointing
    inference.py                    # batch inference from checkpoint
tests/
data/
  features/{training,validation,test}/
    labels.txt                      # per-split label file
    matrices/                       # .npy feature matrices
```

## Linting

```bash
uv run ruff check .
uv run ruff format .
```

Pre-commit hooks run ruff lint, ruff format, and nbstripout automatically.
