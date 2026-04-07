"""Fixed data-loading and evaluation harness for the autoresearch loop.

This file is READ-ONLY for the AI agent. It defines:
- Constants (input shape, label length, data paths)
- Dataset and DataLoader construction
- Metrics accumulation and evaluation
- The structured summary printer

The agent edits train.py only; prepare.py is the stable contract.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_INPUT_SHAPE = (2000, 9)
EXPECTED_LABEL_LENGTH = 10
INPUT_LENGTH = 2000
NUM_FEATURES = 9
NUM_SUBWINDOWS = 10
SUBWINDOW_SIZE = 200

# Default data directories — override with CLI args if needed
DEFAULT_TRAIN_DIRECTORY = Path("data/features/training/matrices")
DEFAULT_VALIDATION_DIRECTORY = Path("data/features/validation/matrices")
DEFAULT_TEST_DIRECTORY = Path("data/features/test/matrices")

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def get_default_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def parse_label_vector(label_text: str) -> Tensor:
    label_parts = [label_part.strip() for label_part in label_text.split(",")]
    if len(label_parts) != EXPECTED_LABEL_LENGTH:
        raise ValueError(
            f"Expected {EXPECTED_LABEL_LENGTH} labels per example, got {len(label_parts)}"
        )

    values: list[float] = []
    for label_part in label_parts:
        if label_part not in {"0", "1"}:
            raise ValueError(f"Labels must be binary 0/1 values, got {label_part!r}")
        values.append(float(label_part))
    return torch.tensor(values, dtype=torch.float32)


def load_labels(labels_file_path: Path) -> dict[str, Tensor]:
    if not labels_file_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file_path}")

    labels: dict[str, Tensor] = {}
    with labels_file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            line_parts = line.split("\t")
            if len(line_parts) != 2:
                raise ValueError(
                    f"Malformed labels.txt line {line_number}: expected <file>\\t<v0,...,v9>"
                )
            file_name, label_text = line_parts
            file_basename = Path(file_name).name
            label_vector = parse_label_vector(label_text)
            if file_basename in labels:
                if not torch.equal(labels[file_basename], label_vector):
                    raise ValueError(f"Conflicting label entry for {file_basename}")
                continue
            labels[file_basename] = label_vector

    if not labels:
        raise ValueError("No labels were loaded from labels.txt")
    return labels


def resolve_labels_file_path(split_directory: Path) -> Path:
    candidate_paths = [
        split_directory / "labels.txt",
        split_directory.parent / "labels.txt",
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        f"Could not find labels.txt for split directory {split_directory}"
    )


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------


class SVWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        split_directory: Path,
        labels: dict[str, Tensor],
        max_samples: int | None = None,
    ) -> None:
        if not split_directory.exists():
            raise FileNotFoundError(f"Split directory not found: {split_directory}")
        if not split_directory.is_dir():
            raise NotADirectoryError(f"Expected a directory: {split_directory}")

        files = sorted(split_directory.glob("*.npy"))
        if not files:
            raise ValueError(
                f"No .npy files found in split directory: {split_directory}"
            )
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be a positive integer")
            files = files[:max_samples]

        missing_labels = [path.name for path in files if path.name not in labels]
        if missing_labels:
            preview = ", ".join(missing_labels[:5])
            raise ValueError(
                f"Missing labels for {len(missing_labels)} files in {split_directory}: {preview}"
            )

        self.samples = [path for path in files if path.name in labels]
        self.labels = labels

        for sample_path in self.samples:
            array = np.load(sample_path, mmap_mode="r")
            if array.shape != EXPECTED_INPUT_SHAPE:
                raise ValueError(
                    f"{sample_path} has shape {array.shape}, expected {EXPECTED_INPUT_SHAPE}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample_path = self.samples[index]
        features = np.load(sample_path).astype(np.float32, copy=False)
        labels = self.labels[sample_path.name]
        return torch.from_numpy(features), labels.clone()


def create_dataloader(
    split_directory: Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_samples: int | None = None,
) -> DataLoader[tuple[Tensor, Tensor]]:
    resolved_labels_file_path = resolve_labels_file_path(
        split_directory=split_directory,
    )
    labels = load_labels(resolved_labels_file_path)
    dataset = SVWindowDataset(
        split_directory=split_directory,
        labels=labels,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass
class EpochMetrics:
    loss: float
    elementwise_precision: float
    elementwise_recall: float
    elementwise_f1: float
    elementwise_accuracy: float
    exact_match_accuracy: float
    any_sv_precision: float
    any_sv_recall: float
    any_sv_f1: float


class MetricsAccumulator:
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.sample_count = 0
        self.elementwise_true_positives = 0
        self.elementwise_false_positives = 0
        self.elementwise_false_negatives = 0
        self.elementwise_correct = 0
        self.elementwise_total = 0
        self.exact_match_count = 0
        self.any_sv_true_positives = 0
        self.any_sv_false_positives = 0
        self.any_sv_false_negatives = 0

    def update(self, loss: float, logits: Tensor, labels: Tensor) -> None:
        batch_size = labels.shape[0]
        self.loss_sum += loss * batch_size
        self.sample_count += batch_size

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).to(dtype=torch.int64)
        targets = labels.to(dtype=torch.int64)

        self.elementwise_true_positives += int(
            ((predictions == 1) & (targets == 1)).sum().item()
        )
        self.elementwise_false_positives += int(
            ((predictions == 1) & (targets == 0)).sum().item()
        )
        self.elementwise_false_negatives += int(
            ((predictions == 0) & (targets == 1)).sum().item()
        )
        self.elementwise_correct += int((predictions == targets).sum().item())
        self.elementwise_total += int(targets.numel())
        self.exact_match_count += int((predictions == targets).all(dim=1).sum().item())

        any_predictions = predictions.any(dim=1).to(dtype=torch.int64)
        any_targets = targets.any(dim=1).to(dtype=torch.int64)
        self.any_sv_true_positives += int(
            ((any_predictions == 1) & (any_targets == 1)).sum().item()
        )
        self.any_sv_false_positives += int(
            ((any_predictions == 1) & (any_targets == 0)).sum().item()
        )
        self.any_sv_false_negatives += int(
            ((any_predictions == 0) & (any_targets == 1)).sum().item()
        )

    def compute(self) -> EpochMetrics:
        elementwise_precision = safe_divide(
            self.elementwise_true_positives,
            self.elementwise_true_positives + self.elementwise_false_positives,
        )
        elementwise_recall = safe_divide(
            self.elementwise_true_positives,
            self.elementwise_true_positives + self.elementwise_false_negatives,
        )
        elementwise_f1 = safe_divide(
            2 * elementwise_precision * elementwise_recall,
            elementwise_precision + elementwise_recall,
        )
        any_sv_precision = safe_divide(
            self.any_sv_true_positives,
            self.any_sv_true_positives + self.any_sv_false_positives,
        )
        any_sv_recall = safe_divide(
            self.any_sv_true_positives,
            self.any_sv_true_positives + self.any_sv_false_negatives,
        )
        any_sv_f1 = safe_divide(
            2 * any_sv_precision * any_sv_recall,
            any_sv_precision + any_sv_recall,
        )

        return EpochMetrics(
            loss=safe_divide(self.loss_sum, self.sample_count),
            elementwise_precision=elementwise_precision,
            elementwise_recall=elementwise_recall,
            elementwise_f1=elementwise_f1,
            elementwise_accuracy=safe_divide(
                self.elementwise_correct, self.elementwise_total
            ),
            exact_match_accuracy=safe_divide(self.exact_match_count, self.sample_count),
            any_sv_precision=any_sv_precision,
            any_sv_recall=any_sv_recall,
            any_sv_f1=any_sv_f1,
        )


# ---------------------------------------------------------------------------
# Evaluation helper (called from train.py)
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    accumulator = MetricsAccumulator()
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            labels = labels.to(device=device, dtype=torch.float32, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, labels)
            accumulator.update(loss.item(), logits.detach(), labels.detach())
    return accumulator.compute()


# ---------------------------------------------------------------------------
# Structured summary printer (grep-friendly output)
# ---------------------------------------------------------------------------


def print_summary(
    val_metrics: EpochMetrics,
    training_seconds: float,
    total_seconds: float,
    num_epochs: int,
    num_params: int,
) -> None:
    print("---")
    print(f"val_elementwise_f1:     {val_metrics.elementwise_f1:.6f}")
    print(f"val_any_sv_f1:          {val_metrics.any_sv_f1:.6f}")
    print(f"val_exact_match:        {val_metrics.exact_match_accuracy:.6f}")
    print(f"val_loss:               {val_metrics.loss:.6f}")
    print(f"training_seconds:       {training_seconds:.1f}")
    print(f"total_seconds:          {total_seconds:.1f}")
    print(f"num_epochs:             {num_epochs}")
    print(f"num_params:             {num_params}")
