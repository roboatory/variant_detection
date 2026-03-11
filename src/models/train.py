from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pysam
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from .architecture import ModelConfig, SVHunterLite, load_checkpoint, save_checkpoint
except ImportError:
    from architecture import ModelConfig, SVHunterLite, load_checkpoint, save_checkpoint


WINDOW_FILE_RE = re.compile(r"^(.+?)_(\d+)_(\d+)\.npy$")


@dataclass(frozen=True)
class WindowRecord:
    contig: str
    start: int
    end: int
    path: str


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    contig: str
    start: int
    end: int
    paths: tuple[str, ...]
    label: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_windows(features_dir: str, window_size: int) -> list[WindowRecord]:
    windows: list[WindowRecord] = []
    for name in os.listdir(features_dir):
        if not name.endswith(".npy"):
            continue
        match = WINDOW_FILE_RE.match(name)
        if match is None:
            continue
        contig, start_str, end_str = match.groups()
        start = int(start_str)
        end = int(end_str)
        if (end - start) != window_size:
            continue
        windows.append(
            WindowRecord(
                contig=contig,
                start=start,
                end=end,
                path=os.path.join(features_dir, name),
            )
        )
    windows.sort(key=lambda w: (w.contig, w.start, w.end))
    return windows


def load_truth_bed(path: str) -> dict[str, list[tuple[int, int]]]:
    truth: dict[str, list[tuple[int, int]]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            contig = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            if end <= start:
                end = start + 1
            truth.setdefault(contig, []).append((start, end))
    for contig in truth:
        truth[contig].sort()
    return truth


def load_truth_vcf(path: str) -> dict[str, list[tuple[int, int]]]:
    truth: dict[str, list[tuple[int, int]]] = {}
    vcf = pysam.VariantFile(path)
    for rec in vcf.fetch():
        contig = rec.contig
        start = int(rec.start)
        stop = int(rec.stop) if rec.stop is not None else (start + 1)
        if stop <= start:
            stop = start + 1
        truth.setdefault(contig, []).append((start, stop))
    vcf.close()
    for contig in truth:
        truth[contig].sort()
    return truth


def load_truth_intervals(
    truth_bed: str | None,
    truth_vcf: str | None,
) -> dict[str, list[tuple[int, int]]]:
    if truth_bed and truth_vcf:
        raise ValueError("Provide only one of --truth-bed or --truth-vcf.")
    if truth_bed:
        return load_truth_bed(truth_bed)
    if truth_vcf:
        return load_truth_vcf(truth_vcf)
    return {}


def load_labels_txt(path: str) -> dict[str, dict[str, Any]]:
    """
    Parse label file lines:
    ./training_data/chr1_930135_932135.npy<TAB>0,0,0,0,0,0,0,1,1,1
    """
    labels: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            file_part, label_part = parts[0], parts[1]
            basename = os.path.basename(file_part)
            try:
                vector = [int(x) for x in label_part.split(",") if x != ""]
            except ValueError:
                continue
            labels[basename] = {
                "vector": vector,
                "binary": int(any(v > 0 for v in vector)),
            }
    return labels


def overlaps_truth(
    contig: str,
    start: int,
    end: int,
    truth: dict[str, list[tuple[int, int]]],
) -> int:
    intervals = truth.get(contig)
    if not intervals:
        return 0
    for t_start, t_end in intervals:
        if t_start >= end:
            break
        if t_end > start and t_start < end:
            return 1
    return 0


def block_is_valid(
    block: list[WindowRecord],
    expected_step: int | None,
) -> bool:
    if len(block) <= 1:
        return True
    for i in range(len(block) - 1):
        if expected_step is None:
            if block[i].end != block[i + 1].start:
                return False
        else:
            if (block[i + 1].start - block[i].start) != expected_step:
                return False
    return True


def build_samples(
    windows: list[WindowRecord],
    sequence_length: int,
    truth: dict[str, list[tuple[int, int]]] | None = None,
    labels_map: dict[str, dict[str, Any]] | None = None,
    expected_step: int | None = None,
) -> list[SampleRecord]:
    per_contig: dict[str, list[WindowRecord]] = {}
    for w in windows:
        per_contig.setdefault(w.contig, []).append(w)

    samples: list[SampleRecord] = []
    for contig, contig_windows in per_contig.items():
        contig_windows.sort(key=lambda w: w.start)
        if len(contig_windows) < sequence_length:
            continue
        for i in range(0, len(contig_windows) - sequence_length + 1):
            block = contig_windows[i : i + sequence_length]
            if not block_is_valid(block, expected_step=expected_step):
                continue

            sample_start = block[0].start
            sample_end = block[-1].end

            if labels_map is not None:
                missing = False
                labels = []
                for rec in block:
                    item = labels_map.get(os.path.basename(rec.path))
                    if item is None:
                        missing = True
                        break
                    labels.append(int(item["binary"]))
                if missing:
                    continue
                label = int(any(labels))
            else:
                label = overlaps_truth(contig, sample_start, sample_end, truth or {})

            sample_id = f"{contig}:{sample_start}-{sample_end}"
            samples.append(
                SampleRecord(
                    sample_id=sample_id,
                    contig=contig,
                    start=sample_start,
                    end=sample_end,
                    paths=tuple(w.path for w in block),
                    label=label,
                )
            )
    return samples


def split_samples_by_contig(
    samples: list[SampleRecord],
    val_frac: float,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    if not samples:
        return [], []

    contigs = sorted({s.contig for s in samples})
    rng = random.Random(seed)
    rng.shuffle(contigs)

    if len(contigs) == 1:
        shuffled = samples[:]
        rng.shuffle(shuffled)
        split = max(1, int(round((1.0 - val_frac) * len(shuffled))))
        split = min(max(1, split), len(shuffled) - 1) if len(shuffled) > 1 else 1
        return shuffled[:split], shuffled[split:]

    n_val = max(1, int(round(len(contigs) * val_frac)))
    n_val = min(max(1, n_val), len(contigs))
    val_contigs = set(contigs[:n_val])
    train_samples = [s for s in samples if s.contig not in val_contigs]
    val_samples = [s for s in samples if s.contig in val_contigs]
    return train_samples, val_samples


class NpySequenceDataset(Dataset):
    def __init__(self, samples: list[SampleRecord], dtype: np.dtype = np.float32) -> None:
        self.samples = samples
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        arrs: list[np.ndarray] = []
        for path in sample.paths:
            arr = np.load(path).astype(self.dtype, copy=False)
            if arr.ndim != 2 or arr.shape[1] != 9:
                raise ValueError(
                    f"Expected feature window shape (window_size, 9), got {arr.shape} for {path}"
                )
            arrs.append(arr)
        x = np.stack(arrs, axis=0)  # (T, W, C=9)
        y = np.array([sample.label], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def average_precision_score_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    y_prob = y_prob.astype(np.float64)
    n_pos = y_true.sum()
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * max(0.0, r - prev_recall)
        prev_recall = r
    return float(ap)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.int32)
    y_pred = (y_prob >= 0.5).astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    ap = average_precision_score_binary(y_true, y_prob)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ap": float(ap),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def evaluate(
    model: SVHunterLite,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            probs.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))
    y_prob = np.concatenate(probs) if probs else np.array([], dtype=np.float32)
    y_true = np.concatenate(targets) if targets else np.array([], dtype=np.float32)
    metrics = compute_binary_metrics(y_true, y_prob) if len(y_true) else {}
    return (float(np.mean(losses)) if losses else math.nan), metrics


def truncate_samples(samples: list[SampleRecord], max_samples: int | None, seed: int) -> list[SampleRecord]:
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return samples
    rng = random.Random(seed)
    copied = samples[:]
    rng.shuffle(copied)
    return copied[:max_samples]


def resolve_labels_and_truth(args: argparse.Namespace) -> tuple[dict[str, list[tuple[int, int]]], dict[str, dict[str, Any]] | None]:
    labels_map = load_labels_txt(args.labels_txt) if args.labels_txt else None
    truth = load_truth_intervals(args.truth_bed, args.truth_vcf)
    if labels_map is not None and truth:
        raise ValueError("Use either --labels-txt or truth interval inputs, not both.")
    return truth, labels_map


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    truth, labels_map = resolve_labels_and_truth(args)
    if labels_map is None and not truth:
        raise ValueError("Training requires --labels-txt or one of --truth-bed/--truth-vcf.")

    windows = collect_windows(args.features_dir, args.window_size)
    samples = build_samples(
        windows=windows,
        sequence_length=args.sequence_length,
        truth=truth,
        labels_map=labels_map,
        expected_step=args.step_size,
    )
    samples = truncate_samples(samples, args.max_samples, args.seed)
    if not samples:
        raise ValueError("No samples could be constructed from features.")

    train_samples, val_samples = split_samples_by_contig(samples, args.val_frac, args.seed)
    if not train_samples or not val_samples:
        raise ValueError("Train/val split failed. Adjust --val-frac or --max-samples.")

    train_dataset = NpySequenceDataset(train_samples)
    val_dataset = NpySequenceDataset(val_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config = ModelConfig(
        sequence_length=args.sequence_length,
        window_size=args.window_size,
        in_channels=9,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = SVHunterLite(config).to(device)

    train_labels = np.array([s.label for s in train_samples], dtype=np.int32)
    n_pos = int(train_labels.sum())
    n_neg = int(len(train_labels) - n_pos)
    if n_pos > 0 and n_neg > 0 and args.use_pos_weight:
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_ap = -1.0
    best_epoch = -1
    best_path = os.path.join(args.output_dir, "best.pt")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else math.nan
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        val_ap = float(val_metrics.get("ap", 0.0))

        epoch_row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics}
        history.append(epoch_row)
        print(json.dumps(epoch_row))

        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            save_checkpoint(best_path, model)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    metrics = {
        "best_epoch": best_epoch,
        "best_ap": best_ap,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "history": history,
        "config": config.__dict__,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved checkpoint: {best_path}")
    print(f"Saved metrics: {metrics_path}")


def run_predict(args: argparse.Namespace) -> None:
    truth, labels_map = resolve_labels_and_truth(args)
    windows = collect_windows(args.features_dir, args.window_size)
    samples = build_samples(
        windows=windows,
        sequence_length=args.sequence_length,
        truth=truth,
        labels_map=labels_map,
        expected_step=args.step_size,
    )
    samples = truncate_samples(samples, args.max_samples, args.seed)
    if not samples:
        raise ValueError("No samples could be constructed from features.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_checkpoint(args.checkpoint, device)
    dataset = NpySequenceDataset(samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    out_rows: list[dict[str, str | int | float]] = []
    idx = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _ = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            labels = y.numpy().reshape(-1).astype(int)
            for prob, label in zip(probs, labels):
                s = samples[idx]
                out_rows.append(
                    {
                        "sample_id": s.sample_id,
                        "contig": s.contig,
                        "start": s.start,
                        "end": s.end,
                        "probability": float(prob),
                        "label": int(label),
                    }
                )
                idx += 1

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "contig", "start", "end", "probability", "label"],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved predictions: {args.output_csv}")


def run_embed(args: argparse.Namespace) -> None:
    truth, labels_map = resolve_labels_and_truth(args)
    windows = collect_windows(args.features_dir, args.window_size)
    samples = build_samples(
        windows=windows,
        sequence_length=args.sequence_length,
        truth=truth,
        labels_map=labels_map,
        expected_step=args.step_size,
    )
    samples = truncate_samples(samples, args.max_samples, args.seed)
    if not samples:
        raise ValueError("No samples could be constructed from features.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_checkpoint(args.checkpoint, device)
    dataset = NpySequenceDataset(samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    embeddings: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, z = model(x)
            embeddings.append(z.cpu().numpy())
            probs.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            labels.append(y.numpy().reshape(-1))

    emb = np.concatenate(embeddings, axis=0)
    p = np.concatenate(probs, axis=0)
    y = np.concatenate(labels, axis=0)
    contigs = np.array([s.contig for s in samples], dtype=object)
    starts = np.array([s.start for s in samples], dtype=np.int64)
    ends = np.array([s.end for s in samples], dtype=np.int64)
    sample_ids = np.array([s.sample_id for s in samples], dtype=object)

    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        embeddings=emb,
        probabilities=p,
        labels=y,
        contigs=contigs,
        starts=starts,
        ends=ends,
        sample_ids=sample_ids,
    )
    print(f"Saved embeddings: {args.output_npz}")


def run_sandbox(args: argparse.Namespace) -> None:
    """
    Lightweight pipeline check:
    - build samples
    - run one minibatch through model
    - optionally run one tiny train epoch
    """
    set_seed(args.seed)
    truth, labels_map = resolve_labels_and_truth(args)
    windows = collect_windows(args.features_dir, args.window_size)
    samples = build_samples(
        windows=windows,
        sequence_length=args.sequence_length,
        truth=truth,
        labels_map=labels_map,
        expected_step=args.step_size,
    )
    samples = truncate_samples(samples, args.max_samples, args.seed)
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples for sandbox check.")

    dataset = NpySequenceDataset(samples)
    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=False, num_workers=0)
    x, y = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config = ModelConfig(
        sequence_length=args.sequence_length,
        window_size=args.window_size,
        in_channels=9,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = SVHunterLite(config).to(device)
    logits, emb = model(x.to(device))

    print(
        json.dumps(
            {
                "num_windows": len(windows),
                "num_samples": len(samples),
                "input_batch_shape": list(x.shape),
                "label_batch_shape": list(y.shape),
                "logits_shape": list(logits.shape),
                "embedding_shape": list(emb.shape),
            }
        )
    )

    if args.sandbox_train_step:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(logits, y.to(device))
        loss.backward()
        optimizer.step()
        print(json.dumps({"sandbox_train_loss": float(loss.item())}))


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--features-dir", required=True, help="Directory with contig_start_end.npy feature windows.")
    parser.add_argument("--window-size", type=int, default=2000, help="Feature window size.")
    parser.add_argument("--sequence-length", type=int, default=1, help="Number of windows per sample.")
    parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Expected start offset between adjacent windows in a sample. "
        "Default requires strict end-to-start contiguity.",
    )
    parser.add_argument("--labels-txt", default=None, help="Optional labels.txt mapping file->label vector.")
    parser.add_argument("--truth-bed", default=None, help="Optional BED with truth intervals.")
    parser.add_argument("--truth-vcf", default=None, help="Optional VCF with truth intervals.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d-model", type=int, default=64, help="Hidden embedding size.")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SVHunter-lite training/inference in MAMNET feature space.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train model from feature windows and labels.")
    add_common_data_args(train_p)
    add_model_args(train_p)
    train_p.add_argument("--output-dir", default="output/models/train", help="Output directory.")
    train_p.add_argument("--epochs", type=int, default=20, help="Max training epochs.")
    train_p.add_argument("--patience", type=int, default=5, help="Early-stopping patience.")
    train_p.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate.")
    train_p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    train_p.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction.")
    train_p.add_argument("--use-pos-weight", action="store_true", help="Use positive-class weighting in BCE.")

    pred_p = sub.add_parser("predict", help="Predict probabilities for samples.")
    add_common_data_args(pred_p)
    pred_p.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
    pred_p.add_argument("--output-csv", default="output/models/predictions.csv", help="Prediction CSV path.")

    emb_p = sub.add_parser("embed", help="Export learned sample embeddings.")
    add_common_data_args(emb_p)
    emb_p.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
    emb_p.add_argument("--output-npz", default="output/models/embeddings.npz", help="Output NPZ path.")

    sandbox_p = sub.add_parser("sandbox", help="Quick shape and one-step training sanity check.")
    add_common_data_args(sandbox_p)
    add_model_args(sandbox_p)
    sandbox_p.add_argument("--sandbox-train-step", action="store_true", help="Run one optimizer step.")
    sandbox_p.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for sandbox step.")
    sandbox_p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for sandbox step.")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "embed":
        run_embed(args)
    elif args.command == "sandbox":
        run_sandbox(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

