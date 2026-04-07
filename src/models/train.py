"""Autoresearch-compatible training script for SV detection.

This is the file the AI agent edits. It contains:
- Model architecture (SVHunterModel and all sub-modules)
- Optimizer configuration
- Training loop
- Hyperparameters as top-level constants

Run: uv run python src/models/train.py > run.log 2>&1
Grep: grep "^val_elementwise_f1:" run.log
"""

from __future__ import annotations

import sys
import time

import torch
from torch import Tensor, nn
from torch.optim import Adam

from prepare import (
    DEFAULT_TRAIN_DIRECTORY,
    DEFAULT_VALIDATION_DIRECTORY,
    EpochMetrics,
    MetricsAccumulator,
    create_dataloader,
    evaluate,
    get_default_device_name,
    print_summary,
    set_seed,
)

# ===== HYPERPARAMETERS (edit these) ==========================================

SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
NUM_WORKERS = 0
MAX_SAMPLES = None

# model
NUM_FEATURES = 9
INPUT_LENGTH = 2000
SUBWINDOW_SIZE = 200
NUM_SUBWINDOWS = 10
EMBEDDING_DIMENSION = 100
NUM_HEADS = 4
KEY_DIMENSION = 32
NUM_TRANSFORMER_BLOCKS = 3
MLP_HIDDEN_DIMENSION = 128
ATTENTION_DROPOUT = 0.3
HEAD_DROPOUT = 0.4

# ===== MODEL ARCHITECTURE ====================================================


class SVHunterSubwindowEncoder(nn.Module):
    def __init__(self, num_features: int = NUM_FEATURES) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(1, num_features), padding="valid"),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(128, 64, kernel_size=(3, 1), padding="valid"),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(2, 1), padding="valid"),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="valid"),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(2, 1), padding="valid"),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(64, 64, kernel_size=(2, 1), padding="valid"),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(64, 64, kernel_size=(2, 1), padding="valid"),
        )
        self.output_dimension = 64

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return torch.flatten(x, start_dim=1)


class SVHunterMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        num_heads: int = NUM_HEADS,
        key_dimension: int = KEY_DIMENSION,
        dropout: float = ATTENTION_DROPOUT,
    ) -> None:
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.key_dimension = key_dimension
        self.attention_inner_dimension = num_heads * key_dimension
        self.query_projection = nn.Linear(
            embedding_dimension, self.attention_inner_dimension
        )
        self.key_projection = nn.Linear(
            embedding_dimension, self.attention_inner_dimension
        )
        self.value_projection = nn.Linear(
            embedding_dimension, self.attention_inner_dimension
        )
        self.output_projection = nn.Linear(
            self.attention_inner_dimension, embedding_dimension
        )
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        query = self.query_projection(x).view(
            batch_size, seq_len, self.num_heads, self.key_dimension
        )
        key = self.key_projection(x).view(
            batch_size, seq_len, self.num_heads, self.key_dimension
        )
        value = self.value_projection(x).view(
            batch_size, seq_len, self.num_heads, self.key_dimension
        )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch_size, seq_len, self.attention_inner_dimension
        )
        return self.output_projection(attention_output)


class SVHunterTransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        num_heads: int = NUM_HEADS,
        key_dimension: int = KEY_DIMENSION,
        dropout: float = ATTENTION_DROPOUT,
    ) -> None:
        super().__init__()
        self.layer_normalization_1 = nn.LayerNorm(embedding_dimension)
        self.attention = SVHunterMultiHeadAttention(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            key_dimension=key_dimension,
            dropout=dropout,
        )
        self.layer_normalization_2 = nn.LayerNorm(embedding_dimension)
        self.feed_forward_network = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.layer_normalization_1(x))
        x = x + self.feed_forward_network(self.layer_normalization_2(x))
        return x


class SVHunterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        if INPUT_LENGTH != SUBWINDOW_SIZE * NUM_SUBWINDOWS:
            raise ValueError("INPUT_LENGTH must equal SUBWINDOW_SIZE * NUM_SUBWINDOWS")

        self.encoder = SVHunterSubwindowEncoder(num_features=NUM_FEATURES)
        self.patch_projection = nn.Linear(
            self.encoder.output_dimension, EMBEDDING_DIMENSION
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, NUM_SUBWINDOWS, EMBEDDING_DIMENSION)
        )
        self.transformer_blocks = nn.ModuleList(
            [
                SVHunterTransformerBlock(
                    embedding_dimension=EMBEDDING_DIMENSION,
                    num_heads=NUM_HEADS,
                    key_dimension=KEY_DIMENSION,
                    dropout=ATTENTION_DROPOUT,
                )
                for _ in range(NUM_TRANSFORMER_BLOCKS)
            ]
        )
        self.sequence_normalization = nn.LayerNorm(EMBEDDING_DIMENSION)
        self.classifier = nn.Sequential(
            nn.Linear(EMBEDDING_DIMENSION, MLP_HIDDEN_DIMENSION),
            nn.ReLU(inplace=True),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(MLP_HIDDEN_DIMENSION, MLP_HIDDEN_DIMENSION),
            nn.ReLU(inplace=True),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(MLP_HIDDEN_DIMENSION, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input shape (batch, 2000, 9)")
        if x.shape[1] != INPUT_LENGTH or x.shape[2] != NUM_FEATURES:
            raise ValueError(
                f"Expected input shape (batch, {INPUT_LENGTH}, {NUM_FEATURES})"
            )

        batch_size = x.shape[0]
        x = x.view(batch_size, NUM_SUBWINDOWS, SUBWINDOW_SIZE, NUM_FEATURES)
        x = x.unsqueeze(2).reshape(
            batch_size * NUM_SUBWINDOWS, 1, SUBWINDOW_SIZE, NUM_FEATURES
        )
        x = self.encoder(x)
        x = x.view(batch_size, NUM_SUBWINDOWS, self.encoder.output_dimension)
        x = self.patch_projection(x)
        x = x + self.position_embedding

        for block in self.transformer_blocks:
            x = block(x)

        x = self.sequence_normalization(x)
        return self.classifier(x).squeeze(-1)


# ===== TRAINING LOOP =========================================================


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    model.train()
    accumulator = MetricsAccumulator()

    for features, labels in dataloader:
        features = features.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.float32, non_blocking=True)

        logits = model(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        accumulator.update(loss.item(), logits.detach(), labels.detach())

    return accumulator.compute()


def main() -> None:
    total_start = time.time()
    set_seed(SEED)

    device = torch.device(get_default_device_name())

    train_loader = create_dataloader(
        split_directory=DEFAULT_TRAIN_DIRECTORY,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        max_samples=MAX_SAMPLES,
    )
    validation_loader = create_dataloader(
        split_directory=DEFAULT_VALIDATION_DIRECTORY,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        max_samples=MAX_SAMPLES,
    )

    model = SVHunterModel().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = float("-inf")
    best_val_metrics = None
    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_elementwise_f1={val_metrics.elementwise_f1:.4f}"
        )

        if val_metrics.elementwise_f1 > best_val_f1:
            best_val_f1 = val_metrics.elementwise_f1
            best_val_metrics = val_metrics

        # fast-fail on NaN
        if val_metrics.loss != val_metrics.loss or val_metrics.loss > 100:
            print("FAIL")
            sys.exit(1)

    training_seconds = time.time() - training_start
    total_seconds = time.time() - total_start

    if best_val_metrics is None:
        print("FAIL")
        sys.exit(1)

    print_summary(
        val_metrics=best_val_metrics,
        training_seconds=training_seconds,
        total_seconds=total_seconds,
        num_epochs=EPOCHS,
        num_params=num_params,
    )


if __name__ == "__main__":
    main()
