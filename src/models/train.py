"""
Autoresearch training script for SV detection. Single-file.
Usage: cd src/models && uv run python train.py
"""

import math
import sys
import time

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from prepare import (
    DEFAULT_TRAIN_DIRECTORY,
    DEFAULT_VALIDATION_DIRECTORY,
    TIME_BUDGET,
    MetricsAccumulator,
    create_dataloader,
    evaluate,
    get_default_device_name,
    print_summary,
    set_seed,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SVHunterSubwindowEncoder(nn.Module):
    def __init__(self, num_features: int = 9) -> None:
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
        embedding_dimension: int,
        num_heads: int,
        key_dimension: int,
        dropout: float,
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
        embedding_dimension: int,
        num_heads: int,
        key_dimension: int,
        dropout: float,
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

        self.input_norm = nn.LayerNorm(NUM_FEATURES)
        self.input_pos_embedding = nn.Parameter(torch.zeros(1, INPUT_LENGTH, 1))
        nn.init.normal_(self.input_pos_embedding, std=0.02)
        self.encoder = SVHunterSubwindowEncoder(num_features=NUM_FEATURES + 1)
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
        x = self.input_norm(x)
        # Add learnable positional channel
        x = torch.cat([x, self.input_pos_embedding.expand(batch_size, -1, -1)], dim=-1)
        x = x.view(batch_size, NUM_SUBWINDOWS, SUBWINDOW_SIZE, NUM_FEATURES + 1)
        x = x.unsqueeze(2).reshape(
            batch_size * NUM_SUBWINDOWS, 1, SUBWINDOW_SIZE, NUM_FEATURES + 1
        )
        x = self.encoder(x)
        x = x.view(batch_size, NUM_SUBWINDOWS, self.encoder.output_dimension)
        x = self.patch_projection(x)
        x = x + self.position_embedding

        for block in self.transformer_blocks:
            x = block(x)

        x = self.sequence_normalization(x)
        return self.classifier(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
NUM_FEATURES = 9
INPUT_LENGTH = 2000
SUBWINDOW_SIZE = 200
NUM_SUBWINDOWS = 10
EMBEDDING_DIMENSION = 128
NUM_HEADS = 4
KEY_DIMENSION = 32
NUM_TRANSFORMER_BLOCKS = 4
MLP_HIDDEN_DIMENSION = 256
ATTENTION_DROPOUT = 0.1
HEAD_DROPOUT = 0.2

# Optimization
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 100  # upper bound on epochs (may stop earlier via time budget)

# Misc
SEED = 42
NUM_WORKERS = 0
MAX_SAMPLES = None  # cap samples per split (None = use all)

# ---------------------------------------------------------------------------
# Setup: seed, device, data, model, optimizer
# ---------------------------------------------------------------------------

t_start = time.time()
set_seed(SEED)
device = torch.device(get_default_device_name())
print(f"Device: {device}")
print(f"Time budget: {TIME_BUDGET}s")

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
print(f"Parameters: {num_params:,}")

criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
best_val_f1 = float("-inf")
best_val_metrics = None
total_training_time = 0.0
epochs_completed = 0

for epoch in range(1, MAX_EPOCHS + 1):
    t_epoch_start = time.time()

    # Train
    model.train()
    accumulator = MetricsAccumulator()
    for features, labels in train_loader:
        features = features.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.float32, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        accumulator.update(loss.item(), logits.detach(), labels.detach())
    train_metrics = accumulator.compute()

    # Validate
    val_metrics = evaluate(
        model=model,
        dataloader=validation_loader,
        criterion=criterion,
        device=device,
    )

    scheduler.step()

    t_epoch_end = time.time()
    total_training_time += t_epoch_end - t_epoch_start
    epochs_completed = epoch
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(
        f"epoch {epoch:03d} | "
        f"train_loss={train_metrics.loss:.4f} | "
        f"val_loss={val_metrics.loss:.4f} | "
        f"val_f1={val_metrics.elementwise_f1:.4f} | "
        f"remaining={remaining:.0f}s"
    )

    if val_metrics.elementwise_f1 > best_val_f1:
        best_val_f1 = val_metrics.elementwise_f1
        best_val_metrics = val_metrics

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(val_metrics.loss) or val_metrics.loss > 100:
        print("FAIL")
        sys.exit(1)

    # Time's up
    if total_training_time >= TIME_BUDGET:
        break

# Final summary
t_end = time.time()

if best_val_metrics is None:
    print("FAIL")
    sys.exit(1)

print_summary(
    val_metrics=best_val_metrics,
    training_seconds=total_training_time,
    total_seconds=t_end - t_start,
    num_epochs=epochs_completed,
    num_params=num_params,
)
