from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    sequence_length: int = 10
    window_size: int = 200
    in_channels: int = 9
    d_model: int = 64
    num_heads: int = 4
    dropout: float = 0.1


class FrameEncoder(nn.Module):
    """Lightweight local encoder for one feature window (W x C)."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, W, C)
        x = x.unsqueeze(1)  # (B*T, 1, W, C)
        x = self.conv(x)
        x = self.proj(x)  # (B*T, d_model)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer encoder block."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class SVHunterLite(nn.Module):
    """
    Minimal SVHunter-like model for MAMNET space:
    local frame encoder + sequence attention + binary head.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = FrameEncoder(
            d_model=config.d_model,
            dropout=config.dropout,
        )
        self.positional = nn.Parameter(torch.zeros(1, config.sequence_length, config.d_model))
        self.transformer = TransformerBlock(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, W, C)
        b, t, w, c = x.shape
        if t != self.config.sequence_length:
            raise ValueError(f"expected sequence length {self.config.sequence_length}, got {t}")
        if c != self.config.in_channels:
            raise ValueError(f"expected channels {self.config.in_channels}, got {c}")
        x = x.reshape(b * t, w, c)
        x = self.encoder(x).reshape(b, t, -1)  # (B, T, d_model)
        x = x + self.positional
        x = self.transformer(x)
        pooled = self.norm(x).mean(dim=1)  # (B, d_model)
        logits = self.classifier(pooled)  # (B, 1)
        return logits, pooled


def save_checkpoint(path: str, model: SVHunterLite) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": model.config.__dict__,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> SVHunterLite:
    payload = torch.load(path, map_location=device)
    config = ModelConfig(**payload["config"])
    model = SVHunterLite(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model

