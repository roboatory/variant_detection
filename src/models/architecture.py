from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 7) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)
        self.multilayer_perceptron = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        average_attention = self.multilayer_perceptron(self.average_pool(x))
        maximum_attention = self.multilayer_perceptron(self.maximum_pool(x))
        attention = self.activation(average_attention + maximum_attention)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: tuple[int, int] = (2, 9)) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding="same", bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        mean_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([mean_map, max_map], dim=1)
        attention = self.activation(self.conv(attention))
        return x * attention


class CBAMBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 7,
        spatial_kernel_size: tuple[int, int] = (2, 9),
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SVHunterSubwindowEncoder(nn.Module):
    def __init__(self, num_features: int = 9) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(2, num_features), padding="same"),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(128, 64, kernel_size=(2, 1), padding="same"),
            nn.MaxPool2d(kernel_size=(2, 1)),
            CBAMBlock(64, reduction_ratio=7, spatial_kernel_size=(2, num_features)),
            nn.Conv2d(64, 64, kernel_size=(2, 1), padding="same"),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(64, 64, kernel_size=(2, 1), padding="same"),
            nn.MaxPool2d(kernel_size=(2, 1)),
            CBAMBlock(64, reduction_ratio=7, spatial_kernel_size=(2, num_features)),
            nn.Conv2d(64, 64, kernel_size=(2, 1), padding="same"),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(64, 64, kernel_size=(2, 1), padding="same"),
            nn.MaxPool2d(kernel_size=(2, 1)),
            CBAMBlock(64, reduction_ratio=7, spatial_kernel_size=(2, num_features)),
            nn.Conv2d(64, 64, kernel_size=(2, 1), padding="same"),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.output_dimension = 64 * num_features

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return torch.flatten(x, start_dim=1)


class SVHunterMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dimension: int = 100,
        num_heads: int = 32,
        key_dimension: int = 32,
        dropout: float = 0.3,
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

        attention_output = F.scaled_dot_product_attention(
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
        embedding_dimension: int = 100,
        num_heads: int = 32,
        key_dimension: int = 32,
        dropout: float = 0.3,
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
    def __init__(
        self,
        input_length: int = 2000,
        num_features: int = 9,
        subwindow_size: int = 200,
        num_subwindows: int = 10,
        embedding_dimension: int = 100,
        num_heads: int = 32,
        key_dimension: int = 32,
        num_transformer_blocks: int = 7,
        multilayer_perceptron_hidden_dimension: int = 128,
        attention_dropout: float = 0.3,
        head_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if input_length != subwindow_size * num_subwindows:
            raise ValueError("input_length must equal subwindow_size * num_subwindows")

        self.input_length = input_length
        self.num_features = num_features
        self.subwindow_size = subwindow_size
        self.num_subwindows = num_subwindows
        self.encoder = SVHunterSubwindowEncoder(num_features=num_features)
        self.patch_projection = nn.Linear(
            self.encoder.output_dimension, embedding_dimension
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_subwindows, embedding_dimension)
        )
        self.transformer_blocks = nn.ModuleList(
            [
                SVHunterTransformerBlock(
                    embedding_dimension=embedding_dimension,
                    num_heads=num_heads,
                    key_dimension=key_dimension,
                    dropout=attention_dropout,
                )
                for _ in range(num_transformer_blocks)
            ]
        )
        self.sequence_normalization = nn.LayerNorm(embedding_dimension)
        self.classifier = nn.Sequential(
            nn.Linear(
                num_subwindows * embedding_dimension,
                multilayer_perceptron_hidden_dimension,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(
                multilayer_perceptron_hidden_dimension,
                multilayer_perceptron_hidden_dimension,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(multilayer_perceptron_hidden_dimension, num_subwindows),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input shape (batch, 2000, 9)")
        if x.shape[1] != self.input_length or x.shape[2] != self.num_features:
            raise ValueError(
                f"Expected input shape (batch, {self.input_length}, {self.num_features})"
            )

        batch_size = x.shape[0]
        x = x.view(
            batch_size, self.num_subwindows, self.subwindow_size, self.num_features
        )
        x = x.unsqueeze(2).reshape(
            batch_size * self.num_subwindows, 1, self.subwindow_size, self.num_features
        )
        x = self.encoder(x)
        x = x.view(batch_size, self.num_subwindows, self.encoder.output_dimension)
        x = self.patch_projection(x)
        x = x + self.position_embedding

        for block in self.transformer_blocks:
            x = block(x)

        x = self.sequence_normalization(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
