"""Temporal Convolutional Attention Network for sequence labeling."""

import torch
from torch import nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Remove right padding so convolutions stay causal."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A residual TCN block with dilated causal convolutions."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class SelfAttentionBlock(nn.Module):
    """Batch-first self-attention block over pulse positions."""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended, _ = self.attention(x, x, x, need_weights=False)
        return self.norm(x + self.dropout(attended))


class TCAN(nn.Module):
    """TCN plus self-attention model for pulse logits or embeddings."""

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_channels=(32, 32, 64, 64),
        kernel_size=3,
        dropout=0.1,
        attention_heads=4,
        embedding_dim=64,
    ):
        super().__init__()
        layers = []
        for level, out_channels in enumerate(hidden_channels):
            in_channels = input_dim if level == 0 else hidden_channels[level - 1]
            dilation = 2**level
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.tcn = nn.Sequential(*layers)
        hidden_dim = hidden_channels[-1]
        self.attention = SelfAttentionBlock(hidden_dim, attention_heads, dropout)
        self.embedding_projection = (
            nn.Identity()
            if embedding_dim == hidden_dim
            else nn.Linear(hidden_dim, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x):
        """Map [B, T, D] input features to [B, T, E] pulse embeddings."""
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        x = self.attention(x)
        return self.embedding_projection(x)

    def forward(self, x, return_embeddings=False):
        """Map [B, T, D] to logits [B, T, C] or embeddings [B, T, E]."""
        embeddings = self.encode(x)
        if return_embeddings:
            return embeddings
        return self.classifier(embeddings)
