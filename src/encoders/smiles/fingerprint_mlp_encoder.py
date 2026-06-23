from __future__ import annotations

import torch
import torch.nn as nn

from encoders.base import Encoder


class FingerprintMLPEncoder(Encoder):
    """Simple MLP encoder for molecular fingerprint vectors."""

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self._output_dim = out_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.mlp(batch)

    @property
    def output_dim(self) -> int:
        return self._output_dim
