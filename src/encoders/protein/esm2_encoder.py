from __future__ import annotations

import torch
import torch.nn as nn

from encoders.base import Encoder


class ESM2Encoder(Encoder):
    """Lightweight encoder for pre-computed ESM-2 embeddings.

    Applies only a trainable projection head (Linear → LayerNorm → ReLU)
    on top of cached ESM-2 embeddings.  The heavy transformer backbone
    is **not** loaded — embeddings are expected to come from the
    ``ESM2Processor``.

    Args:
        input_dim: Dimension of the cached ESM-2 embeddings.
        out_dim: Output embedding dimension after projection.
            If ``None`` or equal to ``input_dim``, no projection is applied.
    """

    def __init__(
        self,
        input_dim: int = 480,
        out_dim: int | None = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if out_dim is not None and out_dim != input_dim:
            hidden_dim = (input_dim + out_dim) // 2
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
            )
            self._output_dim = out_dim
        else:
            self.projection = None
            self._output_dim = input_dim

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Project cached embeddings.

        Args:
            batch: Tensor of shape ``(B, input_dim)`` — stacked cached
                ESM-2 embeddings.

        Returns:
            Tensor of shape ``(B, output_dim)``.
        """
        if self.projection is not None:
            return self.projection(batch)
        return batch

    @property
    def output_dim(self) -> int:
        return self._output_dim
