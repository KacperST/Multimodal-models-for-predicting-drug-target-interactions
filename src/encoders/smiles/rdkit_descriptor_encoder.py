from __future__ import annotations

import torch.nn as nn

from encoders.base import Encoder


class RDKitDescriptorEncoder(Encoder):
    """MLP encoder for RDKit physicochemical descriptor vectors.

    Takes a 210-dimensional vector of molecular descriptors and projects
    it to a lower-dimensional embedding via a two-layer MLP, following
    the same pattern as :class:`FingerprintMLPEncoder` and
    :class:`LincsEncoder`.

    Architecture::

        Linear(input_dim, hidden_dim) → BatchNorm → ReLU → Dropout
        → Linear(hidden_dim, out_dim) → BatchNorm → ReLU

    Args:
        input_dim: Dimension of the input descriptor vector (default: 210).
        hidden_dim: Hidden layer dimension.
        out_dim: Output embedding dimension.
        dropout: Dropout rate after the first hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 210,
        hidden_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self._output_dim = out_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, batch):
        """Encode a batch of descriptor vectors.

        Args:
            batch: Tensor of shape ``(B, input_dim)``.

        Returns:
            Tensor of shape ``(B, output_dim)``.
        """
        return self.mlp(batch)

    @property
    def output_dim(self) -> int:
        return self._output_dim
