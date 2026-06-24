from __future__ import annotations

import torch
import torch.nn as nn

from encoders.base import Encoder


class ProteinCNNEncoder(Encoder):
    """Multi-kernel 1D-CNN encoder for protein sequences.

    Applies several parallel convolution blocks with different kernel
    sizes, performs masked max-pooling, and concatenates the results.
    """

    def __init__(
        self,
        vocab_size: int = 21,
        embed_dim: int = 256,
        num_filters: int = 128,
        kernel_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 7, 15]

        self._output_dim = num_filters * len(kernel_sizes)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv_blocks = nn.ModuleList(
            [self._conv_block(embed_dim, num_filters, k) for k in kernel_sizes]
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, kernel: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    @staticmethod
    def _masked_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Max-pool ignoring padded positions."""
        x = x.masked_fill(mask == 0, float("-inf"))
        return x.max(dim=2).values

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        mask = (batch != 0).unsqueeze(1).float()
        x = self.embedding(batch).transpose(1, 2)  # (B, embed_dim, L)
        pooled = [self._masked_pool(conv(x), mask) for conv in self.conv_blocks]
        return torch.cat(pooled, dim=1)  # (B, num_filters * n_kernels)

    @property
    def output_dim(self) -> int:
        return self._output_dim
