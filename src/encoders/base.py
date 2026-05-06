from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Encoder(nn.Module, ABC):
    """Abstract base class for modality encoders.

    Subclasses take a batch of processed inputs and produce fixed-size
    embedding vectors of shape ``(batch_size, output_dim)``.
    """

    @abstractmethod
    def forward(self, batch) -> torch.Tensor:
        """Encode a batch and return embeddings of shape ``(B, output_dim)``."""
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of the output embedding vector."""
        ...
