from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class FusionModule(nn.Module, ABC):
    """Abstract base class for multi-modal fusion.

    Combines embeddings from a SMILES encoder and a protein encoder
    into a single prediction output.
    """

    @abstractmethod
    def forward(
        self, smiles_emb: torch.Tensor, protein_emb: torch.Tensor
    ) -> torch.Tensor:
        """Fuse two embedding vectors and produce a prediction.

        Args:
            smiles_emb: ``(B, smiles_dim)`` embeddings from the SMILES encoder.
            protein_emb: ``(B, protein_dim)`` embeddings from the protein encoder.

        Returns:
            ``(B, 1)`` prediction logits.
        """
        ...
