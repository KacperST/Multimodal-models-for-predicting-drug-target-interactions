from __future__ import annotations

import torch
import torch.nn as nn

from fusion.base import FusionModule


class MLPFusion(FusionModule):
    """Fuse two modality embeddings via concatenation + MLP classifier.

    Architecture::

        concat(smiles_emb, protein_emb)
          → Linear → BN → ReLU → Dropout
          → Linear → ReLU → Dropout
          → Linear → 1
    """

    def __init__(
        self,
        smiles_dim: int,
        protein_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 64]

        combined_dim = smiles_dim + protein_dim
        layers: list[nn.Module] = []

        in_dim = combined_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self, smiles_emb: torch.Tensor, protein_emb: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([smiles_emb, protein_emb], dim=1)
        return self.classifier(combined)
