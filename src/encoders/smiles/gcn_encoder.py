from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import GCNConv, global_add_pool

from encoders.base import Encoder


class GCNEncoder(Encoder):
    """GCN encoder for molecular graphs.

    Uses OGB ``AtomEncoder`` for initial node features and applies
    multiple GCN convolutional layers followed by global sum pooling.
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 3) -> None:
        super().__init__()
        self._output_dim = hidden_dim

        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, batch) -> torch.Tensor:
        x = self.atom_encoder(batch.x)
        for conv, bn in zip(self.convs, self.bns):
            x = F.leaky_relu(bn(conv(x, batch.edge_index)))
        return global_add_pool(x, batch.batch)

    @property
    def output_dim(self) -> int:
        return self._output_dim
