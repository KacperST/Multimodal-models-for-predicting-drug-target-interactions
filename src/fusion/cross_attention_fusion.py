from __future__ import annotations

import torch
import torch.nn as nn

from fusion.base import FusionModule


class CrossAttentionFusion(FusionModule):
    """Fuse two modality embeddings via bilinear cross-attention + gating.

    Works with *pooled* embeddings (single vectors per sample).
    For full sequence-level cross-attention, encoders would need to
    return pre-pooling representations — a straightforward extension.

    Architecture::

        1. Project each embedding to a shared ``proj_dim``.
        2. Compute bilinear attention scores and gating weights.
        3. Combine attended representations.
        4. Pass through an MLP head for prediction.
    """

    def __init__(
        self,
        smiles_dim: int,
        protein_dim: int,
        proj_dim: int = 256,
        num_heads: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Project both modalities to a shared dimension
        self.smiles_proj = nn.Linear(smiles_dim, proj_dim)
        self.protein_proj = nn.Linear(protein_dim, proj_dim)

        # Multi-head attention (treat each embedding as a 1-token sequence)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(proj_dim)
        self.norm2 = nn.LayerNorm(proj_dim)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.Sigmoid(),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, smiles_emb: torch.Tensor, protein_emb: torch.Tensor
    ) -> torch.Tensor:
        # Project → (B, 1, proj_dim)
        s = self.smiles_proj(smiles_emb).unsqueeze(1)
        p = self.protein_proj(protein_emb).unsqueeze(1)

        # Cross-attention: smiles queries protein
        attn_out, _ = self.cross_attn(query=s, key=p, value=p)
        s = self.norm1(s + attn_out)

        # Cross-attention: protein queries smiles
        attn_out2, _ = self.cross_attn(query=p, key=s, value=s)
        p = self.norm2(p + attn_out2)

        # Squeeze back to (B, proj_dim)
        s = s.squeeze(1)
        p = p.squeeze(1)

        # Gating
        gate_input = torch.cat([s, p], dim=1)
        g = self.gate(gate_input)
        fused = g * s + (1 - g) * p

        return self.classifier(fused)
