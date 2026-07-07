from __future__ import annotations

import torch
import torch.nn as nn

from fusion.base import FusionModule


class CrossAttentionFusion(FusionModule):
    """Fuse two modality embeddings via cross-attention + gating.

    Works with *pooled* embeddings (single vectors per sample).
    Both modality vectors are stacked into a 2-token sequence so that
    ``MultiheadAttention`` produces a data-dependent 2×2 attention
    matrix (avoiding the seq_len=1 degeneracy where softmax of a
    single element is trivially 1.0).

    Architecture::

        1. Project each embedding to a shared ``proj_dim``.
        2. Stack as [drug, protein] — a 2-token sequence.
        3. Two layers of self-attention across modalities (separate
           weights) with residual connections and LayerNorm.
        4. Gating mechanism to combine attended representations.
        5. MLP classifier head for binary prediction.
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

        # Two-layer cross-modal attention (separate weights per layer)
        self.attn1 = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
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
        # Project to shared dimension
        s = self.smiles_proj(smiles_emb)   # (B, proj_dim)
        p = self.protein_proj(protein_emb)  # (B, proj_dim)

        # Stack as a 2-token sequence: [DRUG, PROTEIN]
        # This gives a 2×2 attention matrix with learnable weights
        seq = torch.stack([s, p], dim=1)    # (B, 2, proj_dim)

        # Layer 1: cross-modal attention
        attn_out, _ = self.attn1(query=seq, key=seq, value=seq)
        seq = self.norm1(seq + attn_out)

        # Layer 2: deeper cross-modal interaction
        attn_out2, _ = self.attn2(query=seq, key=seq, value=seq)
        seq = self.norm2(seq + attn_out2)

        # Unpack back to per-modality vectors
        s = seq[:, 0, :]  # (B, proj_dim) — drug attended by protein
        p = seq[:, 1, :]  # (B, proj_dim) — protein attended by drug

        # Gating: learned weighted combination
        gate_input = torch.cat([s, p], dim=1)
        g = self.gate(gate_input)
        fused = g * s + (1 - g) * p

        return self.classifier(fused)
