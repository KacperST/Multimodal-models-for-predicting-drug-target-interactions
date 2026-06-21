from __future__ import annotations

import torch
import torch.nn as nn

from encoders.base import Encoder
from fusion.base import FusionModule


class MultimodalDTI(nn.Module):
    """Multimodal Drug-Target Interaction prediction model.

    Supports **multiple encoders per modality**.  For example, you can
    combine a GCN and a ChemBERT encoder for SMILES while combining a
    CNN and an ESM-2 encoder for proteins.  All embeddings within each
    modality are concatenated before being passed to the fusion module.

    Args:
        smiles_encoders: One or more SMILES encoders.
        protein_encoders: One or more protein encoders.
        fusion: Fusion module that combines the (concatenated) embeddings.
    """

    def __init__(
        self,
        smiles_encoders: list[Encoder] | Encoder,
        protein_encoders: list[Encoder] | Encoder,
        fusion: FusionModule,
    ) -> None:
        super().__init__()
        # Accept a single encoder or a list — normalise to ModuleList
        if isinstance(smiles_encoders, Encoder):
            smiles_encoders = [smiles_encoders]
        if isinstance(protein_encoders, Encoder):
            protein_encoders = [protein_encoders]

        self.smiles_encoders = nn.ModuleList(smiles_encoders)
        self.protein_encoders = nn.ModuleList(protein_encoders)
        self.fusion = fusion

        # Pre-cache which encoders have trainable params (avoid iterating
        # over all parameters on every forward pass — matters for 650M ESM2)
        self._smiles_trainable = [
            any(p.requires_grad for p in enc.parameters())
            for enc in smiles_encoders
        ]
        self._protein_trainable = [
            any(p.requires_grad for p in enc.parameters())
            for enc in protein_encoders
        ]

    # ── Convenience properties ───────────────────────────────────

    @property
    def total_smiles_dim(self) -> int:
        return sum(e.output_dim for e in self.smiles_encoders)

    @property
    def total_protein_dim(self) -> int:
        return sum(e.output_dim for e in self.protein_encoders)

    # ── Forward ──────────────────────────────────────────────────

    def forward(
        self,
        smiles_batches: list | object,
        protein_batches: list | object,
    ) -> torch.Tensor:
        """Run forward pass.

        Args:
            smiles_batches: If there is one SMILES encoder — the single
                batched tensor / PyG Batch.  If there are *N* encoders —
                a list of *N* batched inputs (one per encoder).
            protein_batches: Same convention as ``smiles_batches``.

        Returns:
            ``(B, 1)`` prediction logits.
        """
        # Normalise to lists
        if not isinstance(smiles_batches, (list, tuple)):
            smiles_batches = [smiles_batches]
        if not isinstance(protein_batches, (list, tuple)):
            protein_batches = [protein_batches]

        smiles_embs = [
            self._run_encoder(enc, batch, trainable)
            for enc, batch, trainable in zip(
                self.smiles_encoders, smiles_batches, self._smiles_trainable
            )
        ]
        protein_embs = [
            self._run_encoder(enc, batch, trainable)
            for enc, batch, trainable in zip(
                self.protein_encoders, protein_batches, self._protein_trainable
            )
        ]

        smiles_emb = torch.cat(smiles_embs, dim=1)   # (B, Σ smiles_dims)
        protein_emb = torch.cat(protein_embs, dim=1)  # (B, Σ protein_dims)

        return self.fusion(smiles_emb, protein_emb)

    @staticmethod
    def _run_encoder(enc: Encoder, batch, has_trainable: bool) -> torch.Tensor:
        """Run encoder, skipping grad graph for fully-frozen encoders."""
        if has_trainable:
            return enc(batch)
        with torch.no_grad():
            return enc(batch).detach()
