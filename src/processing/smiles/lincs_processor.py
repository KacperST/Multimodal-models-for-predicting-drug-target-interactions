from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from processing.base import InputProcessor


class LincsProcessor(InputProcessor):
    """Process SMILES strings into LINCS L1000 expression profile vectors.

    Performs a two-step lookup: SMILES → pert_id → L1000 profile (978-d
    z-score vector).  This allows it to be used as a standard SMILES
    processor within the existing multi-encoder pipeline — no changes
    to ``DTIDataset`` or ``collate_fn`` are needed.

    Args:
        cache_path: Path to ``lincs_profiles.pt`` — a dict mapping
            ``{pert_id: torch.Tensor(978,)}``.
        smiles_pert_map_path: Path to ``smiles_to_pert_id.json`` — a dict
            mapping ``{canonical_smiles: pert_id}``.
    """

    def __init__(
        self,
        cache_path: str,
        smiles_pert_map_path: str,
    ) -> None:
        # Load pert_id → profile cache
        self._profiles: dict[str, torch.Tensor] = torch.load(
            cache_path, weights_only=True
        )

        # Load smiles → pert_id mapping
        with open(smiles_pert_map_path) as f:
            self._smiles_to_pert: dict[str, str] = json.load(f)

        # Determine profile dimension from the first entry
        first_profile = next(iter(self._profiles.values()))
        self._profile_dim: int = first_profile.shape[0]

        # Zero vector for missing lookups (should not happen with proper
        # dataset filtering, but provides a safe fallback)
        self._zero = torch.zeros(self._profile_dim, dtype=torch.float32)

    @property
    def profile_dim(self) -> int:
        """Dimension of the L1000 profile vector (typically 978)."""
        return self._profile_dim

    def process(self, raw_input: str) -> torch.Tensor:
        """Look up the L1000 profile for a SMILES string.

        Args:
            raw_input: SMILES string.

        Returns:
            ``torch.Tensor`` of shape ``(profile_dim,)`` with z-scores.
        """
        pert_id = self._smiles_to_pert.get(raw_input)
        if pert_id is None:
            return self._zero.clone()
        profile = self._profiles.get(pert_id)
        if profile is None:
            return self._zero.clone()
        return profile.clone()

    def collate(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """Stack a list of profile tensors into a batch.

        Args:
            batch: List of ``(profile_dim,)`` tensors.

        Returns:
            ``torch.Tensor`` of shape ``(B, profile_dim)``.
        """
        return torch.stack(batch)
