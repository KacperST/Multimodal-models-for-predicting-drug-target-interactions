from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from processing.base import InputProcessor


class FingerprintProcessor(InputProcessor):
    """Convert SMILES strings to Morgan / ECFP fingerprint bit-vectors."""

    def __init__(self, radius: int = 2, n_bits: int = 2048) -> None:
        self.radius = radius
        self.n_bits = n_bits

    @property
    def fingerprint_dim(self) -> int:
        """Length of the fingerprint vector."""
        return self.n_bits

    def process(self, raw_input: str) -> torch.Tensor:
        mol = Chem.MolFromSmiles(raw_input)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {raw_input[:50]}")
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.radius, nBits=self.n_bits
        )
        return torch.tensor(list(fp), dtype=torch.float)

    def collate(self, batch: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(batch)
