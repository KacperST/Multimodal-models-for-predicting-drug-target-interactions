from __future__ import annotations

import math
from typing import Any

import torch

from processing.base import InputProcessor


class RDKitDescriptorProcessor(InputProcessor):
    """Compute RDKit physicochemical descriptors for SMILES strings.

    Uses ``rdkit.Chem.Descriptors.CalcMolDescriptors`` to produce a
    210-dimensional vector of molecular properties (e.g. MolWt, LogP,
    TPSA, QED, NumHDonors, etc.).

    Optionally loads pre-computed descriptors from a ``.pt`` cache file
    to avoid redundant computation at training time.

    Args:
        cache_path: Optional path to a ``.pt`` file containing a dict
            mapping SMILES → descriptor tensor.  If provided, descriptors
            are loaded from cache instead of being computed on the fly.
    """

    NUM_DESCRIPTORS = 210

    def __init__(self, cache_path: str | None = None, scale: bool = True) -> None:
        self._cache: dict[str, torch.Tensor] = {}
        self.scale = scale
        self._min: torch.Tensor | None = None
        self._range: torch.Tensor | None = None

        if cache_path is not None:
            self._cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            # Determine dim from first cached entry
            first = next(iter(self._cache.values()))
            self._dim = first.shape[0]

            if self.scale:
                print(f"Fitting MinMaxScaler on {len(self._cache)} cached RDKit descriptors...")
                all_tensors = torch.stack(list(self._cache.values()))
                all_tensors = torch.nan_to_num(all_tensors, nan=0.0, posinf=1e5, neginf=-1e5)
                all_tensors = torch.clamp(all_tensors, min=-1e5, max=1e5)
                
                self._min = all_tensors.min(dim=0)[0]
                self._max = all_tensors.max(dim=0)[0]
                self._range = torch.clamp(self._max - self._min, min=1e-8)
        else:
            self._dim = self.NUM_DESCRIPTORS

    @property
    def descriptor_dim(self) -> int:
        """Dimension of the descriptor vector."""
        return self._dim

    def process(self, raw_input: str) -> torch.Tensor:
        """Convert a SMILES string to a descriptor vector.

        Args:
            raw_input: SMILES string.

        Returns:
            Tensor of shape ``(descriptor_dim,)``.
        """
        if raw_input in self._cache:
            return self._cache[raw_input]

        return self._compute(raw_input)

    def _compute(self, smiles: str) -> torch.Tensor:
        """Compute descriptors from scratch using RDKit."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Return zeros for invalid SMILES
            return torch.zeros(self._dim, dtype=torch.float)

        desc_dict = Descriptors.CalcMolDescriptors(mol)
        values = []
        for v in desc_dict.values():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                values.append(0.0)
            else:
                values.append(float(v))

        tensor = torch.tensor(values, dtype=torch.float)
        self._cache[smiles] = tensor
        return tensor

    def collate(self, batch: list[torch.Tensor]) -> torch.Tensor:
        tensor = torch.stack(batch)
        # Zabezpieczenie przed overflow w BatchNorm:
        # 1. Zastąp NaN -> 0, inf -> 1e5, -inf -> -1e5 (dla starych cache'ów)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e5, neginf=-1e5)
        # 2. Przytnij astronomicznie duże wartości (np. 10^21)
        tensor = torch.clamp(tensor, min=-1e5, max=1e5)

        # 3. Zastosuj MinMaxScaler do zakresu [0, 1] jeśli włączony
        if self.scale and self._min is not None and self._range is not None:
            tensor = (tensor - self._min) / self._range
            
        return tensor
