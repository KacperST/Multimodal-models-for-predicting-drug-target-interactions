from __future__ import annotations

import torch
from ogb.utils import smiles2graph
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from processing.base import InputProcessor


class GraphProcessor(InputProcessor):
    """Convert SMILES strings to PyTorch Geometric graph objects.

    Uses OGB featurization to produce rich atom/bond features.
    Graphs are pre-computed and cached for efficiency.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Data] = {}

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def build_cache(self, smiles_list: list[str]) -> None:
        """Pre-compute graphs for all unique SMILES and store in cache.

        Invalid SMILES are silently skipped — use ``valid_smiles`` to check
        which molecules were successfully converted.
        """
        print(f"Building graph cache for {len(smiles_list)} SMILES...")
        for smiles in tqdm(smiles_list):
            if smiles in self._cache:
                continue
            try:
                graph = smiles2graph(smiles)
                data = Data(
                    x=torch.tensor(graph["node_feat"], dtype=torch.long),
                    edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
                    edge_attr=torch.tensor(graph["edge_feat"], dtype=torch.long),
                )
                self._cache[smiles] = data
            except Exception:
                continue

    @property
    def valid_smiles(self) -> set[str]:
        """Set of SMILES that were successfully converted to graphs."""
        return set(self._cache.keys())

    # ------------------------------------------------------------------
    # InputProcessor interface
    # ------------------------------------------------------------------

    def process(self, raw_input: str) -> Data:
        """Look up a pre-computed graph by its SMILES string."""
        if raw_input not in self._cache:
            raise KeyError(
                f"SMILES '{raw_input[:50]}...' not in cache. "
                "Call build_cache() first."
            )
        return self._cache[raw_input].clone()

    def collate(self, batch: list[Data]) -> Batch:
        """Create a PyG ``Batch`` from a list of ``Data`` objects."""
        return Batch.from_data_list(batch)
