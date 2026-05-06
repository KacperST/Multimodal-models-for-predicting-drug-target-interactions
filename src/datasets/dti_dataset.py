from __future__ import annotations

import torch
from torch.utils.data import Dataset

from processing.base import InputProcessor


class DTIDataset(Dataset):
    """Universal Drug-Target Interaction dataset.

    Supports **multiple processors per modality** — for example, a
    ``GraphProcessor`` and a ``ChemBERTProcessor`` for SMILES at the
    same time.  Each ``__getitem__`` call produces a list of processed
    items for every processor.

    Args:
        smiles_list: Raw SMILES strings.
        sequence_list: Raw protein sequences.
        labels: Target labels (float).
        smiles_processors: One or more SMILES processors.
        protein_processors: One or more protein processors.
    """

    def __init__(
        self,
        smiles_list: list[str],
        sequence_list: list[str],
        labels: list[float],
        smiles_processors: list[InputProcessor] | InputProcessor,
        protein_processors: list[InputProcessor] | InputProcessor,
    ) -> None:
        assert len(smiles_list) == len(sequence_list) == len(labels), (
            "All input lists must have the same length."
        )
        self.smiles_list = smiles_list
        self.sequence_list = sequence_list
        self.labels = labels

        # Normalise to lists
        if isinstance(smiles_processors, InputProcessor):
            smiles_processors = [smiles_processors]
        if isinstance(protein_processors, InputProcessor):
            protein_processors = [protein_processors]

        self.smiles_processors = smiles_processors
        self.protein_processors = protein_processors

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        smi = self.smiles_list[idx]
        seq = self.sequence_list[idx]
        label = self.labels[idx]

        smiles_data = [p.process(smi) for p in self.smiles_processors]
        protein_data = [p.process(seq) for p in self.protein_processors]

        return smiles_data, protein_data, label


def build_collate_fn(
    smiles_processors: list[InputProcessor] | InputProcessor,
    protein_processors: list[InputProcessor] | InputProcessor,
):
    """Build a collate function that delegates batching to the processors.

    Handles both single-processor and multi-processor setups.

    Returns:
        A ``collate_fn`` suitable for ``torch.utils.data.DataLoader``.
    """
    # Normalise to lists
    if isinstance(smiles_processors, InputProcessor):
        smiles_processors = [smiles_processors]
    if isinstance(protein_processors, InputProcessor):
        protein_processors = [protein_processors]

    def collate_fn(batch):
        smiles_items_list, protein_items_list, labels = zip(*batch)

        # smiles_items_list: tuple of N samples, each is a list of K processed items
        # Transpose → list of K lists (one per processor), each containing N items
        smiles_batches = [
            proc.collate([sample[i] for sample in smiles_items_list])
            for i, proc in enumerate(smiles_processors)
        ]
        protein_batches = [
            proc.collate([sample[i] for sample in protein_items_list])
            for i, proc in enumerate(protein_processors)
        ]

        label_batch = torch.tensor(labels, dtype=torch.float)

        # If only one processor per modality, unwrap the list for convenience
        if len(smiles_batches) == 1:
            smiles_batches = smiles_batches[0]
        if len(protein_batches) == 1:
            protein_batches = protein_batches[0]

        return smiles_batches, protein_batches, label_batch

    return collate_fn
