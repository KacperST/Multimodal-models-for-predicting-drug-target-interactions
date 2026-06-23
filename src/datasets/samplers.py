"""Custom batch samplers for efficient multi-modal DTI training."""
from __future__ import annotations

import random


class ProteinGroupedBatchSampler:
    """BatchSampler that groups samples by protein sequence.

    With only ~3,580 unique proteins but 315K+ samples, each protein
    appears ~88 times (paired with different SMILES).  Standard random
    batching puts ~247 unique proteins in each batch of 256, forcing
    ESM-2 to run a separate forward pass for each.

    This sampler groups samples so that consecutive batches share the
    same protein(s).  Combined with ESM2Encoder's deduplication logic,
    ESM-2 processes only ~3 unique proteins per batch instead of ~247
    — an **~85× reduction** in transformer forward passes.

    Training quality is preserved because:
    - Groups are shuffled each epoch (different protein order)
    - Within each group, SMILES pairings are shuffled
    - Over a full epoch, all data is seen exactly once
    - Adam optimizer handles the per-batch gradient noise well

    Args:
        protein_sequences: List of protein sequences (one per sample,
            in dataset order).
        batch_size: Number of samples per batch.
        drop_last: Whether to drop the last incomplete batch.
    """

    def __init__(
        self,
        protein_sequences: list[str],
        batch_size: int,
        drop_last: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group sample indices by protein sequence
        groups: dict[str, list[int]] = {}
        for i, seq in enumerate(protein_sequences):
            groups.setdefault(seq, []).append(i)
        self.groups = list(groups.values())

    def __iter__(self):
        # Shuffle group order and within-group order each epoch
        groups = [g.copy() for g in self.groups]
        random.shuffle(groups)
        for g in groups:
            random.shuffle(g)

        # Flatten into a single index stream and yield batches
        indices = [idx for g in groups for idx in g]
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        total = sum(len(g) for g in self.groups)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size
