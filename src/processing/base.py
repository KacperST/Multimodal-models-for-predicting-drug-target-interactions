from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class InputProcessor(ABC):
    """Abstract base class for input data processing.

    Subclasses define how raw inputs (SMILES strings, protein sequences)
    are converted into tensors and how individual samples are collated
    into batches.
    """

    @abstractmethod
    def process(self, raw_input: str) -> Any:
        """Convert a single raw input string into a model-ready representation.

        Args:
            raw_input: Raw SMILES string or protein sequence.

        Returns:
            Processed representation (tensor, PyG Data, dict, etc.)
        """
        ...

    def build_cache(self, inputs: list[str]) -> None:
        """Pre-compute and cache representations for a list of inputs.
        
        Optional method. Implement this if processing is CPU intensive
        and should be done in the main process before DataLoader forks.
        """
        pass

    @abstractmethod
    def collate(self, batch: list[Any]) -> Any:
        """Combine a list of processed samples into a single batched tensor.

        Args:
            batch: List of outputs from ``process``.

        Returns:
            Batched representation ready for the encoder.
        """
        ...
