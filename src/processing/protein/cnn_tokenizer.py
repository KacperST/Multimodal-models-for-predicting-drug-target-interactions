from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from processing.base import InputProcessor

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


class CNNTokenizer(InputProcessor):
    """Character-level tokenizer for protein sequences.

    Produces integer token IDs suitable for ``nn.Embedding`` layers.
    Index 0 is reserved for padding.
    """

    def __init__(
        self,
        max_len: int = 1000,
        vocab: str | None = None,
    ) -> None:
        self.max_len = max_len
        chars = sorted(set(vocab)) if vocab else sorted(set(STANDARD_AA))
        # Index 0 is reserved for padding
        self.char_to_idx: dict[str, int] = {
            c: i + 1 for i, c in enumerate(chars)
        }
        self.vocab_size = len(self.char_to_idx) + 1  # +1 for padding token

    def process(self, raw_input: str) -> torch.Tensor:
        seq = raw_input.upper()[: self.max_len]
        ids = [self.char_to_idx.get(c, 0) for c in seq]
        return torch.tensor(ids, dtype=torch.long)

    def collate(self, batch: list[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(batch, batch_first=True, padding_value=0)
