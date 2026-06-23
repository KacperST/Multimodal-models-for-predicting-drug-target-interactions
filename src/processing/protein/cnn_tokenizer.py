from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

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
        self._cache: dict[str, torch.Tensor] = {}

    def build_cache(self, inputs: list[str]) -> None:
        """Pre-compute CNN tokens in the main process."""
        print(f"Building CNN tokenizer cache for {len(inputs)} proteins...")
        for raw_input in tqdm(inputs):
            if raw_input in self._cache:
                continue
            seq = raw_input.upper()[: self.max_len]
            ids = [self.char_to_idx.get(c, 0) for c in seq]
            self._cache[raw_input] = torch.tensor(ids, dtype=torch.long)

    def process(self, raw_input: str) -> torch.Tensor:
        if raw_input not in self._cache:
            seq = raw_input.upper()[: self.max_len]
            ids = [self.char_to_idx.get(c, 0) for c in seq]
            return torch.tensor(ids, dtype=torch.long)
        return self._cache[raw_input].clone()

    def collate(self, batch: list[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(batch, batch_first=True, padding_value=0)
