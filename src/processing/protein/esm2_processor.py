from __future__ import annotations

import torch
from transformers import AutoTokenizer

from processing.base import InputProcessor


class ESM2Processor(InputProcessor):
    """Tokenize protein sequences using the ESM-2 tokenizer.

    Produces a dict with ``input_ids`` and ``attention_mask`` tensors,
    ready for the ESM-2 model.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        max_len: int = 1000,
    ) -> None:
        self.model_name = model_name
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def process(self, raw_input: str) -> dict[str, torch.Tensor]:
        """Tokenize a single protein sequence.

        Returns:
            Dict with ``input_ids`` and ``attention_mask``, each of
            shape ``(L,)`` (no batch dimension).
        """
        encoded = self.tokenizer(
            raw_input,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # Squeeze batch dim (1, L) → (L,)
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad a list of tokenized sequences into a batch.

        Returns:
            Dict with ``input_ids`` and ``attention_mask``, each of
            shape ``(B, max_L)`` where ``max_L`` is the longest
            sequence in the batch.
        """
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]

        # Pad to longest sequence in this batch
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
        }
