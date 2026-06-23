from __future__ import annotations

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from processing.base import InputProcessor


class ChemBERTProcessor(InputProcessor):
    """Processor that tokenizes SMILES strings for ChemBERT fine-tuning.

    Uses the HuggingFace tokenizer to produce ``input_ids`` and
    ``attention_mask`` tensors.  Collation pads sequences to the
    longest in the batch.

    Args:
        model_name: HuggingFace model identifier for the tokenizer.
        max_length: Maximum number of tokens per SMILES string.
    """

    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        max_length: int = 256,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self._cache: dict[str, dict[str, torch.Tensor]] = {}
        print(
            f"ChemBERTProcessor: tokenizer loaded for {model_name} "
            f"(max_length={max_length})"
        )

    def build_cache(self, inputs: list[str]) -> None:
        """Pre-compute tokenization in the main process."""
        print(f"Building ChemBERT cache for {len(inputs)} SMILES...")
        for raw_input in tqdm(inputs):
            if raw_input in self._cache:
                continue
            encoded = self.tokenizer(
                raw_input,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=True,
                return_tensors="pt",
            )
            self._cache[raw_input] = {k: v.squeeze(0) for k, v in encoded.items()}

    def process(self, raw_input: str) -> dict[str, torch.Tensor]:
        """Tokenize a single SMILES string.

        Returns:
            Dict with ``input_ids`` and ``attention_mask``, each of
            shape ``(L,)`` where *L* is the tokenized length (no padding).
        """
        if raw_input not in self._cache:
            encoded = self.tokenizer(
                raw_input,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=True,
                return_tensors="pt",
            )
            return {k: v.squeeze(0) for k, v in encoded.items()}
        return {k: v.clone() for k, v in self._cache[raw_input].items()}

    def collate(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Pad and stack tokenized SMILES into a batch.

        Returns:
            Dict with ``input_ids`` and ``attention_mask``, each of
            shape ``(B, L_max)``.
        """
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
        }
