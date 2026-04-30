from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from encoders.base import Encoder


class ESM2Encoder(Encoder):
    """ESM-2 protein language model encoder.

    Loads a pretrained ESM-2 model and extracts per-protein embeddings
    via mean-pooling over non-padding tokens.  An optional linear
    projection maps from the ESM-2 hidden dimension to ``out_dim``.

    Args:
        model_name: HuggingFace model identifier.
        out_dim: Output embedding dimension. If ``None``, uses the
            native ESM-2 hidden size (e.g. 1280 for t33_650M).
        freeze: If ``True``, freeze all ESM-2 transformer weights
            (only the projection head is trainable).
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        out_dim: int | None = 256,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        self.esm2 = AutoModel.from_pretrained(model_name)
        hidden_size = self.esm2.config.hidden_size  # e.g. 1280

        if freeze:
            for param in self.esm2.parameters():
                param.requires_grad = False

        # Optional projection
        if out_dim is not None and out_dim != hidden_size:
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
            )
            self._output_dim = out_dim
        else:
            self.projection = None
            self._output_dim = hidden_size

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a batch of tokenized protein sequences.

        Args:
            batch: Dict with ``input_ids`` and ``attention_mask``,
                each of shape ``(B, L)``.

        Returns:
            Tensor of shape ``(B, output_dim)``.
        """
        outputs = self.esm2(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        # Mean-pool over non-padding tokens
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        mask = batch["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        summed = (last_hidden * mask).sum(dim=1)  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        pooled = summed / counts  # (B, H)

        if self.projection is not None:
            pooled = self.projection(pooled)

        return pooled

    @property
    def output_dim(self) -> int:
        return self._output_dim
