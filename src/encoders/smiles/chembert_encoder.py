from __future__ import annotations

import torch

from encoders.base import Encoder


class ChemBERTEncoder(Encoder):
    """Placeholder encoder for ChemBERT / ChemBERTa models.

    To implement, install ``transformers`` and load the model::

        from transformers import AutoModel

        model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    Then implement ``forward`` to:
    1. Pass tokenized SMILES through the transformer.
    2. Extract the ``[CLS]`` token embedding (or apply mean-pooling).
    3. Return a tensor of shape ``(B, output_dim)``.

    The matching processor should use the same tokenizer
    (``AutoTokenizer.from_pretrained(...)``).
    """

    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        out_dim: int = 256,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self._output_dim = out_dim
        raise NotImplementedError(
            "ChemBERTEncoder requires the `transformers` library. "
            "Install with: pip install transformers"
        )

    def forward(self, batch) -> torch.Tensor:
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self._output_dim
