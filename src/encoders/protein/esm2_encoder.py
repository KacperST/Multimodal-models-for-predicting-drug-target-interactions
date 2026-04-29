from __future__ import annotations

import torch

from encoders.base import Encoder


class ESM2Encoder(Encoder):
    """Placeholder encoder for ESM-2 protein language model.

    To implement, install ``transformers`` and load the model::

        from transformers import AutoModel

        model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

    Then implement ``forward`` to:
    1. Pass tokenized sequences through the ESM-2 model.
    2. Apply mean-pooling over non-padding tokens.
    3. Optionally project to ``out_dim`` with a linear layer.
    4. Return a tensor of shape ``(B, output_dim)``.

    The matching processor (``ESM2Processor``) should use the same tokenizer.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        out_dim: int = 256,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self._output_dim = out_dim
        raise NotImplementedError(
            "ESM2Encoder requires the `transformers` library. "
            "Install with: pip install transformers"
        )

    def forward(self, batch) -> torch.Tensor:
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self._output_dim
