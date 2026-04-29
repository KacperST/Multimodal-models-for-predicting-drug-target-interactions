from __future__ import annotations

from processing.base import InputProcessor


class ESM2Processor(InputProcessor):
    """Placeholder processor for ESM-2 tokenization.

    To implement, install ``transformers`` and load the ESM-2 tokenizer::

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    Then implement ``process`` (tokenize a single sequence) and ``collate``
    (pad a list of tokenized outputs into a batch dict).
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        max_len: int = 1000,
    ) -> None:
        self.model_name = model_name
        self.max_len = max_len
        raise NotImplementedError(
            "ESM2Processor requires the `transformers` library. "
            "Install with: pip install transformers"
        )

    def process(self, raw_input: str):  # type: ignore[override]
        raise NotImplementedError

    def collate(self, batch: list):  # type: ignore[override]
        raise NotImplementedError
