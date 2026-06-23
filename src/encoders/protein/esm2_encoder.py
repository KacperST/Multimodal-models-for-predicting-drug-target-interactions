from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BitsAndBytesConfig

from encoders.base import Encoder


class ESM2Encoder(Encoder):
    """ESM-2 encoder with QLoRA (4-bit quantisation + LoRA adapters).

    Loads the full ESM-2 transformer backbone in 4-bit precision via
    ``bitsandbytes`` and attaches lightweight LoRA adapters for
    parameter-efficient fine-tuning.  A trainable projection head maps
    the pooled representation to the desired ``out_dim``.

    Args:
        model_name: HuggingFace model identifier.
        out_dim: Output embedding dimension after projection.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout applied to LoRA layers.
        lora_target_modules: Names of modules to attach LoRA to.
        device: Target device for model placement (e.g. ``"cuda:0"``).
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        out_dim: int = 256,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        device: str = "cuda:0",
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()

        if lora_target_modules is None:
            lora_target_modules=["query", "key", "value", "dense"]
        base_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": device},
            add_pooling_layer=False,
        )

        # Gradient checkpointing trades compute for memory.
        # Disable on high-VRAM GPUs (GH200 120GB) for ~1.5x speedup.
        if gradient_checkpointing:
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # ── Attach LoRA adapters ─────────────────────────────────
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        self.model = get_peft_model(base_model, lora_config)

        hidden_size = base_model.config.hidden_size
        self.model.print_trainable_parameters()

        # ── Projection head ──────────────────────────────────────
        if out_dim is not None and out_dim != hidden_size:
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
            ).to(device)
            self._output_dim = out_dim
        else:
            self.projection = None
            self._output_dim = hidden_size

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode tokenized protein sequences through the QLoRA ESM-2 backbone.

        When a batch contains duplicate protein sequences (common with
        protein-grouped sampling), ESM-2 is run only on unique sequences
        and the results are expanded back.  Gradients flow correctly
        through the indexing operation.

        Args:
            batch: Dict with ``input_ids`` and ``attention_mask``,
                each of shape ``(B, L)``.

        Returns:
            Tensor of shape ``(B, output_dim)``.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Deduplicate: find unique sequences to skip redundant forward passes
        unique_ids, inverse = torch.unique(
            input_ids, dim=0, return_inverse=True
        )
        n_unique = unique_ids.shape[0]

        if n_unique < input_ids.shape[0]:
            # Pick attention mask for first occurrence of each unique seq
            first_idx = [
                torch.where(inverse == i)[0][0].item()
                for i in range(n_unique)
            ]
            unique_mask = attention_mask[first_idx]
            pooled = self._encode(unique_ids, unique_mask)
            return pooled[inverse]  # expand back to (B, D)

        return self._encode(input_ids, attention_mask)

    def _encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run ESM-2 backbone + projection on a batch of sequences."""
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Mean-pool over non-padding tokens
        hidden = outputs.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Project to output dim (cast to float32 for projection head)
        pooled = pooled.float()
        if self.projection is not None:
            return self.projection(pooled)
        return pooled

    @property
    def output_dim(self) -> int:
        return self._output_dim
