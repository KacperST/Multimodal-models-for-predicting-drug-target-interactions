from __future__ import annotations

import torch

from processing.base import InputProcessor


class CachedESM2Processor(InputProcessor):
    """Processor that looks up pre-computed ESM-2 embeddings.

    Instead of tokenizing sequences at runtime, this processor loads
    a ``{sequence → tensor}`` mapping from a ``.pt`` file produced by
    ``precompute_esm2.py`` and simply returns the cached embedding.

    Args:
        cache_path: Path to the ``.pt`` file with pre-computed embeddings.
    """

    def __init__(self, cache_path: str) -> None:
        self.cache: dict[str, torch.Tensor] = torch.load(
            cache_path, map_location="cpu", weights_only=True
        )
        sample = next(iter(self.cache.values()))
        self.embedding_dim = sample.shape[0]
        print(
            f"CachedESM2Processor: loaded {len(self.cache)} embeddings "
            f"(dim={self.embedding_dim}) from {cache_path}"
        )

    def process(self, raw_input: str) -> torch.Tensor:
        """Look up the pre-computed embedding for a protein sequence.

        Returns:
            Tensor of shape ``(H,)`` — the cached ESM-2 embedding.

        Raises:
            KeyError: If the sequence was not pre-computed.
        """
        try:
            return self.cache[raw_input]
        except KeyError:
            raise KeyError(
                f"Sequence not found in ESM-2 cache (len={len(raw_input)}). "
                f"Re-run precompute_esm2.py to include all sequences."
            )

    def collate(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """Stack cached embeddings into a batch.

        Returns:
            Tensor of shape ``(B, H)``.
        """
        return torch.stack(batch, dim=0)
