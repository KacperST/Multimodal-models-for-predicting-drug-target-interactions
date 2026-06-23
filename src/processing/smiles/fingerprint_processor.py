from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from processing.base import InputProcessor

# Mapping from short config names → skfp fingerprint classes
_FINGERPRINT_REGISTRY: dict[str, tuple[str, str]] = {
    "ecfp":       ("skfp.fingerprints", "ECFPFingerprint"),
    "morgan":     ("skfp.fingerprints", "ECFPFingerprint"),  # ECFP == Morgan
    "maccs":      ("skfp.fingerprints", "MACCSFingerprint"),
    "rdkit":      ("skfp.fingerprints", "RDKitFingerprint"),
    "atom_pair":  ("skfp.fingerprints", "AtomPairFingerprint"),
    "topological_torsion": ("skfp.fingerprints", "TopologicalTorsionFingerprint"),
    "avalon":     ("skfp.fingerprints", "AvalonFingerprint"),
    "map":        ("skfp.fingerprints", "MAPFingerprint"),
    "secfp":      ("skfp.fingerprints", "SECFPFingerprint"),
    "pubchem":    ("skfp.fingerprints", "PubChemFingerprint"),
    "klekota_roth": ("skfp.fingerprints", "KlekotaRothFingerprint"),
}


def _get_fingerprint_class(name: str):
    """Dynamically import and return an skfp fingerprint class by short name."""
    if name not in _FINGERPRINT_REGISTRY:
        available = ", ".join(sorted(_FINGERPRINT_REGISTRY.keys()))
        raise ValueError(
            f"Unknown fingerprint type: '{name}'. "
            f"Available: {available}"
        )
    module_path, class_name = _FINGERPRINT_REGISTRY[name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class FingerprintProcessor(InputProcessor):
    """Convert SMILES strings to molecular fingerprint vectors.

    Uses ``scikit-fingerprints`` (skfp) for flexible fingerprint
    computation.  Supports 10+ fingerprint types selectable by name.

    Args:
        fp_type: Fingerprint type (e.g. ``"ecfp"``, ``"maccs"``,
            ``"rdkit"``, ``"atom_pair"``).  See ``_FINGERPRINT_REGISTRY``
            for all options.
        fp_params: Extra keyword arguments passed to the skfp
            fingerprint constructor (e.g. ``{"fp_size": 2048, "radius": 2}``
            for ECFP).
    """

    def __init__(
        self,
        fp_type: str = "ecfp",
        fp_params: dict | None = None,
    ) -> None:
        fp_params = fp_params or {}
        fp_cls = _get_fingerprint_class(fp_type)
        self.fp = fp_cls(**fp_params)
        self.fp_type = fp_type

        # Compute one dummy fingerprint to determine output dimension
        self._dim: int | None = None
        self._cache: dict[str, torch.Tensor] = {}

    @property
    def fingerprint_dim(self) -> int:
        """Length of the fingerprint vector."""
        if self._dim is None:
            dummy = self.fp.transform(["C"])
            self._dim = dummy.shape[1]
        return self._dim

    def build_cache(self, inputs: list[str]) -> None:
        """Pre-compute all fingerprints in the main process."""
        print(f"Building {self.fp_type} fingerprint cache for {len(inputs)} SMILES...")
        for raw_input in tqdm(inputs):
            if raw_input in self._cache:
                continue
            arr = self.fp.transform([raw_input])  # (1, n_bits)
            if hasattr(arr, "toarray"):
                arr = arr.toarray()  # sparse → dense
            self._cache[raw_input] = torch.tensor(np.asarray(arr).squeeze(0), dtype=torch.float)

    def process(self, raw_input: str) -> torch.Tensor:
        if raw_input not in self._cache:
            # Fallback just in case (shouldn't happen if build_cache was called)
            arr = self.fp.transform([raw_input])
            if hasattr(arr, "toarray"):
                arr = arr.toarray()
            return torch.tensor(np.asarray(arr).squeeze(0), dtype=torch.float)
        return self._cache[raw_input].clone()

    def collate(self, batch: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(batch)
