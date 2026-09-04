"""Pre-compute RDKit physicochemical descriptors for all unique SMILES.

Reads a clean Parquet dataset, computes 210 RDKit descriptors for each
unique molecule, and saves the result as a ``.pt`` cache file (dict
mapping SMILES → Tensor).

Usage::

    uv run precompute_rdkit_descriptors.py                          # default paths
    uv run precompute_rdkit_descriptors.py --input datasets/clean.parquet --output datasets/rdkit_descriptors.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import polars as pl
import torch
from tqdm import tqdm


def compute_descriptors(smiles: str) -> torch.Tensor | None:
    """Compute all RDKit molecular descriptors for a single SMILES string."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    desc_dict = Descriptors.CalcMolDescriptors(mol)
    values = []
    for v in desc_dict.values():
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            values.append(0.0)
        else:
            values.append(float(v))

    return torch.tensor(values, dtype=torch.float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute RDKit descriptors")
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/clean.parquet",
        help="Path to the clean Parquet dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/rdkit_descriptors.pt",
        help="Output path for the .pt cache file",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    input_path = root / args.input
    output_path = root / args.output

    print(f"Loading dataset from: {input_path}")
    df = pl.read_parquet(str(input_path))

    unique_smiles = df["Ligand SMILES"].unique().sort().to_list()
    print(f"Unique SMILES: {len(unique_smiles)}")

    cache: dict[str, torch.Tensor] = {}
    failed = 0

    for smi in tqdm(unique_smiles, desc="Computing RDKit descriptors"):
        tensor = compute_descriptors(smi)
        if tensor is not None:
            cache[smi] = tensor
        else:
            failed += 1

    print(f"Computed descriptors for {len(cache)} molecules ({failed} failed)")

    if cache:
        first = next(iter(cache.values()))
        print(f"Descriptor dimension: {first.shape[0]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, str(output_path))
    print(f"Saved cache to: {output_path}")


if __name__ == "__main__":
    main()
