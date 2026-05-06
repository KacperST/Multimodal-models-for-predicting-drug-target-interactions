"""Pre-compute ChemBERT embeddings for all unique SMILES strings.

Run this **once** before training.  The script extracts unique SMILES
from the clean Parquet dataset, runs them through the ChemBERT model in
batches, and saves a mapping ``{smiles → embedding}`` as a ``.pt``
file.

Usage::

    python precompute_chembert.py
    python precompute_chembert.py --model seyonec/ChemBERTa-zinc-base-v1 --batch-size 64
    python precompute_chembert.py --output datasets/chembert_embeddings.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent


def _mean_pool(
    last_hidden: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
    summed = (last_hidden * mask).sum(dim=1)     # (B, H)
    counts = mask.sum(dim=1).clamp(min=1)        # (B, 1)
    return summed / counts                       # (B, H)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute ChemBERT embeddings for unique SMILES strings"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT_DIR / "datasets" / "clean.parquet"),
        help="Path to the clean Parquet dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="seyonec/ChemBERTa-zinc-base-v1",
        help="HuggingFace ChemBERT model identifier",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Maximum sequence length (tokens)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pt file path (default: datasets/chemberta_<model_short>.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on (auto / cpu / cuda / cuda:0 …)",
    )
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ── Load data ────────────────────────────────────────────────
    df = pl.read_parquet(args.data)
    sequences = df["Ligand SMILES"].unique().to_list()
    print(f"Unique SMILES sequences: {len(sequences)}")

    # ── Load model + tokenizer ───────────────────────────────────
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    # ── Pre-compute ──────────────────────────────────────────────
    embeddings: dict[str, torch.Tensor] = {}

    for start in tqdm(range(0, len(sequences), args.batch_size), desc="ChemBERT"):
        batch_seqs = sequences[start : start + args.batch_size]

        encoded = tokenizer(
            batch_seqs,
            truncation=True,
            max_length=args.max_len,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = pooled.cpu()  # always store on CPU

        for seq, emb in zip(batch_seqs, pooled):
            embeddings[seq] = emb

    # ── Save ─────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        # Derive short name, e.g. "ChemBERTa-zinc-base-v1"
        short = args.model.split("/")[-1]
        out_path = ROOT_DIR / "datasets" / f"{short}.pt"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, str(out_path))

    sample_dim = next(iter(embeddings.values())).shape[0]
    print(f"\nSaved {len(embeddings)} embeddings (dim={sample_dim}) to: {out_path}")


if __name__ == "__main__":
    main()
