"""Visualise the latent space of selected DTI models using t-SNE.

Extracts embeddings from the penultimate fusion layer (just before the
final Linear → 1 prediction head) and reduces them to 2-D with t-SNE.
Each point is coloured by its ground-truth class (Active / Inactive).

Usage::

    uv run python src/plot_latent_space.py
    uv run python src/plot_latent_space.py --device cuda
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import yaml
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.transform import train_val_test_split_scaffold
from datasets.dti_dataset import DTIDataset, build_collate_fn
from main import (
    ROOT_DIR,
    MultimodalDTI,
    build_fusion,
    build_protein_components,
    build_smiles_components,
    load_data,
)
from processing.smiles.graph_processor import GraphProcessor

# ── Models to visualise ──────────────────────────────────────────────

MODELS_TO_PLOT = [
    "gcn_chembert_vs_cnn",
    "gcn_fp_chembert_vs_cnn_esm2",
]

# ── Helpers ──────────────────────────────────────────────────────────


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT_DIR / path


def _load_config(config_path: Path) -> dict:
    with config_path.open() as handle:
        return yaml.safe_load(handle)


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _to_device(obj, device: torch.device):
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(item, device) for item in obj)
    if hasattr(obj, "to"):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def _load_checkpoint(model: MultimodalDTI, checkpoint_path: Path) -> None:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=False)


# ── Embedding extraction ─────────────────────────────────────────────


@torch.no_grad()
def collect_embeddings(
    model: MultimodalDTI,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model and capture the penultimate fusion layer activations.

    We register a forward hook on the last ReLU (or Dropout) layer
    before the final ``Linear(hidden → 1)`` in the fusion MLP.
    This gives us the most informative latent representation.

    Returns:
        (embeddings, labels) — both as numpy arrays.
    """
    model.eval()

    # Find the penultimate layer in fusion.classifier
    # The Sequential is: [Linear, BN, ReLU, Dropout, ..., Linear(→1)]
    # We hook the layer just before the last Linear.
    classifier = model.fusion.classifier
    hook_layer = classifier[-2]  # Dropout before final Linear

    captured: list[torch.Tensor] = []

    def _hook_fn(_module, _input, output):
        captured.append(output.detach().cpu())

    handle = hook_layer.register_forward_hook(_hook_fn)

    all_labels: list[float] = []

    pbar = tqdm(loader, desc="Extracting embeddings", leave=False)
    for smiles_batch, protein_batch, y in pbar:
        smiles_batch = _to_device(smiles_batch, device)
        protein_batch = _to_device(protein_batch, device)
        _ = model(smiles_batch, protein_batch)
        all_labels.extend(y.numpy().tolist())

    handle.remove()

    embeddings = torch.cat(captured, dim=0).numpy()
    labels = np.asarray(all_labels)
    return embeddings, labels


# ── Plotting ─────────────────────────────────────────────────────────


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    output_path: Path,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    """Reduce embeddings to 2-D with t-SNE and plot."""
    print(f"  Running t-SNE (perplexity={perplexity}) on {embeddings.shape} ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    coords = tsne.fit_transform(embeddings)

    active_mask = labels == 1
    inactive_mask = ~active_mask

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.scatter(
        coords[inactive_mask, 0],
        coords[inactive_mask, 1],
        c="#94a3b8",
        s=8,
        alpha=0.45,
        label=f"Inactive ({inactive_mask.sum():,})",
        rasterized=True,
    )
    ax.scatter(
        coords[active_mask, 0],
        coords[active_mask, 1],
        c="#2563eb",
        s=8,
        alpha=0.55,
        label=f"Active ({active_mask.sum():,})",
        rasterized=True,
    )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(
        f"Latent Space (t-SNE) — {model_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, markerscale=3)
    ax.grid(True, alpha=0.15)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved t-SNE plot → {output_path}")


# ── Per-model pipeline ───────────────────────────────────────────────


def extract_and_plot(
    model_name: str,
    configs_dir: Path,
    checkpoints_dir: Path,
    output_dir: Path,
    split_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
    device_name: str,
) -> None:
    config_path = configs_dir / f"{model_name}.yaml"
    checkpoint_path = checkpoints_dir / f"{model_name}.pt"

    if not config_path.exists():
        print(f"  ✗ Config not found: {config_path}")
        return
    if not checkpoint_path.exists():
        print(f"  ✗ Checkpoint not found: {checkpoint_path}")
        return

    cfg = _load_config(config_path)
    train_cfg = cfg["training"]

    smiles_processors, smiles_encoders = build_smiles_components(cfg)
    protein_processors, protein_encoders = build_protein_components(cfg)

    train_df, val_df, test_df = split_frames
    for processor in smiles_processors:
        if isinstance(processor, GraphProcessor):
            all_smiles = (
                train_df["Ligand SMILES"].unique().sort().to_list()
                + val_df["Ligand SMILES"].unique().sort().to_list()
                + test_df["Ligand SMILES"].unique().sort().to_list()
            )
            processor.build_cache(list(dict.fromkeys(all_smiles)))
            valid_smiles = processor.valid_smiles
            train_df = train_df.filter(pl.col("Ligand SMILES").is_in(valid_smiles))
            val_df = val_df.filter(pl.col("Ligand SMILES").is_in(valid_smiles))
            test_df = test_df.filter(pl.col("Ligand SMILES").is_in(valid_smiles))

    batch_size = train_cfg.get("batch_size", 256)
    num_workers = train_cfg.get("num_workers", 4)
    if "OMP_NUM_THREADS" in os.environ:
        num_workers = 0

    test_ds = DTIDataset(
        smiles_list=test_df["Ligand SMILES"].to_list(),
        sequence_list=test_df["Full_Protein_Sequence"].to_list(),
        labels=test_df["is_active"].cast(pl.Float64).to_list(),
        smiles_processors=smiles_processors,
        protein_processors=protein_processors,
    )
    collate_fn = build_collate_fn(smiles_processors, protein_processors)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    device = _select_device(device_name)
    total_smiles_dim = sum(enc.output_dim for enc in smiles_encoders)
    total_protein_dim = sum(enc.output_dim for enc in protein_encoders)
    fusion = build_fusion(cfg, total_smiles_dim, total_protein_dim)
    model = MultimodalDTI(smiles_encoders, protein_encoders, fusion)
    _load_checkpoint(model, checkpoint_path)
    model = model.to(device)

    embeddings, labels = collect_embeddings(model, test_loader, device)
    print(f"  Embeddings shape: {embeddings.shape}, Positive rate: {labels.mean():.3f}")

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    plot_tsne(embeddings, labels, model_name, model_dir / "tsne_latent_space.png")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise latent space of DTI models with t-SNE"
    )
    parser.add_argument(
        "--configs-dir", type=str, default=str(ROOT_DIR / "configs"),
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default=str(ROOT_DIR / "checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "logs" / "plots"),
        help="Directory where plot images are saved.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
    )
    args = parser.parse_args()

    configs_dir = _resolve_path(args.configs_dir)
    checkpoints_dir = _resolve_path(args.checkpoints_dir)
    output_dir = _resolve_path(args.output_dir)

    reference_cfg = _load_config(sorted(configs_dir.glob("*.yaml"))[0])
    shared_df = load_data(reference_cfg["data"])
    print(f"Loaded dataset: {shared_df.height} rows")

    split_frames = train_val_test_split_scaffold(
        shared_df,
        proportions=reference_cfg["data"].get("split_ratios", [0.7, 0.1, 0.2]),
    )

    for model_name in MODELS_TO_PLOT:
        print(f"\n{'═' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'═' * 60}")
        extract_and_plot(
            model_name=model_name,
            configs_dir=configs_dir,
            checkpoints_dir=checkpoints_dir,
            output_dir=output_dir,
            split_frames=split_frames,
            device_name=args.device,
        )

    print(f"\n✓ All t-SNE plots saved to {output_dir}")


if __name__ == "__main__":
    main()
