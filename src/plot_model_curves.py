"""Generate ROC, Precision-Recall, and Confusion Matrix plots for selected models.

Loads each specified checkpoint, runs inference on the shared test split,
and saves three publication-quality plots per model.

Usage::

    python src/plot_model_curves.py
    python src/plot_model_curves.py --device cpu
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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
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

# ── Models to evaluate ───────────────────────────────────────────────

MODELS_TO_PLOT = [
    "gcn_vs_cnn",
    "gcn_fp_chembert_vs_cnn_esm2",
    "chembert_vs_esm2",
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
    """Move a tensor, PyG Batch, list, or dict of tensors to *device*."""
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(item, device) for item in obj)
    if hasattr(obj, "to"):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


# ── Inference ────────────────────────────────────────────────────────


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on *loader* and return (labels, probabilities)."""
    model.eval()
    all_labels: list[float] = []
    all_probs: list[float] = []

    pbar = tqdm(loader, desc="Inference", leave=False)
    for smiles_batch, protein_batch, y in pbar:
        smiles_batch = _to_device(smiles_batch, device)
        protein_batch = _to_device(protein_batch, device)
        y = y.to(device)

        logits = model(smiles_batch, protein_batch).squeeze(1)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    return np.asarray(all_labels), np.asarray(all_probs)


# ── Plotting ─────────────────────────────────────────────────────────


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2.2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.2, linestyle="--", label="Random")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved ROC curve     → {output_path}")


def plot_pr_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#059669", lw=2.2, label=f"PR (AUPRC = {pr_auc:.4f})")

    # Baseline: fraction of positive class
    baseline = labels.mean()
    ax.axhline(y=baseline, color="#94a3b8", lw=1.2, linestyle="--", label=f"Baseline ({baseline:.3f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved PR curve      → {output_path}")


def plot_confusion_matrix(
    labels: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    output_path: Path,
    threshold: float = 0.5,
) -> None:
    """Plot and save confusion matrix."""
    preds = (probs > threshold).astype(int)
    cm = confusion_matrix(labels.astype(int), preds)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Inactive", "Active"],
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=True,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved Conf. Matrix  → {output_path}")


# ── Per-model pipeline ───────────────────────────────────────────────


def evaluate_and_plot(
    model_name: str,
    configs_dir: Path,
    checkpoints_dir: Path,
    output_dir: Path,
    split_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
    device_name: str,
) -> None:
    """Load one model, run inference, produce all three plots."""
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

    # Build processors & encoders
    smiles_processors, smiles_encoders = build_smiles_components(cfg)
    protein_processors, protein_encoders = build_protein_components(cfg)

    # Filter for graph processors
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

    # Build test loader
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

    # Build model & load weights
    device = _select_device(device_name)
    total_smiles_dim = sum(enc.output_dim for enc in smiles_encoders)
    total_protein_dim = sum(enc.output_dim for enc in protein_encoders)
    fusion = build_fusion(cfg, total_smiles_dim, total_protein_dim)
    model = MultimodalDTI(smiles_encoders, protein_encoders, fusion)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    model = model.to(device)

    # Run inference
    labels, probs = collect_predictions(model, test_loader, device)
    print(f"  Test samples: {len(labels)}, Positive rate: {labels.mean():.3f}")

    # Generate plots
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_curve(labels, probs, model_name, model_dir / "roc_curve.png")
    plot_pr_curve(labels, probs, model_name, model_dir / "pr_curve.png")
    plot_confusion_matrix(labels, probs, model_name, model_dir / "confusion_matrix.png")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ROC, PR, and Confusion Matrix plots for selected models"
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=str(ROOT_DIR / "configs"),
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=str(ROOT_DIR / "checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "logs" / "plots"),
        help="Directory where plot images are saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )
    args = parser.parse_args()

    configs_dir = _resolve_path(args.configs_dir)
    checkpoints_dir = _resolve_path(args.checkpoints_dir)
    output_dir = _resolve_path(args.output_dir)

    # Load dataset once and compute shared split
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
        evaluate_and_plot(
            model_name=model_name,
            configs_dir=configs_dir,
            checkpoints_dir=checkpoints_dir,
            output_dir=output_dir,
            split_frames=split_frames,
            device_name=args.device,
        )

    print(f"\n✓ All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
