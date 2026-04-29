"""Multimodal DTI — main entry point.

Reads a YAML experiment config, builds all components (processors,
encoders, fusion, dataset, trainer), and runs the full pipeline.

Supports **multiple encoders per modality** — e.g. GCN + ChemBERT
for SMILES simultaneously.

Usage::

    python main.py                          # uses configs/default.yaml
    python main.py --config configs/my.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.loader import load_bindingdb_data
from data.transform import (
    add_activity_label,
    remove_cx_notation,
    remove_duplicates,
    remove_nulls,
    train_test_val_split,
    tranform_ki_to_log_ki,
)
from datasets.dti_dataset import DTIDataset, build_collate_fn
from encoders.protein.cnn_encoder import ProteinCNNEncoder
from encoders.smiles.fingerprint_mlp_encoder import FingerprintMLPEncoder
from encoders.smiles.gcn_encoder import GCNEncoder
from fusion.cross_attention_fusion import CrossAttentionFusion
from fusion.mlp_fusion import MLPFusion
from models.multimodal import MultimodalDTI
from processing.base import InputProcessor
from processing.protein.cnn_tokenizer import CNNTokenizer
from processing.smiles.fingerprint_processor import FingerprintProcessor
from processing.smiles.graph_processor import GraphProcessor
from training.metrics import compute_confusion_matrix
from training.trainer import Trainer

ROOT_DIR = Path(__file__).resolve().parent


# ── Single-encoder factory ───────────────────────────────────────────

def _build_one_smiles(enc_cfg: dict) -> tuple[InputProcessor, nn.Module]:
    """Build a single SMILES (processor, encoder) pair."""
    enc_type = enc_cfg["type"]
    params = enc_cfg.get("params", {})

    if enc_type == "gcn":
        processor = GraphProcessor()
        encoder = GCNEncoder(
            hidden_dim=params.get("hidden_dim", 256),
            num_layers=params.get("num_layers", 3),
        )
    elif enc_type == "fingerprint_mlp":
        n_bits = params.get("n_bits", 2048)
        processor = FingerprintProcessor(
            radius=params.get("radius", 2), n_bits=n_bits,
        )
        encoder = FingerprintMLPEncoder(
            input_dim=n_bits,
            hidden_dim=params.get("hidden_dim", 512),
            out_dim=params.get("out_dim", 256),
            dropout=params.get("dropout", 0.2),
        )
    elif enc_type == "chembert":
        # When you implement ChemBERT, import and instantiate here.
        raise NotImplementedError(
            "ChemBERT encoder is a placeholder. "
            "Implement ChemBERTEncoder and its matching processor."
        )
    else:
        raise ValueError(f"Unknown smiles encoder type: {enc_type}")

    return processor, encoder


def _build_one_protein(enc_cfg: dict) -> tuple[InputProcessor, nn.Module]:
    """Build a single protein (processor, encoder) pair."""
    enc_type = enc_cfg["type"]
    params = enc_cfg.get("params", {})

    if enc_type == "cnn":
        max_len = params.get("max_seq_len", 1000)
        processor = CNNTokenizer(max_len=max_len)
        encoder = ProteinCNNEncoder(
            vocab_size=processor.vocab_size,
            embed_dim=params.get("embed_dim", 256),
            num_filters=params.get("num_filters", 128),
            kernel_sizes=params.get("kernel_sizes", [3, 7, 15]),
        )
    elif enc_type == "esm2":
        raise NotImplementedError(
            "ESM-2 encoder is a placeholder. "
            "Implement ESM2Encoder and its matching processor."
        )
    else:
        raise ValueError(f"Unknown protein encoder type: {enc_type}")

    return processor, encoder


# ── Multi-encoder factory ────────────────────────────────────────────

def build_smiles_components(cfg: dict):
    """Build all SMILES (processor, encoder) pairs from config.

    Returns:
        (list[InputProcessor], list[Encoder])
    """
    specs = cfg["smiles_encoders"]
    # Support legacy single-encoder format
    if isinstance(specs, dict):
        specs = [specs]

    processors, encoders = [], []
    for spec in specs:
        proc, enc = _build_one_smiles(spec)
        processors.append(proc)
        encoders.append(enc)
    return processors, encoders


def build_protein_components(cfg: dict):
    """Build all protein (processor, encoder) pairs from config.

    Returns:
        (list[InputProcessor], list[Encoder])
    """
    specs = cfg["protein_encoders"]
    if isinstance(specs, dict):
        specs = [specs]

    processors, encoders = [], []
    for spec in specs:
        proc, enc = _build_one_protein(spec)
        processors.append(proc)
        encoders.append(enc)
    return processors, encoders


def build_fusion(cfg: dict, smiles_dim: int, protein_dim: int):
    """Build fusion module from config."""
    fus_type = cfg["fusion"]["type"]
    params = cfg["fusion"].get("params", {})

    if fus_type == "mlp":
        return MLPFusion(
            smiles_dim=smiles_dim,
            protein_dim=protein_dim,
            hidden_dims=params.get("hidden_dims", [256, 64]),
            dropout=params.get("dropout", 0.3),
        )
    elif fus_type == "cross_attention":
        return CrossAttentionFusion(
            smiles_dim=smiles_dim,
            protein_dim=protein_dim,
            proj_dim=params.get("proj_dim", 256),
            num_heads=params.get("num_heads", 4),
            hidden_dim=params.get("hidden_dim", 128),
            dropout=params.get("dropout", 0.3),
        )
    else:
        raise ValueError(f"Unknown fusion type: {fus_type}")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal DTI training")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    # ── Device ───────────────────────────────────────────────────
    dev_str = train_cfg.get("device", "auto")
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)
    print(f"Using device: {device}")

    # ── 1. Load and clean data ───────────────────────────────────
    data_path = str(ROOT_DIR / data_cfg["path"])
    df = load_bindingdb_data(data_path)
    df = remove_cx_notation(df)
    df = remove_nulls(df)
    df = remove_duplicates(df)
    df = tranform_ki_to_log_ki(df)
    df = add_activity_label(df, pki_threshold=data_cfg.get("pki_threshold", 7.0))
    print(f"Dataset size after cleaning: {df.height}")

    # ── 2. Train / val / test split ──────────────────────────────
    ratios = data_cfg.get("split_ratios", [0.7, 0.1, 0.2])
    train_df, val_df, test_df = train_test_val_split(df, proportions=ratios)
    print(f"Train: {train_df.height}  Val: {val_df.height}  Test: {test_df.height}")

    # ── 3. Build processors and encoders ─────────────────────────
    smiles_processors, smiles_encoders = build_smiles_components(cfg)
    protein_processors, protein_encoders = build_protein_components(cfg)

    print(f"SMILES encoders:  {[type(e).__name__ for e in smiles_encoders]}")
    print(f"Protein encoders: {[type(e).__name__ for e in protein_encoders]}")

    # ── 4. Pre-compute caches (if needed) ────────────────────────
    for proc in smiles_processors:
        if isinstance(proc, GraphProcessor):
            all_smiles = df["Ligand SMILES"].unique().to_list()
            proc.build_cache(all_smiles)
            valid = proc.valid_smiles
            print(f"Valid SMILES graphs: {len(valid)}")
            train_df = train_df.filter(pl.col("Ligand SMILES").is_in(valid))
            val_df = val_df.filter(pl.col("Ligand SMILES").is_in(valid))
            test_df = test_df.filter(pl.col("Ligand SMILES").is_in(valid))

    # ── 5. Build datasets and data loaders ───────────────────────
    def _make_dataset(split_df: pl.DataFrame) -> DTIDataset:
        return DTIDataset(
            smiles_list=split_df["Ligand SMILES"].to_list(),
            sequence_list=split_df["Full_Protein_Sequence"].to_list(),
            labels=split_df["is_active"].cast(pl.Float64).to_list(),
            smiles_processors=smiles_processors,
            protein_processors=protein_processors,
        )

    train_ds = _make_dataset(train_df)
    val_ds = _make_dataset(val_df)
    test_ds = _make_dataset(test_df)

    collate_fn = build_collate_fn(smiles_processors, protein_processors)
    batch_size = train_cfg.get("batch_size", 256)
    n_workers = train_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=n_workers,
        pin_memory=True, persistent_workers=(n_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=n_workers,
        pin_memory=True, persistent_workers=(n_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=n_workers,
        pin_memory=True, persistent_workers=(n_workers > 0),
    )

    # ── 6. Build model ───────────────────────────────────────────
    total_smiles_dim = sum(e.output_dim for e in smiles_encoders)
    total_protein_dim = sum(e.output_dim for e in protein_encoders)
    fusion = build_fusion(cfg, total_smiles_dim, total_protein_dim)
    model = MultimodalDTI(smiles_encoders, protein_encoders, fusion)
    print(model)

    # ── 7. Train ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=train_cfg.get("patience", 8),
    )
    model = trainer.fit(train_loader, val_loader, epochs=train_cfg.get("epochs", 50))

    # ── 8. Final evaluation ──────────────────────────────────────
    test_metrics = trainer.eval_epoch(test_loader)
    print("\n═══ Test Results ═══")
    for k, v in test_metrics.items():
        print(f"  {k:>12s}: {v:.4f}")


if __name__ == "__main__":
    main()
