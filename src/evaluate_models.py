"""Evaluate all trained DTI models and compare them in one table.

This script reuses the existing model-building code from ``main.py`` and
scans ``src/configs`` / ``src/checkpoints`` to produce a single dataframe
with metrics for each saved model.

It also writes a small comparison plot so the differences are easy to inspect.

Example::

    python src/evaluate_models.py
    python src/evaluate_models.py --limit 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.transform import train_val_test_split
from datasets.dti_dataset import DTIDataset, build_collate_fn
from main import (  # type: ignore[import-not-found]
    ROOT_DIR,
    MultimodalDTI,
    build_fusion,
    build_protein_components,
    build_smiles_components,
    load_data,
)
from processing.smiles.graph_processor import GraphProcessor
from training.trainer import Trainer


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


def _build_test_loader(
    test_df: pl.DataFrame,
    smiles_processors,
    protein_processors,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    test_ds = DTIDataset(
        smiles_list=test_df["Ligand SMILES"].to_list(),
        sequence_list=test_df["Full_Protein_Sequence"].to_list(),
        labels=test_df["is_active"].cast(pl.Float64).to_list(),
        smiles_processors=smiles_processors,
        protein_processors=protein_processors,
    )
    collate_fn = build_collate_fn(smiles_processors, protein_processors)

    if "OMP_NUM_THREADS" in os.environ:
        num_workers = 0

    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


def _filter_splits_for_graph_processors(
    split_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
    smiles_processors,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train_df, val_df, test_df = split_frames

    for processor in smiles_processors:
        if isinstance(processor, GraphProcessor):
            all_smiles = train_df["Ligand SMILES"].unique().to_list()
            all_smiles += val_df["Ligand SMILES"].unique().to_list()
            all_smiles += test_df["Ligand SMILES"].unique().to_list()
            processor.build_cache(list(dict.fromkeys(all_smiles)))
            valid_smiles = processor.valid_smiles

            train_df = train_df.filter(pl.col("Ligand SMILES").is_in(valid_smiles))
            val_df = val_df.filter(pl.col("Ligand SMILES").is_in(valid_smiles))
            test_df = test_df.filter(pl.col("Ligand SMILES").is_in(valid_smiles))

    return train_df, val_df, test_df


def _plot_results(results_df: pl.DataFrame, output_path: Path) -> None:
    ok_df = results_df.filter(pl.col("status") == "ok")
    if ok_df.is_empty():
        return

    ok_df = ok_df.sort("auc", descending=True)
    rows = ok_df.to_dicts()
    labels = [row["model_name"] for row in rows]
    auc_values = [row["auc"] for row in rows]
    auprc_values = [row["auprc"] for row in rows]

    height = max(4.5, 0.4 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(12, height))
    positions = list(range(len(labels)))

    ax.barh([pos - 0.18 for pos in positions], auc_values, height=0.35, label="AUC")
    ax.barh([pos + 0.18 for pos in positions], auprc_values, height=0.35, label="AUPRC")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Model comparison on the shared test split")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate_one_model(
    config_path: Path,
    checkpoint_path: Path,
    split_frames: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
    device_override: str | None = None,
) -> dict:
    cfg = _load_config(config_path)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    smiles_processors, smiles_encoders = build_smiles_components(cfg)
    protein_processors, protein_encoders = build_protein_components(cfg)

    train_df, val_df, test_df = split_frames
    train_df, val_df, test_df = _filter_splits_for_graph_processors(
        (train_df, val_df, test_df),
        smiles_processors,
    )

    batch_size = train_cfg.get("batch_size", 256)
    num_workers = train_cfg.get("num_workers", 4)
    test_loader = _build_test_loader(
        test_df=test_df,
        smiles_processors=smiles_processors,
        protein_processors=protein_processors,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    device_name = device_override or train_cfg.get("device", "auto")
    device = _select_device(device_name)

    total_smiles_dim = sum(encoder.output_dim for encoder in smiles_encoders)
    total_protein_dim = sum(encoder.output_dim for encoder in protein_encoders)
    fusion = build_fusion(cfg, total_smiles_dim, total_protein_dim)
    model = MultimodalDTI(smiles_encoders, protein_encoders, fusion)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

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
        checkpoint_dir=checkpoint_path.parent,
        checkpoint_filename=checkpoint_path.name,
        patience=train_cfg.get("patience", 8),
    )

    metrics = trainer.eval_epoch(test_loader)
    metrics.update(
        {
            "model_name": config_path.stem,
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "device": str(device),
            "n_train": train_df.height,
            "n_val": val_df.height,
            "n_test": test_df.height,
            "status": "ok",
            "error": None,
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all saved DTI models")
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=str(ROOT_DIR / "configs"),
        help="Directory containing YAML configs.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=str(ROOT_DIR / "checkpoints"),
        help="Directory containing saved checkpoints.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(ROOT_DIR / "logs" / "model_comparison.csv"),
        help="Where to save the comparison dataframe as CSV.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=str(ROOT_DIR / "logs" / "model_comparison.png"),
        help="Where to save the metric comparison plot.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the evaluation device, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N configs for a quick smoke test.",
    )
    args = parser.parse_args()

    configs_dir = _resolve_path(args.configs_dir)
    checkpoints_dir = _resolve_path(args.checkpoints_dir)
    output_csv = _resolve_path(args.output_csv)
    output_plot = _resolve_path(args.output_plot)

    config_paths = sorted(configs_dir.glob("*.yaml"))
    if args.limit is not None:
        config_paths = config_paths[: args.limit]

    if not config_paths:
        raise FileNotFoundError(f"No YAML configs found in {configs_dir}")

    reference_cfg = _load_config(config_paths[0])
    shared_df = load_data(reference_cfg["data"])
    print(f"Loaded dataset with {shared_df.height} rows")

    split_frames = train_val_test_split(
        shared_df,
        proportions=reference_cfg["data"].get("split_ratios", [0.7, 0.1, 0.2]),
    )

    results: list[dict] = []
    for config_path in config_paths:
        checkpoint_path = checkpoints_dir / f"{config_path.stem}.pt"
        print(f"Evaluating {config_path.stem}...")

        if not checkpoint_path.exists():
            results.append(
                {
                    "model_name": config_path.stem,
                    "config_path": str(config_path),
                    "checkpoint_path": str(checkpoint_path),
                    "status": "missing_checkpoint",
                    "error": f"Checkpoint not found: {checkpoint_path}",
                }
            )
            print(f"  missing checkpoint: {checkpoint_path}")
            continue

        try:
            result = evaluate_one_model(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                split_frames=split_frames,
                device_override=args.device,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch running
            result = {
                "model_name": config_path.stem,
                "config_path": str(config_path),
                "checkpoint_path": str(checkpoint_path),
                "status": "error",
                "error": str(exc),
            }
            print(f"  error: {exc}")

        results.append(result)

    results_df = pl.DataFrame(results)
    results_df = results_df.sort("model_name")

    if "status" in results_df.columns:
        ok_df = results_df.filter(pl.col("status") == "ok")
        if not ok_df.is_empty():
            print("\nTop models by AUC:")
            print(
                ok_df.sort("auc", descending=True).select(
                    ["model_name", "auc", "auprc", "f1", "precision", "recall", "n_test"]
                )
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_csv(output_csv)
    print(f"\nSaved comparison table to {output_csv}")

    _plot_results(results_df, output_plot)
    if output_plot.exists():
        print(f"Saved plot to {output_plot}")


if __name__ == "__main__":
    main()