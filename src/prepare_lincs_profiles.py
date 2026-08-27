"""One-time preprocessing: extract & aggregate LINCS L1000 profiles from GCTX.

Reads the merged BindingDB+CMap dataset to find relevant pert_ids,
extracts their gene expression signatures (z-scores) from the GCTX file,
aggregates multiple signatures per compound via median, and saves
the result as a PyTorch cache file.

Usage::

    python prepare_lincs_profiles.py
    python prepare_lincs_profiles.py --gctx ~/level5_beta_trt_cp_n720216x12328.gctx

Outputs:
    datasets/lincs_profiles.pt      — {pert_id: tensor(978)} mapping
    datasets/smiles_to_pert_id.json — {canonical_smiles: pert_id} mapping
    datasets/lincs_balanced.parquet — balanced (50/50) merged dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LINCS L1000 profiles")
    parser.add_argument(
        "--gctx",
        type=str,
        default=str(Path.home() / "level5_beta_trt_cp_n720216x12328.gctx"),
        help="Path to GCTX file",
    )
    parser.add_argument(
        "--gene-info",
        type=str,
        default=str(Path.home() / "geneinfo_beta.txt"),
        help="Path to geneinfo_beta.txt",
    )
    parser.add_argument(
        "--merged",
        type=str,
        default=str(ROOT_DIR / "datasets" / "bindingdb_cmap_merged_direct_SMILES.parquet"),
        help="Path to merged BindingDB+CMap parquet",
    )
    parser.add_argument(
        "--siginfo",
        type=str,
        default=str(Path.home() / "siginfo_beta.txt"),
        help="Path to siginfo_beta.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "datasets"),
        help="Output directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of signatures to read from GCTX at once",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load merged dataset ───────────────────────────────────
    print("Loading merged dataset...")
    merged = pl.read_parquet(args.merged)
    print(f"  Merged rows: {merged.height}")
    print(f"  Unique pert_ids: {merged['pert_id'].n_unique()}")

    # ── 2. Load siginfo and map pert → sig ───────────────────────
    print("Loading siginfo to map pert_id → sig_id...")
    siginfo = pl.read_csv(args.siginfo, separator="\t", ignore_errors=True, infer_schema_length=10000)
    target_pert_ids = set(merged["pert_id"].unique().to_list())
    
    # Filter siginfo to our relevant pert_ids
    relevant_sigs = siginfo.filter(pl.col("pert_id").is_in(list(target_pert_ids)))
    
    # Optionally filter to high quality signatures if the column exists
    if "is_hiq" in relevant_sigs.columns:
        relevant_sigs = relevant_sigs.filter(pl.col("is_hiq") == 1)
        print(f"  Filtered to high-quality signatures (is_hiq=1).")
        
    print(f"  Relevant signatures found: {relevant_sigs.height}")

    # Build pert_id → [sig_ids] dict
    pert_to_sigs: dict[str, list[str]] = {}
    for row in relevant_sigs.select(["pert_id", "sig_id"]).iter_rows(named=True):
        pid = row["pert_id"]
        sid = row["sig_id"]
        pert_to_sigs.setdefault(pid, []).append(sid)

    print(f"  Unique pert_ids with sigs: {len(pert_to_sigs)}")

    # ── 3. Identify landmark gene indices in GCTX ────────────────
    print("Loading gene info to find landmark genes...")
    genes = pl.read_csv(args.gene_info, separator="\t", ignore_errors=True)
    landmark_gene_ids = set(
        genes.filter(pl.col("feature_space") == "landmark")["gene_id"]
        .cast(pl.Utf8)
        .to_list()
    )
    print(f"  Landmark genes in geneinfo: {len(landmark_gene_ids)}")

    # Find landmark gene indices in the GCTX
    with h5py.File(args.gctx, "r") as f:
        gctx_sig_ids = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in f["0/META/COL/id"][:]
        ]
        gctx_gene_ids = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in f["0/META/ROW/id"][:]
        ]

    # Map gene_id → HDF5 dimension 1 (12328)
    landmark_gene_indices = [
        i for i, gid in enumerate(gctx_gene_ids) if gid in landmark_gene_ids
    ]
    print(f"  Landmark genes found in GCTX: {len(landmark_gene_indices)}")

    # Map sig_id → HDF5 dimension 0 (720216) for fast lookup
    sig_to_idx = {sid: i for i, sid in enumerate(gctx_sig_ids)}

    # Filter to sigs actually present in GCTX
    gctx_sig_set = set(gctx_sig_ids)
    for pid in list(pert_to_sigs.keys()):
        pert_to_sigs[pid] = [s for s in pert_to_sigs[pid] if s in gctx_sig_set]
        if not pert_to_sigs[pid]:
            del pert_to_sigs[pid]

    print(f"  Pert_ids with sigs in GCTX: {len(pert_to_sigs)}")
    total_sigs = sum(len(v) for v in pert_to_sigs.values())
    print(f"  Total signatures to extract: {total_sigs}")

    # ── 4. Extract profiles from GCTX ────────────────────────────
    print("\nExtracting L1000 profiles from GCTX (this may take a while)...")
    n_landmarks = len(landmark_gene_indices)
    gene_indices = sorted(landmark_gene_indices)

    # Collect all sig_ids we need and their row indices
    all_needed_sigs: list[str] = []
    for sigs in pert_to_sigs.values():
        all_needed_sigs.extend(sigs)
    all_needed_row_indices = [sig_to_idx[s] for s in all_needed_sigs]

    # Sort by row index for sequential HDF5 access
    sorted_pairs = sorted(
        zip(all_needed_row_indices, all_needed_sigs), key=lambda x: x[0]
    )

    # Read in batches
    sig_profiles: dict[str, np.ndarray] = {}
    batch_size = args.batch_size

    with h5py.File(args.gctx, "r") as f:
        matrix = f["0/DATA/0/matrix"]
        for batch_start in tqdm(
            range(0, len(sorted_pairs), batch_size),
            desc="Reading GCTX",
            unit="batch",
        ):
            batch_pairs = sorted_pairs[batch_start : batch_start + batch_size]
            batch_row_indices = [p[0] for p in batch_pairs]
            batch_sig_ids = [p[1] for p in batch_pairs]

            # Read rows individually (they may not be contiguous)
            for row_idx, sid in zip(batch_row_indices, batch_sig_ids):
                row_data = matrix[row_idx, gene_indices]
                sig_profiles[sid] = row_data.astype(np.float32)

    print(f"  Extracted {len(sig_profiles)} signature profiles")

    # ── 5. Aggregate per pert_id (median) ────────────────────────
    print("Aggregating profiles per pert_id (median)...")
    pert_profiles: dict[str, torch.Tensor] = {}

    for pid, sigs in tqdm(pert_to_sigs.items(), desc="Aggregating"):
        arrays = [sig_profiles[s] for s in sigs if s in sig_profiles]
        if not arrays:
            continue
        stacked = np.stack(arrays, axis=0)  # (n_sigs, 978)
        median_profile = np.median(stacked, axis=0)  # (978,)
        pert_profiles[pid] = torch.tensor(median_profile, dtype=torch.float32)

    print(f"  Aggregated profiles: {len(pert_profiles)} pert_ids")

    # ── 6. Save profiles cache ───────────────────────────────────
    profiles_path = output_dir / "lincs_profiles.pt"
    torch.save(pert_profiles, str(profiles_path))
    print(f"\n✅ Saved LINCS profiles cache: {profiles_path}")
    print(f"   Keys: {len(pert_profiles)}, vector dim: {n_landmarks}")

    # ── 7. Build & save SMILES → pert_id mapping ─────────────────
    print("Building SMILES → pert_id mapping...")
    smiles_pert_map: dict[str, str] = {}
    available_perts = set(pert_profiles.keys())

    # Use canonical_smiles_rdkit from merged for the mapping
    for row in merged.select(["canonical_smiles_rdkit", "pert_id"]).unique().iter_rows(named=True):
        smi = row["canonical_smiles_rdkit"]
        pid = row["pert_id"]
        if pid in available_perts and smi is not None:
            smiles_pert_map[smi] = pid

    # Also add original Ligand SMILES as keys (some may differ)
    for row in merged.select(["Ligand SMILES", "pert_id"]).unique().iter_rows(named=True):
        smi = row["Ligand SMILES"]
        pid = row["pert_id"]
        if pid in available_perts and smi is not None:
            smiles_pert_map[smi] = pid

    map_path = output_dir / "smiles_to_pert_id.json"
    with open(map_path, "w") as f:
        json.dump(smiles_pert_map, f)
    print(f"✅ Saved SMILES → pert_id mapping: {map_path}")
    print(f"   Entries: {len(smiles_pert_map)}")

    # ── 8. Create balanced dataset ───────────────────────────────
    print("\nCreating balanced dataset (50/50 undersampling)...")

    # Filter merged to only include rows with available profiles
    merged_with_profiles = merged.filter(
        pl.col("pert_id").is_in(list(available_perts))
    )
    print(f"  Rows with L1000 profiles: {merged_with_profiles.height}")

    active = merged_with_profiles.filter(pl.col("is_active") == True)
    inactive = merged_with_profiles.filter(pl.col("is_active") == False)
    print(f"  Active:   {active.height}")
    print(f"  Inactive: {inactive.height}")

    # Undersample the majority class
    n_minority = min(active.height, inactive.height)
    if active.height > inactive.height:
        active = active.sample(n=n_minority, seed=42)
    else:
        inactive = inactive.sample(n=n_minority, seed=42)

    balanced = pl.concat([active, inactive]).sample(fraction=1.0, seed=42, shuffle=True)
    print(f"  Balanced dataset: {balanced.height} rows ({n_minority} per class)")

    balanced_path = output_dir / "lincs_balanced.parquet"
    balanced.write_parquet(str(balanced_path))
    print(f"✅ Saved balanced dataset: {balanced_path}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  LINCS profiles:    {profiles_path} ({len(pert_profiles)} compounds × {n_landmarks} genes)")
    print(f"  SMILES→pert_id:    {map_path} ({len(smiles_pert_map)} entries)")
    print(f"  Balanced dataset:  {balanced_path} ({balanced.height} rows)")
    print(f"\nReady for training with: python main.py --config configs/lincs/<config>.yaml")


if __name__ == "__main__":
    main()
