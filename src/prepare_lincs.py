"""Merge LINCS L1000 gene expression profiles with BindingDB DTI dataset.

Strategy (hybrid exact + similarity matching):
  1. Canonicalise SMILES in both BindingDB and LINCS compoundinfo using RDKit.
  2. Exact match: compounds with identical canonical SMILES (tanimoto_score=1.0).
  3. Similarity match: for unmatched BindingDB compounds, find nearest LINCS
     neighbour by Tanimoto similarity on Morgan fingerprints (radius=2, 2048 bits).
     Accept matches with similarity >= threshold (default 0.7).
  4. Filter siginfo to QC-passed signatures (relaxed: qc_pass=1 only).
  5. For each matched pert_id, load gene expression profiles from GCTX (978 landmark genes).
  6. Aggregate multiple signatures per compound via median.
  7. Join gene expression vectors + tanimoto_score to the clean BindingDB DataFrame.
  8. Save the merged dataset as Parquet.

Usage::

    python prepare_lincs.py
    python prepare_lincs.py --tanimoto-threshold 0.7
    python prepare_lincs.py --tanimoto-threshold 0.5  # more aggressive

Requirements: cmapPy, rdkit, polars, pandas, numpy, h5py
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

# Suppress RDKit warnings for invalid SMILES
RDLogger.DisableLog("rdApp.*")

ROOT_DIR = Path(__file__).resolve().parent

# ── Default paths ──────────────────────────────────────────────────────────
DEFAULT_DTI = str(ROOT_DIR / "datasets" / "clean.parquet")
DEFAULT_OUTPUT = str(ROOT_DIR / "datasets" / "clean_with_lincs.parquet")
DEFAULT_GCTX = str(Path.home() / "level5_beta_trt_cp_n720216x12328.gctx")
DEFAULT_SIGINFO = str(Path.home() / "siginfo_beta.txt")
DEFAULT_COMPOUNDINFO = str(Path.home() / "compoundinfo_beta.txt")
DEFAULT_GENEINFO = str(Path.home() / "geneinfo_beta.txt")


# ── Helpers ────────────────────────────────────────────────────────────────

def canonicalise_smiles(smi: str) -> str | None:
    """Return canonical SMILES via RDKit, or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def smiles_to_fp(smi: str):
    """Compute Morgan fingerprint (radius=2, 2048 bits) from SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


# ── Step 1: Load compound info ────────────────────────────────────────────

def load_and_prepare_compoundinfo(path: str) -> pd.DataFrame:
    """Load compoundinfo, canonicalise SMILES, compute fingerprints."""
    print("[1/7] Loading compoundinfo...")
    ci = pd.read_csv(path, sep="\t", usecols=["pert_id", "canonical_smiles", "target", "moa"])
    ci = ci.dropna(subset=["canonical_smiles"])
    ci = ci[ci["canonical_smiles"].str.strip() != ""]

    print(f"  Compounds with SMILES: {len(ci)}")
    print("  Canonicalising LINCS SMILES...")
    ci["canon_smiles"] = ci["canonical_smiles"].apply(canonicalise_smiles)
    ci = ci.dropna(subset=["canon_smiles"])

    # Deduplicate: keep unique canon_smiles → pert_id mapping
    ci = ci.drop_duplicates(subset=["canon_smiles", "pert_id"])
    print(f"  Valid compounds: {ci['canon_smiles'].nunique()}")
    return ci


# ── Step 2: Load DTI data ─────────────────────────────────────────────────

def load_dti_data(path: str) -> pl.DataFrame:
    """Load cleaned DTI dataset and add canonical SMILES column."""
    print("[2/7] Loading DTI dataset...")
    df = pl.read_parquet(path)
    print(f"  DTI pairs: {df.height:,}, Unique SMILES: {df['Ligand SMILES'].n_unique():,}")

    print("  Canonicalising BindingDB SMILES...")
    canon_list = [canonicalise_smiles(s) for s in df["Ligand SMILES"].to_list()]
    df = df.with_columns(pl.Series("canon_smiles", canon_list))
    df = df.filter(pl.col("canon_smiles").is_not_null())
    print(f"  Valid after canonicalisation: {df.height:,}")
    return df


# ── Step 3: Hybrid matching (exact + Tanimoto) ────────────────────────────

def find_matching_compounds(
    dti_df: pl.DataFrame,
    compound_df: pd.DataFrame,
    tanimoto_threshold: float = 0.7,
) -> dict[str, tuple[str, float]]:
    """Match BindingDB SMILES to LINCS via exact match + Tanimoto similarity.

    Returns:
        dict mapping bdb_canon_smiles → (lincs_canon_smiles, tanimoto_score)
    """
    print(f"[3/7] Matching compounds (exact + Tanimoto ≥ {tanimoto_threshold})...")
    t0 = time.time()

    dti_smiles = dti_df["canon_smiles"].unique().to_list()
    lincs_unique = compound_df.drop_duplicates(subset=["canon_smiles"])
    lincs_smiles_list = lincs_unique["canon_smiles"].tolist()
    lincs_smiles_set = set(lincs_smiles_list)

    print(f"  BindingDB unique SMILES: {len(dti_smiles):,}")
    print(f"  LINCS unique SMILES:     {len(lincs_smiles_list):,}")

    # Phase 1: Exact matches
    matches: dict[str, tuple[str, float]] = {}
    unmatched = []
    for smi in dti_smiles:
        if smi in lincs_smiles_set:
            matches[smi] = (smi, 1.0)
        else:
            unmatched.append(smi)
    print(f"  Exact matches: {len(matches):,}")
    print(f"  Remaining for similarity search: {len(unmatched):,}")

    # Phase 2: Compute LINCS fingerprints
    print("  Computing LINCS fingerprints...")
    lincs_fps = []
    lincs_fps_smiles = []
    for smi in lincs_smiles_list:
        fp = smiles_to_fp(smi)
        if fp is not None:
            lincs_fps.append(fp)
            lincs_fps_smiles.append(smi)
    print(f"  LINCS fingerprints: {len(lincs_fps):,}")

    # Phase 3: Tanimoto nearest neighbour for unmatched
    print(f"  Searching nearest neighbours for {len(unmatched):,} compounds...")
    n_sim_matched = 0
    for i, smi in enumerate(unmatched):
        fp = smiles_to_fp(smi)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, lincs_fps)
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        if best_sim >= tanimoto_threshold:
            matches[smi] = (lincs_fps_smiles[best_idx], float(best_sim))
            n_sim_matched += 1
        if (i + 1) % 25000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1:,}/{len(unmatched):,} done "
                  f"({elapsed:.0f}s, {n_sim_matched:,} similarity matches so far)")

    elapsed = time.time() - t0
    print(f"  ✓ Total matches: {len(matches):,} "
          f"(exact: {len(matches) - n_sim_matched:,}, "
          f"similarity: {n_sim_matched:,}) in {elapsed:.0f}s")

    return matches


# ── Step 4: Filter siginfo ────────────────────────────────────────────────

def filter_siginfo(path: str, pert_ids: set[str]) -> pd.DataFrame:
    """Load siginfo and filter to QC-passed signatures for matched pert_ids.

    Relaxed filters: only qc_pass=1, no is_hiq or is_exemplar requirement.
    """
    print("[4/7] Filtering signatures (siginfo)...")
    cols_needed = ["sig_id", "pert_id", "pert_type", "is_hiq", "qc_pass",
                   "is_exemplar_sig", "cell_iname", "pert_idose", "pert_itime"]
    siginfo = pd.read_csv(path, sep="\t", usecols=cols_needed, low_memory=False)
    print(f"  Total signatures: {len(siginfo):,}")

    # Relaxed filter: compound treatment + QC pass only
    siginfo = siginfo[
        (siginfo["pert_type"] == "trt_cp")
        & (siginfo["qc_pass"] == 1)
    ]
    print(f"  After qc_pass filter: {len(siginfo):,}")

    # Filter to matched pert_ids
    siginfo = siginfo[siginfo["pert_id"].isin(pert_ids)]
    print(f"  After pert_id match: {len(siginfo):,}")

    # Prefer exemplar if available per pert_id, else keep all
    exemplar = siginfo[siginfo["is_exemplar_sig"] == 1]
    non_exemplar_pids = set(siginfo["pert_id"]) - set(exemplar["pert_id"])
    fallback = siginfo[siginfo["pert_id"].isin(non_exemplar_pids)]
    result = pd.concat([exemplar, fallback], ignore_index=True)
    print(f"  Using exemplar where available: {len(result):,} sigs "
          f"({len(exemplar):,} exemplar + {len(fallback):,} fallback)")
    return result


# ── Step 5: Extract gene expression ───────────────────────────────────────

def extract_gene_expression(
    gctx_path: str,
    geneinfo_path: str,
    sig_ids: list[str],
    lincs_smiles_to_pert_ids: dict[str, list[str]],
    siginfo: pd.DataFrame,
    batch_size: int = 5000,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Extract 978-dim landmark gene expression vectors from GCTX.

    Aggregates per LINCS canonical SMILES (median over all signatures).

    Returns:
        (dict mapping lincs_canon_smiles → np.ndarray(978,), list of gene symbols)
    """
    from cmapPy.pandasGEXpress import parse

    print("[5/7] Extracting gene expression profiles from GCTX...")

    # Get landmark gene IDs
    geneinfo = pd.read_csv(geneinfo_path, sep="\t")
    landmark = geneinfo[geneinfo["feature_space"] == "landmark"]
    landmark_ids = [str(gid) for gid in landmark["gene_id"].tolist()]
    landmark_symbols = landmark["gene_symbol"].tolist()
    print(f"  Landmark genes: {len(landmark_ids)}")

    # Build reverse map: pert_id → lincs_canon_smiles
    pert_to_smiles: dict[str, str] = {}
    for smi, pids in lincs_smiles_to_pert_ids.items():
        for pid in pids:
            pert_to_smiles[pid] = smi

    # Build sig_id → lincs_canon_smiles
    sig_to_smiles: dict[str, str] = {}
    for _, row in siginfo.iterrows():
        pid = row["pert_id"]
        sid = row["sig_id"]
        if pid in pert_to_smiles:
            sig_to_smiles[sid] = pert_to_smiles[pid]

    # Collect expression per LINCS compound
    smiles_profiles: dict[str, list[np.ndarray]] = {}

    total = len(sig_ids)
    for i in range(0, total, batch_size):
        batch_sids = sig_ids[i : i + batch_size]
        n_batches = (total + batch_size - 1) // batch_size
        print(f"  Reading batch {i // batch_size + 1}/{n_batches} "
              f"({len(batch_sids)} sigs)...")
        try:
            gctoo = parse.parse(gctx_path, rid=landmark_ids, cid=batch_sids)
            expr_df = gctoo.data_df

            for sid in expr_df.columns:
                smi = sig_to_smiles.get(sid)
                if smi is None:
                    continue
                vec = expr_df[sid].values.astype(np.float32)
                if smi not in smiles_profiles:
                    smiles_profiles[smi] = []
                smiles_profiles[smi].append(vec)

            del gctoo, expr_df
            gc.collect()
        except Exception as e:
            print(f"  ⚠ Error in batch: {e}")
            continue

    # Aggregate: median across all signatures per compound
    print("  Aggregating profiles (median)...")
    result: dict[str, np.ndarray] = {}
    for smi, profiles in smiles_profiles.items():
        stacked = np.stack(profiles, axis=0)
        result[smi] = np.median(stacked, axis=0).astype(np.float32)

    print(f"  ✓ Gene expression profiles for {len(result):,} LINCS compounds")
    return result, landmark_symbols


# ── Step 6: Build mapping ─────────────────────────────────────────────────

def build_bdb_to_ge_mapping(
    bdb_matches: dict[str, tuple[str, float]],
    lincs_profiles: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, float]]:
    """Map BindingDB SMILES → (GE vector, tanimoto_score) via LINCS profiles.

    Returns:
        dict mapping bdb_canon_smiles → (978-dim vector, tanimoto_score)
    """
    print("[6/7] Building BindingDB → GE mapping...")
    result: dict[str, tuple[np.ndarray, float]] = {}
    n_exact = 0
    n_sim = 0
    n_missing = 0

    for bdb_smi, (lincs_smi, score) in bdb_matches.items():
        if lincs_smi in lincs_profiles:
            result[bdb_smi] = (lincs_profiles[lincs_smi], score)
            if score == 1.0:
                n_exact += 1
            else:
                n_sim += 1
        else:
            n_missing += 1

    print(f"  Mapped: {len(result):,} (exact: {n_exact:,}, similarity: {n_sim:,})")
    print(f"  Missing GE profile: {n_missing:,}")
    return result


# ── Step 7: Merge and save ────────────────────────────────────────────────

def merge_and_save(
    dti_df: pl.DataFrame,
    bdb_ge_mapping: dict[str, tuple[np.ndarray, float]],
    gene_symbols: list[str],
    output_path: str,
) -> pl.DataFrame:
    """Join expression profiles + tanimoto scores to DTI DataFrame."""
    print("[7/7] Merging and saving...")

    smiles_list = []
    vectors = []
    scores = []
    for smi, (vec, score) in bdb_ge_mapping.items():
        smiles_list.append(smi)
        vectors.append(vec)
        scores.append(score)

    expr_matrix = np.stack(vectors, axis=0)
    gene_cols = [f"ge_{sym}" for sym in gene_symbols]

    # Build Polars DataFrame directly (avoids pyarrow dependency)
    data_dict: dict[str, list] = {
        "canon_smiles": smiles_list,
        "tanimoto_score": scores,
    }
    for j, col_name in enumerate(gene_cols):
        data_dict[col_name] = expr_matrix[:, j].tolist()
    expr_pl = pl.DataFrame(data_dict)

    # Join
    merged = dti_df.join(expr_pl, on="canon_smiles", how="inner")

    # Report
    n_with_ge = merged.height
    n_total = dti_df.height
    n_exact = merged.filter(pl.col("tanimoto_score") == 1.0).height
    n_sim = merged.filter(pl.col("tanimoto_score") < 1.0).height
    print(f"  DTI pairs with GE: {n_with_ge:,} / {n_total:,} ({100*n_with_ge/n_total:.1f}%)")
    print(f"    Exact match:      {n_exact:,}")
    print(f"    Similarity match: {n_sim:,}")
    print(f"  Unique SMILES:  {merged['canon_smiles'].n_unique():,}")
    print(f"  Unique proteins: {merged['Full_Protein_Sequence'].n_unique():,}")
    print(f"  Tanimoto score stats:")
    ts = merged["tanimoto_score"]
    print(f"    mean={ts.mean():.3f}, median={ts.median():.3f}, "
          f"min={ts.min():.3f}, max={ts.max():.3f}")

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(str(out))
    print(f"\n  ✓ Saved to: {out}")
    print(f"    Shape: {merged.shape}")

    return merged


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LINCS L1000 gene expression with BindingDB DTI dataset"
    )
    parser.add_argument("--dti-input", type=str, default=DEFAULT_DTI)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--gctx", type=str, default=DEFAULT_GCTX)
    parser.add_argument("--siginfo", type=str, default=DEFAULT_SIGINFO)
    parser.add_argument("--compoundinfo", type=str, default=DEFAULT_COMPOUNDINFO)
    parser.add_argument("--geneinfo", type=str, default=DEFAULT_GENEINFO)
    parser.add_argument("--tanimoto-threshold", type=float, default=0.7,
                        help="Min Tanimoto similarity for non-exact matches (default: 0.7)")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="GCTX read batch size")
    args = parser.parse_args()

    # Step 1
    compound_df = load_and_prepare_compoundinfo(args.compoundinfo)

    # Step 2
    dti_df = load_dti_data(args.dti_input)

    # Step 3: Hybrid matching
    bdb_matches = find_matching_compounds(dti_df, compound_df, args.tanimoto_threshold)
    if not bdb_matches:
        print("\n✗ No matching compounds found.")
        return

    # Collect all LINCS SMILES that were matched → their pert_ids
    matched_lincs_smiles = set(lincs_smi for lincs_smi, _ in bdb_matches.values())
    lincs_smiles_to_pert_ids: dict[str, list[str]] = {}
    for smi in matched_lincs_smiles:
        pids = compound_df[compound_df["canon_smiles"] == smi]["pert_id"].unique().tolist()
        lincs_smiles_to_pert_ids[smi] = pids
    all_pert_ids = set()
    for pids in lincs_smiles_to_pert_ids.values():
        all_pert_ids.update(pids)
    print(f"  LINCS compounds to fetch: {len(matched_lincs_smiles):,}")
    print(f"  Total pert_ids: {len(all_pert_ids):,}")

    # Step 4
    siginfo = filter_siginfo(args.siginfo, all_pert_ids)
    if siginfo.empty:
        print("\n✗ No signatures found.")
        return
    sig_ids = siginfo["sig_id"].unique().tolist()

    # Step 5
    lincs_profiles, gene_symbols = extract_gene_expression(
        gctx_path=args.gctx,
        geneinfo_path=args.geneinfo,
        sig_ids=sig_ids,
        lincs_smiles_to_pert_ids=lincs_smiles_to_pert_ids,
        siginfo=siginfo,
        batch_size=args.batch_size,
    )
    if not lincs_profiles:
        print("\n✗ No gene expression profiles extracted.")
        return

    # Step 6
    bdb_ge_mapping = build_bdb_to_ge_mapping(bdb_matches, lincs_profiles)

    # Step 7
    merged = merge_and_save(dti_df, bdb_ge_mapping, gene_symbols, args.output)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Original DTI pairs:       {dti_df.height:,}")
    print(f"  Matched DTI pairs:        {merged.height:,}")
    print(f"  Coverage:                 {100 * merged.height / dti_df.height:.1f}%")
    print(f"  Tanimoto threshold:       {args.tanimoto_threshold}")
    print(f"  Gene expression dims:     {len(gene_symbols)} (landmark genes)")
    print(f"  Output file:              {args.output}")


if __name__ == "__main__":
    main()
