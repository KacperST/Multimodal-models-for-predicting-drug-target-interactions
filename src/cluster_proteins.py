"""Cluster unique protein sequences using MMseqs2.

Exports unique protein sequences from the LINCS dataset to a FASTA file,
runs MMseqs2 easy-cluster, and saves the cluster assignments as a JSON
mapping {cluster_representative: [member_sequences, ...]}.

Usage::

    uv run cluster_proteins.py --input datasets/lincs_balanced.parquet --output datasets/protein_clusters.json
    uv run cluster_proteins.py --input datasets/lincs_balanced.parquet --output datasets/protein_clusters.json --min-seq-id 0.4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import polars as pl


ROOT_DIR = Path(__file__).resolve().parent


def export_fasta(sequences: list[str], fasta_path: Path) -> dict[str, str]:
    """Write sequences to a FASTA file, returning id→sequence mapping."""
    id_to_seq = {}
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(sequences):
            seq_id = f"protein_{i}"
            id_to_seq[seq_id] = seq
            f.write(f">{seq_id}\n{seq}\n")
    return id_to_seq


def run_mmseqs_cluster(
    fasta_path: Path,
    output_prefix: Path,
    tmp_dir: Path,
    mmseqs_bin: str = "mmseqs",
    min_seq_id: float = 0.4,
) -> dict[str, list[str]]:
    """Run MMseqs2 easy-cluster and parse cluster assignments.

    Args:
        fasta_path: Path to input FASTA file.
        output_prefix: Prefix for MMseqs2 output files.
        tmp_dir: Temporary directory for MMseqs2.
        mmseqs_bin: Path to mmseqs binary.
        min_seq_id: Minimum sequence identity threshold (0.0–1.0).

    Returns:
        Dictionary mapping representative_id → list of member_ids.
    """
    cmd = [
        mmseqs_bin,
        "easy-cluster",
        str(fasta_path),
        str(output_prefix),
        str(tmp_dir),
        "--min-seq-id", str(min_seq_id),
        "-c", "0.8",                # coverage threshold
        "--cov-mode", "0",           # bidirectional coverage
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("MMseqs2 STDERR:")
        print(result.stderr)
        raise RuntimeError(f"MMseqs2 failed with exit code {result.returncode}")

    # Parse the TSV output: columns are [representative_id, member_id]
    tsv_path = Path(str(output_prefix) + "_cluster.tsv")
    clusters = defaultdict(list)
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                rep_id, member_id = parts
                clusters[rep_id].append(member_id)

    return dict(clusters)


def main():
    parser = argparse.ArgumentParser(description="Cluster protein sequences using MMseqs2")
    parser.add_argument("--input", type=str, default="datasets/lincs_balanced.parquet",
                        help="Path to input Parquet file")
    parser.add_argument("--output", type=str, default="datasets/protein_clusters.json",
                        help="Path to output JSON file with cluster assignments")
    parser.add_argument("--min-seq-id", type=float, default=0.4,
                        help="Minimum sequence identity threshold for clustering (default: 0.4 = 40%%)")
    parser.add_argument("--mmseqs-bin", type=str, default=None,
                        help="Path to mmseqs binary (auto-detected if not provided)")
    args = parser.parse_args()

    # Auto-detect mmseqs binary
    if args.mmseqs_bin is None:
        local_bin = ROOT_DIR.parent / "mmseqs" / "bin" / "mmseqs"
        if local_bin.exists():
            args.mmseqs_bin = str(local_bin)
        else:
            args.mmseqs_bin = "mmseqs"

    # Load data and extract unique sequences
    input_path = str(ROOT_DIR / args.input)
    print(f"Loading dataset from: {input_path}")
    df = pl.read_parquet(input_path)
    unique_seqs = df["Full_Protein_Sequence"].unique().sort().to_list()
    print(f"Found {len(unique_seqs)} unique protein sequences")

    with tempfile.TemporaryDirectory(dir=str(ROOT_DIR)) as tmpdir:
        tmpdir = Path(tmpdir)

        # Export FASTA
        fasta_path = tmpdir / "proteins.fasta"
        id_to_seq = export_fasta(unique_seqs, fasta_path)
        seq_to_id = {v: k for k, v in id_to_seq.items()}

        # Run MMseqs2
        output_prefix = tmpdir / "clusters"
        mmseqs_tmp = tmpdir / "mmseqs_tmp"
        mmseqs_tmp.mkdir()

        id_clusters = run_mmseqs_cluster(
            fasta_path=fasta_path,
            output_prefix=output_prefix,
            tmp_dir=mmseqs_tmp,
            mmseqs_bin=args.mmseqs_bin,
            min_seq_id=args.min_seq_id,
        )

    # Convert from id-based clusters to sequence-based clusters
    # Output format: list of lists, where each inner list is a cluster of sequences
    seq_clusters: list[list[str]] = []
    for rep_id, member_ids in id_clusters.items():
        cluster_seqs = [id_to_seq[mid] for mid in member_ids]
        seq_clusters.append(cluster_seqs)

    print(f"\nClustering results (min_seq_id={args.min_seq_id}):")
    print(f"  Total unique sequences: {len(unique_seqs)}")
    print(f"  Number of clusters: {len(seq_clusters)}")
    sizes = sorted([len(c) for c in seq_clusters], reverse=True)
    print(f"  Largest cluster: {sizes[0]} sequences")
    print(f"  Singleton clusters: {sum(1 for s in sizes if s == 1)}")
    print(f"  Top 10 cluster sizes: {sizes[:10]}")

    # Save
    output_path = str(ROOT_DIR / args.output)
    with open(output_path, "w") as f:
        json.dump(seq_clusters, f)
    print(f"\nSaved {len(seq_clusters)} clusters to: {output_path}")


if __name__ == "__main__":
    main()
