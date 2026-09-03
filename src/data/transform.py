import polars as pl

import skfp.model_selection.splitters.randomized_scaffold_split as rs_module
rs_module.check_random_state = lambda x: x
from skfp.model_selection import randomized_scaffold_train_valid_test_split

import json
import random
from pathlib import Path


def remove_nulls(df: pl.DataFrame) -> pl.DataFrame:
    return df.drop_nulls()


def tranform_ki_to_log_ki(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [pl.col("Ki (nM)").clip(lower_bound=0.00001, upper_bound=100_000).alias("Ki (nM)")]
    ).with_columns([(9 - pl.col("Ki (nM)").log10()).alias("pKi")])


def train_test_val_split(df, proportions=[0.7, 0.1, 0.2]):
    unique_smiles = df.select("Ligand SMILES").unique().sample(fraction=1.0, shuffle=True)
    total_len = len(unique_smiles)

    train_size = int(total_len * proportions[0])
    val_size = int(total_len * proportions[1])
    train_smiles = unique_smiles.slice(0, train_size)
    val_smiles = unique_smiles.slice(train_size, val_size)
    test_smiles = unique_smiles.slice(train_size + val_size, None)

    df_train = df.join(train_smiles, on="Ligand SMILES", how="semi")
    df_val = df.join(val_smiles, on="Ligand SMILES", how="semi")
    df_test = df.join(test_smiles, on="Ligand SMILES", how="semi")

    return df_train, df_val, df_test

def train_val_test_split_scaffold(df, proportions=[0.7, 0.1, 0.2], random_state=42):
    smiles = df["Ligand SMILES"].unique().sort().to_list()
    train_size, val_size, test_size = proportions
    train_smiles, val_smiles, test_smiles = randomized_scaffold_train_valid_test_split(
        smiles,
        train_size=train_size,
        valid_size=val_size,
        test_size=test_size,
        random_state=random_state
    )
    df_train = df.join(pl.DataFrame({"Ligand SMILES": train_smiles}), on="Ligand SMILES", how="semi")
    df_val = df.join(pl.DataFrame({"Ligand SMILES": val_smiles}), on="Ligand SMILES", how="semi")
    df_test = df.join(pl.DataFrame({"Ligand SMILES": test_smiles}), on="Ligand SMILES", how="semi")
    return df_train, df_val, df_test


def _load_protein_clusters(cluster_path: str | Path) -> list[list[str]]:
    """Load protein cluster assignments from a JSON file."""
    with open(cluster_path) as f:
        return json.load(f)


def _split_clusters(
    clusters: list[list[str]],
    proportions: list[float],
    random_state: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    """Assign whole protein clusters to train/val/test splits.

    Clusters are shuffled and then greedily assigned to splits
    in order to approximate the target proportions.
    """
    rng = random.Random(random_state)
    shuffled = list(range(len(clusters)))
    rng.shuffle(shuffled)

    total_seqs = sum(len(c) for c in clusters)
    train_target = int(total_seqs * proportions[0])
    val_target = int(total_seqs * proportions[1])

    train_seqs: set[str] = set()
    val_seqs: set[str] = set()
    test_seqs: set[str] = set()

    train_count = 0
    val_count = 0

    for idx in shuffled:
        members = clusters[idx]
        if train_count < train_target:
            train_seqs.update(members)
            train_count += len(members)
        elif val_count < val_target:
            val_seqs.update(members)
            val_count += len(members)
        else:
            test_seqs.update(members)

    return train_seqs, val_seqs, test_seqs


def train_val_test_split_cold_target(
    df: pl.DataFrame,
    proportions: list[float] = [0.7, 0.1, 0.2],
    random_state: int = 42,
    cluster_path: str | Path = "datasets/protein_clusters.json",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """K3 cold-target split: proteins in test are unseen during training.

    Uses MMseqs2 cluster assignments to ensure that homologous proteins
    always end up in the same split.
    """
    ROOT = Path(__file__).resolve().parent.parent
    clusters = _load_protein_clusters(ROOT / cluster_path)

    train_seqs, val_seqs, test_seqs = _split_clusters(clusters, proportions, random_state)

    df_train = df.filter(pl.col("Full_Protein_Sequence").is_in(list(train_seqs)))
    df_val = df.filter(pl.col("Full_Protein_Sequence").is_in(list(val_seqs)))
    df_test = df.filter(pl.col("Full_Protein_Sequence").is_in(list(test_seqs)))

    print(f"  K3 cold-target split: {len(train_seqs)} train proteins, "
          f"{len(val_seqs)} val proteins, {len(test_seqs)} test proteins")

    return df_train, df_val, df_test


def train_val_test_split_cold_both(
    df: pl.DataFrame,
    proportions: list[float] = [0.7, 0.1, 0.2],
    random_state: int = 42,
    cluster_path: str | Path = "datasets/protein_clusters.json",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """K4 cold-both split: both ligands AND proteins in test are unseen.

    Ligands are split via scaffold split, proteins via MMseqs2 clustering.
    A row is assigned to split X only if BOTH its ligand AND protein belong
    to split X.
    """
    # 1. Split ligands by scaffold
    smiles = df["Ligand SMILES"].unique().sort().to_list()
    train_size, val_size, test_size = proportions
    train_smiles, val_smiles, test_smiles = randomized_scaffold_train_valid_test_split(
        smiles,
        train_size=train_size,
        valid_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )
    train_smiles_set = set(train_smiles)
    val_smiles_set = set(val_smiles)
    test_smiles_set = set(test_smiles)

    # 2. Split proteins by MMseqs2 clusters
    ROOT = Path(__file__).resolve().parent.parent
    clusters = _load_protein_clusters(ROOT / cluster_path)
    train_seqs, val_seqs, test_seqs = _split_clusters(clusters, proportions, random_state)

    # 3. Intersect: assign each row based on BOTH ligand and protein membership
    df_train = df.filter(
        pl.col("Ligand SMILES").is_in(list(train_smiles_set))
        & pl.col("Full_Protein_Sequence").is_in(list(train_seqs))
    )
    df_val = df.filter(
        pl.col("Ligand SMILES").is_in(list(val_smiles_set))
        & pl.col("Full_Protein_Sequence").is_in(list(val_seqs))
    )
    df_test = df.filter(
        pl.col("Ligand SMILES").is_in(list(test_smiles_set))
        & pl.col("Full_Protein_Sequence").is_in(list(test_seqs))
    )

    print(f"  K4 cold-both split: train={df_train.height}, val={df_val.height}, test={df_test.height}")
    total_kept = df_train.height + df_val.height + df_test.height
    print(f"  Rows retained: {total_kept}/{df.height} ({100*total_kept/df.height:.1f}%)")

    return df_train, df_val, df_test


def remove_cx_notation(df: pl.DataFrame):
    return df.with_columns(
        [
            pl.col("Ligand SMILES")
            .str.split_exact("|", 1)
            .struct.field("field_0")
            .str.strip_chars()
            .alias("Ligand SMILES")
        ]
    )


def remove_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by(
                ["Ligand SMILES", "Full_Protein_Sequence"]
            ).agg(
                [
                    pl.col("Ki (nM)").mean().alias("Ki (nM)"),
                ]
            ).select(["Ligand SMILES", "Full_Protein_Sequence", "Ki (nM)"]).sort(
                ["Ligand SMILES", "Full_Protein_Sequence"]
            )


def add_activity_label(df: pl.DataFrame, pki_threshold: float = 7.0) -> pl.DataFrame:
    """Add a boolean ``is_active`` column based on pKi threshold."""
    return df.with_columns([(pl.col("pKi") >= pki_threshold).alias("is_active")])

def remove_invalid_smiles(df: pl.DataFrame) -> pl.DataFrame:
    from rdkit import Chem
    
    def is_valid(smi: str) -> bool:
        try:
            return Chem.MolFromSmiles(smi) is not None
        except Exception:
            return False
            
    print("Validating SMILES strings with RDKit...")
    valid_mask = df["Ligand SMILES"].map_elements(is_valid, return_dtype=pl.Boolean)
    return df.filter(valid_mask)