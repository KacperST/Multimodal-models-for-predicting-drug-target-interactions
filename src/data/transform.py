import polars as pl
from skfp.model_selection import randomized_scaffold_train_valid_test_split


def remove_nulls(df: pl.DataFrame) -> pl.DataFrame:
    return df.drop_nulls()


def tranform_ki_to_log_ki(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [pl.col("Ki (nM)").clip(lower_bound=0.00001, upper_bound=100_000).alias("Ki (nM)")]
    ).with_columns([(9 - pl.col("Ki (nM)").log10()).alias("pKi")])


def train_val_test_split(df, proportions=[0.7, 0.1, 0.2]):
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
    smiles = df["Ligand SMILES"].unique().to_list()
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
            ).select(["Ligand SMILES", "Full_Protein_Sequence", "Ki (nM)"])


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