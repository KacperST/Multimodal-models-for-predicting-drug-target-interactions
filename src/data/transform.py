import polars as pl


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