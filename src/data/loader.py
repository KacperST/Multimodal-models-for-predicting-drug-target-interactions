import polars as pl

def load_bindingdb_data(data_path: str = "datasets/BindingDB_All.tsv", sep = "\t") -> pl.DataFrame:
    sequence_cols = [f"BindingDB Target Chain Sequence {i}" for i in range(1, 51)]

    lazy_df = pl.scan_csv(data_path, separator=sep, ignore_errors=False, quote_char=None)

    df = (
        lazy_df
        .with_columns([
            pl.col("Number of Protein Chains in Target (>1 implies a multichain complex)")
            .cast(pl.Int32, strict=False)
            .fill_null(1)
            .alias("n_chains"),
            
            pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in sequence_cols])
            .alias("actual_seq_count")
        ])
        .with_columns([
            (pl.col("n_chains") == pl.col("actual_seq_count")).alias("is_consistent"),
            pl.concat_str(
                [pl.col(c).fill_null("") for c in sequence_cols],
                separator=":"
            ).str.strip_chars(":").alias("Full_Protein_Sequence")
        ])
        .with_columns([
            pl.col("Full_Protein_Sequence").str.to_uppercase().alias("Full_Protein_Sequence")
        ])
        .with_columns([
            pl.col("Ki (nM)")
            .str.replace_all(r"[^0-9.]", "")
            .replace("", None)
            .cast(pl.Float64, strict=False)
        ])
        .filter(
            (pl.col("Ligand SMILES").is_not_null()) & 
            (pl.col("Full_Protein_Sequence") != "") &
            (pl.col("is_consistent") == True) 
        )
        .select([
            "Ligand SMILES",
            "Full_Protein_Sequence",
            "Ki (nM)",
        ])
        .collect()
        )
    return df