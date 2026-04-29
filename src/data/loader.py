import polars as pl

def load_bindingdb_data(data_path: str = "datasets/BindingDB_All.tsv", sep="\t") -> pl.DataFrame:
    standard_aa = "ACDEFGHIKLMNPQRSTVWY"
    
    lazy_df = pl.scan_csv(data_path, separator=sep, ignore_errors=False, quote_char=None)

    df = (
        lazy_df
        .select([
            pl.col("Ligand SMILES"),
            pl.col("BindingDB Target Chain Sequence 1").alias("Full_Protein_Sequence"),
            pl.col("Ki (nM)"),
            pl.col("Number of Protein Chains in Target (>1 implies a multichain complex)")
              .cast(pl.Int32, strict=False)
              .fill_null(1)
              .alias("n_chains")
        ])
        .filter(
            (pl.col("n_chains") == 1) & 
            (pl.col("Ligand SMILES").is_not_null()) &
            (pl.col("Full_Protein_Sequence").is_not_null())
        )
        .with_columns([
            pl.col("Full_Protein_Sequence").str.to_uppercase().str.strip_chars(),
            pl.col("Ki (nM)").str.replace_all(r"[^0-9.]", "").replace("", None).cast(pl.Float64, strict=False)
        ])
        .filter(
            pl.col("Full_Protein_Sequence").str.contains(f"^[{standard_aa}]+$") &
            pl.col("Ki (nM)").is_not_null() &
            (pl.col("Ki (nM)") > 0)
        )
        .collect()
    )
    
    return df