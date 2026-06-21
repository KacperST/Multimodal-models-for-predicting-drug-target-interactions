"""Script to generate statistics and plots from the cleaned dataset.
Saves plots to the 'plots/' directory for use in the thesis.
"""
import sys
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# Dodanie src do ścieżki, żeby importy z data.transform zadziałały
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.transform import train_test_val_split

# Configure plot style (requires seaborn)
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def main():
    data_path = ROOT_DIR / "datasets" / "clean.parquet"
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run prepare_data.py first.")
        return
        
    print("Loading data...")
    df = pl.read_parquet(data_path)
    print(f"Total number of interactions: {df.height:,}")
    
    plots_dir = ROOT_DIR / "statistics" / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # 1. Class distribution (Active vs Inactive)
    print("\n--- 1. Class Distribution ---")
    class_counts = df["is_active"].value_counts()
    active_count = class_counts.filter(pl.col("is_active") == True)["count"][0]
    inactive_count = class_counts.filter(pl.col("is_active") == False)["count"][0]
    print(f"Active (True): {active_count:,} ({active_count/df.height*100:.1f}%)")
    print(f"Inactive (False): {inactive_count:,} ({inactive_count/df.height*100:.1f}%)")
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Inactive (pKi < 7.0)", "Active (pKi >= 7.0)"], [inactive_count, active_count], color=['#e74c3c', '#2ecc71'])
    ax.set_title("Activity Class Distribution (Class Balance)")
    ax.set_ylabel("Number of interaction pairs")
    for i, v in enumerate([inactive_count, active_count]):
        ax.text(i, v + (df.height * 0.01), f"{v:,}", ha='center')
    plt.tight_layout()
    fig.savefig(plots_dir / "class_balance.png", dpi=300)
    plt.close(fig)

    # 2. pKi Distribution
    print("\n--- 2. pKi Distribution ---")
    pki_mean = df["pKi"].mean()
    pki_median = df["pKi"].median()
    print(f"Mean pKi: {pki_mean:.2f}")
    print(f"Median pKi: {pki_median:.2f}")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["pKi"].to_numpy(), bins=50, kde=True, ax=ax, color="#3498db")
    ax.axvline(x=7.0, color='red', linestyle='--', label='Activity threshold (7.0)')
    ax.set_title("Distribution of pKi values")
    ax.set_xlabel("pKi")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "pki_distribution.png", dpi=300)
    plt.close(fig)

    # 3. Protein sequence lengths
    print("\n--- 3. Protein Statistics ---")
    unique_proteins = df["Full_Protein_Sequence"].n_unique()
    print(f"Unique protein sequences: {unique_proteins:,}")
    
    # Add column with sequence length
    df = df.with_columns(pl.col("Full_Protein_Sequence").str.len_chars().alias("Protein_Length"))
    prot_mean = df["Protein_Length"].mean()
    prot_median = df["Protein_Length"].median()
    prot_max = df["Protein_Length"].max()
    print(f"Mean length: {prot_mean:.1f}")
    print(f"Median length: {prot_median:.1f}")
    print(f"Max length: {prot_max:,}")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    # Cut off plot at 2000 to avoid extreme outliers stretching the x-axis
    plot_data = df.filter(pl.col("Protein_Length") <= 2000)["Protein_Length"].to_numpy()
    sns.histplot(plot_data, bins=50, ax=ax, color="#9b59b6")
    ax.set_title("Distribution of protein sequence lengths (<= 2000)")
    ax.set_xlabel("Number of amino acids")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(plots_dir / "protein_length_distribution.png", dpi=300)
    plt.close(fig)

    # 4. SMILES lengths (as a proxy for molecule size)
    print("\n--- 4. Ligand Statistics ---")
    unique_smiles = df["Ligand SMILES"].n_unique()
    print(f"Unique ligands (SMILES): {unique_smiles:,}")
    
    df = df.with_columns(pl.col("Ligand SMILES").str.len_chars().alias("SMILES_Length"))
    smiles_mean = df["SMILES_Length"].mean()
    print(f"Mean SMILES length: {smiles_mean:.1f} characters")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data = df.filter(pl.col("SMILES_Length") <= 200)["SMILES_Length"].to_numpy()
    sns.histplot(plot_data, bins=50, ax=ax, color="#e67e22")
    ax.set_title("Distribution of SMILES lengths (<= 200 characters)")
    ax.set_xlabel("Number of characters in SMILES string")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(plots_dir / "smiles_length_distribution.png", dpi=300)
    plt.close(fig)

    # 5. Data Partitioning Analysis
    print("\n--- 5. Data Partitioning Analysis (SMILES Random Split) ---")
    print("Calculating split (this might take a moment)...")
    train_df, val_df, test_df = train_test_val_split(df, proportions=[0.7, 0.1, 0.2])
    
    print("\nSplit Results:")
    print("Set     | Interaction Pairs | Unique Ligands | Unique Proteins")
    print("-" * 65)
    print(f"Train   | {train_df.height:17,} | {train_df['Ligand SMILES'].n_unique():14,} | {train_df['Full_Protein_Sequence'].n_unique():15,}")
    print(f"Valid.  | {val_df.height:17,} | {val_df['Ligand SMILES'].n_unique():14,} | {val_df['Full_Protein_Sequence'].n_unique():15,}")
    print(f"Test    | {test_df.height:17,} | {test_df['Ligand SMILES'].n_unique():14,} | {test_df['Full_Protein_Sequence'].n_unique():15,}")
    print("-" * 65)
    
    print("\nPlots have been successfully generated and saved in src/statistics/plots/")

if __name__ == "__main__":
    main()
