import polars as pl
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy import stats
import numpy as np
from tqdm import tqdm

def main():
    print("Loading full dataset...")
    df = pl.read_parquet('datasets/clean.parquet')
    print(f"Total rows: {df.height}")

    actives_arom = []
    inactives_arom = []
    actives_rot = []
    inactives_rot = []

    print("Calculating descriptors for the ENTIRE dataset (this may take 1-2 minutes)...")
    for row in tqdm(df.iter_rows(named=True), total=df.height):
        smi = row['Ligand SMILES']
        is_active = row['is_active']
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                arom = Descriptors.NumAromaticRings(mol)
                rot = Descriptors.NumRotatableBonds(mol)
                if is_active:
                    actives_arom.append(arom)
                    actives_rot.append(rot)
                else:
                    inactives_arom.append(arom)
                    inactives_rot.append(rot)
        except Exception:
            pass

    print("\n--- Aromatic Rings (Core Rigidity) ---")
    print(f"Active median: {np.median(actives_arom)}")
    print(f"Inactive median: {np.median(inactives_arom)}")
    u_arom, p_arom = stats.mannwhitneyu(actives_arom, inactives_arom, alternative='two-sided')
    print(f"Mann-Whitney U statistic: {u_arom}")
    print(f"p-value: {p_arom}")

    print("\n--- Rotatable Bonds (Spatial Flexibility) ---")
    print(f"Active median: {np.median(actives_rot)}")
    print(f"Inactive median: {np.median(inactives_rot)}")
    u_rot, p_rot = stats.mannwhitneyu(actives_rot, inactives_rot, alternative='two-sided')
    print(f"Mann-Whitney U statistic: {u_rot}")
    print(f"p-value: {p_rot}")

if __name__ == '__main__':
    main()
