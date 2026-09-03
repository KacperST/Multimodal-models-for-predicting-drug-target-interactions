import polars as pl
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import multiprocessing
from functools import partial

def get_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except:
        return None

def get_unique_scaffolds(parquet_path):
    df = pl.read_parquet(parquet_path)
    smiles_list = df["Ligand SMILES"].unique().to_list()
    
    scaffolds = set()
    
    # Przetwarzanie wielowątkowe dla prędkości
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(get_scaffold, smiles_list), total=len(smiles_list), desc=f"Scaffolds for {parquet_path}"))
        
    for res in results:
        if res is not None:
            scaffolds.add(res)
            
    return len(scaffolds), len(smiles_list)

if __name__ == "__main__":
    n_scaffolds_lincs, n_smiles_lincs = get_unique_scaffolds("datasets/lincs_balanced.parquet")
    n_scaffolds_clean, n_smiles_clean = get_unique_scaffolds("datasets/clean.parquet")
    
    print("\n===============================")
    print(f"LINCS (lincs_balanced.parquet):")
    print(f" - Unikalne SMILES: {n_smiles_lincs}")
    print(f" - Unikalne szkielety (scaffolds): {n_scaffolds_lincs}")
    
    print(f"\nFULL (clean.parquet):")
    print(f" - Unikalne SMILES: {n_smiles_clean}")
    print(f" - Unikalne szkielety (scaffolds): {n_scaffolds_clean}")
    print("===============================\n")
