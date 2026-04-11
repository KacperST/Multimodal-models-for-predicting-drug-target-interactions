import polars as pl
import torch
from torch_geometric.data import Data
from ogb.utils import smiles2graph
from rdkit import Chem
from tqdm import tqdm


def remove_nulls(df: pl.DataFrame) -> pl.DataFrame:
    return df.drop_nulls()

def tranform_ki_to_log_ki(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .with_columns([
            pl.col("Ki (nM)")
            .clip(lower_bound=0.00001, upper_bound=100_000)
            .alias("Ki (nM)")
        ])
        .with_columns([
            (9 - pl.col("Ki (nM)").log10()).alias("pKi")
        ])
        )

def train_test_val_split(df, proportions = [0.7, 0.1, 0.2]):
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
    return (
        df
        .with_columns([
            pl.col("Ligand SMILES")
            .str.split_exact("|", 1)
            .struct.field("field_0")
            .str.strip_chars()
            .alias("Ligand SMILES")
        ])
    )

def create_pyg_dataset(smiles_list, y_values=None, with_features=False):
    """
    Konwertuje listę SMILES na listę obiektów PyTorch Geometric Data.
    
    Args:
        smiles_list: Lista lub Series ze strukturami SMILES.
        y_values: Opcjonalna lista wartości docelowych (np. Twoje pKi).
        with_features: Jeśli True, używa bogatych cech OGB. Jeśli False, tworzy "czysty" graf.
    """
    pyg_graphs = []
    
    print(f"Konwersja {len(smiles_list)} SMILES na grafy (with_features={with_features})...")
    
    for i, smiles in enumerate(smiles_list):
        try:
            if with_features:
                # 1. Features from OGB
                graph = smiles2graph(smiles)
                
                x = torch.tensor(graph['node_feat'], dtype=torch.long)
                edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
                edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.long)
                
            else:
                # 2. No features at all
                mol = Chem.MolFromSmiles(smiles)
                if mol is None: continue
                
                node_features = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
                x = torch.tensor(node_features, dtype=torch.float)
                
                edges = []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    edges.append([i, j])
                    edges.append([j, i]) 
                
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_attr = None 
            
            y = torch.tensor([y_values[i]], dtype=torch.float) if y_values is not None else None
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            pyg_graphs.append(data)
            
        except Exception as e:
            print(e)
            continue
            
    return pyg_graphs

def tokenize_sequences(df: pl.DataFrame, char_to_idx: dict, max_len: int = 1000) -> pl.DataFrame:
    unk_idx = char_to_idx.get('<UNK>', 0)

    return df.with_columns(
        pl.col("Full_Protein_Sequence")
        .str.to_uppercase()
        .str.split("")
        .list.eval(pl.element().filter(pl.element() != ""))
        .list.tail(max_len)
        .list.eval(
            pl.element()
            .replace(char_to_idx, default=unk_idx)
            .cast(pl.Int64)
        )
        .alias("Tokenized_Sequence")
    )