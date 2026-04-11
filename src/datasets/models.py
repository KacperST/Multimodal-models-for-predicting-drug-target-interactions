from torch.utils.data import Dataset
import polars as pl
import torch


class ProteinLigandDataset(Dataset):
    def __init__(self, df: pl.DataFrame, smiles_map: dict):
        self.smiles = df["Ligand SMILES"].to_list()
        self.sequences = df["Tokenized_Sequence"].to_list()
        self.labels = df["is_active"].to_list()
        self.smiles_map = smiles_map

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        graph = self.smiles_map[self.smiles[idx]].clone()
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return graph, seq, y
