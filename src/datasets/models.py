from torch.utils.data import Dataset
import polars as pl
import torch


class ProteinLigandDataset(Dataset):
    def __init__(self, df, smiles_map):
        self.sequences = df["Tokenized_Sequence"].to_list()
        self.labels    = df["is_active"].to_list()
        self.smiles    = df["Ligand SMILES"].to_list()

        self.graph_store = {}
        for smi, data in smiles_map.items():
            self.graph_store[smi] = {
                "x":          data.x,
                "edge_index": data.edge_index,
                "edge_attr":  data.edge_attr,
            }

    def __getitem__(self, idx):
        g   = self.graph_store[self.smiles[idx]]
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        y   = torch.tensor(self.labels[idx],    dtype=torch.float)
        return g, seq, y

    def __len__(self):
        return len(self.labels)
