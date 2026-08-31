import yaml
import copy
from pathlib import Path
import os

# Konfiguracja bazowa (parametry z Fazy 1)
base_config = {
    "data": {
        "clean_path": "datasets/clean.parquet",
        "path": "datasets/BindingDB_All.tsv",
        "pki_threshold": 7.0,
        "split_ratios": [0.7, 0.1, 0.2]
    },
    "smiles_encoders": [],
    "protein_encoders": [],
    "fusion": {
        "type": "mlp",
        "params": {
            "hidden_dims": [256, 64],  # Z fazy 1
            "dropout": 0.45            # Z fazy 1
        }
    },
    "training": {
        "batch_size": 512,             # Z fazy 1
        "learning_rate": 0.0003,       # Z fazy 1
        "weight_decay": 1e-5,
        "epochs": 150,
        "patience": 10,
        "num_workers": 4,
        "device": "auto"
    }
}

# Parametry enkoderów z Fazy 1 (większe wymiary)
smiles_options = {
    "gcn": {"type": "gcn", "params": {"hidden_dim": 256, "num_layers": 3}},
    "fp": {"type": "fingerprint_mlp", "params": {"fp_type": "ecfp", "fp_params": {"radius": 2, "fp_size": 1024}, "hidden_dim": 512, "out_dim": 256, "dropout": 0.2}},
    "chembert": {"type": "chembert", "params": {"cache_path": "datasets/ChemBERTa-zinc-base-v1.pt", "out_dim": 256}},
    "rdkit_desc": {"type": "rdkit_descriptors", "params": {"cache_path": "datasets/rdkit_descriptors_full.pt", "hidden_dim": 512, "out_dim": 256, "dropout": 0.2}},
}

protein_options = {
    "cnn": {"type": "cnn", "params": {"embed_dim": 256, "num_filters": 128, "kernel_sizes": [3, 7, 15], "max_seq_len": 1000}}
}

# Wszystkie modele strukturalne z Fazy 1 wzbogacone o rdkit_desc
smiles_combos = [
    ["rdkit_desc"],                              # Tylko RDKit (baseline)
    ["fp", "rdkit_desc"],                        # FP + RDKit
    ["gcn", "rdkit_desc"],                       # GCN + RDKit
    ["chembert", "rdkit_desc"],                  # ChemBERT + RDKit
    ["fp", "gcn", "rdkit_desc"],                 # FP + GCN + RDKit
    ["fp", "chembert", "rdkit_desc"],            # FP + ChemBERT + RDKit
    ["gcn", "chembert", "rdkit_desc"],           # GCN + ChemBERT + RDKit
    ["fp", "gcn", "chembert", "rdkit_desc"]      # FP + GCN + ChemBERT + RDKit
]

out_dir = Path("./configs/descriptors_phase1")
out_dir.mkdir(parents=True, exist_ok=True)

# Usunięcie starych configów w tym folderze
for f in os.listdir(out_dir):
    if f.endswith(".yaml"):
        os.remove(out_dir / f)

for s_combo in smiles_combos:
    cfg = copy.deepcopy(base_config)
    
    cfg["smiles_encoders"] = [smiles_options[k] for k in s_combo]
    cfg["protein_encoders"] = [protein_options["cnn"]]
    
    name = "_".join(s_combo) + "_vs_cnn.yaml"
    with open(out_dir / name, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
        
print(f"Generated {len(smiles_combos)} configs for Phase 1 + RDKit in {out_dir}")
