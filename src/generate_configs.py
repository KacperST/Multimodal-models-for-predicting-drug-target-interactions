from pathlib import Path
import yaml
import copy

# Base dictionary for default configs
base_config = {
    "data": {
        "clean_path": "datasets/lincs_balanced.parquet",
        "path": "datasets/BindingDB_All.tsv",
        "pki_threshold": 7.0,
        "split_ratios": [0.7, 0.1, 0.2]
    },
    "smiles_encoders": [],
    "protein_encoders": [],
    "fusion": {
        "type": "mlp",
        "params": {
            "hidden_dims": [128, 32],
            "dropout": 0.5
        }
    },
    "training": {
        "batch_size": 128,
        "learning_rate": 0.00005,
        "weight_decay": 1e-5,
        "epochs": 150,
        "patience": 10,
        "num_workers": 4,
        "device": "auto"
    }
}

smiles_options = {
    "gcn": {"type": "gcn", "params": {"hidden_dim": 128, "out_dim": 128, "num_layers": 3, "dropout": 0.5}},
    "fp": {"type": "fingerprint_mlp", "params": {"fp_type": "ecfp", "fp_params": {"radius": 2, "fp_size": 1024}, "hidden_dim": 256, "out_dim": 128, "dropout": 0.5}},
    "chembert": {"type": "chembert", "params": {"cache_path": "datasets/ChemBERTa-zinc-base-v1.pt", "out_dim": 128}},
    "rdkit_desc": {"type": "rdkit_descriptors", "params": {"cache_path": "datasets/rdkit_descriptors.pt", "hidden_dim": 256, "out_dim": 128, "dropout": 0.5}},
    "lincs": {"type": "lincs", "params": {"cache_path": "datasets/lincs_profiles.pt", "smiles_pert_map_path": "datasets/smiles_to_pert_id.json", "hidden_dim": 128, "out_dim": 128, "dropout": 0.5}},
    "lincs_graph": {"type": "lincs_graph", "params": {"cache_path": "datasets/lincs_profiles.pt", "smiles_pert_map_path": "datasets/smiles_to_pert_id.json", "hidden_dim": 128, "out_dim": 128, "num_layers": 2, "theta": 1.0, "dropout": 0.5}}
}

protein_options = {
    "cnn": {"type": "cnn", "params": {"embed_dim": 128, "num_filters": 64, "kernel_sizes": [3, 7, 15], "max_seq_len": 1000}}
}

import itertools

# Ręcznie wybrana lista 13 najważniejszych kombinacji (odcinamy szum, zostawiamy najlepsze z Fazy 1 + nowości)
smiles_combos = [
    ["rdkit_desc"],                                         # Baseline: Samo RDKit
    ["fp", "lincs", "rdkit_desc"],                          # FP + LINCS + RDKit
    ["gcn", "lincs", "rdkit_desc"],                         # GCN + LINCS + RDKit
    ["chembert", "lincs", "rdkit_desc"],                    # ChemBERT + LINCS + RDKit
    ["fp", "gcn", "lincs", "rdkit_desc"],                   # FP + GCN + LINCS + RDKit
    ["fp", "chembert", "lincs", "rdkit_desc"],              # FP + ChemBERT + LINCS + RDKit
    ["gcn", "chembert", "lincs", "rdkit_desc"],             # GCN + ChemBERT + LINCS + RDKit
    ["fp", "gcn", "chembert", "lincs", "rdkit_desc"]        # FP + GCN + ChemBERT + LINCS + RDKit
]

protein_keys = list(protein_options.keys())
protein_combos = []
for i in range(1, len(protein_keys) + 1):
    for combo in itertools.combinations(protein_keys, i):
        protein_combos.append(list(combo))

out_dir = Path("./configs/descriptors")
out_dir.mkdir(parents=True, exist_ok=True)

import os
# delete existing config files so we don't have overlapping old variants like gcn_cnn_esm2.yaml if we want to replace them all with systematic names
for f in os.listdir(out_dir):
    if f.endswith(".yaml"):
        os.remove(out_dir / f)

for s_combo in smiles_combos:
    for p_combo in protein_combos:
        cfg = copy.deepcopy(base_config)
        
        cfg["smiles_encoders"] = [smiles_options[k] for k in s_combo]
        cfg["protein_encoders"] = [protein_options[k] for k in p_combo]
        
        name = "_".join(s_combo) + "_vs_" + "_".join(p_combo) + ".yaml"
        with open(out_dir / name, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
            
print(f"Generated {len(smiles_combos) * len(protein_combos)} configs.")
