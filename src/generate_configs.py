from pathlib import Path
import yaml
import copy

# Base dictionary for default configs
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
        "type": "cross_attention",
    },
    "training": {
        "batch_size": 256,
        "learning_rate": 1e-4,
        "lora_lr": 2e-5,
        "weight_decay": 0.01,
        "epochs": 30,
        "patience": 8,
        "num_workers": 8,
        "device": "auto"
    }
}

smiles_options = {
    "gcn": {"type": "gcn", "params": {"hidden_dim": 256, "num_layers": 3}},
    "fp": {"type": "fingerprint_mlp", "params": {"fp_type": "ecfp", "fp_params": {"radius": 2, "fp_size": 1024}, "hidden_dim": 512, "out_dim": 256, "dropout": 0.2}},
    "chembert": {"type": "chembert", "params": {"model_name": "seyonec/ChemBERTa-zinc-base-v1", "max_length": 256, "out_dim": 256, "lora_r": 16, "lora_alpha": 16, "lora_dropout": 0.1}}
}

protein_options = {
    "cnn": {"type": "cnn", "params": {"embed_dim": 256, "num_filters": 128, "kernel_sizes": [3, 7, 15], "max_seq_len": 1000}},
    "esm2": {"type": "esm2", "params": {"model_name": "facebook/esm2_t33_650M_UR50D", "max_length": 1024, "out_dim": 256, "lora_r": 16, "lora_alpha": 16, "lora_dropout": 0.1}}
}

import itertools

# generate combinations
smiles_keys = list(smiles_options.keys())
protein_keys = list(protein_options.keys())

smiles_combos = []
for i in range(1, len(smiles_keys) + 1):
    for combo in itertools.combinations(smiles_keys, i):
        smiles_combos.append(list(combo))

protein_combos = []
for i in range(1, len(protein_keys) + 1):
    for combo in itertools.combinations(protein_keys, i):
        protein_combos.append(list(combo))

out_dir = Path(__file__).resolve().parent / "configs"
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
        
        # Scale dropout with model complexity to combat overfitting
        n_encoders = len(s_combo) + len(p_combo)
        dropout = 0.3 + 0.05 * max(0, n_encoders - 2)
        dropout = min(dropout, 0.5)
        cfg["fusion"]["params"] = {"dropout": round(dropout, 2)}
        # hidden_dims intentionally omitted — MLPFusion auto-scales
        # based on the combined encoder output dimensions
        
        name = "_".join(s_combo) + "_vs_" + "_".join(p_combo) + ".yaml"
        with open(out_dir / name, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
            
print(f"Generated {len(smiles_combos) * len(protein_combos)} configs.")
