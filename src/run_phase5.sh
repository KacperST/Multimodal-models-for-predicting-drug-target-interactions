#!/bin/bash
set -e

# SCRIPT TO RUN ALL PHASE 5 MODELS ACROSS K2, K3, K4 SPLITS

# List of the 7 best models selected for Phase 5
CONFIGS=(
  "configs/phase5/fp_lincs_graph_vs_cnn.yaml"
  "configs/phase5/gcn_vs_cnn.yaml"
  "configs/phase5/gcn_vs_cnn_esm2.yaml"
  "configs/phase5/gcn_fp_lincs_graph_vs_cnn.yaml"
  "configs/phase5/gcn_lincs_graph_rdkit_vs_cnn.yaml"
  "configs/phase5/fp_gcn_lincs_graph_rdkit_vs_cnn.yaml"
  "configs/phase5/gcn_chemberta_fp_lincs_graph_rdkit_vs_cnn_esm2.yaml"
)

# K2 (scaffold), K3 (cold_target), K4 (cold_both)
SPLITS=("scaffold" "cold_target" "cold_both")

for config in "${CONFIGS[@]}"; do
  for split in "${SPLITS[@]}"; do
    echo "============================================================"
    echo "Running Phase 5: $config"
    echo "Split Strategy: $split"
    echo "============================================================"
    uv run python run_multiseed.py "$config" --split-strategy "$split"
  done
done

echo "Phase 5 fully completed!"
