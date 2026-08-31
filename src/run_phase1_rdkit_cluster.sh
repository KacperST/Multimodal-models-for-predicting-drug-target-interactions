#!/bin/bash
# Zatrzymanie skryptu w przypadku jakiegokolwiek błędu
set -e

echo "============================================================"
echo "Rozpoczynam przygotowania do Fazy 1 + RDKit na klastrze..."
echo "Data: $(date)"
echo "============================================================"

# Krok 1: Prekompilacja deskryptorów dla pełnego zbioru (zajmie ok. 20-30 min)
echo "[1/3] Generowanie deskryptorów RDKit dla całego zbioru (clean.parquet)..."
uv run precompute_rdkit_descriptors.py --input datasets/clean.parquet --output datasets/rdkit_descriptors_full.pt
echo "  -> Prekompilacja zakończona sukcesem."
echo "------------------------------------------------------------"

# Krok 2: Uruchomienie batcha dla 8 wygenerowanych modeli
echo "[2/3] Uruchamianie kolejki treningowej dla modeli Fazy 1 + RDKit..."
# uv run_batch przechodzi przez wszystkie pliki YAML w folderze
uv run run_batch.py configs/descriptors_phase1
echo "  -> Trening wszystkich modeli zakończony sukcesem."
echo "------------------------------------------------------------"

# Krok 3: Ewaluacja na zbiorze testowym
echo "[3/3] Ocena modeli (ewaluacja testowa)..."
uv run evaluate_models.py --configs-dir configs/descriptors_phase1 --checkpoints-dir checkpoints
echo "  -> Ewaluacja zakończona."
echo "------------------------------------------------------------"

echo "Wszystkie zadania wykonane poprawnie!"
echo "Gotowe tabele i wykresy znajdziesz w: logs/descriptors_phase1/"
echo "Zakończono: $(date)"
echo "============================================================"
