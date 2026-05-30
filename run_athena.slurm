#!/bin/bash
#SBATCH --job-name=dti_train                 # Nazwa zadania
#SBATCH --partition=plgrid-gpu-a100          # Kolejka (partycja) z kartami A100 na Athenie
#SBATCH --time=24:00:00                      # Maksymalny czas działania (HH:MM:SS) - dostosuj do potrzeb
#SBATCH --nodes=1                            # Liczba węzłów
#SBATCH --ntasks-per-node=1                  # Liczba zadań na węzeł
#SBATCH --cpus-per-task=8                    # Liczba rdzeni procesora (zapas dla num_workers=4)
#SBATCH --gres=gpu:1                         # Żądana liczba kart graficznych (1x A100 40GB)
#SBATCH --mem=60G                            # Ilość pamięci RAM
#SBATCH --output=slurm_logs/train_%j.out     # Plik ze standardowym wyjściem (%j zostanie zastąpione ID zada)
#SBATCH --error=slurm_logs/train_%j.err      # Plik z błędami
#SBATCH --account=TWOJ_ID_GRANTU             # TUTAJ WPISZ ID SWOJEGO GRANTU PLGRID!

# 1. Stworzenie katalogu na logi slurm, jeśli nie istnieje
mkdir -p slurm_logs

# 2. Załadowanie odpowiednich modułów środowiskowych na Athenie (Python i ew. CUDA)
# Te moduły mogą się zmieniać w zależności od aktualnej konfiguracji Atheny,
# często "module avail python" pozwala sprawdzić dostępne wersje.
module load Python/3.11.5
module load CUDA/12.4.0

# 3. Przejście do katalogu, z którego uruchomiono komendę sbatch
cd $SLURM_SUBMIT_DIR

# 4. Jeśli używasz narzędzia "uv" lub zwykłego venv, opcjonalnie aktywuj środowisko:
# source .venv/bin/activate

echo "Rozpoczynam trening na węźle: $(hostname)"
echo "Przydzielona karta: $CUDA_VISIBLE_DEVICES"

# 5. Uruchomienie samego treningu przez srun
# Możesz tu zmienić nazwę pliku .yaml na docelową!
srun uv run python src/main.py --config configs/chembert_vs_cnn_esm2.yaml

echo "Zadanie zakończone."
