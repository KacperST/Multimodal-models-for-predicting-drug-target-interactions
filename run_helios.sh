#!/bin/bash
#SBATCH --job-name=dti_train
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgdtiprediction-gpu-gh200
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --output=slurm_logs/train_%j.out
#SBATCH --error=slurm_logs/train_%j.err

mkdir -p slurm_logs

# 1. Załadowanie modułów systemowych Cyfronetu
module load CUDA/12.4.0
module load GCCcore/13.2.0
module load bzip2/1.0.8
module load Python/3.11.5

# 2. Hack naprawiający bibliotekę BZ2 dla NetworkX
mkdir -p "$SLURM_SUBMIT_DIR/.local_libs"
ln -sf /usr/lib64/libbz2.so.1 "$SLURM_SUBMIT_DIR/.local_libs/libbz2.so.1.0"
export LD_LIBRARY_PATH="$SLURM_SUBMIT_DIR/.local_libs:$EBROOTBZIP2/lib:${LD_LIBRARY_PATH:-}"

cd "$SLURM_SUBMIT_DIR"

# 3. OSTATECZNE CZYSZCZENIE ZEPSUTYCH KOMPILATORÓW:
# Zdejmujemy nałożoną przez uv blokadę "tylko do odczytu" ze środowiska
chmod -R u+w .venv

# Pobieramy ścieżki do prawdziwych, natywnych narzędzi NVIDII na procesory ARM
SYS_NVCC=$(which nvcc)
SYS_PTXAS=$(dirname "$SYS_NVCC")/ptxas

# Szukamy wszystkich zepsutych kompilatorów w paczkach PyTorcha i podmieniemy je na systemowe
for bad_nvcc in $(find .venv -name "nvcc" -type f 2>/dev/null); do
    echo "Naprawiam plik: $bad_nvcc"
    rm -f "$bad_nvcc"
    ln -sf "$SYS_NVCC" "$bad_nvcc"
done

for bad_ptxas in $(find .venv -name "ptxas" -type f 2>/dev/null); do
    rm -f "$bad_ptxas"
    ln -sf "$SYS_PTXAS" "$bad_ptxas"
done

export CUDACXX="$SYS_NVCC"

echo "Rozpoczynam trening na węźle: $(hostname)"
echo "Przydzielona karta: $CUDA_VISIBLE_DEVICES"

# 4. START TRENINGU PRZEZ UV
uv run --no-sync python src/main.py --config src/configs/gcn_fp_chembert_vs_cnn_esm2.yaml

echo "Zadanie zakończone."