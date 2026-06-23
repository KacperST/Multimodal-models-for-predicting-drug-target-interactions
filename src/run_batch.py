import os
import glob
import subprocess
from pathlib import Path

# Definiujemy rdzenie CPU (taskset), na których chcemy pracować
CPU_RANGE = "0-7"

def main():
    # Pobierz wszystkie wygenerowane configi
    configs = [
        "configs/gcn_vs_cnn_esm2.yaml",
        "configs/gcn_vs_cnn.yaml",
        "configs/gcn_vs_esm2.yaml",
        "configs/gcn_chembert_vs_cnn_esm2.yaml",
        "configs/gcn_chembert_vs_cnn.yaml",
        "configs/gcn_fp_chembert_vs_cnn_esm2.yaml",
        "configs/gcn_fp_vs_esm2.yaml",
        "configs/gcn_fp_vs_cnn.yaml",
        "configs/gcn_fp_chembert_vs_cnn.yaml",
        "configs/fp_vs_esm2.yaml",
    ]
    print(f"Znaleziono {len(configs)} plików .yaml w folderze configs/")
    if not configs:
        print("Nie znaleziono plików .yaml w folderze configs/")
        return

    print(f"Znaleziono {len(configs)} modeli. Uruchamiam trening sekwencyjny...")
    print(f"Logi poszczególnych modeli będą zapisywane w folderze src/logs/")
    print("-" * 60)

    # Trenuj sekwencyjnie (jeden po drugim)
    for i, config_path in enumerate(configs, 1):
        config_name = Path(config_path).name
        print(f"[{i}/{len(configs)} | CPU {CPU_RANGE}] Start: {config_name}")
        
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{Path(config_path).stem}.log"
        
        cmd = [
            "uv", "run", "main.py",
            "--config", config_path
        ]
        
        # Zabezpieczenie przed tzw. "Thread Oversubscription Deadlock"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "8"
        env["MKL_NUM_THREADS"] = "8"
        env["OPENBLAS_NUM_THREADS"] = "8"
        env["POLARS_MAX_THREADS"] = "8"
        # Wymusza natychmiastowy zapis logów do pliku bez bufforowania
        env["PYTHONUNBUFFERED"] = "1"

        with open(log_file, "w") as f:
            # Uruchomienie procesu podrzędnego
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            
        if result.returncode == 0:
            print(f"[{i}/{len(configs)} | CPU {CPU_RANGE}] SUKCES: {config_name}")
        else:
            print(f"[{i}/{len(configs)} | CPU {CPU_RANGE}] BŁĄD ({result.returncode}): {config_name} (zobacz {log_file})")
            
    print("-" * 60)
    print("Wszystkie zadania treningowe zostały zakończone!")

if __name__ == "__main__":
    main()
