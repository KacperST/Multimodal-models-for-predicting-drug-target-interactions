import os
import glob
import subprocess
from pathlib import Path
import sys

def main():
    configs = sorted(glob.glob("configs/*.yaml"), key=lambda x: x.replace('.', '~'))
    print(f"Znaleziono {len(configs)} plików .yaml w folderze configs/")
    if not configs:
        print("Nie znaleziono plików .yaml w folderze configs/")
        return

    print(f"Uruchamiam trening sekwencyjny (lokalnie)...")
    print(f"Logi poszczególnych modeli będą zapisywane w folderze logs/")
    print("-" * 60)

    config_path = "configs/gcn_fp_chembert_vs_cnn_esm2.yaml"
    config_name = Path(config_path).name
    print(f"Start: {config_name}")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{Path(config_path).stem}.log"
    
    cmd = [
        "uv", "run", "main.py",
        "--config", config_path
    ]
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1" 
    

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        
    if result.returncode == 0:
        print(f"SUKCES: {config_name}")
    else:
        print(f"BŁĄD ({result.returncode}): {config_name} (zobacz {log_file})")
            
    print("-" * 60)
    print("Wszystkie zadania treningowe zostały zakończone!")

if __name__ == "__main__":
    main()