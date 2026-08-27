import os
import glob
import subprocess
from pathlib import Path
import sys

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "configs"
    configs = sorted(glob.glob(f"{target_dir}/*.yaml"), key=lambda x: x.replace('.', '~'))
    print(f"Znaleziono {len(configs)} plików .yaml w folderze {target_dir}")
    if not configs:
        print(f"Nie znaleziono plików .yaml w folderze {target_dir}")
        return

    print(f"Uruchamiam trening sekwencyjny (lokalnie)...")
    print(f"Logi poszczególnych modeli będą zapisywane w folderze logs/")
    print("-" * 60)

    for i, config_path in enumerate(configs, 1):
        config_name = Path(config_path).name
        print(f"[{i}/{len(configs)}] Start: {config_name}")
        # Mirror the folder structure (e.g. configs/lincs -> logs/lincs)
        target_path = Path(target_dir)
        try:
            rel_path = target_path.relative_to("configs")
            log_dir = Path("logs") / rel_path
        except ValueError:
            log_dir = Path("logs") / target_path.name if target_path.name != "." else Path("logs")
            
        log_dir.mkdir(parents=True, exist_ok=True)
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
            print(f"[{i}/{len(configs)}] SUKCES: {config_name}")
        else:
            print(f"[{i}/{len(configs)}] BŁĄD ({result.returncode}): {config_name} (zobacz {log_file})")
            
    print("-" * 60)
    print("Wszystkie zadania treningowe zostały zakończone!")

if __name__ == "__main__":
    main()