"""Run Phase 2 training for the top N models from Phase 1.

Reads ``logs/model_comparison.csv`` (sorted by AUC), picks the top N
model names, matches them to configs in ``configs/``, and trains each
one sequentially.

Usage::

    uv run run_batch.py              # top 10 by default
    uv run run_batch.py --top 5      # only top 5
    uv run run_batch.py --all        # all available configs
"""

import argparse
import csv
import os
import sys
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR / "configs"
LOGS_DIR = SCRIPT_DIR / "logs"
PHASE1_CSV = LOGS_DIR / "model_comparison.csv"

# Definiujemy rdzenie CPU (taskset), na których chcemy pracować
CPU_RANGE = "0-7"


def _read_top_models(csv_path: Path, top_n: int) -> list[str]:
    """Read model_comparison.csv and return top N model names by AUC."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # CSV is already sorted by AUC descending, but sort explicitly to be safe
    rows.sort(key=lambda r: float(r["auc"]), reverse=True)
    return [row["model_name"] for row in rows[:top_n]]


def main():
    parser = argparse.ArgumentParser(description="Phase 2 batch training")
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of top models from Phase 1 to retrain (default: 10)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Train all available configs instead of top N",
    )
    args = parser.parse_args()

    if args.all:
        configs = sorted(CONFIGS_DIR.glob("*.yaml"))
    else:
        if not PHASE1_CSV.exists():
            print(f"Nie znaleziono wyników Fazy 1: {PHASE1_CSV}")
            print("Uruchom --all lub najpierw wygeneruj model_comparison.csv")
            return

        top_names = _read_top_models(PHASE1_CSV, args.top)
        print(f"Top {len(top_names)} modeli z Fazy 1 (wg AUC):")
        for i, name in enumerate(top_names, 1):
            print(f"  {i:2d}. {name}")
        print()

        configs = []
        for name in top_names:
            cfg = CONFIGS_DIR / f"{name}.yaml"
            if cfg.exists():
                configs.append(cfg)
            else:
                print(f"  ⚠ Brak konfiguracji: {cfg.name} — pomijam")

    if not configs:
        print("Nie znaleziono plików .yaml do trenowania!")
        return
    configs = configs[9:]
    print(f"Uruchamiam trening sekwencyjny dla {len(configs)} modeli...")
    print(f"Logi poszczególnych modeli będą zapisywane w folderze logs/")
    print("-" * 60)

    for i, config_path in enumerate(configs, 1):
        config_name = config_path.name
        print(f"[{i}/{len(configs)} | CPU {CPU_RANGE}] Start: {config_name}")

        LOGS_DIR.mkdir(exist_ok=True)
        log_file = LOGS_DIR / f"{config_path.stem}.log"

        cmd = [
            sys.executable, "main.py",
            "--config", str(config_path),
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
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

        if result.returncode == 0:
            print(f"[{i}/{len(configs)} | CPU {CPU_RANGE}] SUKCES: {config_name}")
        else:
            print(f"[{i}/{len(configs)} | CPU {CPU_RANGE}] BŁĄD ({result.returncode}): {config_name} (zobacz {log_file})")

    print("-" * 60)
    print("Wszystkie zadania treningowe zostały zakończone!")


if __name__ == "__main__":
    main()
