import os
import glob
import argparse
import subprocess
from pathlib import Path
import re
import numpy as np

def run_multiseed_for_config(config_path, seeds, env):
    print(f"\n============================================================")
    print(f"Starting Multi-Seed Training for: {config_path.name}")
    print(f"Seeds to evaluate: {seeds}")
    print(f"============================================================")

    # Mirror log folder structure
    try:
        rel_path = config_path.parent.relative_to("configs")
        log_dir = Path("logs") / rel_path
    except ValueError:
        log_dir = Path("logs") / config_path.parent.name if config_path.parent.name != "." else Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = log_dir / f"{config_path.stem}_multiseed_report.txt"
    
    metrics = {
        "auc": [],
        "auprc": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "loss": []
    }

    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] Running training with seed {seed}...")
        log_file = log_dir / f"{config_path.stem}_seed{seed}.log"
        
        cmd = [
            "uv", "run", "main.py",
            "--config", str(config_path),
            "--seed", str(seed)
        ]
        
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            
        if result.returncode != 0:
            print(f"  -> ERROR during seed {seed}. Check log: {log_file}")
            continue
            
        # Parse log for Test Results
        with open(log_file, "r") as f:
            log_content = f.read()
            
        # Extract metrics using regex
        auc_match = re.search(r"auc:\s+([0-9.]+)", log_content)
        auprc_match = re.search(r"auprc:\s+([0-9.]+)", log_content)
        f1_match = re.search(r"f1:\s+([0-9.]+)", log_content)
        precision_match = re.search(r"precision:\s+([0-9.]+)", log_content)
        recall_match = re.search(r"recall:\s+([0-9.]+)", log_content)
        loss_match = re.search(r"loss:\s+([0-9.]+)", log_content)
        
        if auc_match:
            metrics["auc"].append(float(auc_match.group(1)))
            metrics["auprc"].append(float(auprc_match.group(1)))
            metrics["f1"].append(float(f1_match.group(1)))
            metrics["precision"].append(float(precision_match.group(1)))
            metrics["recall"].append(float(recall_match.group(1)))
            metrics["loss"].append(float(loss_match.group(1)))
            print(f"  -> SUCCESS! Test AUC: {metrics['auc'][-1]:.4f}")
        else:
            print(f"  -> SUCCESS but could not parse metrics from {log_file}")

    print(f"\n============================================================")
    print(f"FINAL MULTI-SEED REPORT for {config_path.name}")
    print(f"============================================================")
    
    report_lines = []
    report_lines.append(f"Model: {config_path.name}")
    report_lines.append(f"Seeds: {seeds}")
    report_lines.append(f"Runs completed successfully: {len(metrics['auc'])}/{len(seeds)}\n")
    
    report_lines.append("Aggregated Metrics (Mean ± Std):")
    for metric_name, values in metrics.items():
        if values:
            mean = np.mean(values)
            std = np.std(values)
            report_lines.append(f"  {metric_name:>12s}: {mean:.4f} ± {std:.4f}")
            
    final_report = "\n".join(report_lines)
    print(final_report)
    print(f"============================================================")
    
    with open(report_file, "w") as f:
        f.write(final_report)
    print(f"Saved report to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Run training multiple times with different seeds for one config or a whole directory.")
    parser.add_argument("target", type=str, help="Path to a single config file or a directory (e.g. configs/lincs/)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999, 1024, 2026], help="List of seeds to use")
    args = parser.parse_args()

    target_path = Path(args.target)
    if not target_path.exists():
        print(f"Error: Target {target_path} does not exist.")
        return
        
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if target_path.is_file():
        configs = [target_path]
    elif target_path.is_dir():
        configs = sorted([Path(p) for p in glob.glob(f"{target_path}/*.yaml")], key=lambda x: x.name.replace('.', '~'))
        print(f"Znaleziono {len(configs)} plików .yaml w folderze {target_path}")
    else:
        print(f"Error: Target is neither a file nor a directory.")
        return

    if not configs:
        print("Nie znaleziono żadnych plików .yaml do uruchomienia.")
        return

    for i, config_path in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Processing configuration: {config_path.name}")
        run_multiseed_for_config(config_path, args.seeds, env)

    print("\n------------------------------------------------------------")
    print("Wszystkie zadania wielokrotnego treningu (multi-seed) zostały zakończone!")
    print("------------------------------------------------------------")

if __name__ == "__main__":
    main()
