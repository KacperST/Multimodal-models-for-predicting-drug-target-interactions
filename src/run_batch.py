import os
import glob
import queue
import threading
import subprocess
from pathlib import Path

# Definiujemy 4 sloty "workerów" i ich przypisanie do rdzeni CPU (taskset)
# Dostosuj te wartości, jeśli masz mniej lub więcej rdzeni w procesorze!
CPU_SLOTS = [
    "0-3",    # Worker 1: rdzenie 0-3
    "4-7",    # Worker 2: rdzenie 4-7
    "8-11",   # Worker 3: rdzenie 8-11
    "12-15"   # Worker 4: rdzenie 12-15
]

def worker(slot_id, cpu_range, q):
    while True:
        try:
            config_path = q.get_nowait()
        except queue.Empty:
            # Brak kolejnych zadań w kolejce
            break
            
        config_name = Path(config_path).name
        print(f"[Slot {slot_id} | CPU {cpu_range}] Start: {config_name}")
        
        # Logowanie do pliku, żeby 4 procesy nie mieszały outputu w terminalu
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{Path(config_path).stem}.log"
        
        cmd = [
            "taskset", "-c", cpu_range,
            "uv", "run", "main.py",
            "--config", config_path
        ]
        
        # Zabezpieczenie przed tzw. "Thread Oversubscription Deadlock"
        env = os.environ.copy()
        # Skoro dajesz 4 rdzenie na model (np. 0-3), powiedzmy bibliotekom C++, żeby nie próbowały
        # używać domyślnych 32/64 wątków (bo system wciąż widzi wszystkie procesory).
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        env["OPENBLAS_NUM_THREADS"] = "4"
        env["POLARS_MAX_THREADS"] = "4"
        # Zapobiega opóźnieniom w logach (wymusza natychmiastowy zapis do pliku bez bufforowania)
        env["PYTHONUNBUFFERED"] = "1"

        with open(log_file, "w") as f:
            # Uruchomienie procesu podrzędnego
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            
        if result.returncode == 0:
            print(f"[Slot {slot_id} | CPU {cpu_range}] SUKCES: {config_name}")
        else:
            print(f"[Slot {slot_id} | CPU {cpu_range}] BŁĄD ({result.returncode}): {config_name} (zobacz {log_file})")
            
        q.task_done()

def main():
    # Pobierz wszystkie wygenerowane configi
    configs = sorted(glob.glob("configs/*.yaml"))
    if not configs:
        print("Nie znaleziono plików .yaml w folderze configs/")
        return

    # Umieść je w bezpiecznej dla wątków kolejce
    q = queue.Queue()
    for c in configs:
        q.put(c)

    print(f"Kolejka: {len(configs)} modeli. Uruchamiam {len(CPU_SLOTS)} procesów jednocześnie...")
    print(f"Logi poszczególnych modeli będą zapisywane w folderze src/logs/")
    print("-" * 60)

    # Uruchom 4 workery (wątki)
    threads = []
    for i, cpu_range in enumerate(CPU_SLOTS):
        t = threading.Thread(target=worker, args=(i + 1, cpu_range, q))
        t.start()
        threads.append(t)

    # Czekaj aż wszystkie wątki pobiorą i skończą zadania z kolejki
    for t in threads:
        t.join()
        
    print("-" * 60)
    print("Wszystkie zadania treningowe zostały zakończone!")

if __name__ == "__main__":
    main()
