import polars as pl
from collections import Counter

def calc_aa_freq(proteins: list[str]) -> dict[str, float]:
    """Oblicza częstotliwość występowania każdego aminokwasu w liście sekwencji."""
    total_length = 0
    counts = Counter()
    for p in proteins:
        counts.update(p)
        total_length += len(p)
    
    # Zwracamy procentowy udział każdego aminokwasu
    return {aa: (count / total_length) * 100 for aa, count in counts.items()}

def main():
    print("Wczytywanie zbioru danych...")
    df = pl.read_parquet('datasets/clean.parquet')
    
    # Pobieramy UNIKALNE białka z podziałem na aktywność, aby uniknąć biasu "hubów"
    active_proteins = df.filter(pl.col('is_active') == True)['Full_Protein_Sequence'].unique().to_list()
    inactive_proteins = df.filter(pl.col('is_active') == False)['Full_Protein_Sequence'].unique().to_list()
    
    print(f"Liczba unikalnych białek (aktywne): {len(active_proteins)}")
    print(f"Liczba unikalnych białek (nieaktywne): {len(inactive_proteins)}\n")
    
    # Obliczamy częstotliwości
    active_freq = calc_aa_freq(active_proteins)
    inactive_freq = calc_aa_freq(inactive_proteins)
    
    # Wyświetlamy statystyki z boku na bok
    print(f"{'Aminokwas':<12} | {'Aktywne (%)':<12} | {'Nieaktywne (%)':<12}")
    print("-" * 42)
    
    all_aas = sorted(set(list(active_freq.keys()) + list(inactive_freq.keys())))
    for aa in all_aas:
        if aa.isalpha():
            act_val = active_freq.get(aa, 0.0)
            inact_val = inactive_freq.get(aa, 0.0)
            print(f"{aa:<12} | {act_val:<12.2f} | {inact_val:<12.2f}")

if __name__ == '__main__':
    main()
