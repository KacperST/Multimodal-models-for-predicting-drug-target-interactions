import torch
import sys

def analyze_descriptors(file_path: str):
    print(f"Loading {file_path}...")
    try:
        cache = torch.load(file_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    num_mols = len(cache)
    print(f"Liczba cząsteczek: {num_mols}")

    if num_mols == 0:
        return

    # Złączenie wszystkich wektorów w jedną dużą macierz (Molecules x 210)
    all_tensors = torch.stack(list(cache.values()))
    
    num_elements = all_tensors.numel()
    
    # Obliczanie statystyk
    num_zeros = (all_tensors == 0.0).sum().item()
    num_infs = torch.isinf(all_tensors).sum().item()
    num_nans = torch.isnan(all_tensors).sum().item()

    print(f"\nOgólne statystyki (Łącznie wartości: {num_elements}):")
    print(f"Zera: {num_zeros} ({num_zeros/num_elements*100:.2f}%)")
    print(f"Nieskończoności (inf): {num_infs} ({num_infs/num_elements*100:.6f}%)")
    print(f"NaN-y: {num_nans} ({num_nans/num_elements*100:.6f}%)")

    # Sprawdzamy ile cząsteczek zawiera chociaż jedno zero/inf/NaN
    mols_with_zeros = ((all_tensors == 0.0).sum(dim=1) > 0).sum().item()
    mols_with_infs = (torch.isinf(all_tensors).sum(dim=1) > 0).sum().item()
    mols_with_nans = (torch.isnan(all_tensors).sum(dim=1) > 0).sum().item()

    print(f"\nStatystyki per cząsteczka (Łącznie cząsteczek: {num_mols}):")
    print(f"Cząsteczki z co najmniej 1 zerem: {mols_with_zeros} ({mols_with_zeros/num_mols*100:.2f}%)")
    print(f"Cząsteczki z co najmniej 1 inf: {mols_with_infs} ({mols_with_infs/num_mols*100:.6f}%)")
    print(f"Cząsteczki z co najmniej 1 NaN: {mols_with_nans} ({mols_with_nans/num_mols*100:.6f}%)")

    # Wartości min, max i średnie (tylko na prawidłowych danych)
    valid_tensors = all_tensors[~(torch.isinf(all_tensors) | torch.isnan(all_tensors))]
    if valid_tensors.numel() > 0:
        print(f"\nAnaliza tylko poprawnych wartości:")
        print(f"Min: {valid_tensors.min().item():.4f}")
        print(f"Max: {valid_tensors.max().item():.4f}")
        print(f"Mean: {valid_tensors.mean().item():.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_descriptors(sys.argv[1])
    else:
        print("Usage: python check_rdkit_stats.py <path_to.pt>")
