from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import torch
from torch_geometric.loader import DataLoader

from data.loader import load_bindingdb_data
from data.transform import (
    create_pyg_dataset,
    remove_cx_notation,
    remove_nulls,
    train_test_val_split,
    tranform_ki_to_log_ki,
)
from models.gcn import SimpleGCN


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "datasets" / "BindingDB_All.tsv"
NUM_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
HIDDEN_CHANNELS = 64


def analyze_weird_smiles(df: pl.DataFrame, col_name: str = "Ligand SMILES") -> pl.DataFrame:
    weird_df = df.filter(pl.col(col_name).str.contains(r"\|"))

    total_records = df.height
    weird_count = weird_df.height
    percentage = (weird_count / total_records) * 100 if total_records else 0.0
    unique_weird = weird_df.select(col_name).unique()

    print("--- Analiza formatu CXSMILES ---")
    print(f"Suma wszystkich rekordow: {total_records}")
    print(f"Liczba rekordow z '|...|': {weird_count} ({percentage:.2f}%)")
    print(f"Liczba unikalnych struktur z tym zapisem: {unique_weird.height}")
    print("-" * 32)

    if weird_count > 0:
        print("Przykladowe smilesy z dziwnym zapisem:")
        for smiles in unique_weird.head(5).to_series():
            print(f"-> {smiles}")
    else:
        print("Nie znaleziono zadnych rekordow z tym formatem.")

    return unique_weird


def describe_protein_sequence_lengths(df: pl.DataFrame) -> pl.DataFrame:
    df_with_lengths = df.with_columns(
        pl.col("Full_Protein_Sequence")
        .str.strip_chars()
        .str.len_chars()
        .alias("sequence_length")
    )

    print(df_with_lengths["sequence_length"].describe())
    print(f"Liczba unikalnych dlugosci: {df_with_lengths['sequence_length'].n_unique()}")
    print(df_with_lengths["sequence_length"].value_counts(sort=True).head(20))

    return df_with_lengths


def plot_smiles_frequency_distribution(df: pl.DataFrame) -> None:
    smiles_hist = df["Ligand SMILES"].value_counts(sort=True)
    count_distribution = (
        smiles_hist["count"]
        .alias("c")
        .value_counts()
        .sort("count")
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(
        count_distribution["c"],
        count_distribution["count"],
        alpha=0.6,
        color="darkorange",
        edgecolors="white",
        s=50,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Rozklad 'Frequency-of-Frequency' dla SMILES (Skala Log-Log)", fontsize=14)
    plt.xlabel("Liczba wystapien (Jak popularny jest lek)", fontsize=12)
    plt.ylabel("Liczba unikalnych SMILES (Ile jest takich lekow)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_protein_length_distribution(df: pl.DataFrame) -> None:
    lengths = (
        df.with_columns(
            pl.col("Full_Protein_Sequence").str.strip_chars().str.len_chars().alias("sequence_length")
        )["sequence_length"]
        .to_list()
    )

    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=60, color="steelblue", edgecolor="white")
    plt.title("Rozklad dlugosci Full_Protein_Sequence")
    plt.xlabel("Dlugosc sekwencji (aa)")
    plt.ylabel("Liczba rekordow")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def build_graph_dataset(df: pl.DataFrame, smiles_map: dict[str, torch.Tensor], target_col: str = "pKi"):
    dataset = []

    for smiles, target in zip(df["Ligand SMILES"].to_list(), df[target_col].to_list()):
        graph = smiles_map[smiles].clone()
        graph.y = torch.tensor([float(target)], dtype=torch.float)
        dataset.append(graph)

    return dataset


def train(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        predictions = model(batch).view(-1)
        loss = criterion(predictions, batch.y.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        predictions = model(batch).view(-1)
        loss = criterion(predictions, batch.y.view(-1))
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def main() -> None:
    pl.Config.set_fmt_str_lengths(1000)

    df = load_bindingdb_data(str(DATA_PATH))
    print(df.head())

    analyze_weird_smiles(df)

    df = remove_cx_notation(df)
    analyze_weird_smiles(df)

    describe_protein_sequence_lengths(df)

    df = remove_nulls(df)
    df = tranform_ki_to_log_ki(df)
    print(df["pKi"].describe())

    plot_smiles_frequency_distribution(df)
    plot_protein_length_distribution(df)

    train_df, val_df, test_df = train_test_val_split(df)
    print(train_df.shape)
    print(val_df.shape)
    print(test_df.shape)

    all_unique_smiles = df["Ligand SMILES"].unique().to_list()
    all_graphs = create_pyg_dataset(all_unique_smiles, with_features=False)
    smiles_map = dict(zip(all_unique_smiles, all_graphs))

    valid_smiles = set(smiles_map.keys())
    print(f"Liczba poprawnych grafow w slowniku: {len(valid_smiles)}")

    train_df = train_df.filter(pl.col("Ligand SMILES").is_in(valid_smiles))
    val_df = val_df.filter(pl.col("Ligand SMILES").is_in(valid_smiles))
    test_df = test_df.filter(pl.col("Ligand SMILES").is_in(valid_smiles))

    print(f"Rekordy w treningu po synchronizacji: {train_df.height}")
    print(f"Rekordy w walidacji po synchronizacji: {val_df.height}")
    print(f"Rekordy w teście po synchronizacji: {test_df.height}")

    train_dataset = build_graph_dataset(train_df, smiles_map)
    val_dataset = build_graph_dataset(val_df, smiles_map)
    test_dataset = build_graph_dataset(test_df, smiles_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SimpleGCN(hidden_channels=HIDDEN_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoka: {epoch:03d}, "
            f"Train loss: {train_loss:.4f}, "
            f"Val loss: {val_loss:.4f}, "
            f"Test loss: {test_loss:.4f}"
        )


if __name__ == "__main__":
    main()