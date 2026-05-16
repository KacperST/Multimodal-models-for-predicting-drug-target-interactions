import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Dataset Analysis\n",
                "Ten notebook ładuje zbiór danych DTI i analizuje rozkład długości sekwencji białek oraz uśmiechów (SMILES), a także podstawowe statystyki par DTI."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import polars as pl\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import numpy as np\n",
                "\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "plt.rcParams['figure.figsize'] = (12, 5)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Wczytanie danych\n",
                "df = pl.read_parquet('datasets/clean.parquet')\n",
                "print(f'Liczba wszystkich par: {df.height:,}')\n",
                "print(f'Unikalne leki (SMILES): {df[\"Ligand SMILES\"].n_unique():,}')\n",
                "print(f'Unikalne białka: {df[\"Full_Protein_Sequence\"].n_unique():,}')\n",
                "\n",
                "active_counts = df[\"is_active\"].value_counts()\n",
                "print(f'\\nRozkład klas (aktywne/nieaktywne):\\n{active_counts}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Obliczanie długości znakowych (jest to rewelacyjne przybliżenie liczby tokenów dla SMILES i białek)\n",
                "df = df.with_columns([\n",
                "    pl.col(\"Ligand SMILES\").str.len_chars().alias(\"SMILES_Length\"),\n",
                "    pl.col(\"Full_Protein_Sequence\").str.len_chars().alias(\"Protein_Length\")\n",
                "])\n",
                "\n",
                "print(\"Statystyki długości znakowej SMILES:\")\n",
                "print(df[\"SMILES_Length\"].describe())\n",
                "\n",
                "print(\"\\nStatystyki długości znakowej Białek:\")\n",
                "print(df[\"Protein_Length\"].describe())"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Wykresy rozkładów\n",
                "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
                "\n",
                "# Używamy 99. percentyla by usunąć ekstrema do ładnego wykresu\n",
                "smiles_p99 = df[\"SMILES_Length\"].quantile(0.99)\n",
                "protein_p99 = df[\"Protein_Length\"].quantile(0.99)\n",
                "\n",
                "sns.histplot(df.filter(pl.col(\"SMILES_Length\") <= smiles_p99)[\"SMILES_Length\"].to_numpy(), \n",
                "             bins=50, ax=axes[0], color='skyblue')\n",
                "axes[0].set_title(f'Rozkład długości SMILES (odcięto 1% ekstremów > {int(smiles_p99)})')\n",
                "axes[0].set_xlabel('Liczba znaków')\n",
                "axes[0].set_ylabel('Liczba próbek')\n",
                "\n",
                "sns.histplot(df.filter(pl.col(\"Protein_Length\") <= protein_p99)[\"Protein_Length\"].to_numpy(), \n",
                "             bins=50, ax=axes[1], color='salmon')\n",
                "axes[1].set_title(f'Rozkład długości Białek (odcięto 1% ekstremów > {int(protein_p99)})')\n",
                "axes[1].set_xlabel('Liczba znaków')\n",
                "axes[1].set_ylabel('Liczba próbek')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Analiza ucięcia długości (Truncation)\n",
                "W plikach konfiguracyjnych mamy ustawione `max_length` (np. 1024 dla ESM2 i 256 dla ChemBERT).\n",
                "Zobaczmy, jaki procent naszego zbioru danych przekracza te wartości."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "max_smiles = 256\n",
                "max_protein = 1024\n",
                "\n",
                "smiles_cut = df.filter(pl.col(\"SMILES_Length\") > max_smiles).height / df.height * 100\n",
                "protein_cut = df.filter(pl.col(\"Protein_Length\") > max_protein).height / df.height * 100\n",
                "\n",
                "print(f\"Procent leków, które zostaną obcięte przy długości {max_smiles}: {smiles_cut:.2f}%\")\n",
                "print(f\"Procent białek, które zostaną obcięte przy długości {max_protein}: {protein_cut:.2f}%\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("dataset_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook utworzony!")
