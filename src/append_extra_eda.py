import json

with open('src/dataset_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = [
  # ── pKi Distribution ──
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "\n",
    "### 4. pKi Value Distribution\n",
    "\n",
    "The target variable in its continuous form. The red dashed line marks the activity threshold (pKi = 7.0, i.e. Ki = 100 nM)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
    "\n",
    "# Histogram + KDE\n",
    "sns.histplot(data=df.to_pandas(), x=\"pKi\", bins=60, kde=True, ax=axes[0], color=\"#3b82f6\")\n",
    "axes[0].axvline(7.0, color=\"red\", linestyle=\"--\", lw=2, label=\"Activity threshold (pKi = 7.0)\")\n",
    "axes[0].set_title(\"pKi Value Distribution\")\n",
    "axes[0].set_xlabel(\"pKi = -log10(Ki)\")\n",
    "axes[0].set_ylabel(\"Count\")\n",
    "axes[0].legend()\n",
    "\n",
    "# Box plot by class\n",
    "df_pd = df.to_pandas()\n",
    "df_pd[\"Class\"] = df_pd[\"is_active\"].map({True: \"Active\", False: \"Inactive\"})\n",
    "sns.boxplot(data=df_pd, x=\"Class\", y=\"pKi\", ax=axes[1],\n",
    "            palette={\"Active\": \"#22c55e\", \"Inactive\": \"#ef4444\"})\n",
    "axes[1].axhline(7.0, color=\"red\", linestyle=\"--\", lw=1, alpha=0.5)\n",
    "axes[1].set_title(\"pKi Distribution by Class\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f\"pKi range: [{df['pKi'].min():.2f}, {df['pKi'].max():.2f}]\")\n",
    "print(f\"Mean: {df['pKi'].mean():.3f}  |  Median: {df['pKi'].median():.3f}  |  Std: {df['pKi'].std():.3f}\")"
   ]
  },
  # ── Scaffold Analysis ──
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "\n",
    "### 5. Murcko Scaffold Analysis\n",
    "\n",
    "Since we use **scaffold-based splitting** to divide the data into train/val/test sets, it is important to understand the chemical diversity at the scaffold level. A scaffold is the core ring system of a molecule (Murcko decomposition). If a few scaffolds dominate the dataset, the model might overfit to those particular chemotypes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from rdkit import Chem\n",
    "from rdkit.Chem.Scaffolds import MurckoScaffold\n",
    "from collections import Counter\n",
    "\n",
    "def get_scaffold(smiles: str) -> str:\n",
    "    \"\"\"Return the generic Murcko scaffold SMILES for a given molecule.\"\"\"\n",
    "    mol = Chem.MolFromSmiles(smiles)\n",
    "    if mol is None:\n",
    "        return \"\"\n",
    "    try:\n",
    "        scaffold = MurckoScaffold.MakeScaffoldGeneric(\n",
    "            MurckoScaffold.GetScaffoldForMol(mol)\n",
    "        )\n",
    "        return Chem.MolToSmiles(scaffold)\n",
    "    except Exception:\n",
    "        return \"\"\n",
    "\n",
    "unique_smiles = df[\"Ligand SMILES\"].unique().to_list()\n",
    "print(f\"Computing scaffolds for {len(unique_smiles):,} unique SMILES ...\")\n",
    "scaffolds = [get_scaffold(s) for s in unique_smiles]\n",
    "\n",
    "scaffold_counter = Counter(scaffolds)\n",
    "n_scaffolds = len([s for s in scaffold_counter if s != \"\"])\n",
    "\n",
    "print(f\"Unique SMILES:    {len(unique_smiles):,}\")\n",
    "print(f\"Unique scaffolds: {n_scaffolds:,}\")\n",
    "print(f\"Ratio (SMILES / scaffolds): {len(unique_smiles) / n_scaffolds:.1f}x\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Top 15 most common scaffolds\n",
    "top_scaffolds = scaffold_counter.most_common(16)\n",
    "# skip empty scaffold if present\n",
    "top_scaffolds = [(s, c) for s, c in top_scaffolds if s != \"\"][:15]\n",
    "\n",
    "labels = [s if len(s) < 40 else s[:37] + \"...\" for s, _ in top_scaffolds]\n",
    "counts = [c for _, c in top_scaffolds]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12, 6))\n",
    "bars = ax.barh(range(len(counts)), counts, color=\"#6366f1\", edgecolor=\"white\")\n",
    "ax.set_yticks(range(len(counts)))\n",
    "ax.set_yticklabels(labels, fontsize=9, fontfamily=\"monospace\")\n",
    "ax.invert_yaxis()\n",
    "ax.set_xlabel(\"Number of unique molecules\")\n",
    "ax.set_title(\"Top 15 Most Common Murcko Scaffolds\")\n",
    "for bar, c in zip(bars, counts):\n",
    "    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,\n",
    "            f\"{c:,}\", va=\"center\", fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Distribution of molecules per scaffold (long-tail)\n",
    "scaffold_sizes = sorted(scaffold_counter.values(), reverse=True)\n",
    "# remove empty scaffold\n",
    "scaffold_sizes = [s for s in scaffold_sizes if s > 0]\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "axes[0].hist(scaffold_sizes, bins=50, color=\"#14b8a6\", log=True, edgecolor=\"white\")\n",
    "axes[0].set_title(\"Molecules per scaffold (log scale)\")\n",
    "axes[0].set_xlabel(\"Number of molecules sharing a scaffold\")\n",
    "axes[0].set_ylabel(\"Number of scaffolds\")\n",
    "\n",
    "# Cumulative coverage\n",
    "cumsum = np.cumsum(scaffold_sizes) / sum(scaffold_sizes) * 100\n",
    "axes[1].plot(range(1, len(cumsum)+1), cumsum, color=\"#f97316\", lw=2)\n",
    "axes[1].set_title(\"Cumulative molecule coverage by scaffold rank\")\n",
    "axes[1].set_xlabel(\"Number of scaffolds (ranked by frequency)\")\n",
    "axes[1].set_ylabel(\"% of all unique molecules covered\")\n",
    "axes[1].axhline(80, color=\"gray\", linestyle=\"--\", lw=1, alpha=0.5, label=\"80%\")\n",
    "axes[1].axhline(50, color=\"gray\", linestyle=\":\", lw=1, alpha=0.5, label=\"50%\")\n",
    "axes[1].legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# How many scaffolds cover 50% and 80% of molecules?\n",
    "n_50 = np.searchsorted(cumsum, 50) + 1\n",
    "n_80 = np.searchsorted(cumsum, 80) + 1\n",
    "print(f\"Scaffolds needed to cover 50% of molecules: {n_50:,} (out of {n_scaffolds:,})\")\n",
    "print(f\"Scaffolds needed to cover 80% of molecules: {n_80:,} (out of {n_scaffolds:,})\")"
   ]
  },
  # ── Amino Acid Composition ──
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "\n",
    "### 6. Amino Acid Composition\n",
    "\n",
    "Frequency of individual amino acids across all protein sequences. We compare the composition between Active and Inactive classes to check if certain protein types (e.g. cysteine-rich) are overrepresented in one class."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from collections import Counter as Ctr\n",
    "\n",
    "STANDARD_AA = list(\"ACDEFGHIKLMNPQRSTVWY\")\n",
    "\n",
    "def count_aa(sequences: list[str]) -> dict:\n",
    "    total = Ctr()\n",
    "    for seq in sequences:\n",
    "        total.update(seq)\n",
    "    # Normalize to frequency\n",
    "    s = sum(total[aa] for aa in STANDARD_AA)\n",
    "    return {aa: total[aa] / s * 100 for aa in STANDARD_AA}\n",
    "\n",
    "active_seqs = df.filter(pl.col(\"is_active\"))[\"Full_Protein_Sequence\"].to_list()\n",
    "inactive_seqs = df.filter(~pl.col(\"is_active\"))[\"Full_Protein_Sequence\"].to_list()\n",
    "\n",
    "freq_active = count_aa(active_seqs)\n",
    "freq_inactive = count_aa(inactive_seqs)\n",
    "\n",
    "x = np.arange(len(STANDARD_AA))\n",
    "width = 0.35\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(14, 5))\n",
    "ax.bar(x - width/2, [freq_active[aa] for aa in STANDARD_AA], width,\n",
    "       label=\"Active\", color=\"#22c55e\", alpha=0.8)\n",
    "ax.bar(x + width/2, [freq_inactive[aa] for aa in STANDARD_AA], width,\n",
    "       label=\"Inactive\", color=\"#ef4444\", alpha=0.8)\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(STANDARD_AA, fontsize=12, fontweight=\"bold\")\n",
    "ax.set_ylabel(\"Frequency (%)\")\n",
    "ax.set_title(\"Amino Acid Composition: Active vs Inactive\")\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# Show the biggest differences\n",
    "diffs = {aa: abs(freq_active[aa] - freq_inactive[aa]) for aa in STANDARD_AA}\n",
    "sorted_diffs = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:5]\n",
    "print(\"Top 5 amino acids with the largest frequency difference between classes:\")\n",
    "for aa, d in sorted_diffs:\n",
    "    print(f\"  {aa}: Active={freq_active[aa]:.2f}%  Inactive={freq_inactive[aa]:.2f}%  (Δ={d:.2f}pp)\")"
   ]
  },
  # ── Duplicate Analysis ──
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "\n",
    "### 7. Duplicate Pair Analysis\n",
    "\n",
    "We check whether the same (SMILES, protein) pair appears more than once in the dataset. Duplicates could inflate evaluation metrics if the same pair lands in both the training and test sets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "pair_counts = (\n",
    "    df\n",
    "    .group_by([\"Ligand SMILES\", \"Full_Protein_Sequence\"])\n",
    "    .agg(pl.len().alias(\"n_occurrences\"))\n",
    ")\n",
    "\n",
    "n_unique_pairs = pair_counts.height\n",
    "n_duplicated_pairs = pair_counts.filter(pl.col(\"n_occurrences\") > 1).height\n",
    "n_duplicated_rows = pair_counts.filter(pl.col(\"n_occurrences\") > 1)[\"n_occurrences\"].sum()\n",
    "\n",
    "print(f\"Total rows in dataset:          {df.height:,}\")\n",
    "print(f\"Unique (SMILES, protein) pairs:  {n_unique_pairs:,}\")\n",
    "print(f\"Pairs appearing more than once:  {n_duplicated_pairs:,}\")\n",
    "if n_duplicated_rows is not None:\n",
    "    print(f\"Total rows from duplicated pairs: {n_duplicated_rows:,}\")\n",
    "    print(f\"Duplicate rate: {n_duplicated_pairs / n_unique_pairs * 100:.2f}%\")\n",
    "else:\n",
    "    print(\"No duplicates found.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if n_duplicated_pairs > 0:\n",
    "    # Check if duplicated pairs have conflicting labels\n",
    "    conflicting = (\n",
    "        df\n",
    "        .group_by([\"Ligand SMILES\", \"Full_Protein_Sequence\"])\n",
    "        .agg([\n",
    "            pl.len().alias(\"n\"),\n",
    "            pl.col(\"is_active\").n_unique().alias(\"n_labels\"),\n",
    "            pl.col(\"pKi\").mean().alias(\"mean_pKi\"),\n",
    "            pl.col(\"pKi\").std().alias(\"std_pKi\"),\n",
    "        ])\n",
    "        .filter(pl.col(\"n\") > 1)\n",
    "    )\n",
    "    n_conflicting = conflicting.filter(pl.col(\"n_labels\") > 1).height\n",
    "    print(f\"Duplicated pairs with CONFLICTING labels (Active in one, Inactive in another): {n_conflicting:,}\")\n",
    "    print(f\"Duplicated pairs with CONSISTENT labels: {n_duplicated_pairs - n_conflicting:,}\")\n",
    "    \n",
    "    if n_conflicting > 0:\n",
    "        print(f\"\\nThis means {n_conflicting:,} pairs have measurements on both sides of the pKi=7.0 threshold.\")\n",
    "        print(\"This is expected — experimental Ki values have inherent measurement noise.\")\n",
    "        print(f\"\\nSample conflicting pairs:\")\n",
    "        sample = conflicting.filter(pl.col(\"n_labels\") > 1).sort(\"n\", descending=True).head(5)\n",
    "        print(sample.select([\"n\", \"n_labels\", \"mean_pKi\", \"std_pKi\"]))\n",
    "else:\n",
    "    print(\"No duplicate pairs found — each (drug, protein) combination is unique.\")"
   ]
  }
]

nb['cells'].extend(new_cells)

with open('src/dataset_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("4 new sections appended successfully!")
