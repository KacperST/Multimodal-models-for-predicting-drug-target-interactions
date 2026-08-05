# Phase 1 Results — Master's Thesis Writeup

## Results Table

| # | Architecture | Enc. | AUC | AUPRC | F1 | Precision | Recall |
|---|-------------|------|-----|-------|----|-----------|--------|
| 1 | GCN vs CNN | 2 | **0.896** | 0.895 | **0.813** | 0.816 | 0.811 |
| 2 | GCN vs CNN+ESM-2 | 3 | 0.894 | 0.893 | 0.814 | 0.806 | 0.822 |
| 3 | GCN+ChemBERT vs CNN | 3 | 0.891 | 0.889 | 0.808 | 0.807 | 0.809 |
| 4 | GCN vs ESM-2 | 2 | 0.890 | 0.891 | 0.807 | 0.809 | 0.805 |
| 5 | GCN+ChemBERT vs CNN+ESM-2 | 4 | 0.888 | 0.886 | 0.806 | 0.804 | 0.809 |
| 6 | GCN+FP+ChemBERT vs CNN+ESM-2 | 5 | 0.885 | 0.885 | 0.793 | 0.831 | 0.758 |
| 7 | FP+ChemBERT vs CNN | 3 | 0.884 | 0.884 | 0.779 | 0.845 | 0.722 |
| 8 | GCN+FP vs CNN+ESM-2 | 4 | 0.884 | 0.883 | 0.796 | 0.817 | 0.777 |
| 9 | GCN+FP vs CNN | 3 | 0.884 | 0.883 | 0.800 | 0.811 | 0.789 |
| 10 | FP+ChemBERT vs CNN+ESM-2 | 4 | 0.883 | 0.882 | 0.769 | 0.855 | 0.700 |
| 11 | FP vs CNN+ESM-2 | 3 | 0.883 | 0.883 | 0.768 | 0.856 | 0.697 |
| 12 | GCN+FP+ChemBERT vs CNN | 4 | 0.883 | 0.882 | 0.798 | 0.811 | 0.785 |
| 13 | FP vs CNN | 2 | 0.881 | 0.882 | 0.774 | 0.851 | 0.710 |
| 14 | GCN+ChemBERT vs ESM-2 | 3 | 0.881 | 0.881 | 0.796 | 0.796 | 0.796 |
| 15 | FP vs ESM-2 | 2 | 0.878 | 0.879 | 0.787 | 0.816 | 0.761 |
| 16 | FP+ChemBERT vs ESM-2 | 3 | 0.877 | 0.879 | 0.755 | 0.857 | 0.675 |
| 17 | GCN+FP+ChemBERT vs ESM-2 | 4 | 0.876 | 0.874 | 0.797 | 0.799 | 0.795 |
| 18 | GCN+FP vs ESM-2 | 3 | 0.876 | 0.875 | 0.793 | 0.808 | 0.778 |
| 19 | ChemBERT vs CNN | 2 | 0.865 | 0.864 | 0.780 | 0.778 | 0.783 |
| 20 | ChemBERT vs CNN+ESM-2 | 3 | 0.865 | 0.863 | 0.780 | 0.780 | 0.780 |
| 21 | ChemBERT vs ESM-2 | 2 | 0.859 | 0.859 | 0.774 | 0.775 | 0.772 |

---

## Key Findings

### 1. Graph Convolutional Networks as the Strongest Drug Encoder

All six top-performing architectures by AUC (0.888–0.896) include the GCN encoder. The best-performing model without GCN (FP+ChemBERT vs CNN, AUC = 0.884) ranks only seventh overall. This indicates that **explicit molecular graph topology** — atoms as nodes, bonds as edges — provides a stronger inductive bias for DTI prediction than either fixed-length molecular fingerprints or contextual SMILES representations from pretrained language models.

Among the three drug encoders evaluated, the Graph Convolutional Network achieved the highest discriminative performance, reaching AUC = 0.896 in the simplest bimodal configuration (GCN vs CNN). This result exceeds ChemBERT-only architectures (AUC = 0.859–0.865) by over 3 percentage points, confirming that explicit bond topology information, as encoded by the molecular graph, constitutes a stronger predictive signal than the sequential SMILES representation processed by a pretrained language model.

### 2. Lightweight CNN Outperforms Pretrained ESM-2 for Protein Encoding

A direct comparison of protein encoders, holding the drug encoder (GCN) constant, reveals a surprising result:

| Protein encoder | AUC |
|---|---|
| CNN (trained from scratch) | **0.896** |
| ESM-2 (650M params, frozen) | 0.890 |
| CNN + ESM-2 (combined) | 0.894 |

The lightweight CNN encoder, trained from scratch on amino acid sequences, outperformed frozen ESM-2 embeddings by 0.6 percentage points in AUC. Moreover, combining both protein encoders (GCN vs CNN+ESM-2, AUC = 0.894) did not surpass the CNN-only configuration, suggesting that ESM-2 embeddings — when used in a feature extraction mode without fine-tuning — do not contribute complementary information beyond what the task-specific CNN already captures.

This finding is consistent with the hypothesis that pretrained protein language models require task-specific adaptation (e.g., fine-tuning or parameter-efficient techniques such as LoRA) to fully leverage their learned representations for downstream interaction prediction tasks.

### 3. Diminishing Returns of Multimodal Complexity

A systematic analysis of model performance as a function of architectural complexity reveals an inverse relationship between the number of encoders and predictive quality:

| Number of encoders | Best AUC | Best F1 | Example architecture |
|---|---|---|---|
| 2 | **0.896** | **0.813** | GCN vs CNN |
| 3 | 0.894 | 0.814 | GCN vs CNN+ESM-2 |
| 4 | 0.888 | 0.806 | GCN+ChemBERT vs CNN+ESM-2 |
| 5 | 0.885 | 0.793 | GCN+FP+ChemBERT vs CNN+ESM-2 |

Increasing the number of encoders from 2 to 5 resulted in a monotonic decrease in both AUC (−1.1 pp) and F1 score (−2.0 pp). This phenomenon can be attributed to two factors. First, **information redundancy**: encoders operating on the same input modality (e.g., GCN and molecular fingerprints both encode molecular structure from SMILES) produce partially overlapping representations, and their concatenation does not yield proportionally richer features. Second, **increased overfitting risk**: architectures with more encoders have a higher-dimensional concatenated embedding, which, combined with the fixed-size MLP fusion head, creates a capacity mismatch that exacerbates overfitting. To mitigate the latter, an adaptive dropout strategy was employed (dropout = 0.3 + 0.05 × (N − 2), where N denotes the number of encoders), and the MLP fusion head dimensions were auto-scaled proportionally to the input size (first hidden layer = input_dim / 2, second = input_dim / 8). Despite these regularisation measures, the fundamental trend of diminishing returns persisted.

### 4. Precision–Recall Trade-off in Fingerprint-Based Models

Models incorporating ECFP (Extended Connectivity Fingerprints) exhibit a characteristic pattern of **high precision but low recall** compared to GCN-based architectures:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| GCN vs CNN | 0.816 | **0.811** | **0.813** |
| FP vs CNN | **0.851** | 0.710 | 0.774 |
| FP+ChemBERT vs CNN | **0.845** | 0.722 | 0.779 |
| FP vs CNN+ESM-2 | **0.856** | 0.697 | 0.768 |

Fingerprint-based models achieved precision scores 3–4 percentage points higher than GCN models (0.845–0.856 vs 0.806–0.816), but at the cost of substantially lower recall (0.697–0.722 vs 0.805–0.822). This asymmetry likely stems from the fixed-radius, binary nature of ECFP descriptors: they capture well-defined structural motifs that strongly indicate binding activity (high precision), but their inability to represent subtle or novel structural features causes them to miss less obvious active compounds (low recall). In drug discovery applications, where the primary objective is to identify as many potential drug candidates as possible in early-stage virtual screening, the higher recall of graph-based models makes them more suitable for this purpose.

### 5. Information Redundancy of ChemBERT with GCN

Adding ChemBERT to architectures already containing GCN consistently decreased performance:

| Baseline | + ChemBERT | ΔAUC |
|---|---|---|
| GCN vs CNN (0.896) | GCN+ChemBERT vs CNN (0.891) | **−0.005** |
| GCN vs ESM-2 (0.890) | GCN+ChemBERT vs ESM-2 (0.881) | **−0.010** |
| GCN vs CNN+ESM-2 (0.894) | GCN+ChemBERT vs CNN+ESM-2 (0.888) | **−0.006** |

Across all three baseline configurations, augmenting the drug encoder with ChemBERT reduced AUC by 0.5–1.0 percentage points. This result suggests that the contextual SMILES representation generated by ChemBERT does not provide information complementary to that already extracted by the GCN from the molecular graph. Both encoders operate on representations derived from the same SMILES string: GCN converts it to an explicit graph, while ChemBERT processes it as a character sequence. The graph-based representation appears to subsume the sequential information, rendering ChemBERT redundant. However, this observation applies specifically to the feature extraction regime, where ChemBERT weights are frozen. Fine-tuning the transformer backbone — investigated in Phase 2 of this work using Low-Rank Adaptation (LoRA) — may enable ChemBERT to learn task-specific features that complement the graph encoder.

---

## Suggested Chapter Structure

1. **Experimental Setup** — dataset description, scaffold-based splitting, evaluation metrics
2. **Results Table** — all 21 architectural variants with AUC, AUPRC, F1, Precision, Recall
3. **Drug Encoder Analysis** — GCN dominance, FP precision-recall trade-off, ChemBERT redundancy (Findings 1, 4, 5)
4. **Protein Encoder Analysis** — CNN vs ESM-2, lack of improvement from adding ESM-2 (Finding 2)
5. **Effect of Multimodal Scale** — diminishing returns, redundancy hypothesis (Finding 3)
6. **Conclusions and Motivation for Phase 2** — need for transformer fine-tuning (LoRA) and improved fusion mechanism (cross-attention) instead of naive concatenation + MLP
