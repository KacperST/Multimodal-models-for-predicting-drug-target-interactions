# Model comparison

Base submodels used in these combinations.

| submodel | architecture |
| --- | --- |
| GCN | OGB AtomEncoder -> 3x [GCNConv() + BatchNorm1d() + LeakyReLU] -> global add pooling.|
| Fingerprint (ecfp) | Linear() -> BatchNorm1d() -> ReLU -> Dropout() -> Linear() -> BatchNorm1d() -> ReLU. |
| ChemBERT | SMILES embedding -> Linear() -> LayerNorm() -> ReLU. |
| CNN | Embedding() -> parallel Conv1d blocks for each kernel size -> masked max pooling -> concatenate. |
| ESM2 | Protein embedding -> Linear() -> LayerNorm() -> ReLU. |

| model_name | auc | auprc | f1 | precision | recall | loss |
| --- | --- | --- | --- | --- | --- | --- |
| fp_vs_cnn_esm2 | 96.545 | 96.241 | 89.674 | 91.913 | 87.542 | 24.290 |
| gcn_fp_chembert_vs_cnn | 96.165 | 95.893 | 89.415 | 89.167 | 89.665 | 25.248 |
| gcn_fp_chembert_vs_esm2 | 96.189 | 95.918 | 89.361 | 90.361 | 88.383 | 25.207 |
| gcn_fp_vs_esm2 | 95.783 | 95.480 | 88.484 | 90.092 | 86.933 | 26.624 |
| gcn_fp_chembert_vs_cnn_esm2 | 95.537 | 95.202 | 88.430 | 87.125 | 89.775 | 27.329 |
| gcn_chembert_vs_cnn_esm2 | 95.442 | 95.062 | 88.044 | 85.951 | 90.241 | 27.541 |
| gcn_vs_cnn | 95.420 | 95.055 | 87.904 | 85.921 | 89.980 | 27.625 |
| gcn_fp_vs_cnn | 95.223 | 94.907 | 87.832 | 86.842 | 88.845 | 28.237 |
| gcn_chembert_vs_cnn | 95.302 | 94.890 | 87.765 | 84.879 | 90.855 | 28.082 |
| gcn_vs_cnn_esm2 | 95.298 | 94.933 | 87.637 | 87.190 | 88.089 | 27.914 |
| fp_chembert_vs_cnn | 95.100 | 94.699 | 87.471 | 88.347 | 86.612 | 28.560 |
| fp_vs_esm2 | 95.652 | 95.335 | 87.452 | 90.825 | 84.322 | 27.540 |
| fp_vs_cnn | 94.714 | 94.357 | 87.006 | 86.712 | 87.303 | 29.534 |
| gcn_fp_vs_cnn_esm2 | 94.873 | 94.482 | 87.002 | 88.254 | 85.785 | 29.133 |
| gcn_vs_esm2 | 94.796 | 94.471 | 86.935 | 86.028 | 87.862 | 29.241 |
| chembert_vs_cnn | 94.144 | 93.529 | 86.177 | 84.318 | 88.119 | 31.085 |
| fp_chembert_vs_esm2 | 94.983 | 94.646 | 86.051 | 90.135 | 82.321 | 29.845 |
| gcn_chembert_vs_esm2 | 94.117 | 93.620 | 85.791 | 85.984 | 85.599 | 31.029 |
| fp_chembert_vs_cnn_esm2 | 94.164 | 93.763 | 85.641 | 87.720 | 83.658 | 31.343 |
| chembert_vs_esm2 | 92.584 | 91.850 | 84.025 | 83.010 | 85.066 | 34.711 |
| chembert_vs_cnn_esm2 | 92.575 | 91.931 | 83.489 | 83.880 | 83.102 | 34.636 |
