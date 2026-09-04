import re
import os
from pathlib import Path

def parse_report(filepath):
    if not os.path.exists(filepath):
        return {m: '-' for m in ['auc', 'auprc', 'f1', 'precision', 'recall']}
    
    metrics = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    for metric in ['auc', 'auprc', 'f1', 'precision', 'recall']:
        match = re.search(fr'{metric}:\s+([0-9.]+)\s+±\s+([0-9.]+)', content)
        if match:
            mean = float(match.group(1))
            std = float(match.group(2))
            metrics[metric] = f'{mean:.3f} $\\pm$ {std:.3f}'
        else:
            metrics[metric] = '-'
    return metrics

models = [
    ('FP', 'fp', 'fp_lincs', 'fp_lincs_graph'),
    ('GCN', 'gcn', 'gcn_lincs', 'gcn_lincs_graph'),
    ('ChemBERTa', 'chembert', 'chembert_lincs', 'chembert_lincs_graph'),
    ('FP + GCN', 'gcn_fp', 'gcn_fp_lincs', 'gcn_fp_lincs_graph'),
    ('FP + ChemBERTa', 'fp_chembert', 'fp_chembert_lincs', 'fp_chembert_lincs_graph'),
    ('GCN + ChemBERTa', 'gcn_chembert', 'gcn_chembert_lincs', 'gcn_chembert_lincs_graph'),
    ('FP + GCN + ChemBERTa', 'gcn_fp_chembert', 'gcn_fp_chembert_lincs', 'gcn_fp_chembert_lincs_graph')
]

print(r'''\begin{table}[H]
    \centering
    \caption{Comparison of Base Structural Models vs. Models with LINCS (MLP and Dynamic Graph). All metrics are averaged over 5 random seeds to ensure statistical robustness. In all configurations, the protein target is encoded using a 1D CNN.}
    \label{tab:lincs_fusion_comparison}
    \resizebox{0.9\textwidth}{!}{
    \begin{tabular}{l ccccc}
        \toprule
        \textbf{Model / Configuration} & \textbf{AUC} & \textbf{AUPRC} & \textbf{F1} & \textbf{Prec.} & \textbf{Rec.} \\
        \midrule''')

for i, (name, base, mlp, graph) in enumerate(models):
    if i == 0:
        print(r'        \multicolumn{6}{l}{\textit{Single Structural Modality}} \\')
    elif i == 3:
        print(r'        \midrule')
        print(r'        \multicolumn{6}{l}{\textit{Dual Structural Modalities}} \\')
    elif i == 6:
        print(r'        \midrule')
        print(r'        \multicolumn{6}{l}{\textit{Triple Structural Modalities}} \\')
        
    root_dir = Path(__file__).resolve().parent
    m_base = parse_report(str(root_dir / f'logs/lincs/{base}_vs_cnn_multiseed_report.txt'))
    m_mlp = parse_report(str(root_dir / f'logs/lincs/{mlp}_vs_cnn_multiseed_report.txt'))
    m_graph = parse_report(str(root_dir / f'logs/lincs/{graph}_vs_cnn_multiseed_report.txt'))
    
    def highlight_best(metric_key):
        vals = []
        for m in [m_base, m_mlp, m_graph]:
            if m[metric_key] != '-':
                mean_val = float(m[metric_key].split(' $\\pm$ ')[0])
                vals.append(mean_val)
        if not vals:
            return m_base[metric_key], m_mlp[metric_key], m_graph[metric_key]
        
        best = max(vals)
        out = []
        for m in [m_base, m_mlp, m_graph]:
            if m[metric_key] != '-':
                mean_val = float(m[metric_key].split(' $\\pm$ ')[0])
                if mean_val == best:
                    parts = m[metric_key].split(' $\\pm$ ')
                    out.append(r'\textbf{' + parts[0] + r'} $\pm$ ' + parts[1])
                else:
                    out.append(m[metric_key])
            else:
                out.append(m[metric_key])
        return out[0], out[1], out[2]
        
    auc_b, auc_m, auc_g = highlight_best('auc')
    
    print(r'        \textbf{' + name + r'} & & & & & \\')
    print(f"        \\hspace{{3mm}} Base (No LINCS) & {auc_b} & {m_base['auprc']} & {m_base['f1']} & {m_base['precision']} & {m_base['recall']} \\\\")
    print(f"        \\hspace{{3mm}} + LINCS (MLP) & {auc_m} & {m_mlp['auprc']} & {m_mlp['f1']} & {m_mlp['precision']} & {m_mlp['recall']} \\\\")
    print(f"        \\hspace{{3mm}} + LINCS (Graph) & {auc_g} & {m_graph['auprc']} & {m_graph['f1']} & {m_graph['precision']} & {m_graph['recall']} \\\\")
    if i < len(models) - 1 and i not in [2, 5]:
        print(r'        \addlinespace')

print(r'''        \bottomrule
    \end{tabular}
    }
\end{table}''')
