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
            # std = float(match.group(2))
            # Just print the mean for the LaTeX table to fit the user's example style
            metrics[metric] = f'{mean:.3f}'
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
    \caption{Comparison of Base Structural Models vs. Models with LINCS (MLP and Dynamic Graph) averaged over 5 random seeds.}
    \label{tab:lincs_fusion_comparison_multiseed}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{l ccccc ccccc ccccc}
        \toprule
        \multirow{2}{*}{\textbf{Structural Encoders}} & \multicolumn{5}{c}{\textbf{Base Model (No LINCS)}} & \multicolumn{5}{c}{\textbf{+ LINCS (MLP)}} & \multicolumn{5}{c}{\textbf{+ LINCS (Graph)}} \\
        \cmidrule(lr){2-6} \cmidrule(lr){7-11} \cmidrule(lr){12-16}
        & \textbf{AUC} & \textbf{AUPRC} & \textbf{F1} & \textbf{Prec.} & \textbf{Rec.} & \textbf{AUC} & \textbf{AUPRC} & \textbf{F1} & \textbf{Prec.} & \textbf{Rec.} & \textbf{AUC} & \textbf{AUPRC} & \textbf{F1} & \textbf{Prec.} & \textbf{Rec.} \\
        \midrule''')

for i, (name, base, mlp, graph) in enumerate(models):
    if i == 0:
        print(r'        \multicolumn{16}{l}{\textit{Single Structural Modality}} \\')
    elif i == 3:
        print(r'        \midrule')
        print(r'        \multicolumn{16}{l}{\textit{Dual Structural Modalities}} \\')
    elif i == 6:
        print(r'        \midrule')
        print(r'        \multicolumn{16}{l}{\textit{Triple Structural Modalities}} \\')
        
    m_base = parse_report(f'logs/lincs/{base}_vs_cnn_multiseed_report.txt')
    m_mlp = parse_report(f'logs/lincs/{mlp}_vs_cnn_multiseed_report.txt')
    m_graph = parse_report(f'logs/lincs/{graph}_vs_cnn_multiseed_report.txt')
    
    def highlight_best(metric_key):
        vals = []
        for m in [m_base, m_mlp, m_graph]:
            if m[metric_key] != '-':
                vals.append(float(m[metric_key]))
        if not vals:
            return m_base[metric_key], m_mlp[metric_key], m_graph[metric_key]
        
        best = max(vals)
        out = []
        for m in [m_base, m_mlp, m_graph]:
            if m[metric_key] != '-' and float(m[metric_key]) == best:
                out.append(r'\textbf{' + m[metric_key] + '}')
            else:
                out.append(m[metric_key])
        return out[0], out[1], out[2]
        
    auc_b, auc_m, auc_g = highlight_best('auc')
    
    line = f'        {name}'
    for idx, m in enumerate([m_base, m_mlp, m_graph]):
        if idx == 0: auc = auc_b
        elif idx == 1: auc = auc_m
        else: auc = auc_g
        
        line += f" & {auc} & {m['auprc']} & {m['f1']} & {m['precision']} & {m['recall']}"
    line += ' \\\\'
    print(line)

print(r'''        \bottomrule
    \end{tabular}
    }
\end{table}''')
