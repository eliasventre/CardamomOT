#!/usr/bin/env python3
"""
Table S3 — Candidate Hub Regulators by Outgoing Regulatory Strength
===================================================================

Generates Table S3: ranked lists of candidate hub regulators for each
experimental dataset (Semrau, Kameneva, Schiebinger), based on outgoing
regulatory strength in the GRN inferred by CardamomOT.

Uses 5 independent inference replicates from for_figureS9/<Dataset>/.
The GRN matrix is averaged across replicates before computing outgoing
strength; per-replicate std is also reported.

Definition: Outgoing regulatory strength = sum_{j != i} |W_{ij}|,
i.e. the sum of absolute values of all outgoing edge weights from gene i,
excluding self-loops.

Outputs:
  - tableS3_hub_regulators.csv : Combined CSV of all hub regulators.
  - tableS3_hub_regulators.tex : LaTeX table (top 10 per dataset, vertical layout).
"""

import numpy as np
import scanpy as sc
import pandas as pd
import os
import sys
import re
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_hub_table(datapath, for_figS9_path, dataset_name, stimulus_names):
    """
    Build a ranked hub regulator table from inferred GRN replicates.

    Parameters
    ----------
    datapath : str
        Path to the experimental dataset folder (for gene names from .h5ad).
    for_figS9_path : str
        Path to for_figureS9/<Dataset>/ containing inter_run_*.npy files.
    dataset_name : str
        Name of the dataset (e.g. 'Semrau').
    stimulus_names : list of str
        Names of stimulus genes (e.g. ['RA'] or ['Dox', 'Serum']).

    Returns
    -------
    pd.DataFrame with columns:
        Rank, Dataset, Gene, Type, Outgoing_Strength, Outgoing_Strength_Std,
        N_Outgoing_Edges, Top5_Outgoing_Targets
    """
    # --- Load gene names ---
    full_path = os.path.join(datapath, 'Data', 'data_full.h5ad')
    train_path = os.path.join(datapath, 'Data', 'data_train.h5ad')
    adata = sc.read_h5ad(full_path if os.path.exists(full_path) else train_path)
    gene_names = list(stimulus_names) + list(adata.var_names)

    # --- Load all GRN replicates from for_figureS9 ---
    replicate_files = sorted(glob.glob(os.path.join(for_figS9_path, 'inter_run_*.npy')))
    if not replicate_files:
        raise FileNotFoundError(f'No inter_run_*.npy files found in {for_figS9_path}')
    print(f'  [{dataset_name}] Loading {len(replicate_files)} replicate(s): '
          f'{os.path.basename(for_figS9_path)}/')

    matrices = []
    for f in replicate_files:
        mat = np.load(f)
        if mat.ndim == 3:
            mat = mat[:, :, 0]
        matrices.append(mat)

    n_genes = len(gene_names)
    n_stim = len(stimulus_names)
    n_reps = len(matrices)

    # --- Average GRN matrix across replicates ---
    avg_matrix = np.mean(matrices, axis=0)

    # --- Per-replicate outgoing strengths (for std) ---
    per_rep_strengths = []
    for mat in matrices:
        off_diag_rep = mat.copy()
        np.fill_diagonal(off_diag_rep, 0)
        per_rep_strengths.append(np.abs(off_diag_rep).sum(axis=1))
    per_rep_strengths = np.array(per_rep_strengths)  # shape: (n_reps, n_genes)

    # --- Remove self-loops from averaged matrix ---
    off_diag = avg_matrix.copy()
    np.fill_diagonal(off_diag, 0)

    # --- Outgoing regulatory strength (from averaged matrix) ---
    out_strength = np.abs(off_diag).sum(axis=1)
    out_strength_std = per_rep_strengths.std(axis=0, ddof=1) if n_reps > 1 else np.zeros(n_genes)
    n_out_edges = (off_diag != 0).sum(axis=1)

    # --- Build table ---
    rows = []
    for i in range(n_genes):
        name = gene_names[i]
        gene_type = 'Stimulus' if i < n_stim else 'Gene'
        strength = out_strength[i]
        strength_std = out_strength_std[i]
        n_edges = n_out_edges[i]

        # Top 5 outgoing targets with signed weights (from averaged matrix)
        out_weights = off_diag[i, :]
        top5_idx = np.argsort(np.abs(out_weights))[::-1][:5]
        top5_parts = []
        for j in top5_idx:
            w = out_weights[j]
            if w != 0:
                top5_parts.append(f'{gene_names[j]} ({w:+.2f})')
        top5_targets = '; '.join(top5_parts) if top5_parts else '—'

        rows.append({
            'Dataset': dataset_name,
            'Gene': name,
            'Type': gene_type,
            'Outgoing_Strength': round(strength, 3),
            'Outgoing_Strength_Std': round(strength_std, 3),
            'N_Outgoing_Edges': n_edges,
            'Top5_Outgoing_Targets': top5_targets,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('Outgoing_Strength', ascending=False)
    df['Rank'] = range(1, len(df) + 1)
    df = df[['Rank', 'Dataset', 'Gene', 'Type', 'Outgoing_Strength',
             'Outgoing_Strength_Std', 'Top5_Outgoing_Targets']]
    return df


def display_top_hubs(df, dataset_name, top_n=15):
    """Display top N hub genes for a dataset, excluding stimuli."""
    subset = df[(df['Dataset'] == dataset_name) & (df['Type'] == 'Gene')].head(top_n)
    return subset[['Rank', 'Gene', 'Outgoing_Strength', 'Outgoing_Strength_Std',
                   'Top5_Outgoing_Targets']]


def escape_latex(s):
    """Escape special LaTeX characters in a string."""
    return (str(s)
            .replace('\\', r'\textbackslash ')
            .replace('_', r'\_')
            .replace('&', r'\&')
            .replace('%', r'\%')
            .replace('#', r'\#')
            .replace('$', r'\$')
            .replace('{', r'\{')
            .replace('}', r'\}')
            .replace('~', r'\textasciitilde ')
            .replace('^', r'\textasciicircum '))


def _top4_targets_no_weights(targets_str):
    """Extract top 4 target gene names (without interaction weights) from the targets string."""
    parts = [p.strip() for p in targets_str.split(';')]
    gene_names = []
    for p in parts:
        # Strip the weight in parentheses, e.g., "Zfp42 (-28.19)" -> "Zfp42"
        name = re.sub(r'\s*\([+-]?[\d.]+\)\s*$', '', p).strip()
        if name and name != '—':
            gene_names.append(escape_latex(name))
    return ', '.join(gene_names[:4]) if gene_names else '—'


def generate_latex_vertical_table(df_all, top_n=10):
    """
    Generate a single vertical LaTeX table spanning the full page width.
    Datasets are stacked vertically with grouping headers.
    Uses tabularx with booktabs for a clean, compact layout.
    """
    newline = '\n'
    lines = []

    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\caption{Top \detokenize{10} candidate hub regulators ranked by '
                  r'outgoing regulatory strength $\sum_{j\neq i}|W_{ij}|$ '
                  r'in the GRN inferred by \textsc{CardamomOT} for each '
                  r'experimental dataset. Values show mean $\pm$ standard deviation across '
                  r'5 independent inference runs.}')
    lines.append(r'\label{tableS3}')
    lines.append(r'\small')
    lines.append(r'\begin{tabularx}{\textwidth}{r l l l X}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Rank & Gene & Out.\ Strength & Top 4 Outgoing Targets \\')
    lines.append(r'\midrule')

    datasets = ['Semrau', 'Kameneva', 'Schiebinger']
    for ds_idx, ds_name in enumerate(datasets):
        sub = df_all[(df_all['Dataset'] == ds_name) & (df_all['Type'] == 'Gene')].head(top_n)

        for row_idx, (_, row) in enumerate(sub.iterrows()):
            gene = escape_latex(row['Gene'])
            strength = f"{row['Outgoing_Strength']:.3f} $\\pm$ {row['Outgoing_Strength_Std']:.3f}"
            targets = _top4_targets_no_weights(row['Top5_Outgoing_Targets'])

            # Show dataset name only on first row of each group
            ds_cell = ds_name if row_idx == 0 else ''
            rank = row_idx + 1

            lines.append(f'{ds_cell} & {rank} & {gene} & {strength} & {targets} \\\\')

        # Add a separating rule between datasets (but not after the last)
        if ds_idx < len(datasets) - 1:
            lines.append(r'\cmidrule{1-5}')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabularx}')
    lines.append(r'\end{table}')

    return newline.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # --- Build hub tables for all three datasets ---
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'experimental_datasets')
    for_figS9_dir = os.path.join(os.path.dirname(__file__), 'for_figureS9')

    df_semrau = build_hub_table(
        os.path.join(base_dir, 'Semrau'),
        os.path.join(for_figS9_dir, 'Semrau'),
        'Semrau', ['RA'])
    df_kameneva = build_hub_table(
        os.path.join(base_dir, 'Kameneva'),
        os.path.join(for_figS9_dir, 'Kameneva'),
        'Kameneva', ['RA'])
    df_schiebinger = build_hub_table(
        os.path.join(base_dir, 'Schiebinger'),
        os.path.join(for_figS9_dir, 'Schiebinger'),
        'Schiebinger', ['Dox', 'Serum'])

    # Combine into a single table
    df_all = pd.concat([df_semrau, df_kameneva, df_schiebinger], ignore_index=True)

    # Save to CSV
    out_dir = os.path.dirname(__file__)
    csv_path = os.path.join(out_dir, 'tableS3_hub_regulators.csv')
    df_all.to_csv(csv_path, index=False)
    print(f'Saved {len(df_all)} rows to {csv_path}')
    print(f'  Semrau:    {len(df_semrau)} genes')
    print(f'  Kameneva:  {len(df_kameneva)} genes')
    print(f'  Schiebinger: {len(df_schiebinger)} genes')

    # -----------------------------------------------------------------------
    # Top 15 hub regulators per dataset
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('Semrau — Top 15 hub regulators (genes only, excluding stimuli)')
    print('=' * 70)
    print(display_top_hubs(df_all, 'Semrau').to_string(index=False))

    print('\n' + '=' * 70)
    print('Kameneva — Top 15 hub regulators')
    print('=' * 70)
    print(display_top_hubs(df_all, 'Kameneva').to_string(index=False))

    print('\n' + '=' * 70)
    print('Schiebinger — Top 15 hub regulators')
    print('=' * 70)
    print(display_top_hubs(df_all, 'Schiebinger').to_string(index=False))

    # -----------------------------------------------------------------------
    # Stimulus regulatory strength (for reference)
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('Stimulus regulatory strength (for reference)')
    print('Shows how strongly each stimulus gene regulates the GRN.')
    print('=' * 70)
    print(df_all[df_all['Type'] == 'Stimulus'][
        ['Dataset', 'Gene', 'Outgoing_Strength', 'Top5_Outgoing_Targets']
    ].to_string(index=False))

    # -----------------------------------------------------------------------
    # LaTeX Table — Top 10 Hub Regulators per Dataset (Vertical Layout)
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('LaTeX table output:')
    print('=' * 70)
    latex_output = generate_latex_vertical_table(df_all, top_n=10)
    print(latex_output)

    # Save to a .tex file for direct inclusion
    tex_path = os.path.join(out_dir, 'tableS3_hub_regulators.tex')
    with open(tex_path, 'w') as f:
        f.write(latex_output)
    print(f'\nSaved to {tex_path}')
