"""
figure8.py
==========
Compares GRN inference methods (CardamomOT, PEARSON, Reference Fitting) against a
prior reference network across three datasets (Semrau, Kameneva/Olivier, Schiebinger).

Metrics computed (diagonal and Stimulus row excluded):
  - AUPR  (Area Under Precision-Recall curve)
  - AUROC (Area Under ROC curve)
  - Early Precision @ top-k% (EP10: k=10%, EP20: k=20%)

Output: figure_8.pdf  (A4 portrait, ~827x1169 pts)

Usage:
    python figure8.py

Paths follow the same conventions as figure_s1-3.ipynb:
    base/<dataset>/Data/data_full.h5ad  (or data_train.h5ad)
    base/<dataset>/cardamomOT/ref_network.csv   ← signed prior (1/0)
    base/<dataset>/cardamomOT/inter.npy   ← CardamomOT inferred GRN
    base/<dataset>/PEARSON/inter.npy or .csv
    base/<dataset>/RF/inter.npy or .csv
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_curve,
)

# ──────────────────────────────────────────────
# 1.  DATASET PATHS  (edit to match your layout)
# ──────────────────────────────────────────────
BASE = './../../experimental_datasets'   # same root as the notebook

DATASETS = {
    'Semrau':    os.path.join(BASE, 'Semrau'),
    'Kameneva':  os.path.join(BASE, 'Kameneva'),
    'Schiebinger': os.path.join(BASE, 'Schiebinger'),
}

# ──────────────────────────────────────────────
# 2.  STYLE CONSTANTS
# ──────────────────────────────────────────────
COLORS = {
    'CardamomOT': '#D7191C',   # rouge  (comme benchmark)
    'GENIE3':     '#1F5FA6',   # bleu foncé
    'RF':         '#00BCD4',   # cyan/turquoise
    'PEARSON':    '#808080',   # gris
}
LINESTYLES = {
    'CardamomOT': '-',
    'GENIE3':     '--',
    'RF':         ':',
    'PEARSON':    '-.',
}
MARKERS = {
    'CardamomOT': 'o',
    'GENIE3':     's',
    'RF':         '^',
    'PEARSON':    'D',
}
METHODS = list(COLORS.keys())

EARLY_K = [0.10, 0.25]   # early precision thresholds (top-10%, top-25%)

# ──────────────────────────────────────────────
# 3.  DATA LOADING HELPERS
# ──────────────────────────────────────────────

def load_gene_names(dataset_path):
    """Return list of gene names from h5ad (no Stimulus)."""
    for fname in ('data_full.h5ad', 'data_train.h5ad'):
        fpath = os.path.join(dataset_path, 'Data', fname)
        if os.path.exists(fpath):
            adata = sc.read_h5ad(fpath)
            return list(adata.var_names)
    raise FileNotFoundError(f"No h5ad found in {dataset_path}/Data/")


def _load_npy(path):
    """Load .npy, handle 3-D tensors (take slice 0)."""
    m = np.load(path)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m


def load_method_matrix(dataset_path, method, n_genes):
    """
    Load inferred GRN matrix for a given method.
    Returns absolute-value matrix (n_genes x n_genes), or None if not found.
    The Stimulus row/col (index 0 in raw files) is removed here.
    """
    if method == 'CardamomOT':
        p = os.path.join(dataset_path, 'cardamomOT', 'inter.npy')
        if os.path.exists(p):
            m = _load_npy(p)
            # inter includes Stimulus as row/col 0 → remove
            if m.shape[0] == n_genes + 1:
                m = m[1:, 1:]
                m -= np.diag(np.diag(m))
            return np.abs(m)
        return None

    else:
        p = os.path.join(dataset_path, method, 'inter.npy')
        if os.path.exists(p):
            m = _load_npy(p)
            # inter includes Stimulus as row/col 0 → remove
            if m.shape[0] == n_genes + 1:
                m = m[1:, 1:]
                m -= np.diag(np.diag(m))
            return np.abs(m)
        return None


def load_reference(dataset_path):
    """
    Load binary reference network (0/1).
    Removes Stimulus row/col (first row/col) and diagonal.
    Checks that gene order matches adata.var_names and reorders if needed.
    Returns a (n_genes x n_genes) float array.
    """
    ref_path = os.path.join(dataset_path, 'cardamomOT', 'ref_network.csv')
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"ref_network.csv not found in {ref_path}")

    df = pd.read_csv(ref_path, index_col=0)

    # Drop Stimulus row/col (first entry)
    df = df.iloc[1:, 1:]

    # Load gene order from adata
    genes_adata = None
    for fname in ('data_full.h5ad', 'data_train.h5ad'):
        fpath = os.path.join(dataset_path, 'Data', fname)
        if os.path.exists(fpath):
            genes_adata = list(sc.read_h5ad(fpath).var_names)
            break

    if genes_adata is not None:
        genes_ref = list(df.index)
        genes_adata = [g.upper() for g in genes_adata]
        genes_ref = [g.upper() for g in genes_ref]
        if genes_ref != genes_adata:
            missing = set(genes_adata) - set(genes_ref)
            extra   = set(genes_ref) - set(genes_adata)
            if missing:
                print(f"  [WARNING] {len(missing)} genes in adata missing from ref_network: {list(missing)[:5]}...")
            if extra:
                print(f"  [WARNING] {len(extra)} genes in ref_network not in adata: {list(extra)[:5]}...")
            common = [g for g in genes_adata if g in set(genes_ref)]
            print(f"  [INFO] Reordering ref_network: {genes_ref[:3]}... -> {genes_adata[:3]}...")
            df = df.loc[common, common]
        else:
            print(f"  [INFO] ref_network gene order matches adata ({len(genes_ref)} genes)")

    ref_bin = df.values.astype(float)
    ref_bin -= np.diag(np.diag(ref_bin))

    return ref_bin

# ──────────────────────────────────────────────
# 4.  METRIC HELPERS
# ──────────────────────────────────────────────

def compute_aupr(y_true, y_score):
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.sum() == 0:
        return np.nan, None, None
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return auc(rec, prec), rec, prec


def compute_auroc(y_true, y_score):
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan, None, None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr), fpr, tpr


def compute_early_precision(y_true, y_score, k_frac):
    """Precision in the top-k% predicted edges."""
    n_top = max(1, int(np.ceil(k_frac * len(y_true))))
    top_idx = np.argsort(y_score)[::-1][:n_top]
    return y_true[top_idx].mean()


def random_aupr(y_true):
    """Expected AUPR for a random classifier = prevalence."""
    return y_true.mean()

# ──────────────────────────────────────────────
# 5.  MAIN DATA ASSEMBLY
# ──────────────────────────────────────────────

def gather_all_results():
    """
    Returns a nested dict:
        results[dataset][method] = {
            'aupr': float,
            'auroc': float,
            'ep10': float,
            'ep20': float,
            'pr_curve': (rec, prec),   # arrays for plotting
            'roc_curve': (fpr, tpr),
            'random_aupr': float,
        }
    """
    results = {}
    for ds_name, ds_path in DATASETS.items():
        results[ds_name] = {}
        try:
            genes = load_gene_names(ds_path)
        except FileNotFoundError as e:
            print(f"[SKIP] {ds_name}: {e}")
            continue
        n_genes = len(genes)
        y_true = load_reference(ds_path).astype(bool).ravel()
        print(np.sum(y_true), y_true.shape)

        rand_aupr = random_aupr(y_true)

        for method in METHODS:
            y_score = load_method_matrix(ds_path, method, n_genes)
            if y_score is None:
                print(f"[MISSING] {ds_name} / {method}")
                continue
            y_score = y_score.ravel()
            print(np.sum(y_score), y_score.shape)

            # Normalise scores to [0,1] for comparability
            s_min, s_max = y_score.min(), y_score.max()
            if s_max > s_min:
                y_score = (y_score - s_min) / (s_max - s_min)

            aupr_val, rec_arr, prec_arr = compute_aupr(y_true, y_score)
            auroc_val, fpr_arr, tpr_arr = compute_auroc(y_true, y_score)
            ep10 = compute_early_precision(y_true, y_score, 0.10)
            ep20 = compute_early_precision(y_true, y_score, 0.20)

            results[ds_name][method] = {
                'aupr':       aupr_val,
                'auroc':      auroc_val,
                'ep10':       ep10,
                'ep20':       ep20,
                'pr_curve':   (rec_arr, prec_arr),
                'roc_curve':  (fpr_arr, tpr_arr),
                'random_aupr': rand_aupr,
            }
            print(f"  {ds_name:15s} | {method:20s} | AUPR={aupr_val:.3f}  AUROC={auroc_val:.3f}  EP10={ep10:.3f}")

    return results

# ──────────────────────────────────────────────
# 6.  FIGURE LAYOUT
# ──────────────────────────────────────────────

def make_figure(results):
    """
    Layout (A4 portrait  8.27 x 11.69 in):

    Row 0 (top half): PR curves — one panel per dataset  [3 panels]
    Row 1 (middle):   ROC curves — one panel per dataset [3 panels]
    Row 2 (bottom):   Bar charts for scalar metrics (AUPR, AUROC, EP10, EP20)
                      grouped by dataset                 [4 panels]
    """
    DS_NAMES = list(DATASETS.keys())
    N_DS = len(DS_NAMES)
    N_M  = len(METHODS)   # 4

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    # ---------- outer grid: 3 rows ----------
    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[2.2, 2.2, 2.2],
        hspace=0.45,
        left=0.09, right=0.97, top=0.96, bottom=0.05
    )

    # Row 0: PR curves
    gs_pr  = gridspec.GridSpecFromSubplotSpec(1, N_DS, subplot_spec=outer[0], wspace=0.32)
    # Row 1: ROC curves
    gs_roc = gridspec.GridSpecFromSubplotSpec(1, N_DS, subplot_spec=outer[1], wspace=0.32)
    # Row 2: 4 bar‑chart metrics
    gs_bar = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2], wspace=0.40)

    ax_pr  = [fig.add_subplot(gs_pr[0, i])  for i in range(N_DS)]
    ax_roc = [fig.add_subplot(gs_roc[0, i]) for i in range(N_DS)]
    ax_bar = [fig.add_subplot(gs_bar[0, i]) for i in range(4)]

    panel_labels = iter('ABCDEFGHIJKLMNOP')

    # ── helper: add panel letter ──────────────
    def label_ax(ax, letter, x=-0.14, y=1.05):
        ax.text(x, y, letter, transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='bottom', ha='left', clip_on=False)

    # ── PR curves ─────────────────────────────
    for i, ds in enumerate(DS_NAMES):
        ax = ax_pr[i]
        label_ax(ax, next(panel_labels))
        ax.set_title(ds, fontsize=8, fontweight='bold', pad=4)
        ax.set_xlabel('Recall', fontsize=7)
        ax.set_ylabel('Precision', fontsize=7) if i == 0 else None
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, lw=0.4, alpha=0.4)

        ds_res = results.get(ds, {})
        rand_p = None
        for method in METHODS:
            mres = ds_res.get(method)
            if mres is None:
                continue
            rec, prec = mres['pr_curve']
            if rec is None:
                continue
            ax.plot(rec, prec,
                    color=COLORS[method],
                    ls=LINESTYLES[method],
                    lw=1.4,
                    label=f"{method} (AUPR={mres['aupr']:.2f})")
            if rand_p is None:
                rand_p = mres['random_aupr']

        if rand_p is not None:
            ax.axhline(rand_p, color='gray', ls=':', lw=0.8, label=f'Random ({rand_p:.2f})')

        if i == N_DS - 1:
            ax.legend(fontsize=5.5, loc='upper right', framealpha=0.8,
                      handlelength=1.8, borderpad=0.5)

        ax.text(0.02, 0.02, 'PR curve', transform=ax.transAxes,
                fontsize=5.5, color='gray', va='bottom')

    # ── ROC curves ────────────────────────────
    for i, ds in enumerate(DS_NAMES):
        ax = ax_roc[i]
        label_ax(ax, next(panel_labels))
        ax.set_title(ds, fontsize=8, fontweight='bold', pad=4)
        ax.set_xlabel('FPR', fontsize=7)
        ax.set_ylabel('TPR', fontsize=7) if i == 0 else None
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.plot([0, 1], [0, 1], color='gray', ls=':', lw=0.8, label='Random (0.50)')

        ds_res = results.get(ds, {})
        for method in METHODS:
            mres = ds_res.get(method)
            if mres is None:
                continue
            fpr, tpr = mres['roc_curve']
            if fpr is None:
                continue
            ax.plot(fpr, tpr,
                    color=COLORS[method],
                    ls=LINESTYLES[method],
                    lw=1.4,
                    label=f"{method} (AUROC={mres['auroc']:.2f})")

        if i == N_DS - 1:
            ax.legend(fontsize=5.5, loc='lower right', framealpha=0.8,
                      handlelength=1.8, borderpad=0.5)

        ax.text(0.02, 0.02, 'ROC curve', transform=ax.transAxes,
                fontsize=5.5, color='gray', va='bottom')

    # ── Bar charts ────────────────────────────
    metric_keys  = ['aupr', 'auroc', 'ep10',  'ep20']
    metric_labels = ['AUPR', 'AUROC', 'Early Precision\n(top 10%)', 'Early Precision\n(top 20%)']

    x = np.arange(N_DS)
    # FIX: 4 offsets centrés pour 4 méthodes
    width   = 0.18
    offsets = (np.arange(N_M) - (N_M - 1) / 2.0) * width  # [-0.27, -0.09, +0.09, +0.27]

    for mi, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
        ax = ax_bar[mi]
        label_ax(ax, next(panel_labels), x=-0.20)
        ax.set_title(ml, fontsize=7.5, fontweight='bold', pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([d[:4] for d in DS_NAMES], fontsize=6.5)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_ylim(0, 1.08)
        ax.grid(True, axis='y', lw=0.4, alpha=0.4, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)

        for ji, method in enumerate(METHODS):
            vals = []
            for ds in DS_NAMES:
                mres = results.get(ds, {}).get(method)
                vals.append(mres[mk] if (mres and not np.isnan(mres[mk])) else 0.0)
            bars = ax.bar(x + offsets[ji], vals, width=width * 0.88,
                          color=COLORS[method], alpha=0.85,
                          label=method, zorder=3,
                          edgecolor='white', linewidth=0.4)
            # value labels on bars
            for bar, v in zip(bars, vals):
                if v > 0.02:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.015,
                            f'{v:.2f}',
                            ha='center', va='bottom', fontsize=4.5, rotation=90,
                            color='#333333')

        # Random baseline for AUPR
        if mk == 'aupr':
            for ds_i, ds in enumerate(DS_NAMES):
                any_mres = next((results.get(ds, {}).get(m) for m in METHODS
                                 if results.get(ds, {}).get(m)), None)
                if any_mres:
                    ax.plot([ds_i - 0.38, ds_i + 0.38],
                            [any_mres['random_aupr']] * 2,
                            color='gray', ls=':', lw=1.0)

    # ── Shared legend at bottom ────────────────
    legend_handles = [
        Line2D([0], [0], color=COLORS[m], ls=LINESTYLES[m], lw=1.8,
               marker=MARKERS[m], markersize=5, label=m)
        for m in METHODS
    ]
    fig.legend(handles=legend_handles,
               loc='lower center',
               ncol=4,
               fontsize=7,
               framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01),
               borderpad=0.6)

    plt.savefig('figure_8.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print("Saved figure_8.png")
    try:
        from PIL import Image
        Image.open('figure_8.png').convert('RGB').save('figure_8.pdf', 'PDF', resolution=300)
        print("Saved figure_8.pdf")
    except ImportError:
        print("(PIL not available — skipping PDF export)")
    return fig


# ──────────────────────────────────────────────
# 7.  ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Loading and computing metrics ===\n")
    results = gather_all_results()

    if not any(results[ds] for ds in results):
        print("\n[ERROR] No data could be loaded. Check your BASE path and folder structure.")
        sys.exit(1)

    print("\n=== Generating figure ===")
    make_figure(results)