"""
figureS9.py
===========
Variability of GRN networks inferred by CardamomOT across 3 datasets.

For each dataset (Semrau, Kameneva, Schiebinger), the inference pipeline
(fit_mixture → fit_network → refine_network_degradations) is run N_RUNS
times independently. All C(N_RUNS, 2) pairs are compared symmetrically
(no run is used as a fixed reference).

Results cached in for_figureS9/<dataset>/.

Parameters per dataset (from run_pipeline.sh):
  Semrau     : full,  mean_forcing=1.0
  Kameneva   : full,  mean_forcing=0.5
  Schiebinger: train, mean_forcing=0.0, stimulus=1.0, prior=1.0,
               force_basins=0.0, temporal_basins=0

Figure (A4 landscape, 2 rows × 3 columns):
  Panel A : pairwise cosine similarity (all C(N,2) pairs) —
            violin + stripplot per dataset
  Panel B : mean pairwise AUPR ± σ for top-k genes
            (k = 10, 20, 30, 40, 50 ; capped at G if G < 50)
            Genes ranked by mean L1 centrality across all runs
            For each pair (i,j): average of both directions
            (predict i from j and j from i)
"""

import sys
sys.path += ['./../../']

import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from CardamomOT import NetworkModel as NetworkModel_beta

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_RUNS = 5  # number of independent re-fits (run 0 = existing inference)

DATASETS = [
    dict(
        name            = 'Semrau',
        path            = './../../experimental_datasets/Semrau',
        split           = 'full',
        mean_forcing    = 1.0,
        stimulus        = 1.0,
        prior           = 1.0,
        force_basins    = 1.0,
        temporal_basins = 1,
    ),
    dict(
        name            = 'Kameneva',
        path            = './../../experimental_datasets/Kameneva',
        split           = 'full',
        mean_forcing    = 0.5,
        stimulus        = 0.2,
        prior           = 1.0,
        force_basins    = 1.0,
        temporal_basins = 1,
    ),
    dict(
        name            = 'Schiebinger',
        path            = './../../experimental_datasets/Schiebinger',
        split           = 'train',
        mean_forcing    = 0.0,
        stimulus        = 1.0,
        prior           = 1.0,
        force_basins    = 0.0,
        temporal_basins = 0,
    ),
]

COLORS = ['#4878CF', '#E87B30', '#6DB36D']

# ─────────────────────────────────────────────────────────────────────────────
# Helpers : inference
# ─────────────────────────────────────────────────────────────────────────────

def _load_stimulus_schedule(p):
    path = os.path.join(p, 'Data', 'stimulus_schedule.txt')
    if os.path.exists(path):
        return np.loadtxt(path)
    return None


def _detect_n_stimuli(stim_sched):
    if stim_sched is None:
        return 1
    arr = np.asarray(stim_sched)
    return int(arr.shape[1]) if arr.ndim == 2 else 1


def run_single_fit(cfg):
    """
    Run a full inference on the dataset and return model.inter (copy).
    Equivalent to fit_mixture + fit_network + refine_network_degradations.
    """
    p     = cfg['path']
    split = cfg['split']

    data_path = os.path.join(p, 'Data', f'data_{split}.h5ad')
    adata     = ad.read_h5ad(data_path)

    stim_sched = _load_stimulus_schedule(p)
    n_stimuli  = _detect_n_stimuli(stim_sched)

    model = NetworkModel_beta(adata.shape[1], n_stimuli=n_stimuli)
    model.mean_forcing_em   = cfg['mean_forcing']
    model.stimulus          = cfg['stimulus']
    model.prior_network_pen = cfg['prior']
    model.force_basins      = cfg['force_basins']
    model.temporal_basins   = int(cfg['temporal_basins'])

    # ── Step 1 : mixture ─────────────────────────────────────────────────────
    model.fit_mixture(
        adata,
        gene_names      = list(adata.var_names),
        min_components  = 2,
        max_components  = 2,
        max_iter_kinetics = 0,
        verb            = True,
        stimulus_schedule = stim_sched,
    )

    # ── Step 2 : degradation rates (from adata.var) ─────────────────────────
    ns    = model.n_stimuli
    G_tot = adata.shape[1] + ns
    model.d = np.ones((2, G_tot))
    if 'd1' in adata.var.columns:
        model.d[1, ns:] = adata.var['d1'].values
    if 'd0' in adata.var.columns:
        model.d[0, ns:] = adata.var['d0'].values

    # ── Step 2b : prior reference network (optional CSV) ────────────────────
    model.ref_network = np.ones((G_tot, G_tot, model.n_networks))
    ref_csv = os.path.join(p, 'Data', 'ref_network.csv')
    if os.path.exists(ref_csv):
        try:
            ref_df     = pd.read_csv(ref_csv, index_col=0, sep='\t')
            genes_up   = [g.upper() for g in adata.var_names]
            row_genes  = [g.upper() for g in ref_df.index]
            col_genes  = [g.upper() for g in ref_df.columns]
            row_idxs   = [ns + genes_up.index(g) for g in row_genes if g in genes_up]
            col_idxs   = [ns + genes_up.index(g) for g in col_genes if g in genes_up]
            ref_mat    = ref_df.values
            for n in range(model.n_networks):
                for ii, ri in enumerate(row_idxs):
                    model.ref_network[ri, col_idxs, n] = ref_mat[ii, :]
        except Exception as e:
            print(f"  Warning: ref_network.csv not loaded ({e})")

    # ── Step 3 : network ─────────────────────────────────────────────────────
    model.fit_network(adata, intensity_prior=100, verb=True, stimulus_schedule=stim_sched)

    # ── Step 4 : degradation refinement ─────────────────────────────────────
    model.refine_network_degradations(stimulus_schedule=stim_sched)

    return model.inter.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Cache management
# ─────────────────────────────────────────────────────────────────────────────

def get_inter_list(cfg, n_runs=N_RUNS, cache_root='for_figureS9'):
    """
    Load inter matrices from cache (for_figureS9/<dataset>/)
    or compute them if missing.

    Cache structure:
      for_figureS9/<dataset>/inter_run_00.npy  ← independent fit 0
      for_figureS9/<dataset>/inter_run_01.npy  ← independent fit 1
      …

    All runs are independent inferences (no external reference run).
    If files already exist, inference is skipped.
    Returns (list of inter matrices, n_stimuli).
    """
    p       = cfg['path']
    var_dir = os.path.join(cache_root, cfg['name'])
    os.makedirs(var_dir, exist_ok=True)

    inters = []
    for i in range(n_runs):
        cache_i = os.path.join(var_dir, f'inter_run_{i:02d}.npy')
        if os.path.exists(cache_i):
            print(f"  [{cfg['name']}] Run {i:02d} : loaded from cache ({cache_i})")
            inters.append(np.load(cache_i))
        else:
            print(f"  [{cfg['name']}] Run {i:02d} : inference in progress…")
            inter = run_single_fit(cfg)
            np.save(cache_i, inter)
            print(f"  [{cfg['name']}] Run {i:02d} : saved → {cache_i}")
            inters.append(inter)

    ns = 1
    return inters, ns


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_cosine(inters, ns):
    """Pairwise cosine similarities. Returns upper-triangle list."""
    vecs = []
    for inter in inters:
        block = inter[ns:, ns:, 0] if inter.ndim == 3 else inter[ns:, ns:]
        v     = block.flatten()
        norm  = np.linalg.norm(v)
        vecs.append(v / norm if norm > 1e-12 else v)
    n   = len(vecs)
    upper = [np.dot(vecs[i], vecs[j])
             for i in range(n) for j in range(i + 1, n)]
    return upper


# ─────────────────────────────────────────────────────────────────────────────
# AUPR vs top-k genes
# ─────────────────────────────────────────────────────────────────────────────

K_VALUES = [10, 20, 30, 40, 50]


def _gene_block(inter, ns):
    """Extract the gene×gene block (diagonal set to 0)."""
    b = inter[ns:, ns:, 0] if inter.ndim == 3 else inter[ns:, ns:]
    b = b.copy()
    np.fill_diagonal(b, 0.0)
    return b


def _l1_degree(block):
    """
    Weighted L1 centrality of each gene: sum of |outgoing edges| +
    sum of |incoming edges|.  Could be replaced by spectral centrality,
    but remains simple and highly interpretable for GRNs.
    """
    return np.abs(block).sum(axis=1) + np.abs(block).sum(axis=0)


EDGE_THRESHOLD = 0.5   # threshold to binarize the reference (|inter| > 1 → edge)


def compute_aupr_pairwise(inters, ns, k_values=None):
    """
    For all C(N, 2) pairs (i, j) with i < j:
      1. Select top-k genes by mean L1 centrality across all runs.
      2. Extract the k×k sub-blocks from both runs.
      3. Compute AUPR in both directions:
           - binarize run_i (|val| > EDGE_THRESHOLD) → predict with |run_j|
           - binarize run_j (|val| > EDGE_THRESHOLD) → predict with |run_i|
         Then average the two valid values.
      4. Collect all C(N,2) AUPR values per k.

    Baseline = mean sparsity across all runs per k.

    Returns:
        k_eff       : list[int]
        aupr_pairs  : (C(N,2), len(k_eff)) array, NaN if undefined
        baseline    : (len(k_eff),) array
    """
    try:
        from sklearn.metrics import average_precision_score as aps
    except ImportError:
        print("  Warning: sklearn not available — AUPR skipped")
        return [], np.zeros((0, 0)), np.zeros(0)

    if k_values is None:
        k_values = K_VALUES
    n = len(inters)
    if n < 2:
        return [], np.zeros((0, len(k_values))), np.zeros(len(k_values))

    blocks = [_gene_block(inter, ns) for inter in inters]
    G      = blocks[0].shape[0]

    # Rank genes by mean L1 centrality (no favored run)
    avg_degree = np.mean([_l1_degree(b) for b in blocks], axis=0)
    order = np.argsort(avg_degree)[::-1]

    # Effective k values (deduplicated, capped at G)
    k_eff = []
    for k in k_values:
        ke = min(k, (G // 10) * 10)
        if ke not in k_eff:
            k_eff.append(ke)

    n_pairs    = n * (n - 1) // 2
    aupr_pairs = np.full((n_pairs, len(k_eff)), np.nan)
    baseline   = np.full(len(k_eff), np.nan)

    for ki, k in enumerate(k_eff):
        idx = order[:k]
        # baseline = mean sparsity across all runs
        sparsities = [(np.abs(b[np.ix_(idx, idx)].flatten()) > EDGE_THRESHOLD).mean()
                      for b in blocks]
        baseline[ki] = float(np.mean(sparsities))

        pair = 0
        for i in range(n):
            for j in range(i + 1, n):
                bi = blocks[i][np.ix_(idx, idx)].flatten()
                bj = blocks[j][np.ix_(idx, idx)].flatten()
                yi = (np.abs(bi) > EDGE_THRESHOLD).astype(int)
                yj = (np.abs(bj) > EDGE_THRESHOLD).astype(int)

                vals = []
                if 0 < yi.sum() < len(yi):
                    vals.append(aps(yi, np.abs(bj)))
                if 0 < yj.sum() < len(yj):
                    vals.append(aps(yj, np.abs(bi)))
                if vals:
                    aupr_pairs[pair, ki] = float(np.mean(vals))
                pair += 1

    return k_eff, aupr_pairs, baseline


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def _draw_violin(ax, vals, color, pos=0):
    """Violin + stripplot for a list of scalar values."""
    if len(vals) < 2:
        ax.scatter([pos], vals, s=30, color=color, zorder=4)
        return
    parts = ax.violinplot([vals], positions=[pos], widths=0.6,
                           showmedians=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor(color); pc.set_alpha(0.60); pc.set_edgecolor('none')
    parts['cmedians'].set_color('#222222'); parts['cmedians'].set_linewidth(1.2)
    for key in ('cmins', 'cmaxes', 'cbars'):
        if key in parts:
            parts[key].set_color('#888888'); parts[key].set_linewidth(0.8)
    jitter = np.random.default_rng(42).uniform(-0.10, 0.10, size=len(vals))
    ax.scatter(np.full(len(vals), pos) + jitter, vals,
               s=16, color=color, zorder=5, alpha=0.9,
               edgecolors='white', linewidths=0.4)


def make_figure(datasets=DATASETS, n_runs=N_RUNS, k_values=None,
                save_path='figureS9.png'):

    if k_values is None:
        k_values = K_VALUES

    # ── Data collection ──────────────────────────────────────────────────────
    all_cos   = {}   # {name: list of pairwise cosine sims}
    all_aupr  = {}   # {name: (k_eff list, aupr_mat)}

    for cfg in datasets:
        print(f"\n══ {cfg['name']} ══")
        inters, ns = get_inter_list(cfg, n_runs)

        cos = pairwise_cosine(inters, ns)
        all_cos[cfg['name']] = cos
        print(f"  cosine : {np.mean(cos):.4f} ± {np.std(cos):.4f}  ({len(inters)} runs)")

        k_eff, aupr_mat, baseline = compute_aupr_pairwise(inters, ns, k_values)
        all_aupr[cfg['name']] = (k_eff, aupr_mat, baseline)
        if len(k_eff):
            mu = np.nanmean(aupr_mat, axis=0)
            print(f"  AUPR   : " + "  ".join(f"k={k}: {v:.3f}" for k, v in zip(k_eff, mu)))
            print(f"  random : " + "  ".join(f"k={k}: {v:.3f}" for k, v in zip(k_eff, baseline)))

    # ── Layout (A4 landscape) ────────────────────────────────────────────────
    n_ds = len(datasets)
    fig  = plt.figure(figsize=(11.69, 8.27))

    gs_A = gridspec.GridSpec(
        1, n_ds,
        figure=fig,
        left=0.07, right=0.97,
        top=0.91, bottom=0.56,
        wspace=0.35,
    )
    gs_B = gridspec.GridSpec(
        1, n_ds,
        figure=fig,
        left=0.07, right=0.97,
        top=0.47, bottom=0.09,
        wspace=0.35,
    )

    axes_A = [fig.add_subplot(gs_A[0, i]) for i in range(n_ds)]
    axes_B = [fig.add_subplot(gs_B[0, i]) for i in range(n_ds)]

    # ── Panel A : cosine similarity ──────────────────────────────────────────
    for i, (cfg, ax) in enumerate(zip(datasets, axes_A)):
        vals  = all_cos[cfg['name']]
        color = COLORS[i]
        _draw_violin(ax, vals, color)
        mu, sd = np.mean(vals), np.std(vals)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylim(0.6, 1.0)
        ax.set_title(cfg['name'], fontsize=9, fontweight='bold', pad=3)
        ax.set_xlabel(f'μ={mu:.3f}  σ={sd:.3f}', fontsize=6.5, color='#444444')
        ax.set_ylabel('Cosine similarity' if i == 0 else '', fontsize=7)
        ax.tick_params(axis='y', labelsize=6.5)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)

    # ── Panel B : AUPR vs top-k ──────────────────────────────────────────────
    for i, (cfg, ax) in enumerate(zip(datasets, axes_B)):
        color                    = COLORS[i]
        k_eff, aupr_mat, baseline = all_aupr[cfg['name']]

        if len(k_eff) == 0 or aupr_mat.shape[0] == 0:
            ax.text(0.5, 0.5, 'AUPR not available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=7)
            ax.axis('off')
            continue

        x = np.arange(len(k_eff))

        # Individual lines (thin, semi-transparent)
        for r in range(aupr_mat.shape[0]):
            row = aupr_mat[r]
            mask = ~np.isnan(row)
            if mask.sum() < 2:
                continue
            ax.plot(x[mask], row[mask],
                    color=color, lw=0.9, alpha=0.35, zorder=2)

        # Mean ± 1σ band
        mu_k  = np.nanmean(aupr_mat, axis=0)
        sd_k  = np.nanstd(aupr_mat,  axis=0)
        valid = ~np.isnan(mu_k)
        ax.plot(x[valid], mu_k[valid],
                color=color, lw=2.0, zorder=4, label='mean')
        ax.fill_between(x[valid],
                        mu_k[valid] - sd_k[valid],
                        mu_k[valid] + sd_k[valid],
                        color=color, alpha=0.18, zorder=3)

        # Random baseline = actual sparsity per k (varies with k)
        bl_valid = ~np.isnan(baseline)
        if bl_valid.any():
            ax.plot(x[bl_valid], baseline[bl_valid],
                    color='#888888', linestyle='--', lw=0.9,
                    alpha=0.8, zorder=3, label='random')

        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_eff], fontsize=7)
        ax.set_xlabel('Top-k genes (L1 centrality)', fontsize=7)
        ax.set_ylabel('AUPR' if i == 0 else '', fontsize=7)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(cfg['name'], fontsize=9, fontweight='bold', pad=3)
        ax.tick_params(axis='y', labelsize=6.5)
        ax.spines[['top', 'right']].set_visible(False)
        if i == 0:
            ax.legend(fontsize=6, frameon=False)

    # ── Panel labels ─────────────────────────────────────────────────────────
    fig.text(0.01, 0.93, 'A', fontsize=13, fontweight='bold', color='#111111', va='top')
    fig.text(0.01, 0.49, 'B', fontsize=13, fontweight='bold', color='#111111', va='top')

    # ── Save ─────────────────────────────────────────────────────────────────
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"\nSaved {save_path}")
    try:
        from PIL import Image
        pdf_path = save_path.replace('.png', '.pdf')
        Image.open(save_path).convert('RGB').save(pdf_path, 'PDF', resolution=300)
        print(f"Saved {pdf_path}")
    except ImportError:
        print("(PIL not available — PDF export skipped)")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-runs',   type=int, default=N_RUNS,
                    help='Number of independent re-fits per dataset')
    ap.add_argument('--k-values', type=int, nargs='+', default=K_VALUES,
                    help='Top-k values for AUPR (e.g.: 10 20 30 40 50)')
    ap.add_argument('--out',      type=str, default='figureS9.png',
                    help='Output file name')
    args = ap.parse_args()
    make_figure(n_runs=args.n_runs, k_values=args.k_values, save_path=args.out)
