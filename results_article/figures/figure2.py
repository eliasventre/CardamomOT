"""
figure_2.py
-----------
CardamomOT — model fit quality on three experimental datasets
(Semrau, Kameneva, Schiebinger).

Layout  (A4 portrait, 8.27 × 11.69 in)
-----------------------------------------
Row A  : UMAP embeddings (reference | CardamomOT) × 3 datasets, coloured by cell type
Row B  : Mean ± SD gene expression across time — Schiebinger (top) & Semrau (bottom)
Row C  : Marginal mRNA distributions — 2 genes × (all times + 3 timepoints), Schiebinger

Usage:
    python figure_2.py
"""

import sys
import numpy as np
import scanpy as sc
import anndata as ad
import scipy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from umap import UMAP

sys.path += ['./../../']
from CardamomOT import NetworkModel as CardamomNetworkModel   # noqa: F401 (kept for env)

# ---------------------------------------------------------------------------
# Paths — adjust to your environment
# ---------------------------------------------------------------------------
SCHIE_PATH   = './../../experimental_datasets/Schiebinger'
SEMRAU_PATH  = './../../experimental_datasets/Semrau'
KAMENEVA_PATH = './../../experimental_datasets/Kameneva'

# ---------------------------------------------------------------------------
# Visual constants  (harmonised with figure7 style)
# ---------------------------------------------------------------------------
SOFT_COLORS = [
    '#4878CF', '#E87B30', '#6DB36D', '#C94040', '#8C6DBF',
    '#57A8A8', '#C97BAA', '#8C7B6B', '#9CBF57', '#CF9B3A',
]
REF_COLOR = '#555555'   # grey  — reference data
SIM_COLOR = '#C94040'   # red   — CardamomOT model

CELL_TYPE_ORDER = {
    'semrau':  ['Pluripotency', 'Post-implantation epiblast',
                'Extraembryonic endoderm', 'Neurectoderm'],
    'kameneva': ['Precursors', 'Intermediate', 'Sympathoblasts', 'Chromaffin'],
    'schie':   ['Mef', 'Neural', 'Stromal', 'Epithelial', 'Trophoblast', 'iPSC'],
}

# Gene indices for expression panels
SCHIE_GENES  = [24, 83, 17, 48]   # Row B — Schiebinger
SEMRAU_GENES = [36, 23, 38, 34]   # Row B — Semrau
HIST_GENES   = [83, 17]           # Row C — distributions (Schiebinger)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _soft_color(i):
    return SOFT_COLORS[i % len(SOFT_COLORS)]


def _to_dense(X):
    return X.toarray() if scipy.sparse.issparse(X) else np.asarray(X, dtype=float)


def format_ct(s):
    s = s.replace('_', ' ').capitalize()
    return 'iPSC' if s == 'Ipsc' else s


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(base_path, n_per_time=100):
    """
    Load reference + CardamomOT-beta data, subsample uniformly across
    timepoints, and project both into a shared UMAP space.
    No re-normalisation (already done upstream on full gene set); log1p is
    applied on raw counts before UMAP.
    """
    is_schie   = 'Schiebinger' in base_path
    train_file = (f'{base_path}/Data/data_train.h5ad' if is_schie
                  else f'{base_path}/Data/data_full.h5ad')

    adata_train = sc.read(train_file)
    adata_beta  = sc.read(f'{base_path}/cardamomOT/adata_beta.h5ad')
    print(adata_beta)

    rna_full   = _to_dense(adata_train.X)
    beta_full  = _to_dense(adata_beta.X)
    t_rna_full  = np.array(adata_train.obs['time'])
    t_beta_full = np.array(adata_beta.obs['time'])

    idx_rna  = np.arange(len(t_rna_full))
    idx_beta = np.arange(len(t_beta_full))

    rna_sub  = rna_full[idx_rna];   t_rna  = t_rna_full[idx_rna]
    beta_sub = beta_full[idx_beta];  t_beta = t_beta_full[idx_beta]
    obs_rna  = adata_train.obs.iloc[idx_rna]
    obs_beta = adata_beta.obs.iloc[idx_beta]

    # No re-normalisation; log1p on raw counts before UMAP
    combined = ad.AnnData(np.vstack([rna_sub, beta_sub]))
    sc.pp.log1p(combined)
    X_combined = _to_dense(combined.X)

    umap_model = UMAP(n_components=2, random_state=42, min_dist=0.7)
    umap_model.fit(X_combined)
    rna_2d  = umap_model.transform(X_combined[:len(rna_sub)])
    beta_2d = umap_model.transform(X_combined[len(rna_sub):])

    return dict(
        adata_train=adata_train, adata_beta=adata_beta,
        rna_2d=rna_2d,   beta_2d=beta_2d,
        t_rna=t_rna,     t_beta=t_beta,
        obs_rna=obs_rna, obs_beta=obs_beta,
    )


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def draw_umap(ax, coords, ct_arr, unique_cts, ct_colors, title, s=10, show_xlabel=True):
    for ct in unique_cts:
        mask = ct_arr == ct
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[ct_colors[ct]], alpha=0.6, s=s, lw=0, label=ct)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    if show_xlabel:
        ax.set_xlabel('UMAP 1', fontsize=6)
    ax.set_ylabel('UMAP 2', fontsize=6)


def draw_mean_sd(ax, x_data, x_sim, t_data, t_sim, gene_idx, gene_name,
                 dataset_name=None, show_legend=False, show_xlabel=True, show_ylabel=True):
    unique_t = np.sort(np.unique(t_data))
    for X, time_arr, color, label in [
        (x_data, t_data, REF_COLOR, 'Reference'),
        (x_sim,  t_sim,  SIM_COLOR, 'Model'),
    ]:
        means = np.array([np.mean(X[time_arr == t, gene_idx]) for t in unique_t])
        stds  = np.array([np.std( X[time_arr == t, gene_idx]) for t in unique_t])
        ax.fill_between(unique_t, np.maximum(means - stds, 0), means + stds,
                        color=color, alpha=0.18)
        ax.plot(unique_t, means, color=color, lw=1.5, label=label)
    title = f'{dataset_name} — {gene_name}' if dataset_name else gene_name
    ax.set_title(title, fontsize=7, pad=2)
    if show_xlabel:
        ax.set_xlabel('Time', fontsize=6, labelpad=2)
    if show_ylabel:
        ax.set_ylabel('Mean ± SD', fontsize=6, labelpad=2)
    ax.tick_params(labelsize=5, pad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    if show_legend:
        ax.legend(fontsize=5, frameon=False, loc='upper center')


def draw_hist(ax, vals_data, vals_sim, title, n_bins=22, show_legend=False, show_xlabel=True, show_ylabel=True):
    q99  = max(np.quantile(vals_data, 0.99), np.quantile(vals_sim, 0.99))
    bins = np.linspace(0, q99 + 1, n_bins)
    ax.hist(vals_data, density=True, bins=bins,
            color=REF_COLOR, alpha=0.45, histtype='bar')
    ax.hist(vals_sim,  density=True, bins=bins,
            ec=SIM_COLOR, histtype='step', lw=1.5)
    ax.set_title(title, fontsize=6.5, pad=2)
    if show_xlabel:
        ax.set_xlabel('mRNA (copies/cell)', fontsize=6, labelpad=2)
    if show_ylabel:
        ax.set_ylabel('Density', fontsize=6, labelpad=2)
    ax.tick_params(labelsize=5, pad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    if show_legend:
        ax.legend(
            handles=[
                Line2D([0], [0], color=REF_COLOR, lw=2, label='Reference'),
                Line2D([0], [0], color=SIM_COLOR,  lw=2, label='Model'),
            ],
            fontsize=5, frameon=False, loc='upper right',
        )



def draw_umap_time(ax, coords, time_vals, title, s=10, fig=None, add_colorbar=False):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=time_vals, cmap='viridis', alpha=0.6, s=s, lw=0)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('UMAP 1', fontsize=6)
    ax.set_ylabel('UMAP 2', fontsize=6)
    if add_colorbar and fig is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        plt.colorbar(sc, cax=cax).ax.tick_params(labelsize=5)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading Semrau...")
    semrau   = load_dataset(SEMRAU_PATH,   n_per_time=200)
    print("Loading Kameneva...")
    kameneva = load_dataset(KAMENEVA_PATH, n_per_time=100)
    print("Loading Schiebinger...")
    schie    = load_dataset(SCHIE_PATH,    n_per_time=100)

    ct_colors = {
        key: {ct: _soft_color(i) for i, ct in enumerate(cts)}
        for key, cts in CELL_TYPE_ORDER.items()
    }

    x_sd = _to_dense(schie['adata_train'].X);  t_sd = np.array(schie['adata_train'].obs['time'])
    x_ss = _to_dense(schie['adata_beta'].X);   t_ss = np.array(schie['adata_beta'].obs['time'])
    x_rd = _to_dense(semrau['adata_train'].X); t_rd = np.array(semrau['adata_train'].obs['time'])
    x_rs = _to_dense(semrau['adata_beta'].X);  t_rs = np.array(semrau['adata_beta'].obs['time'])

    unique_ts = np.sort(np.unique(t_sd))
    sel_times = unique_ts[[1, len(unique_ts) // 2, -2]]

    # ---- Layout (all coords in figure units 0-1, top->bottom) ----
    #  A  gs_tref   0.970 - 0.865   h=0.105
    #  B  gs_tmod   0.840 - 0.735   h=0.105
    #     gs_tcbar  0.726 - 0.720   h=0.006   colorbar even thinner
    #  C  gs_ref    0.680 - 0.575   h=0.105   (gap B-C: 0.040 → was 0.028)
    #  D  gs_mod    0.550 - 0.445   h=0.105   (gap C-D: 0.025)
    #     gs_leg    0.435 - 0.395   h=0.040
    #  E  gs_Bt     0.370 - 0.308   h=0.062
    #  F  gs_Br     0.285 - 0.223   h=0.062   (gap E-F: 0.023 → was 0.015)
    #  G  gs_Ct     0.193 - 0.100   h=0.093   (gap F-G: 0.030, reduced height)
    #  H  gs_Cb     0.068 - 0.000   h=0.068   (reduced height)
    fig = plt.figure(figsize=(8.27, 11.0))
    L, R = 0.07, 0.97
    WU = 0.22   # wspace for UMAP grids
    WE = 0.42   # wspace for expression grids

    gs_tref  = gridspec.GridSpec(1, 3, figure=fig, left=L, right=R, top=0.970, bottom=0.865, wspace=WU)
    gs_tmod  = gridspec.GridSpec(1, 3, figure=fig, left=L, right=R, top=0.840, bottom=0.735, wspace=WU)
    gs_tcbar = gridspec.GridSpec(1, 3, figure=fig, left=L, right=R, top=0.720, bottom=0.714, wspace=WU)
    gs_ref   = gridspec.GridSpec(1, 3, figure=fig, left=L, right=R, top=0.680, bottom=0.575, wspace=WU)
    gs_mod   = gridspec.GridSpec(1, 3, figure=fig, left=L, right=R, top=0.550, bottom=0.445, wspace=WU)
    gs_leg   = gridspec.GridSpec(1, 3, figure=fig, left=L, right=R, top=0.435, bottom=0.395, wspace=WU)
    gs_Bt    = gridspec.GridSpec(1, 4, figure=fig, left=L, right=R, top=0.370, bottom=0.308, wspace=WE)
    gs_Br    = gridspec.GridSpec(1, 4, figure=fig, left=L, right=R, top=0.283, bottom=0.221, wspace=WE)
    gs_Ct    = gridspec.GridSpec(1, 4, figure=fig, left=L, right=R, top=0.185, bottom=0.105, wspace=WE)
    gs_Cb    = gridspec.GridSpec(1, 4, figure=fig, left=L, right=R, top=0.080, bottom=0.000, wspace=WE)

    umap_datasets = [
        ('semrau',   semrau,   'Semrau',      15),
        ('kameneva', kameneva, 'Kameneva',     15),
        ('schie',    schie,    'Schiebinger',   4),
    ]

    for col, (key, d, dset_label, s) in enumerate(umap_datasets):
        cts_rna  = np.array([format_ct(c) for c in d['obs_rna']['cell_type']])
        cts_beta = np.array([format_ct(c) for c in d['obs_beta']['cell_type']])
        unique_cts = CELL_TYPE_ORDER[key]
        colors     = ct_colors[key]

        # A: ref by time
        ax_a = fig.add_subplot(gs_tref[0, col])
        sc_a = ax_a.scatter(d['rna_2d'][:, 0], d['rna_2d'][:, 1],
                            c=d['t_rna'], cmap='viridis', alpha=0.6, s=s, lw=0)
        ax_a.set_title(f'{dset_label} -- Reference', fontsize=7)
        ax_a.set_xticks([]); ax_a.set_yticks([])
        ax_a.set_ylabel('UMAP 2', fontsize=6)

        # B: CardamomOT by time
        ax_b = fig.add_subplot(gs_tmod[0, col])
        ax_b.scatter(d['beta_2d'][:, 0], d['beta_2d'][:, 1],
                     c=d['t_beta'], cmap='viridis', alpha=0.6, s=s, lw=0)
        ax_b.set_title(f'{dset_label} -- NB mixture', fontsize=7)
        ax_b.set_xticks([]); ax_b.set_yticks([])
        ax_b.set_xlabel('UMAP 1', fontsize=6); ax_b.set_ylabel('UMAP 2', fontsize=6)

        # Thin horizontal colorbar (one per column) — "Time" label only on leftmost
        ax_cb = fig.add_subplot(gs_tcbar[0, col])
        ax_cb.set_axis_off()
        pos = ax_cb.get_position()
        # Place bar at bottom of the slot to maximise gap with C below
        bar_h = pos.height
        cax = fig.add_axes([pos.x0 + pos.width * 0.08, pos.y0,
                            pos.width * 0.84, bar_h])
        cb = fig.colorbar(sc_a, cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=4, length=2, pad=1)
        cb.locator = plt.MaxNLocator(nbins=4)
        cb.update_ticks()
        if col == 0:
            # "Time" as a ylabel to the left of the bar, vertically centred
            cax.set_ylabel('Time', fontsize=5, labelpad=3, rotation=0, va='center', ha='right')

        # C: ref by cell type (no UMAP 1 label — shown on D below)
        ax_c = fig.add_subplot(gs_ref[0, col])
        draw_umap(ax_c, d['rna_2d'], cts_rna, unique_cts, colors,
                  f'{dset_label} -- Reference', s=s, show_xlabel=False)

        # D: CardamomOT by cell type
        ax_d = fig.add_subplot(gs_mod[0, col])
        draw_umap(ax_d, d['beta_2d'], cts_beta, unique_cts, colors,
                  f'{dset_label} -- NB mixture', s=s, show_xlabel=True)

        # Cell-type legend strip (2 rows x ceil(n/2) cols)
        ax_leg = fig.add_subplot(gs_leg[0, col])
        ax_leg.set_axis_off()
        handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=colors[ct], markersize=5, label=ct)
                   for ct in unique_cts]
        ncol_leg = int(np.ceil(len(unique_cts) / 2))
        ax_leg.legend(handles=handles, fontsize=5.5, markerscale=1.1,
                      loc='center', ncol=ncol_leg, frameon=False,
                      borderaxespad=0, handletextpad=0.3, columnspacing=0.8)

    # E: mean+/-SD Schiebinger (no x-label — shared with F below)
    gnames_s = schie['adata_train'].var_names
    gnames_r = semrau['adata_train'].var_names

    for i, gene_idx in enumerate(SCHIE_GENES):
        ax = fig.add_subplot(gs_Bt[0, i])
        draw_mean_sd(ax, x_sd, x_ss, t_sd, t_ss,
                     gene_idx, gnames_s[gene_idx], dataset_name='Schiebinger',
                     show_legend=(i == 0), show_xlabel=False, show_ylabel=(i == 0))

    # F: mean+/-SD Semrau (x-label shown here)
    for i, gene_idx in enumerate(SEMRAU_GENES):
        ax = fig.add_subplot(gs_Br[0, i])
        draw_mean_sd(ax, x_rd, x_rs, t_rd, t_rs,
                     gene_idx, gnames_r[gene_idx], dataset_name='Semrau',
                     show_legend=False, show_xlabel=True, show_ylabel=(i == 0))

    # G: histos gene 1 (no x-label)
    gene_name_0 = gnames_s[HIST_GENES[0]]
    ax0 = fig.add_subplot(gs_Ct[0, 0])
    draw_hist(ax0, x_sd[:, HIST_GENES[0]], x_ss[:, HIST_GENES[0]],
              f'Schiebinger - {gene_name_0} - All times', show_legend=True, show_xlabel=False, show_ylabel=True)
    for col, tp in enumerate(sel_times):
        ax = fig.add_subplot(gs_Ct[0, col + 1])
        draw_hist(ax, x_sd[t_sd == tp, HIST_GENES[0]],
                  x_ss[t_ss == tp, HIST_GENES[0]],
                  f't = {tp:.0f}h', show_xlabel=False, show_ylabel=False)

    # H: histos gene 2 (x-label shown here)
    gene_name_1 = gnames_s[HIST_GENES[1]]
    ax0 = fig.add_subplot(gs_Cb[0, 0])
    draw_hist(ax0, x_sd[:, HIST_GENES[1]], x_ss[:, HIST_GENES[1]],
              f'Schiebinger - {gene_name_1} - All times', show_legend=False, show_xlabel=True, show_ylabel=True)
    for col, tp in enumerate(sel_times):
        ax = fig.add_subplot(gs_Cb[0, col + 1])
        draw_hist(ax, x_sd[t_sd == tp, HIST_GENES[1]],
                  x_ss[t_ss == tp, HIST_GENES[1]],
                  f't = {tp:.0f}h', show_xlabel=True, show_ylabel=False)

    # Panel labels
    fig.canvas.draw()

    def _label(gs_obj, row, col, letter, dx=-0.050, dy=0.004):
        ax_tmp = fig.add_subplot(gs_obj[row, col])
        fig.canvas.draw()
        pos = ax_tmp.get_position()
        fig.text(pos.x0 + dx, pos.y1 + dy, letter,
                 ha='left', va='bottom', fontsize=10, fontweight='bold', clip_on=False)
        ax_tmp.remove()

    _label(gs_tref, 0, 0, 'A')
    _label(gs_tmod, 0, 0, 'B')
    _label(gs_ref,  0, 0, 'C')
    _label(gs_mod,  0, 0, 'D')
    _label(gs_Bt,   0, 0, 'E')
    _label(gs_Br,   0, 0, 'F')
    _label(gs_Ct,   0, 0, 'G')
    _label(gs_Cb,   0, 0, 'H')

    plt.savefig('figure_2.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print("Saved figure_2.png")
    try:
        from PIL import Image
        Image.open('figure_2.png').convert('RGB').save('figure_2.pdf', 'PDF', resolution=600)
        print("Saved figure_2.pdf")
    except ImportError:
        print("(PIL not available -- skipping PDF export)")


if __name__ == '__main__':
    main()