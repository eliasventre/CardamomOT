"""
figure_6.py
-----------
CardamomOT — model fit quality on three experimental datasets
(Semrau, Kameneva, Schiebinger).

Layout  (A4 portrait, 8.27 × 11.69 in)
-----------------------------------------
Row A–F : UMAP embeddings (Reference | CardamomOT) × 3 datasets, coloured by time
Row G–I : Cell-type composition timeline (stacked bar) × 3 datasets
Row J–O : Mean ± SD gene expression × 3 datasets (1–3 genes each)

Usage:
    python figure_6.py
"""

import sys
import numpy as np
import scanpy as sc
import anndata as ad
import scipy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from umap import UMAP

sys.path += ['./../../']
from CardamomOT import NetworkModel as CardamomNetworkModel   # noqa: F401
from CardamomOT import train_classifier, predict_cell_types

# ---------------------------------------------------------------------------
# Paths — adjust to your environment
# ---------------------------------------------------------------------------
SCHIE_PATH     = './../../experimental_datasets/Schiebinger'
SEMRAU_PATH    = './../../experimental_datasets/Semrau'
KAMENEVA_PATH  = './../../experimental_datasets/Kameneva'

# ---------------------------------------------------------------------------
# Visual constants  (harmonised with figure_2 style)
# ---------------------------------------------------------------------------
REF_COLOR = '#555555'   # grey  — reference data
SIM_COLOR = '#C94040'   # red   — CardamomOT model
NB_COLOR  = '#4878CF'   # blue  — NB baseline

# Gene indices for mean±SD panels (same as notebook)
SEMRAU_GENES    = [4]
KAMENEVA_GENES  = [86, 12]
SCHIE_GENES     = [71, 107, 83]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_dense(X):
    return X.toarray() if scipy.sparse.issparse(X) else np.asarray(X, dtype=float)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# STIM hyperparameter per dataset (from figure_7)
STIM  = {'Semrau': 1.0, 'Kameneva': 0.2, 'Schiebinger': 1.0}
PRIOR = 1.0


def load_dataset(base_path, dataset_name):
    """
    Load adata_beta, adata_sim and rna_traj directly from cardamomOT/ h5ad files,
    using the correct stim/prior hyperparameters per dataset.

    Returns:
        adata_beta  – AnnData from adata_beta_stim{s}_prior{p}.h5ad
        adata_sim   – AnnData from adata_sim_stim{s}_prior{p}.h5ad
        adata_traj  – AnnData wrapping rna_traj with time + var_names + cell_type
        rna_traj    – np.ndarray (cells, genes), from data_rna.npy col 1:
        time_traj   – np.ndarray (cells,),       from data_rna.npy col 0
        adata_ref   – AnnData from Data/data_full.h5ad or data_train.h5ad
    """
    import os
    cp   = f'{base_path}/cardamomOT'
    stim = STIM[dataset_name]

    # Reference AnnData (for classifier + var_names)
    full_path  = f'{base_path}/Data/data_full.h5ad'
    train_path = f'{base_path}/Data/data_train.h5ad'
    adata_ref  = sc.read(train_path if 'Schiebinger' in base_path else full_path)

    # adata_beta and adata_sim from cardamomOT h5ad files
    adata_beta = sc.read_h5ad(f'{cp}/adata_beta_stim{stim}_prior{PRIOR}.h5ad')
    adata_sim  = sc.read_h5ad(f'{cp}/adata_sim_stim{stim}_prior{PRIOR}.h5ad')

    # AnnData wrapper for rna_traj
    adata_traj = sc.read_h5ad(f'{cp}/adata_rna_traj_stim{stim}_prior{PRIOR}.h5ad')

    # Predict cell types
    clf        = train_classifier(adata_ref, label_key='cell_type')
    adata_beta = predict_cell_types(adata_beta.copy(), clf)
    adata_sim  = predict_cell_types(adata_sim.copy(),  clf)
    adata_traj = predict_cell_types(adata_traj.copy(), clf)

    return adata_beta, adata_sim, adata_traj, adata_ref


# ---------------------------------------------------------------------------
# Panel drawing helpers
# ---------------------------------------------------------------------------

def draw_umap(ax, coords, time_vals, title, s=5, show_colorbar=False):
    """UMAP scatter coloured by time, figure_2 style."""
    sc_obj = ax.scatter(coords[:, 0], coords[:, 1],
                        c=time_vals, cmap='viridis', alpha=0.6, s=s, lw=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('UMAP 1', fontsize=8)
    ax.set_ylabel('UMAP 2', fontsize=8)
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        cb = plt.colorbar(sc_obj, cax=cax)
        cb.set_label('Time', fontsize=7)
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        cb.locator = plt.MaxNLocator(nbins=4)
        cb.update_ticks()


def draw_mean_sd(ax, adata_traj, adata_beta, adata_sim,
                 gene_idx, gene_name,
                 show_legend=False, show_xlabel=True, show_ylabel=True):
    """
    Mean ± SD gene expression — figure_2 style.
    Three curves: rna_traj/Reference (grey), adata_beta/NB (blue), adata_sim/Model (red).
    Each dataset uses its own time vector.
    Legend shown only when show_legend=True.
    """
    x_beta    = _to_dense(adata_beta.X)
    time_beta = np.array(adata_beta.obs['time'])
    x_sim     = _to_dense(adata_sim.X)
    time_sim  = np.array(adata_sim.obs['time'])
    rna_traj     = _to_dense(adata_traj.X)
    time_traj  = np.array(adata_traj.obs['time'])

    curves = [
        (rna_traj[:, gene_idx], time_traj, REF_COLOR, 'Reference'),
        (x_beta[:,   gene_idx], time_beta, NB_COLOR,  'NB mixture'),
        (x_sim[:,    gene_idx], time_sim,  SIM_COLOR,  'Sim'),
    ]

    for vals, tvec, color, label in curves:
        unique_t = np.sort(np.unique(tvec))
        means = np.array([np.mean(vals[tvec == t]) for t in unique_t])
        stds  = np.array([np.std( vals[tvec == t]) for t in unique_t])
        ax.fill_between(unique_t, np.maximum(means - stds, 0), means + stds,
                        color=color, alpha=0.18)
        ax.plot(unique_t, means, color=color, lw=1.5, label=label)

    ax.set_title(gene_name, fontsize=10, pad=2)
    if show_xlabel:
        ax.set_xlabel('Time', fontsize=7, labelpad=2)
    if show_ylabel:
        ax.set_ylabel('Mean ± SD', fontsize=7, labelpad=2)
    else:
        ax.set_ylabel('')
    ax.tick_params(labelsize=5, pad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    if show_legend:
        ax.legend(fontsize=6, frameon=False, loc='upper center')


def draw_celltype_timeline(ax, adata_beta, adata_sim, title,
                           time_key='time', cell_type_key='cell_type',
                           n_timepoints=6, show_legend=False,
                           box_aspect=0.6):
    """
    Stacked-bar cell-type composition — 2 bar groups per timepoint:
      left  (solid)       = adata_beta
      right (alpha=0.45)  = adata_sim
    6 timepoints chosen uniformly (linspace) from adata_beta available times.
    """
    all_cts = sorted(
        set(adata_beta.obs[cell_type_key].astype(str).unique()) |
        set(adata_sim.obs[cell_type_key].astype(str).unique())
    )
    palette   = sns.color_palette('tab10', len(all_cts))
    color_map = {ct: palette[i] for i, ct in enumerate(all_cts)}

    # Pick n_timepoints uniformly spaced from adata_beta available times
    avail_beta = np.sort(np.unique(adata_beta.obs[time_key].astype(float)))
    indices      = np.round(np.linspace(0, len(avail_beta) - 1, n_timepoints)).astype(int)
    display_times = avail_beta[indices]
    n_display     = len(display_times)

    def _props(adata, t_display):
        ser   = adata.obs[time_key].astype(float)
        ct    = adata.obs[cell_type_key].astype(str)
        avail = np.sort(np.unique(ser))
        out   = []
        for t in t_display:
            t_near = avail[np.argmin(np.abs(avail - t))]
            mask = ser == t_near; n_t = mask.sum()
            out.append({c: (ct[mask] == c).sum() / n_t if n_t > 0 else 0.0
                        for c in all_cts})
        return out

    props_beta = _props(adata_beta, display_times)
    props_sim  = _props(adata_sim,  display_times)

    w    = 0.35
    x    = np.arange(n_display)
    bots = [np.zeros(n_display), np.zeros(n_display)]
    handles, used_labels = [], []

    def _legend_label(cell_type):
        label = str(cell_type).replace('_', ' ')
        return label[:1].upper() + label[1:] if label else label

    for ct in all_cts:
        col = color_map[ct]
        pb  = np.array([d[ct] for d in props_beta])
        ps  = np.array([d[ct] for d in props_sim])

        bar = ax.bar(x - w/2, pb, w, bottom=bots[0], color=col, label=ct,
                edgecolor='black', linewidth=0.2)
        ax.bar(x + w/2, ps, w, bottom=bots[1], color=col,
             edgecolor='black', linewidth=0.2)

        if ct not in used_labels:
            handles.append(bar); used_labels.append(ct)
        bots[0] += pb; bots[1] += ps

    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(t)}' for t in display_times],
                       fontsize=6, rotation=0)
    ax.set_ylabel('Proportion', fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis='both', labelsize=5)
    ax.set_box_aspect(box_aspect)
    ax.set_anchor('N')
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)

    if show_legend:
        n_cols = max(1, len(used_labels) // 2)
        legend_labels = [_legend_label(ct) for ct in used_labels]
        ax.legend(handles=handles, labels=legend_labels,
                  fontsize=6, loc='upper center', ncol=n_cols,
                  bbox_to_anchor=(0.5, -0.16),
                  title='Reference | Simulated', title_fontsize=6, framealpha=0.5)


# ---------------------------------------------------------------------------
# UMAP projection: adata_beta vs adata_sim
# ---------------------------------------------------------------------------

def compute_umap_beta_sim(adata_beta, adata_sim,
                           umap_min_dist=0.7, umap_random_state=42):
    """
    Compute UMAP for adata_beta and adata_sim:
      1. normalize_total each dataset independently (on a copy)
      2. concatenate
      3. log1p the concatenation
      4. fit_transform (no projection — all cells fit together)

    Returns:
        beta_2d   – UMAP coords for adata_beta cells
        sim_2d    – UMAP coords for adata_sim cells
        t_beta    – time values for adata_beta cells
        t_sim     – time values for adata_sim cells
    """
    t_beta = np.array(adata_beta.obs['time'])
    t_sim  = np.array(adata_sim.obs['time'])

    # Step 1: normalize_total each dataset independently (copy to avoid side effects)
    ab = ad.AnnData(_to_dense(adata_beta.X).copy())
    sc.pp.normalize_total(ab, target_sum=1e4)

    as_ = ad.AnnData(_to_dense(adata_sim.X).copy())
    sc.pp.normalize_total(as_, target_sum=1e4)

    # Step 2+3: concatenate then log1p
    n_beta = ab.n_obs
    combined = ad.AnnData(np.vstack([_to_dense(ab.X), _to_dense(as_.X)]))
    sc.pp.log1p(combined)
    combined_norm = _to_dense(combined.X)

    # Step 4: fit_transform on the full concatenation
    umap_model = UMAP(n_components=2, random_state=umap_random_state,
                      min_dist=umap_min_dist)
    coords = umap_model.fit_transform(combined_norm)

    beta_2d = coords[:n_beta]
    sim_2d  = coords[n_beta:]

    return beta_2d, sim_2d, t_beta, t_sim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Load Semrau ──────────────────────────────────────────────────────────
    print("Loading Semrau...")
    (adata_beta_semrau, adata_sim_semrau, adata_traj_semrau, _) = load_dataset(SEMRAU_PATH, 'Semrau')
    beta_semrau_2d, sim_semrau_beta_2d, t_beta_semrau, t_sim_semrau = compute_umap_beta_sim(
        adata_traj_semrau, adata_sim_semrau)

    # ── Load Kameneva ────────────────────────────────────────────────────────
    print("Loading Kameneva...")
    (adata_beta_kameneva, adata_sim_kameneva, adata_traj_kameneva, _) = load_dataset(KAMENEVA_PATH, 'Kameneva')
    beta_kameneva_2d, sim_kameneva_beta_2d, t_beta_kameneva, t_sim_kameneva = compute_umap_beta_sim(
        adata_traj_kameneva, adata_sim_kameneva)

    # ── Load Schiebinger ─────────────────────────────────────────────────────
    print("Loading Schiebinger...")
    (adata_beta_schie, adata_sim_schie, adata_traj_schie, _) = load_dataset(SCHIE_PATH, 'Schiebinger')
    beta_schie_2d, sim_schie_beta_2d, t_beta_schie, t_sim_schie = compute_umap_beta_sim(
        adata_beta_schie, adata_sim_schie)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69))
    gs   = gridspec.GridSpec(2, 2, figure=fig,
                              height_ratios=[3, 1.2], width_ratios=[1.8, 1],
                              hspace=0.2, wspace=0.28)
    gs01 = gs[0, 0].subgridspec(3, 2, hspace=0.25, wspace=0.2)
    gs02 = gs[0, 1].subgridspec(3, 1, hspace=0.60, wspace=0.3)
    gs03 = gs[1, :].subgridspec(2, 3, hspace=0.45, wspace=0.3)

    axes = (
        [fig.add_subplot(gs01[r, c]) for r in range(3) for c in range(2)] +
        [fig.add_subplot(gs02[r, 0]) for r in range(3)] +
        [fig.add_subplot(gs03[r, c]) for r in range(2) for c in range(3)]
    )

    # Panel labels A–O
    for ax, lbl in zip(axes, "ABCDEFGHIJKLMNO"):
        ax.text(-0.15, 1.05, lbl, transform=ax.transAxes,
                ha='left', va='bottom', fontsize=8, fontweight='bold', clip_on=False)

    # ── UMAP panels A–F  (adata_beta | adata_sim, shared UMAP space) ──────────
    umap_configs = [
        (axes[0], beta_semrau_2d,       t_beta_semrau,    'Semrau — Reference',        5, False),
        (axes[1], sim_semrau_beta_2d,   t_sim_semrau,     'Semrau — Sim',          5, True),
        (axes[2], beta_kameneva_2d,     t_beta_kameneva,  'Kameneva — Reference',      5, False),
        (axes[3], sim_kameneva_beta_2d, t_sim_kameneva,   'Kameneva — Sim',       5, True),
        (axes[4], beta_schie_2d,        t_beta_schie,     'Schiebinger — Reference',   2, False),
        (axes[5], sim_schie_beta_2d,    t_sim_schie,      'Schiebinger — Sim',    2, True),
    ]
    for ax, coords, t_vals, title, s, with_cb in umap_configs:
        draw_umap(ax, coords, t_vals, title, s=s, show_colorbar=with_cb)

    # Remove UMAP axis labels from panels A–D (keep only E=axes[4], F=axes[5])
    for ax in axes[:4]:
        ax.set_xlabel('')
    for ax in axes[1:6:2]:
        ax.set_ylabel('')

    # ── Cell-type timeline panels G–I ─────────────────────────────────────────
    # 3 bars per timepoint: rna_traj (solid) | adata_beta (mid) | adata_sim (light)
    celltype_configs = [
        (axes[6], adata_traj_semrau,   adata_sim_semrau,   'Semrau — Cell-type timeline'),
        (axes[7], adata_traj_kameneva, adata_sim_kameneva, 'Kameneva — Cell-type timeline'),
        (axes[8], adata_traj_schie,    adata_sim_schie,    'Schiebinger — Cell-type timeline'),
    ]
    for ax, a_beta, a_sim, title in celltype_configs:
        draw_celltype_timeline(ax, a_beta, a_sim, title, show_legend=True)

    # ── Gene mean±SD panels J–O ───────────────────────────────────────────────
    # 3 curves per panel: rna_traj (grey) | adata_beta (blue) | adata_sim (red)
    # Legend only on first gene of each dataset (no repetition).
    gene_configs = [
        # (axes_slice, adata_traj, adata_beta, adata_sim, rna_traj, time_traj, gene_indices, dataset_name)
        (axes[9:10],  adata_traj_semrau,   adata_beta_semrau,   adata_sim_semrau,
         SEMRAU_GENES,   'Semrau'),
        (axes[10:12], adata_traj_kameneva, adata_beta_kameneva, adata_sim_kameneva,
         KAMENEVA_GENES, 'Kameneva'),
        (axes[12:15], adata_traj_schie,    adata_beta_schie,    adata_sim_schie,
         SCHIE_GENES,    'Schiebinger'),
    ]
    for i, (ax_slice, a_traj, a_beta, a_sim, gene_idxs, dset) in enumerate(gene_configs):
        is_last_dataset = (i == len(gene_configs) - 1)   # Schiebinger → rangée M N O
        gnames = a_traj.var_names
        for k, (ax, gene_idx) in enumerate(zip(ax_slice, gene_idxs)):
            show_main_annotations = ax in (axes[9], axes[12])
            draw_mean_sd(
                ax,
                a_traj, a_beta, a_sim,
                gene_idx, f'{dset} — {gnames[gene_idx]}',
                show_legend=(ax in (axes[9], axes[12])),
                show_xlabel=is_last_dataset,   # 'Time' only for M N O
                show_ylabel=show_main_annotations,
            )

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.savefig('figure_6.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print("Saved figure_6.png")
    try:
        from PIL import Image
        Image.open('figure_6.png').convert('RGB').save('figure_6.pdf', 'PDF', resolution=300)
        print("Saved figure_6.pdf")
    except ImportError:
        print("(PIL not available — skipping PDF export)")


if __name__ == '__main__':
    main()