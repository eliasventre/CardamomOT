"""
figure_6.py
-----------
CardamomOT — model fit quality on three experimental datasets
(Semrau, Kameneva, Schiebinger).

Layout  (A4 portrait, 8.27 × 11.69 in)
-----------------------------------------
Row A–F : PHATE embeddings (Reference | CardamomOT) × 3 datasets, coloured by time
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
import phate 

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

def get_common_limits(coords1, coords2):
    """Compute xlim and ylim spanning both coordinate arrays."""
    all_coords = np.vstack([coords1, coords2])
    xlim = (all_coords[:, 0].min(), all_coords[:, 0].max())
    ylim = (all_coords[:, 1].min(), all_coords[:, 1].max())
    return xlim, ylim


def draw_phate(ax, coords, time_vals, title, s=5, show_colorbar=False, xlim=None, ylim=None):
    """PHATE scatter coloured by time, figure_2 style."""
    sc_obj = ax.scatter(coords[:, 0], coords[:, 1],
                        c=time_vals, cmap='viridis', alpha=0.6, s=s, lw=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('PHATE 1', fontsize=8)
    ax.set_ylabel('PHATE 2', fontsize=8)
    
    # Set consistent limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        cb = plt.colorbar(sc_obj, cax=cax)
        cb.set_label('Time', fontsize=7)
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        cb.locator = plt.MaxNLocator(nbins=4)
        cb.update_ticks()


# ---------------------------------------------------------------------------
# UMAP projection: adata_beta vs adata_sim
# ---------------------------------------------------------------------------

def compute_phate_beta_sim(adata_beta, adata_sim,
                            phate_random_state=42):
    """
    Compute PHATE for adata_beta and adata_sim:
      1. normalize_total each dataset independently (on a copy)
      2. concatenate
      3. log1p the concatenation
      4. fit_transform (no projection — all cells fit together)

    Returns:
        beta_2d   – PHATE coords for adata_beta cells
        sim_2d    – PHATE coords for adata_sim cells
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

    # Step 2+3: concatenate then sqrt (PHATE's default is sqrt(counts), not log1p)
    n_beta = ab.n_obs
    combined = ad.AnnData(np.vstack([_to_dense(ab.X), _to_dense(as_.X)]))
    combined_norm = np.sqrt(combined.X)

    # Step 4: fit_transform on the full concatenation
    phate_op = phate.PHATE(n_components=2, random_state=phate_random_state)
    coords = phate_op.fit_transform(combined_norm)

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
    beta_semrau_2d, sim_semrau_beta_2d, t_beta_semrau, t_sim_semrau = compute_phate_beta_sim(
        adata_traj_semrau, adata_sim_semrau)

    # ── Load Kameneva ────────────────────────────────────────────────────────
    print("Loading Kameneva...")
    (adata_beta_kameneva, adata_sim_kameneva, adata_traj_kameneva, _) = load_dataset(KAMENEVA_PATH, 'Kameneva')
    beta_kameneva_2d, sim_kameneva_beta_2d, t_beta_kameneva, t_sim_kameneva = compute_phate_beta_sim(
        adata_traj_kameneva, adata_sim_kameneva)

    # ── Load Schiebinger ─────────────────────────────────────────────────────
    print("Loading Schiebinger...")
    (adata_beta_schie, adata_sim_schie, adata_traj_schie, _) = load_dataset(SCHIE_PATH, 'Schiebinger')
    beta_schie_2d, sim_schie_beta_2d, t_beta_schie, t_sim_schie = compute_phate_beta_sim(
        adata_beta_schie, adata_sim_schie)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.27, 6.5))
    gs   = gridspec.GridSpec(3, 2, figure=fig,
                              hspace=0.25, wspace=0.2)

    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]

    # Panel labels A–O
    for ax, lbl in zip(axes, "ABCDEF"):
        ax.text(-0.15, 1.05, lbl, transform=ax.transAxes,
                ha='left', va='bottom', fontsize=8, fontweight='bold', clip_on=False)

    # ── PHATE panels A–F  (adata_beta | adata_sim, shared PHATE space) ────────
    # Compute common limits for each dataset pair
    xlim_semrau, ylim_semrau = get_common_limits(beta_semrau_2d, sim_semrau_beta_2d)
    xlim_kameneva, ylim_kameneva = get_common_limits(beta_kameneva_2d, sim_kameneva_beta_2d)
    xlim_schie, ylim_schie = get_common_limits(beta_schie_2d, sim_schie_beta_2d)
    
    phate_configs = [
        (axes[0], beta_semrau_2d,       t_beta_semrau,    'Semrau — Reference', 5, False, xlim_semrau, ylim_semrau),
        (axes[1], sim_semrau_beta_2d,   t_sim_semrau,     'Semrau — Sim',       5, True, xlim_semrau, ylim_semrau),
        (axes[2], beta_kameneva_2d,     t_beta_kameneva,  'Kameneva — Reference', 5, False, xlim_kameneva, ylim_kameneva),
        (axes[3], sim_kameneva_beta_2d, t_sim_kameneva,   'Kameneva — Sim',      5, True, xlim_kameneva, ylim_kameneva),
        (axes[4], beta_schie_2d,        t_beta_schie,     'Schiebinger — Reference', 2, False, xlim_schie, ylim_schie),
        (axes[5], sim_schie_beta_2d,    t_sim_schie,      'Schiebinger — Sim',   2, True, xlim_schie, ylim_schie),
    ]
    for ax, coords, t_vals, title, s, with_cb, xlim, ylim in phate_configs:
        draw_phate(ax, coords, t_vals, title, s=s, show_colorbar=with_cb, xlim=xlim, ylim=ylim)

    # Remove PHATE axis labels from panels A–D (keep only E=axes[4], F=axes[5])
    for ax in axes[:4]:
        ax.set_xlabel('')
    for ax in axes[1:6:2]:
        ax.set_ylabel('')

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.savefig('figureS10.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print("Saved figureS10.png")
    try:
        from PIL import Image
        Image.open('figureS10.png').convert('RGB').save('figureS10.pdf', 'PDF', resolution=300)
        print("Saved figureS10.pdf")
    except ImportError:
        print("(PIL not available — skipping PDF export)")


if __name__ == '__main__':
    main()