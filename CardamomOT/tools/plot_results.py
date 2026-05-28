"""
Plotting utilities for comparing data, mixture, trajectories, and simulations.
"""
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import scipy
from umap import UMAP


def _enforce_cell_type_order(adata, class_order, label_key='cell_type'):
    if label_key in adata.obs:
        adata.obs[label_key] = pd.Categorical(
            adata.obs[label_key], categories=class_order, ordered=True
        )


def _umap_scatter(ax, coords, values, vmin, vmax, cmap='viridis', s=6):
    return ax.scatter(
        coords[:, 0], coords[:, 1], c=values, cmap=cmap,
        vmin=vmin, vmax=vmax, s=s, linewidths=0, rasterized=True,
    )


def _plot_time_panel(adatas, names):
    time_values_all = []
    for A in adatas:
        if "time" in A.obs:
            tv = pd.to_numeric(A.obs["time"], errors='coerce').dropna()
            if tv.size > 0:
                time_values_all.append(tv.values)
    if not time_values_all:
        return
    time_cat = np.concatenate(time_values_all)
    vmin, vmax = float(time_cat.min()), float(time_cat.max())
    fig, axes = plt.subplots(1, len(adatas), figsize=(4 * len(adatas), 4), squeeze=False)
    scatter_ref = None
    for i, (A, name) in enumerate(zip(adatas, names)):
        ax = axes[0, i]
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if "time" in A.obs:
            tv = pd.to_numeric(A.obs["time"], errors='coerce').values
            scatter_ref = _umap_scatter(ax, A.obsm["X_umap"], tv, vmin, vmax)
        else:
            ax.axis('off')
    if scatter_ref is not None:
        cbar_ax = fig.add_axes([0.12, 0.06, 0.76, 0.03])
        cb = fig.colorbar(scatter_ref, cax=cbar_ax, orientation='horizontal')
        cb.set_label('time')
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    plt.show()


def _plot_celltype_panel(adatas, names, class_order=None):
    if not any("cell_type" in A.obs for A in adatas):
        return
    if class_order is not None:
        categories = list(class_order)
    else:
        cats = []
        for A in adatas:
            if "cell_type" in A.obs:
                cats.extend(A.obs["cell_type"].astype(str).unique().tolist())
        categories = list(dict.fromkeys(cats))

    colors = plt.cm.tab10.colors
    color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}

    fig, axes = plt.subplots(1, len(adatas), figsize=(4 * len(adatas), 4), squeeze=False)
    for i, (A, name) in enumerate(zip(adatas, names)):
        ax = axes[0, i]
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if "cell_type" in A.obs:
            coords = A.obsm["X_umap"]
            for cat in categories:
                mask = (A.obs["cell_type"].astype(str) == cat).values
                if np.any(mask):
                    ax.scatter(coords[mask, 0], coords[mask, 1],
                               color=color_map[cat], s=6, linewidths=0, rasterized=True)
        else:
            ax.axis('off')

    present = [cat for cat in categories if any(
        ("cell_type" in A.obs) and (A.obs["cell_type"].astype(str) == cat).any()
        for A in adatas
    )]
    handles = [mpatches.Patch(color=color_map[cat], label=cat) for cat in present]
    if handles:
        fig.legend(handles, [h.get_label() for h in handles],
                   loc='lower center', ncol=len(handles),
                   frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    plt.show()


def _fit_umap(adatas, names, reference_idx=0, norm=True, log=True):
    """Fit UMAP on adatas[reference_idx] and project all others onto it."""
    reducer = UMAP(random_state=42, min_dist=0.7)
    ref = adatas[reference_idx]
    if norm:
        sc.pp.normalize_total(ref, target_sum=1e4)
    if log:
        sc.pp.log1p(ref)
    X_ref = ref.X.toarray() if scipy.sparse.issparse(ref.X) else ref.X
    proj = reducer.fit(X_ref)
    ref.obsm["X_umap"] = proj.embedding_
    for idx, A in enumerate(adatas):
        if idx == reference_idx:
            continue
        if norm:
            sc.pp.normalize_total(A, target_sum=1e4)
        if log:
            sc.pp.log1p(A)
        X = A.X.toarray() if scipy.sparse.issparse(A.X) else A.X
        A.obsm["X_umap"] = proj.transform(X)


def _joint_umap(adatas, names, norm=True, log=True):
    """Fit UMAP jointly on all adatas concatenated."""
    reducer = UMAP(random_state=42, min_dist=0.7)
    adata_all = ad.concat(adatas, join='inner', label='source', keys=names)
    if norm:
        sc.pp.normalize_total(adata_all, target_sum=1e4)
    if log:
        sc.pp.log1p(adata_all)
    X = adata_all.X.toarray() if scipy.sparse.issparse(adata_all.X) else adata_all.X
    proj = reducer.fit(X)
    adata_all.obsm["X_umap"] = proj.transform(X)
    for name, A in zip(names, adatas):
        A.obsm["X_umap"] = adata_all[adata_all.obs['source'] == name].obsm["X_umap"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_results_rna_mixture(split, project_on_full, p, normtransform=True, logtransform=True):
    """Compare observed data to the NB mixture fit (UMAP by time and cell type).

    Parameters
    ----------
    split : str
        Data split to load (``"full"`` or ``"train"``).
    project_on_full : bool
        If True, fit UMAP on data and project mixture; otherwise fit jointly.
    p : str
        Path to the project directory (trailing slash included).
    normtransform : bool
        Apply total-count normalisation before UMAP.
    logtransform : bool
        Apply log1p transformation before UMAP.
    """
    adata_full_init = ad.read_h5ad(p + f'Data/data_{split}.h5ad')
    adata_beta_init = ad.read_h5ad(p + 'cardamomOT/adata_beta.h5ad')

    if "Semrau" in p:
        exclude = (adata_full_init.var_names != "Sparc")
        adata_full = adata_full_init[:, exclude].copy()
        adata_beta = adata_beta_init[:, exclude].copy()
    else:
        adata_full = adata_full_init.copy()
        adata_beta = adata_beta_init.copy()

    adatas = [adata_full, adata_beta]
    names = ["Exp Data", "NB Mixture"]

    class_order = None
    if "cell_type" in adata_full.obs:
        class_order = adata_full.obs['cell_type'].astype(str).unique().tolist()
        for A in adatas:
            _enforce_cell_type_order(A, class_order)

    for name, A in zip(names, adatas):
        A.obs_names = [f"{name}_{i}" for i in range(A.n_obs)]
        A.obs["source"] = name

    if project_on_full:
        _fit_umap(adatas, names, reference_idx=0, norm=normtransform, log=logtransform)
    else:
        _joint_umap(adatas, names, norm=normtransform, log=logtransform)

    _plot_time_panel(adatas, names)
    _plot_celltype_panel(adatas, names, class_order)


def plot_results_rna_clean(split, project_on_full, p, stim=1.0, prior=1.0,
                           normtransform=True, logtransform=True):
    """Compare inferred RNA trajectories across all pipeline stages (UMAP).

    Loads reference (NB mixture), network modes, RNA trajectories, and
    simulated data, then plots time and cell-type UMAPs for all four.

    Parameters
    ----------
    split : str
        Data split (``"full"`` or ``"train"``).
    project_on_full : bool
        If True, fit UMAP on the NB mixture and project others.
    p : str
        Path to the project directory (trailing slash included).
    stim : float
        Stimulus-edge penalisation value used during inference.
    prior : float
        Prior-network weighting value used during inference.
    normtransform : bool
        Apply total-count normalisation before UMAP.
    logtransform : bool
        Apply log1p transformation before UMAP.
    """
    s, q = stim, prior
    adata_beta = ad.read_h5ad(p + f'cardamomOT/adata_beta_stim{s}_prior{q}.h5ad')
    adata_theta = ad.read_h5ad(p + f'cardamomOT/adata_theta_stim{s}_prior{q}.h5ad')
    adata_sim = ad.read_h5ad(p + f'cardamomOT/adata_sim_stim{s}_prior{q}.h5ad')
    adata_rna = ad.read_h5ad(p + f'cardamomOT/adata_rna_traj_stim{s}_prior{q}.h5ad')

    adatas = [adata_rna, adata_beta, adata_theta, adata_sim]
    names = ['Reference', "NB Mixture", "Network", "Sim"]

    class_order = None
    if "cell_type" in adata_beta.obs:
        class_order = adata_beta.obs['cell_type'].astype(str).unique().tolist()
        for A in adatas:
            _enforce_cell_type_order(A, class_order)

    for name, A in zip(names, adatas):
        A.obs_names = [f"{name}_{i}" for i in range(A.n_obs)]
        A.obs["source"] = name

    if project_on_full:
        _fit_umap(adatas, names, reference_idx=1, norm=normtransform, log=logtransform)
    else:
        _joint_umap(adatas, names, norm=normtransform, log=logtransform)

    _plot_time_panel(adatas, names)
    _plot_celltype_panel(adatas, names, class_order)


def plot_results_prot(project_on_full, p, stim=1.0, prior=1.0):
    """Compare protein trajectories and simulations (UMAP colored by time).

    Parameters
    ----------
    project_on_full : bool
        If True, fit UMAP on trajectories and project simulations.
    p : str
        Path to the project directory (trailing slash included).
    stim : float
        Stimulus-edge penalisation value used during inference.
    prior : float
        Prior-network weighting value used during inference.
    """
    s, q = stim, prior
    adata_traj = ad.read_h5ad(p + f'cardamomOT/adata_prot_traj_stim{s}_prior{q}.h5ad')
    adata_simul = ad.read_h5ad(p + f'cardamomOT/adata_prot_simul_stim{s}_prior{q}.h5ad')

    adatas = [adata_traj, adata_simul]
    names = ["Traj", "Sim"]

    for name, adata in zip(names, adatas):
        adata.obs_names = [f"{name}_{i}" for i in range(adata.n_obs)]
        adata.obs["source"] = name

    reducer = UMAP(random_state=42, min_dist=0.7)
    if project_on_full:
        X = adata_traj.X.toarray() if scipy.sparse.issparse(adata_traj.X) else adata_traj.X
        proj = reducer.fit(X)
        adata_traj.obsm["X_umap"] = proj.transform(X)
        for adata in [adata_simul]:
            X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
            adata.obsm["X_umap"] = proj.transform(X)
    else:
        adata_all = ad.concat(adatas, join='inner', label='source', keys=names)
        X = adata_all.X.toarray() if scipy.sparse.issparse(adata_all.X) else adata_all.X
        proj = reducer.fit(X)
        adata_all.obsm["X_umap"] = proj.transform(X)
        for name, adata in zip(names, adatas):
            adata.obsm["X_umap"] = adata_all[adata.obs_names].obsm["X_umap"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i, (adata, name) in enumerate(zip(adatas, names)):
        if "time" in adata.obs:
            sc.pl.umap(adata, color="time", ax=axes[i], show=False, title=f"{name} - time")
        else:
            axes[i].axis('off')
    plt.tight_layout()
    plt.show()


def plot_results_rna_clean_kov(project_on_full, p, combo, stim=1.0, prior=1.0, normlog=True):
    """Compare wild-type simulation to a KO/OV perturbation (UMAP).

    Parameters
    ----------
    project_on_full : bool
        If True, fit UMAP on NB mixture and project others.
    p : str
        Path to the project directory (trailing slash included).
    combo : str
        Perturbation label, e.g. ``"KO_Gata6_OV_none"``.
    stim : float
        Stimulus-edge penalisation value used during inference.
    prior : float
        Prior-network weighting value used during inference.
    normlog : bool
        Apply normalisation and log1p before UMAP.
    """
    import os
    s, q = stim, prior
    adata_traj = ad.read_h5ad(p + f'cardamomOT/adata_beta_stim{s}_prior{q}.h5ad')
    adata_sim = ad.read_h5ad(p + f'cardamomOT/adata_sim_stim{s}_prior{q}.h5ad')
    adata_KOV = ad.read_h5ad(
        os.path.join(p, f'cardamomOT/adata_sim_{combo}_stim{s}_prior{q}.h5ad')
    )

    adatas = [adata_traj, adata_sim, adata_KOV]
    names = ["Train", "Sim", combo]

    class_order = None
    if "cell_type" in adata_traj.obs:
        class_order = adata_traj.obs['cell_type'].astype(str).unique().tolist()
        for A in adatas:
            _enforce_cell_type_order(A, class_order)

    for name, A in zip(names, adatas):
        A.obs_names = [f"{name}_{i}" for i in range(A.n_obs)]
        A.obs["source"] = name

    if project_on_full:
        _fit_umap(adatas, names, reference_idx=0, norm=normlog, log=normlog)
    else:
        _joint_umap(adatas, names, norm=normlog, log=normlog)

    _plot_time_panel(adatas, names)

    if class_order is not None:
        categories = list(class_order)
    else:
        cats = []
        for A in adatas:
            if "cell_type" in A.obs:
                cats.extend(A.obs["cell_type"].astype(str).unique().tolist())
        categories = list(dict.fromkeys(cats))

    cmap_cat = plt.get_cmap('Dark2')
    color_map = {cat: cmap_cat(i % 8) for i, cat in enumerate(categories)}

    if any("cell_type" in A.obs for A in adatas):
        fig, axes = plt.subplots(1, len(adatas), figsize=(4 * len(adatas), 4), squeeze=False)
        for i, (A, name) in enumerate(zip(adatas, names)):
            ax = axes[0, i]
            ax.set_title(name, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if "cell_type" in A.obs:
                coords = A.obsm["X_umap"]
                for cat in categories:
                    mask = (A.obs["cell_type"].astype(str) == cat).values
                    if np.any(mask):
                        ax.scatter(coords[mask, 0], coords[mask, 1],
                                   color=color_map[cat], s=6, linewidths=0, rasterized=True)
            else:
                ax.axis('off')
        present = [cat for cat in categories if any(
            ("cell_type" in A.obs) and (A.obs["cell_type"].astype(str) == cat).any()
            for A in adatas
        )]
        handles = [mpatches.Patch(color=color_map[cat], label=cat) for cat in present]
        if handles:
            fig.legend(handles, [h.get_label() for h in handles],
                       loc='lower center', ncol=len(handles),
                       frameon=False, bbox_to_anchor=(0.5, -0.12))
        fig.tight_layout(rect=[0, 0.18, 1, 1])
        plt.show()
