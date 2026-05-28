"""
Plotting and analysis utilities for KO/OV perturbation results.
"""
import os
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import scipy
from umap import UMAP

from .characterize_cell_type import train_classifier, predict_cell_types, plot_cell_type_proportions


def _enforce_cell_type_order(adata, class_order, label_key='cell_type'):
    if label_key in adata.obs:
        adata.obs[label_key] = pd.Categorical(
            adata.obs[label_key], categories=class_order, ordered=True
        )


def _fit_umap(adatas, names, reference_idx=0, norm=True, log=True):
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
            sc = ax.scatter(A.obsm["X_umap"][:, 0], A.obsm["X_umap"][:, 1],
                            c=tv, cmap='viridis', vmin=vmin, vmax=vmax,
                            s=6, linewidths=0, rasterized=True)
            scatter_ref = sc
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

    cmap_cat = plt.get_cmap('Dark2')
    color_map = {cat: cmap_cat(i % 8) for i, cat in enumerate(categories)}

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_results_sim_kov(p, combo, stim=1.0, prior=1.0, project_on_full=False, normlog=True):
    """Compare wild-type simulation to a KO/OV perturbation (UMAP).

    Parameters
    ----------
    p : str
        Path to the project directory (trailing slash included).
    combo : str
        Perturbation label, e.g. ``"KO_Gata6_OV_none"``.
    stim : float
        Stimulus-edge penalisation value used during inference.
    prior : float
        Prior-network weighting value used during inference.
    project_on_full : bool
        If True, fit UMAP on NB mixture and project others.
    normlog : bool
        Apply normalisation and log1p before UMAP.
    """
    s, q = stim, prior
    adata_traj = ad.read_h5ad(p + f'cardamomOT/adata_beta_stim{s}_prior{q}.h5ad')
    adata_sim = ad.read_h5ad(p + f'cardamomOT/adata_sim_stim{s}_prior{q}.h5ad')
    adata_kov = ad.read_h5ad(
        os.path.join(p, f'cardamomOT/adata_sim_{combo}_stim{s}_prior{q}.h5ad')
    )

    adatas = [adata_traj, adata_sim, adata_kov]
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
    _plot_celltype_panel(adatas, names, class_order)


def compare_cell_types(p, combo_name, split="full", stim=1.0, prior=1.0, label_key='cell_type'):
    """Compare cell-type proportions between wild-type and a KO/OV perturbation.

    Trains a classifier on the observed data, predicts cell types on WT
    trajectories, WT simulation, and the perturbation simulation, then plots
    stacked bar proportions for all three.

    Parameters
    ----------
    p : str
        Path to the project directory (trailing slash included).
    combo_name : str
        Perturbation label, e.g. ``"KO_CHGA_OV_none"``.
    split : str
        Data split used to load the reference data (``"full"`` or ``"train"``).
    stim : float
        Stimulus value used during inference.
    prior : float
        Prior value used during inference.
    label_key : str
        ``obs`` key used to store/read cell-type predictions.
    """
    s, q = stim, prior
    adata_full = ad.read_h5ad(os.path.join(p, "Data", f"data_{split}.h5ad"))
    adata_traj = ad.read_h5ad(p + f"cardamomOT/adata_rna_traj_stim{s}_prior{q}.h5ad")
    adata_sim = ad.read_h5ad(p + f"cardamomOT/adata_sim_stim{s}_prior{q}.h5ad")

    perturb_path = p + f"cardamomOT/adata_sim_{combo_name}_stim{s}_prior{q}.h5ad"
    adata_perturb = ad.read_h5ad(perturb_path)

    clf = train_classifier(adata_full, label_key=label_key)
    adata_traj = predict_cell_types(adata_traj, clf, label_key=label_key)
    adata_sim = predict_cell_types(adata_sim, clf, label_key=label_key)
    adata_perturb = predict_cell_types(adata_perturb, clf, label_key=label_key)

    adata_perturb.write(perturb_path)

    cmap_cat = plt.get_cmap("Dark2")
    cats = adata_full.obs[label_key].astype(str).unique().tolist()
    colors = [cmap_cat(i % 8) for i in range(len(cats))]

    plot_cell_type_proportions(
        adatas=[adata_traj, adata_sim, adata_perturb],
        labels=["data", "sim wt", combo_name],
        label_key=label_key,
        colors=colors,
    )
