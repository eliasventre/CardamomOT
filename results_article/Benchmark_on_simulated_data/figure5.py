"""
figure_5.py
-----------
Simule des jeux de données cross-sectionnels (chaque cellule simulée indépendamment depuis t=0)
avec HARISSA pour 4 réseaux (BN8, FN8, CN5, FN4), calibre CardamomOT et ReferenceFitting,
évalue les champs de vitesse reconstruits via cosine similarity, et génère la figure finale.

Usage:
    python figure_5.py

Sorties:
    figure_5.png   (et figure_5.pdf via PIL)

Dépendances:
    numpy, matplotlib, anndata, scvelo, umap-learn, sklearn, seaborn, torch,
    CardamomOT, harissa, rf (dans ./_scripts/rf.py)
"""

import numpy as np
import scanpy as sc
import anndata as ad
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.cbook as cbook
if not hasattr(cbook, 'mplDeprecation'):
    cbook.mplDeprecation = DeprecationWarning
import scvelo as scv
import sys

# --- Adjust paths to your environment ---
sys.path += ['./../../']          # CardamomOT parent
sys.path.append('./_scripts/')    # rf module

from CardamomOT import NetworkModel as CardamomNetworkModel
from CardamomOT.inference.simulations import kon_ref, simulate_next_prot_ode
from harissa import NetworkModel as HarissaNetworkModel
import rf

# ---------------------------------------------------------------------------
# Path towards pre-computed Schiebinger CardamomOT results — adjust if needed
# ---------------------------------------------------------------------------
SCHIE_DIR = './../../experimental_datasets/Schiebinger'

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
cmap_tab = plt.get_cmap('tab20')
colors_methods = {
    'CardamomOT':          (cmap_tab(6),  cmap_tab(7)),
    'REFERENCE_FITTING':   (cmap_tab(18), cmap_tab(19)),
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def step_ode_modif(d1, ks, inter, basal, scale, P):
    """Euler step for the deterministic limit model."""
    a = kon_ref(P, ks, inter, basal)
    delta_P = d1 * (scale * a - P)
    return P + delta_P, delta_P

def step_ode_rna(d0, ks, inter, basal, scale, P, M):
    """Euler step for the deterministic limit model."""
    a = kon_ref(P, ks, inter, basal)
    delta_M = d0 * (scale * a - M)
    return M + delta_M, delta_M


def kon_harissa(basal, inter, k0, k1, P):
    """Compute the kon rates according to the Harissa model."""
    linear = basal + P @ inter
    phi = 1 / (1 + np.exp(-linear))
    kon = k0 + (k1 - k0) * phi
    kon[0] = 0
    return kon


def make_adata_plot(embedding, velocity, time_vals):
    """Build a minimal AnnData for scv.pl.velocity_embedding_stream."""
    ap = ad.AnnData(X=embedding.astype(np.float32))
    ap.obs['time'] = time_vals.astype(float)
    ap.obsm['X_umap'] = embedding
    ap.obsm['velocity_umap'] = velocity
    return ap


def plot_stream(ax, embedding, velocity, time_vals, title, size=15, add_colorbar=False,
                density=1, smooth=0.5):
    scv.pl.velocity_embedding_stream(
        make_adata_plot(embedding, velocity, time_vals),
        basis='umap', ax=ax, color='time', cmap='viridis',
        alpha=1, size=size, show=False, colorbar=False,
        density=density, smooth=smooth)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_colorbar:
        sm = ScalarMappable(cmap='viridis',
                            norm=Normalize(vmin=time_vals.min(), vmax=time_vals.max()))
        sm.set_array([])
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.1)
        plt.colorbar(sm, cax=cax, label='Time')


def weighted_cosine_similarity(delta_i, delta_j):
    """Weighted cosine similarity (same weighting as in the notebook)."""
    norms_i = np.linalg.norm(delta_i, axis=1)
    norms_j = np.linalg.norm(delta_j, axis=1)
    weights = norms_i * norms_j
    denom = weights.sum()
    if denom == 0:
        return 0.0
    weights /= denom
    return np.sum(cosine_similarity(delta_i, delta_j).diagonal() * weights)


# ---------------------------------------------------------------------------
# Core pipeline: simulate + calibrate + compute velocity deltas
# ---------------------------------------------------------------------------

def run_pipeline(dataset_name, model_harissa, time, data, rna_traj, prot_traj_scaled,
                 degradation_rates, run=0):
    """
    Run CardamomOT and ReferenceFitting on the provided (pre-simulated) data,
    compute velocity deltas for mRNAs and proteins, embed in UMAP, and return
    similarity matrices + UMAP embeddings + raw cell data.

    Parameters
    ----------
    dataset_name : str
    model_harissa : HarissaNetworkModel  (already configured & used for simulation)
    time : 1-D int array  (C,)
    data : array (C+1, G+2)  raw HARISSA data matrix
    rna_traj : array (C, G+1)  [stimulus col included]
    prot_traj_scaled : array (C, G+1)
    degradation_rates_path : str or None

    Returns
    -------
    sim_matrix_rna, sim_matrix_prot : (3,3) cosine-similarity matrices
    UMAPs : dict of 2-D embeddings / velocity vectors
    cells : dict of raw data arrays
    """

    # ---- Harissa velocity field ----
    k0_h = model_harissa.a[0] * model_harissa.d[0]
    k1_h = model_harissa.a[1] * model_harissa.d[0]
    delta_rna_harissa  = np.zeros_like(rna_traj,        dtype=float)
    delta_prot_harissa = np.zeros_like(prot_traj_scaled, dtype=float)
    for n, cell in enumerate(prot_traj_scaled):
        kon = kon_harissa(model_harissa.basal, model_harissa.inter, k0_h, k1_h, cell)
        delta_rna_harissa[n]  = kon / model_harissa.a[-1] - model_harissa.d[0] * rna_traj[n]
        delta_prot_harissa[n] = model_harissa.d[1] * (kon / k1_h - cell)

    # ---- CardamomOT calibration ----
    x = data[1:, 1:].copy()
    x[:, 0] = time
    G_cols = x.shape[1]
    model_carda = CardamomNetworkModel(G_cols - 1)
    model_carda.d = degradation_rates
    model_carda.fit(x)
    model_carda.adapt_to_unitary()

    c          = model_carda.a[-1]
    ks_cells   = np.max(model_carda.a[:-1], axis=0)
    ks         = (model_carda.a[:-1] / ks_cells).T
    d0         = model_carda.d[0]
    d1         = model_carda.d[1]

    # delta_rna_carda  = d0 * (model_carda.kon_theta * ks_cells / c - rna_traj)
    delta_rna_carda = np.zeros_like(rna_traj, dtype=float)
    for ci in range(rna_traj.shape[0]):
        _, delta_M = step_ode_rna(d0, ks, model_carda.inter, model_carda.basal,
                                    ks_cells / c, prot_traj_scaled[ci].reshape(1, -1), rna_traj[ci].reshape(1, -1))
        delta_rna_carda[ci] = delta_M[0]
    delta_prot_carda = np.zeros_like(prot_traj_scaled, dtype=float)
    for ci in range(prot_traj_scaled.shape[0]):
        _, delta_P = step_ode_modif(d1, ks, model_carda.inter, model_carda.basal,
                                    1, prot_traj_scaled[ci].reshape(1, -1))
        delta_prot_carda[ci] = delta_P[0]

    # ---- ReferenceFitting calibration ----
    adata_rf = ad.AnnData(X=rna_traj, obs={'time': time})
    options_rf = dict(lr=0.05, reg_sinkhorn=0.1, reg_A=1, reg_A_elastic=0.5,
                      iter=1000, ot_coupling=True, optimizer=torch.optim.Adam,
                      n_pca_components=-1)
    estim = rf.Estimator(
        [adata_rf], kos=[None],
        lr=options_rf['lr'], reg_sinkhorn=options_rf['reg_sinkhorn'],
        reg_A=options_rf['reg_A'], reg_A_elastic=options_rf['reg_A_elastic'],
        iter=options_rf['iter'], ot_coupling=options_rf['ot_coupling'],
        optimizer=options_rf['optimizer'], norm=False, t_key='time')
    estim.fit(print_iter=10, alg='alternating', update_couplings_iter=250)

    A_rf  = estim.A.cpu().numpy()
    b_rf  = estim.b.cpu().numpy()

    # RNA delta: linear prediction
    delta_rna_reffit  = rna_traj @ A_rf + b_rf
    # Protein delta: OT-coupling-based prediction
    delta_prot_reffit = prot_traj_scaled @ A_rf + b_rf 
    unique_times = np.unique(time)
    for number, cell in enumerate(prot_traj_scaled):
        ti = number // 100
        if ti < len(unique_times) - 1:
            row_weights = estim.Ts[0][ti][number % 100].cpu().numpy()
            next_cells  = prot_traj_scaled[100 * (ti + 1): 100 * (ti + 2)]
            P2 = (np.dot(row_weights, next_cells) * 100)
            delta_prot_reffit[number] = P2 - cell

    # ---- Strip stimulus column ----
    delta_rna_carda_s    = delta_rna_carda[:, 1:]
    delta_prot_carda_s   = delta_prot_carda[:, 1:]
    delta_rna_harissa_s  = delta_rna_harissa[:, 1:]
    delta_prot_harissa_s = delta_prot_harissa[:, 1:]
    delta_rna_reffit_s   = delta_rna_reffit[:, 1:]
    delta_prot_reffit_s  = delta_prot_reffit[:, 1:]
    rna_s   = rna_traj[:, 1:]
    prot_s  = prot_traj_scaled[:, 1:]

    # ---- Subsample 100 cells per timepoint ----
    n_sub = 100
    np.random.seed(42)
    sub_idx = []
    for t in np.unique(time):
        idx = np.where(time == t)[0]
        sampled = np.random.choice(idx, size=min(n_sub, len(idx)), replace=False)
        sub_idx.extend(sampled)
    sub_idx = np.array(sub_idx)

    time_sub           = time[sub_idx]
    rna_sub            = rna_s[sub_idx]
    prot_sub           = prot_s[sub_idx]
    dr_carda_sub       = delta_rna_carda_s[sub_idx]
    dp_carda_sub       = delta_prot_carda_s[sub_idx]
    dr_harissa_sub     = delta_rna_harissa_s[sub_idx]
    dp_harissa_sub     = delta_prot_harissa_s[sub_idx]
    dr_reffit_sub      = delta_rna_reffit_s[sub_idx]
    dp_reffit_sub      = delta_prot_reffit_s[sub_idx]

    if run == 0:
        # ---- UMAP fits ----
        umap_rna  = UMAP(n_components=2, random_state=42, min_dist=0.7)
        umap_prot = UMAP(n_components=2, random_state=42, min_dist=0.7)

        umap_rna.fit(np.vstack([rna_sub,
                                rna_sub + dr_reffit_sub,
                                rna_sub + dr_carda_sub,
                                rna_sub + dr_harissa_sub]))
        umap_prot.fit(np.vstack([prot_sub,
                                prot_sub + dp_reffit_sub,
                                prot_sub + dp_carda_sub,
                                prot_sub + dp_harissa_sub]))

        rna_2d  = umap_rna.transform(rna_sub)
        prot_2d = umap_prot.transform(prot_sub)

        dr_reffit_2d  = (umap_rna.transform(rna_sub + dr_reffit_sub)  - rna_2d) * 0.5
        dr_carda_2d   = (umap_rna.transform(rna_sub + dr_carda_sub)   - rna_2d)
        dr_harissa_2d = (umap_rna.transform(rna_sub + dr_harissa_sub) - rna_2d)
        dp_reffit_2d  = (umap_prot.transform(prot_sub + dp_reffit_sub) - prot_2d) * 0.2
        dp_carda_2d   = (umap_prot.transform(prot_sub + dp_carda_sub)  - prot_2d)
        dp_harissa_2d = (umap_prot.transform(prot_sub + dp_harissa_sub)- prot_2d)

        UMAPs = {
        'rna': rna_2d, 'prot': prot_2d,
        'rna_reffit':  dr_reffit_2d,  'prot_reffit':  dp_reffit_2d,
        'rna_carda':   dr_carda_2d,   'prot_carda':   dp_carda_2d,
        'rna_harissa': dr_harissa_2d, 'prot_harissa': dp_harissa_2d,
        }
    else:
        UMAPs=None

    # ---- Cosine similarity matrices (exclure t=0) ----
    mask_nonzero = time > 0
    methods_rna  = [delta_rna_reffit_s[mask_nonzero],   delta_rna_carda_s[mask_nonzero],   delta_rna_harissa_s[mask_nonzero]]
    methods_prot = [delta_prot_reffit_s[mask_nonzero],  delta_prot_carda_s[mask_nonzero],  delta_prot_harissa_s[mask_nonzero]]

    sim_rna  = np.array([[weighted_cosine_similarity(methods_rna[i],  methods_rna[j])
                          for j in range(3)] for i in range(3)])
    sim_prot = np.array([[weighted_cosine_similarity(methods_prot[i], methods_prot[j])
                          for j in range(3)] for i in range(3)])
    cells = {
        'rna': rna_sub, 'prot': prot_sub,
        'rna_reffit':  dr_reffit_sub,  'prot_reffit':  dp_reffit_sub,
        'rna_carda':   dr_carda_sub,   'prot_carda':   dp_carda_sub,
        'rna_harissa': dr_harissa_sub, 'prot_harissa': dp_harissa_sub,
        'time': time_sub,
    }
    return sim_rna, sim_prot, UMAPs, cells


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

def make_harissa_BN8(seed=0):
    np.random.seed(seed)
    C, G = 1000, 8
    t = [0, 6, 12, 24, 36, 48, 60, 72, 84, 96]
    k = np.linspace(0, C, len(t) + 1, dtype='int')
    time = np.zeros(C, dtype='int')
    for i in range(len(t)):
        time[k[i]:k[i+1]] = t[i]

    data = np.zeros((C + 1, G + 2), dtype='int')
    data[0][1:] = np.arange(G + 1)
    data[1:, 0] = time
    data[1:, 1] = 100 * (time > 0)

    mh = HarissaNetworkModel(G)
    mh.d[0] = 0.25;  mh.d[1] = 0.05
    mh.basal[1:] = -4
    for (i, j, v) in [(0,1,10),(1,2,10),(1,3,10),(3,2,-10),(2,3,-10),(2,2,5),(3,3,5),
                       (2,4,10),(3,5,10),(2,5,-10),(3,4,-10),(4,7,-10),(5,6,-10),
                       (4,6,10),(5,7,10),(7,8,10),(6,8,-10)]:
        mh.inter[i, j] = v

    prot = np.ones((C, G + 1), dtype='float32')
    for ki in range(C):
        sim = mh.simulate(time[ki], burnin=5)
        prot[ki, 1:] = sim.p[-1]
        data[ki + 1, 2:] = np.random.poisson(sim.m[-1])
    rna = data[1:, 1:].copy()
    prot_scaled = prot / np.max(prot, axis=0)
    return mh, time, data, rna, prot_scaled, mh.d.copy()


def make_harissa_FN8(seed=0):
    np.random.seed(seed)
    C, G = 1000, 8
    t = [0, 6, 12, 24, 36, 48, 60, 72, 84, 96]
    k = np.linspace(0, C, len(t) + 1, dtype='int')
    time = np.zeros(C, dtype='int')
    for i in range(len(t)):
        time[k[i]:k[i+1]] = t[i]

    data = np.zeros((C + 1, G + 2), dtype='int')
    data[0][1:] = np.arange(G + 1)
    data[1:, 0] = time
    data[1:, 1] = 100 * (time > 0)

    mh = HarissaNetworkModel(G)
    mh.d[0] = 0.4;  mh.d[1] = 0.08
    mh.basal[1:] = -5
    for (i, j, v) in [(0,1,10),(1,2,10),(2,3,10),(3,4,10),(3,5,10),(3,6,10),
                       (4,1,-10),(5,1,-10),(6,1,-10),(4,4,10),(5,5,10),(6,6,10),
                       (4,8,-10),(4,7,-10),(6,7,10),(7,6,10),(8,8,10)]:
        mh.inter[i, j] = v

    prot = np.ones((C, G + 1), dtype='float32')
    for ki in range(C):
        sim = mh.simulate(time[ki], burnin=5)
        prot[ki, 1:] = sim.p[-1]
        data[ki + 1, 2:] = np.random.poisson(sim.m[-1])
    rna = data[1:, 1:].copy()
    prot_scaled = prot / np.max(prot, axis=0)
    return mh, time, data, rna, prot_scaled, mh.d.copy()


def make_harissa_CN5(seed=0):
    np.random.seed(seed)
    C, G = 1000, 5
    t = [0, 6, 12, 24, 36, 48, 60, 72, 84, 96]
    k = np.linspace(0, C, len(t) + 1, dtype='int')
    time = np.zeros(C, dtype='int')
    for i in range(len(t)):
        time[k[i]:k[i+1]] = t[i]

    data = np.zeros((C + 1, G + 2), dtype='int')
    data[0][1:] = np.arange(G + 1)
    data[1:, 0] = time
    data[1:, 1] = 100 * (time > 0)

    mh = HarissaNetworkModel(G)
    mh.d[0] = 0.5;  mh.d[1] = 0.1
    mh.basal[1:] = [-5, 4, 4, -5, -5]
    for (i, j, v) in [(0,1,10),(1,2,-10),(2,3,-10),(3,4,10),(4,5,10),(5,1,-10)]:
        mh.inter[i, j] = v

    prot = np.ones((C, G + 1), dtype='float32')
    for ki in range(C):
        sim = mh.simulate(time[ki], burnin=5)
        prot[ki, 1:] = sim.p[-1]
        data[ki + 1, 2:] = np.random.poisson(sim.m[-1])
    rna = data[1:, 1:].copy()
    prot_scaled = prot / np.max(prot, axis=0)
    return mh, time, data, rna, prot_scaled, mh.d.copy()


def make_harissa_FN4(seed=0):
    np.random.seed(seed)
    C, G = 1000, 4
    t = [0, 6, 12, 24, 36, 48, 60, 72, 84, 96]
    k = np.linspace(0, C, len(t) + 1, dtype='int')
    time = np.zeros(C, dtype='int')
    for i in range(len(t)):
        time[k[i]:k[i+1]] = t[i]

    data = np.zeros((C + 1, G + 2), dtype='int')
    data[0][1:] = np.arange(G + 1)
    data[1:, 0] = time
    data[1:, 1] = 100 * (time > 0)

    mh = HarissaNetworkModel(G)
    mh.d[0] = 1;  mh.d[1] = 0.2
    mh.d /= 5
    mh.basal[1:] = -5
    for (i, j, v) in [(0,1,10),(1,2,10),(1,3,10),(3,4,10),(4,1,-10),(2,2,10),(3,3,10)]:
        mh.inter[i, j] = v

    prot = np.ones((C, G + 1), dtype='float32')
    for ki in range(C):
        sim = mh.simulate(time[ki], burnin=5)
        prot[ki, 1:] = sim.p[-1]
        data[ki + 1, 2:] = np.random.poisson(sim.m[-1])
    rna = data[1:, 1:].copy()
    prot_scaled = prot / np.max(prot, axis=0)
    return mh, time, data, rna, prot_scaled, mh.d.copy()


# ---------------------------------------------------------------------------
# Schiebinger: load pre-computed CardamomOT results and build UMAP embeddings
# Reproduit exactement les cellules 20 et 22 du notebook original.
# ---------------------------------------------------------------------------

def load_schiebinger():
    """
    Charge les résultats CardamomOT pré-calculés sur les données Schiebinger
    et retourne les embeddings UMAP 2D + vecteurs vitesse, prêts pour plot_stream.

    Returns
    -------
    rna_traj_2d, delta_rna_carda_2d, prot_traj_2d, delta_prot_carda_2d,
    time_schiebinger  (tous arrays numpy)
    """
    base = f'{SCHIE_DIR}/cardamomOT'

    tmp = np.load(f'{base}/data_rna.npy')
    G               = tmp.shape[1]
    model           = CardamomNetworkModel(G)
    model.d         = np.load(f'{base}/degradations.npy')
    model.basal     = np.load(f'{base}/basal_simul.npy')
    model.inter     = np.load(f'{base}/inter_simul.npy')
    model.prot      = np.load(f'{base}/data_prot_unitary.npy')
    model.rna        = np.load(f'{base}/data_rna.npy')  
    model.kon_theta = np.load(f'{base}/data_kon_theta.npy')
    model.a         = np.load(f'{base}/mixture_parameters.npy')
    time_s          = np.load(f'{base}/data_times.npy')

    # Subsample 200 cells per timepoint 
    n_per_time = 200
    np.random.seed(0)
    sub_idx = []
    for t in np.unique(time_s):
        idx = np.where(time_s == t)[0]
        sub_idx.extend(np.random.choice(idx, size=min(n_per_time, len(idx)), replace=False))
    sub_idx = np.array(sub_idx)

    time_s          = time_s[sub_idx]
    rna_traj        = model.rna[sub_idx, 1:]
    prot_traj      = model.prot[sub_idx, 1:]
    kon_traj = model.kon_theta[sub_idx, 1:]

    c        = model.a[-1]
    ks_cells = np.max(model.a[:-1], axis=0)
    ks         = (model.a[:-1] / ks_cells).T
    d0       = model.d[0]
    d1       = model.d[1]

    # RNA delta (cellule 22 du notebook)
    # delta_rna_carda  = 10 * d0[1:] * (kon_traj * ks_cells[1:] / c[1:] - rna_traj)
    delta_rna_carda = np.zeros_like(prot_traj, dtype=float)
    for ci in range(rna_traj.shape[0]):
        _, delta_M = step_ode_rna(d0 * 10, ks, model.inter, model.basal,
                                    ks_cells / c, model.prot[sub_idx, :][ci, :].reshape(1, -1), model.rna[sub_idx, :][ci, :].reshape(1, -1))
        delta_rna_carda[ci] = delta_M[0, 1:]
    # Protein delta
    delta_prot_carda = np.zeros_like(prot_traj, dtype=float)
    for ci in range(prot_traj.shape[0]):
        _, delta_P = step_ode_modif(d1 * 50, ks, model.inter, model.basal,
                                    1, model.prot[sub_idx, :][ci, :].reshape(1, -1))
        delta_prot_carda[ci] = delta_P[0, 1:]

    # Top-8 masking on RNA (identique au notebook)
    k_top = 8
    n_rows, n_cols = delta_rna_carda.shape
    topk_idx = np.argpartition(np.abs(delta_rna_carda), -min(k_top, n_cols), axis=1)[:, -min(k_top, n_cols):]
    mask = np.zeros_like(delta_rna_carda, dtype=bool)
    mask[np.arange(n_rows)[:, None], topk_idx] = True
    delta_rna_carda = np.where(mask, delta_rna_carda, 0.0)

    # Normalisation log1p pour l'UMAP RNA (identique au notebook, cellule 22)
    rna_endpoints  = np.maximum(rna_traj     + delta_rna_carda,  0)
    prot_endpoints = np.maximum(prot_traj + delta_prot_carda, 0)

    rna_ad    = ad.AnnData(X=rna_traj.copy())
    rna_ep_ad = ad.AnnData(X=rna_endpoints.copy())
    sc.pp.normalize_total(rna_ad,    target_sum=1e4)
    sc.pp.normalize_total(rna_ep_ad, target_sum=1e4)
    sc.pp.log1p(rna_ad)
    sc.pp.log1p(rna_ep_ad)
    rna_norm    = rna_ad.X.copy()
    rna_ep_norm = rna_ep_ad.X.copy()

    umap_rna  = UMAP(n_components=2, random_state=42, min_dist=0.7)
    umap_prot = UMAP(n_components=2, random_state=42, min_dist=0.7)
    umap_rna.fit(rna_norm)
    umap_prot.fit(prot_traj)

    rna_traj_2d        = umap_rna.transform(rna_norm)
    prot_traj_2d       = umap_prot.transform(prot_traj)
    delta_rna_carda_2d = umap_rna.transform(rna_ep_norm)    - rna_traj_2d
    delta_prot_carda_2d= umap_prot.transform(prot_endpoints) - prot_traj_2d

    return rna_traj_2d, delta_rna_carda_2d, prot_traj_2d, delta_prot_carda_2d, time_s.astype(float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N_RUNS = 5   # nombre de jeux de données simulés par réseau

    DATASETS = [
        ('BN8', make_harissa_BN8),
        ('FN8', make_harissa_FN8),
        ('CN5', make_harissa_CN5),
        ('FN4', make_harissa_FN4),
    ]

    # scores[name] = {'rna': (N_RUNS, 2), 'prot': (N_RUNS, 2)}
    #   axis-1 : [ReferenceFitting, CardamomOT]  (m[2,0] et m[2,1])
    scores = {name: {'rna': [], 'prot': []} for name, _ in DATASETS}
    # UMAPs et cells du run 0 (seed=0) — utilisés pour les stream plots
    vis = {}

    print("\n=== Loading Schiebinger ===")
    rna_traj_2d, delta_rna_carda_2d, prot_traj_2d, delta_prot_carda_2d, time_schiebinger = \
        load_schiebinger()
    
    for name, builder in DATASETS:
        for run in range(N_RUNS):
            print(f"\n=== {name}  run {run}/{N_RUNS-1} ===")
            mh, time, data, rna, prot, degradation_rates = builder(seed=run)
            sim_rna, sim_prot, UMAPs, cells = run_pipeline(
                name, mh, time, data, rna, prot, degradation_rates, run=run)
            # sim_rna[2, 0] = RF vs Harissa,  sim_rna[2, 1] = Carda vs Harissa
            scores[name]['rna'].append([sim_rna[2, 0],  sim_rna[2, 1]])
            scores[name]['prot'].append([sim_prot[2, 0], sim_prot[2, 1]])
            if run == 0:
                vis[name] = {'UMAPs': UMAPs, 'cells': cells}
        scores[name]['rna']  = np.array(scores[name]['rna'])   # (N_RUNS, 2)
        scores[name]['prot'] = np.array(scores[name]['prot'])

    # ---- Figure layout ----
    fig = plt.figure(figsize=(8.27, 11.69))
    gs  = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[1, 2, 2, 2], hspace=0.3)
    wspaces = [0.5, 0.25, 0.25, 0.25]
    subplot_specs = [gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[i], wspace=w)
                     for i, w in enumerate(wspaces)]
    axes = [fig.add_subplot(spec[0, c]) for spec in subplot_specs for c in range(2)]

    # Panel labels
    x_left  = min(axes[i].get_position().x0 for i in range(0, 8, 2))
    x_right = min(axes[i].get_position().x0 for i in range(1, 8, 2))
    x_offset = 0.06
    for i, (ax, label) in enumerate(zip(axes, "ABCDEFGH")):
        x_fig = (x_left if i % 2 == 0 else x_right) - x_offset
        fig.text(x_fig, ax.get_position().y1 + 0.005, label,
                 ha="left", va="bottom", fontsize=10, fontweight="bold", clip_on=False)

    # --- Bar charts : moyenne ± erreur standard sur N_RUNS runs ---
    dataset_titles = ['BN8', 'FN8', 'CN5', 'FN4']
    bar_methods    = ['ReferenceFitting', 'CardamomOT']
    bar_colors     = [colors_methods['REFERENCE_FITTING'][0], colors_methods['CardamomOT'][0]]
    x = np.arange(len(dataset_titles))
    width = 0.35

    bar_configs = [
        (axes[0], 'rna',  'Comparison of RNA Similarities Across Datasets',     (-0.5, 1), False),
        (axes[1], 'prot', 'Comparison of Protein Similarities Across Datasets', (0,    1), True),
    ]
    for ax, score_key, title, ylim, show_legend in bar_configs:
        for method_idx, (method, color) in enumerate(zip(bar_methods, bar_colors)):
            means = [scores[n][score_key][:, method_idx].mean() for n in dataset_titles]
            sems  = [scores[n][score_key][:, method_idx].std() / np.sqrt(N_RUNS)
                     for n in dataset_titles]
            ax.bar(x + method_idx * width, means, width, label=method, color=color,
                   yerr=sems, capsize=3, error_kw=dict(elinewidth=0.8, ecolor='black'))
        ax.set_ylabel('Cosine Similarity', fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(dataset_titles)
        ax.set_ylim(*ylim)
        ax.yaxis.grid(True, color='lightgray', linewidth=0.8)
        ax.set_axisbelow(True)
        if show_legend:
            ax.legend(loc='lower right', fontsize=6)

    # --- CN5 protein UMAP (run 0) ---
    CN5_UMAPs = vis['CN5']['UMAPs']
    CN5_cells = vis['CN5']['cells']
    fn4_time              = CN5_cells['time'].astype(float)
    prot_CN5              = CN5_cells['prot']
    delta_prot_carda_CN5  = CN5_cells['prot_carda']
    delta_prot_reffit_CN5 = CN5_cells['prot_reffit']

    umap_prot_CN5 = UMAP(n_components=2, random_state=42, min_dist=0.7)
    umap_prot_CN5.fit(np.vstack([prot_CN5,
                                  prot_CN5 + delta_prot_reffit_CN5,
                                  prot_CN5 + delta_prot_carda_CN5]))
    prot_traj_2d_CN5         = umap_prot_CN5.transform(prot_CN5)
    delta_prot_carda_2d_CN5  = umap_prot_CN5.transform(prot_CN5 + delta_prot_carda_CN5)        - prot_traj_2d_CN5
    delta_prot_reffit_2d_CN5 = umap_prot_CN5.transform(prot_CN5 + delta_prot_reffit_CN5 * 0.1) - prot_traj_2d_CN5

    # --- Stream plot config ---
    DENSITY, SMOOTH = 1, 0.5
    stream_configs = [
        (axes[2], CN5_UMAPs['rna'],  CN5_UMAPs['rna_carda'],    fn4_time,
         'RNA trajectories - CardamomOT',           15, False),
        (axes[3], prot_traj_2d_CN5,  delta_prot_carda_2d_CN5,   fn4_time,
         'Protein trajectories - CardamomOT',       15, True),
        (axes[4], CN5_UMAPs['rna'],  CN5_UMAPs['rna_reffit'],   fn4_time,
         'RNA trajectories - Reference Fitting',    15, False),
        (axes[5], prot_traj_2d_CN5,  delta_prot_reffit_2d_CN5,  fn4_time,
         'Protein trajectories - Reference Fitting',15, True),
        (axes[6], rna_traj_2d,       delta_rna_carda_2d,        time_schiebinger,
         'RNA trajectories - CardamomOT - Schiebinger',              5, False),
        (axes[7], prot_traj_2d,      delta_prot_carda_2d,       time_schiebinger,
         'Inferred protein trajectories - CardamomOT - Schiebinger', 5, True),
    ]
    for ax, emb, vel, t, title, size, cbar in stream_configs:
        plot_stream(ax, emb, vel, t, title, size=size, add_colorbar=cbar,
                    density=DENSITY, smooth=SMOOTH)

    plt.subplots_adjust(hspace=0.2, wspace=0.25)
    plt.savefig('figure_5.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print("Saved figure_5.png")

    try:
        from PIL import Image
        Image.open('figure_5.png').convert('RGB').save('figure_5.pdf', 'PDF', resolution=300)
        print("Saved figure_5.pdf")
    except ImportError:
        print("(PIL not available – skipping PDF export)")


if __name__ == '__main__':
    main()