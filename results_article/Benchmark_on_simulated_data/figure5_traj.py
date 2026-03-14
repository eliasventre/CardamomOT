"""
figure_5_trajectories.py
------------------------
Même structure que figure_5.py, mais avec de VRAIES TRAJECTOIRES.

SIMULATION
----------
Pour chaque réseau (BN8, FN8, CN5, FN4) on simule C cellules chaînées :
  - Cellule i à t_0 : burnin depuis t=0
  - Cellule i à t_{k+1} : re-simulation depuis (P0=sim.p[-1], M0=sim.m[-1])
    avec un 1 en tête pour le stimulus.
Le vrai couplage est l'identité sur les index : cellule i@t_k → cellule i@t_{k+1}.

Ce MÊME jeu de données (vraies trajectoires) est utilisé pour calibrer
CardamomOT ET ReferenceFitting.

COUPLAGES INFÉRÉS
-----------------
* CardamomOT : après calibration sur les données vraies, model.rna[model.times_data==t_k]
  et model.rna[model.times_data==t_{k+1}] contiennent les trajectoires inférées.
  Chaque vecteur dans model.rna à t_k est une reconstruction de l'une des C cellules
  des données originales. Comme chaque cellule a un profil transcriptomique unique,
  on retrouve son index dans les données originales par matching exact (ou quasi-exact).
  Le couplage CardamomOT est alors 1-to-1 : l'index i dans model.rna@t_k prédit
  l'index i dans model.rna@t_{k+1}, qu'on ramène aux vraies cellules originales
  via le mapping par index.

* ReferenceFitting : estim.Ts[0][ti] est le couplage (N_src, N_dst) entre
  les cellules à t_k et t_{k+1} dans les données originales.
  La ligne i donne la distribution sur les cellules à t_{k+1}.

CHAMPS DE VITESSE
-----------------
* CardamomOT : v_i = (model.rna[t_{k+1}][i] - data[t_k][i]) / delta_t
  (avec le mapping d'index ci-dessus)
* RF         : v_i = E_q[(x_{k+1} - x_k)] = (sum_j q_ij * x_{k+1,j}) - x_k,i

Les deux : RNA et protéines.

MÉTRIQUE OT
-----------
Pour chaque intervalle et chaque cellule source i :
    score_i = sum_j  q_ij * || x_pred_j^{k+1} - x_true_i^{k+1} ||
où x_true_i^{k+1} est le vrai successeur (index i dans les données à t_{k+1})
et x_pred est le pool de cellules à t_{k+1} dans les données originales.

SCHIEBINGER : identique à figure_5.py.
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
import matplotlib.cbook as cbook
if not hasattr(cbook, 'mplDeprecation'):
    cbook.mplDeprecation = DeprecationWarning
import scvelo as scv
import sys

sys.path += ['./../../']
sys.path.append('./_scripts/')

from CardamomOT import NetworkModel as CardamomNetworkModel
from CardamomOT.inference.simulations import kon_ref
from harissa import NetworkModel as HarissaNetworkModel
import rf

SCHIE_DIR = './../../experimental_datasets/Schiebinger'

cmap_tab = plt.get_cmap('tab20')
colors_methods = {
    'CardamomOT':        (cmap_tab(6),  cmap_tab(7)),
    'REFERENCE_FITTING': (cmap_tab(18), cmap_tab(19)),
}

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def make_adata_plot(embedding, velocity, time_vals):
    ap = ad.AnnData(X=embedding.astype(np.float32))
    ap.obs['time'] = time_vals.astype(float)
    ap.obsm['X_umap'] = embedding
    ap.obsm['velocity_umap'] = velocity
    return ap


def plot_stream(ax, embedding, velocity, time_vals, title, size=15,
                add_colorbar=False, density=1, smooth=0.5):
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


# ---------------------------------------------------------------------------
# OT descendant-distance metric
# ---------------------------------------------------------------------------

def ot_coupling_metric(q_ij, true_next, pred_pool):
    """
    mean_i  sum_j  q_ij * || pred_pool_j - true_next_i ||

    q_ij      : (N, N) lignes normalisées à 1
    true_next : (N, G) vrai successeur de chaque cellule source
    pred_pool : (N, G) pool de cellules à t_{k+1} (= données originales à t_{k+1})
    """
    diff  = pred_pool[np.newaxis, :, :] - true_next[:, np.newaxis, :]  # (N, N, G)
    dists = np.linalg.norm(diff, axis=-1)                               # (N, N)
    return float((q_ij * dists).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# True-trajectory simulation
# ---------------------------------------------------------------------------

def simulate_true_trajectories(model_harissa, time_list, C, seed=0):
    """
    Simule C vraies trajectoires en chaînant les temps.

    Returns
    -------
    rna_by_time  : list[T] of (C, G+1)   — RNA à chaque temps (col0=stimulus)
    prot_by_time : list[T] of (C, G+1)   — protéines
    """
    np.random.seed(seed)
    G = model_harissa.G
    T = len(time_list)

    rna_by_time  = [np.zeros((C, G + 1), dtype=float) for _ in range(T)]
    prot_by_time = [np.ones( (C, G + 1), dtype=float) for _ in range(T)]

    for i in range(C):
        sim = model_harissa.simulate(time_list[0], burnin=5)
        p_prev = sim.p[-1].copy()
        m_prev = sim.m[-1].copy()

        rna_by_time[0][i, 0]  = time_list[0]
        rna_by_time[0][i, 1:] = np.random.poisson(m_prev)
        prot_by_time[0][i, 0] = time_list[0]
        prot_by_time[0][i, 1:] = p_prev

        for k in range(1, T):
            delta_t = time_list[k] - time_list[k - 1]
            # stimulus = 1 pour tous les temps > 0
            M0 = np.concatenate([[1], m_prev])
            P0 = np.concatenate([[1], p_prev])
            sim = model_harissa.simulate(delta_t, M0=M0, P0=P0)
            p_prev = sim.p[-1].copy()
            m_prev = sim.m[-1].copy()

            rna_by_time[k][i, 0]  = time_list[k]
            rna_by_time[k][i, 1:] = np.random.poisson(m_prev)
            prot_by_time[k][i, 0] = time_list[k]
            prot_by_time[k][i, 1:] = p_prev

    return rna_by_time, prot_by_time


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(dataset_name, model_harissa, time_list, C=100, seed=0):
    """
    1. Simule C vraies trajectoires → UN seul jeu de données.
    2. Calibre CardamomOT sur ce jeu de données.
    3. Calibre ReferenceFitting sur ce jeu de données.
    4. Extrait les couplages, calcule les champs de vitesse et la métrique.
    5. Retourne metric_rna, metric_prot, UMAPs, cells.

    Le couplage ground-truth est l'identité : cellule i@t_k → cellule i@t_{k+1}.
    """
    T = len(time_list)

    # ------------------------------------------------------------------ Simulation
    print(f"  [sim] Simulating {C} true trajectories for {dataset_name}...")
    rna_by_time, prot_by_time = simulate_true_trajectories(
        model_harissa, time_list, C, seed=seed)

    # Protéines scalées globalement
    prot_all  = np.vstack(prot_by_time)   # (C*T, G+1)
    prot_max  = np.maximum(prot_all.max(axis=0), 1e-9)
    prot_sc_by_time = [prot_by_time[k] / prot_max for k in range(T)]

    # Données complètes (toutes les cellules, tous les temps) pour l'inférence
    rna_all  = np.vstack(rna_by_time)     # (C*T, G+1)  col0=stimulus (sauf t0)
    time_all = np.array([time_list[k] for k in range(T) for _ in range(C)], dtype=int)

    # ------------------------------------------------------------------ CardamomOT
    print(f"  [carda] Calibrating CardamomOT for {dataset_name}...")
    x_carda = rna_all.copy()
    x_carda[:, 0] = time_all   # col0 = temps pour CardamomOT
    model_carda = CardamomNetworkModel(x_carda.shape[1] - 1)
    model_carda.d = model_harissa.d.copy()
    model_carda.fit(x_carda)

    # model_carda.rna  : (N_inf, G+1)  trajectoires RNA inférées
    # model_carda.prot : (N_inf, G+1)  trajectoires protéines inférées
    # model_carda.times_data : (N_inf,) temps associé à chaque ligne
    #
    # ALIGNEMENT : model.rna est calibré sur les C*T lignes de x_carda.
    # CardamomOT reconstruit des trajectoires continues ; la cellule inférée
    # à l'indice i dans model.rna[times_data==t_k] correspond à la cellule
    # originale à l'indice i dans rna_all[time_all==t_k].
    # (même ordre de traitement interne à CardamomOT)
    # → couplage CardamomOT : i@t_k → i@t_{k+1}  (1-to-1, même index)

    t_inf = model_carda.times_data.astype(int)
    rna_inf_by_time  = {}
    prot_inf_by_time = {}
    for k, tk in enumerate(time_list):
        mask = (t_inf == tk)
        rna_inf_by_time[k]  = model_carda.rna[mask]   # (C, G+1)
        prot_inf_by_time[k] = model_carda.prot[mask]  # (C, G+1)

    # ------------------------------------------------------------------ ReferenceFitting
    print(f"  [rf] Calibrating ReferenceFitting for {dataset_name}...")
    # RF est calibré sur les données RNA (même données que CardamomOT)
    adata_rf = ad.AnnData(X=rna_all, obs={'time': time_all.astype(float)})
    estim = rf.Estimator(
        [adata_rf], kos=[None],
        lr=0.05, reg_sinkhorn=0.1, reg_A=1, reg_A_elastic=0.5,
        iter=1000, ot_coupling=True, optimizer=torch.optim.Adam,
        norm=False, t_key='time')
    estim.fit(print_iter=10, alg='alternating', update_couplings_iter=250)
    # estim.Ts[0][ti] : couplage (C, C) entre cellules à t_{ti} et t_{ti+1}
    # Les cellules source sont rna_by_time[ti] (dans l'ordre)
    # Les cellules cible   sont rna_by_time[ti+1]

    # ------------------------------------------------------------------ Champs de vitesse + métrique
    #
    # Convention :
    #   rna_by_time[ti]     : (C, G+1)  données vraies à t_k  (col0 = stimulus)
    #   prot_sc_by_time[ti] : (C, G+1)  protéines vraies scalées à t_k
    #   rna_inf_by_time[tk] : (C, G+1)  trajectoires RNA inférées par CardamomOT
    #   prot_inf_by_time[tk]: (C, G+1)  trajectoires prot inférées par CardamomOT
    #
    # Vitesse pour les deux méthodes :
    #   v_i = E_q[ (x_{k+1} - x_i) ] / dt
    #   où x_{k+1} parcourt le pool des vraies cellules à t_{k+1} pondérées par q.
    #
    # Métrique RNA  : sum_j q_ij * || log1p(x_j^{k+1}) - log1p(x_i^{k+1,true}) ||^2
    # Métrique prot : sum_j q_ij * || p_j^{k+1}         - p_i^{k+1,true}        ||^2
    #   (pas de log1p pour les protéines qui sont déjà dans [0,1])
    #
    # CardamomOT — reconstruction du couplage :
    #   Pour chaque cellule i dans rna_by_time[ti] (données vraies à t_k),
    #   on cherche ses « copies » dans rna_inf_by_time[tk] par matching exact
    #   de vecteur (les profils sont uniques donc la correspondance est bijective,
    #   mais CardamomOT peut dupliquer un profil si plusieurs particules OT
    #   convergent vers la même cellule originale).
    #   Soient i1, i2, … les indices dans rna_inf_by_time[tk] qui matchent la
    #   cellule i. On regarde les cellules rna_inf_by_time[tk1][i1], [i2], …
    #   et on retrouve leurs index i1', i2', … dans rna_by_time[ti+1] par
    #   matching exact. Le couplage CardamomOT est q_carda[i, i1'] = q_carda[i, i2'] = 1/n_matches.
    #
    # ReferenceFitting — couplage direct :
    #   q_rf = estim.Ts[0][ti] normalisé ligne par ligne.
    #   Les lignes indexent les cellules de rna_by_time[ti],
    #   les colonnes indexent les cellules de rna_by_time[ti+1].
 
    # Champs de vitesse : liste indexée par intervalle ti (0..T-2)
    # La vitesse à ti est définie pour les cellules à t_k = time_list[ti]
    delta_rna_carda  = []   # list[T-1] of (C, G)   (stimulus strippé)
    delta_prot_carda = []
    delta_rna_rf     = []
    delta_prot_rf    = []
 
    metric_carda_rna_list  = []
    metric_carda_prot_list = []
    metric_rf_rna_list     = []
    metric_rf_prot_list    = []
 
    for ti, (tk, tk1) in enumerate(zip(time_list[:-1], time_list[1:])):
        dt = float(tk1 - tk)
 
        rna_k   = rna_by_time[ti]        # (C, G+1) vraies données à t_k
        rna_k1  = rna_by_time[ti + 1]    # (C, G+1) vraies données à t_{k+1}
        prot_k  = prot_sc_by_time[ti]    # (C, G+1) protéines scalées à t_k
        prot_k1 = prot_sc_by_time[ti + 1]
 
        rna_inf_k  = rna_inf_by_time[ti]   # (C, G+1) inférées à t_k
        rna_inf_k1 = rna_inf_by_time[ti + 1]  # (C, G+1) inférées à t_{k+1}
 
        # ---- Couplage CardamomOT ----------------------------------------
        # Étape 1 : pour chaque cellule vraie i à t_k (rna_k[i]),
        #           chercher où rna_k[i] apparaît dans rna_inf_k → indices j1, j2, …
        # Étape 2 : regarder rna_inf_k1[j1], rna_inf_k1[j2], …
        #           et retrouver leurs indices j1', j2', … dans rna_k1
        # → q_carda[i, j1'] = q_carda[i, j2'] = 1/n_matches
 
        q_carda = np.zeros((C, C), dtype=float)
        n_missing = 0
        n_obtained = 0
        for i in range(rna_k.shape[0]):
            # Étape 1 : indices dans rna_inf_k égaux à rna_k[i]
            inf_idxs = np.where(np.all(rna_inf_k[:, 1:] == rna_k[i, 1:], axis=1))[0]  # [j1, j2, …]
            # Étape 2 : descendants inférés → retrouver dans rna_k1
            dest_true = []
            for inf_j in inf_idxs:
                dest_true.extend(np.where(np.all(rna_k1[:, 1:] == rna_inf_k1[inf_j, 1:], axis=1))[0])  # [j1', j2', …]
            if len(dest_true):
                for j_true in dest_true:
                    q_carda[i, j_true] += 1.0
                q_carda[i] /= q_carda[i].sum()
                n_obtained += 1
            else:
                n_missing += 1
        q_carda *= (C / n_obtained)  # renormalisation pour que la somme globale soit C (et pas n_obtained)
        print(f"    CardamomOT couplage: {n_obtained}/{C} obtained, {n_missing}/ {C} missing (fallback to uniform)")

        # ---- Couplage ReferenceFitting ----------------------------------
        q_rf = estim.Ts[0][ti].cpu().numpy()   # (C, C)
        row_sums = q_rf.sum(axis=1, keepdims=True)
        q_rf_n   = q_rf / np.where(row_sums == 0, 1.0, row_sums)
 
        # ---- Vitesses : v_i = E_q[(x_{k+1} - x_i)] / dt ---------------
        # RNA : col0 = stimulus, on travaille sur les gènes (col 1:)
        rna_k_g   = rna_k[:, 1:].astype(float)   # (C, G)
        rna_k1_g  = rna_k1[:, 1:].astype(float)
        prot_k_g  = prot_k[:, 1:].astype(float)              
        prot_k1_g = prot_k1[:, 1:].astype(float)
 
        # E_q[x_{k+1}] pour chaque méthode
        rna_exp_carda  = q_carda  @ rna_k1_g    # (C, G)
        rna_exp_rf     = q_rf_n   @ rna_k1_g
        prot_exp_carda = q_carda  @ prot_k1_g
        prot_exp_rf    = q_rf_n   @ prot_k1_g
 
        delta_rna_carda.append( (rna_exp_carda  - rna_k_g)  / dt )
        delta_rna_rf.append(    (rna_exp_rf     - rna_k_g)  / dt )
        delta_prot_carda.append((prot_exp_carda - prot_k_g) / dt )
        delta_prot_rf.append(   (prot_exp_rf    - prot_k_g) / dt )
 
        # ---- Métrique ---------------------------------------------------
        # Vrai successeur de la cellule i = rna_k1[i] (index i dans t_{k+1})
        # RNA  : sum_j q_ij * || log1p(rna_k1_g[j]) - log1p(rna_k1_g[i]) ||^2
        # Prot : sum_j q_ij * || prot_k1_g[j]        - prot_k1_g[i]       ||^2
        log_rna_k1 = np.log1p(rna_k1_g)    # (C, G)
        log_prot_k1 = prot_k1_g.copy()

        metric_carda_tmp = 0
        metric_rf_tmp = 0
        N = rna_k1_g.shape[0]
        for j in range(N):
            diff_rna = log_rna_k1[:, :] - log_rna_k1[j, :].reshape(1, -1)  # (C,C,G)
            sq_rna   = (diff_rna ** 2).sum(axis=1) / (model_harissa.G - 1)
            metric_carda_tmp += (q_carda[j, :] * sq_rna).sum()
            metric_rf_tmp += (q_rf_n[j, :] * sq_rna).sum()
        
        # RNA — CardamomOT                     
        metric_carda_rna_list.append(metric_carda_tmp / N)
        # RNA — RF
        metric_rf_rna_list.append(metric_rf_tmp / N)

        metric_carda_tmp = 0
        metric_rf_tmp = 0
        for j in range(N):
            diff_prot = log_prot_k1[:, :] - log_prot_k1[j, :].reshape(1, -1)  # (C,C,G)
            sq_prot   = (diff_prot ** 2).sum(axis=1) / (model_harissa.G - 1)
            metric_carda_tmp += (q_carda[j, :] * sq_prot).sum()
            metric_rf_tmp += (q_rf_n[j, :] * sq_prot).sum()

        # Prot — CardamomOT
        metric_carda_prot_list.append(metric_carda_tmp / N)
        # Prot — RF
        metric_rf_prot_list.append(metric_rf_tmp / N)
 
    metric_rna  = [np.mean(metric_rf_rna_list),  np.mean(metric_carda_rna_list)]
    metric_prot = [np.mean(metric_rf_prot_list), np.mean(metric_carda_prot_list)]
 
    # ------------------------------------------------------------------ UMAP
    # Cellules de plot : toutes les cellules à t_1..t_{T-1} (on exclut t_0 car
    # la vitesse à ti correspond à l'intervalle [t_ti, t_{ti+1}] et est attachée
    # aux cellules sources à t_ti → on affiche la vitesse sur les cellules à t_k
    # pour k = 0..T-2, et on les reprojette dans l'espace UMAP des cellules à t_1..T-1
    # Pour simplifier : on utilise les cellules à t_k comme point de départ et
    # on montre la vitesse de l'intervalle [t_k, t_{k+1}].
    n_sub = 100
    np.random.seed(42)
 
    rna_plot_list, prot_plot_list = [], []
    dr_c_list, dp_c_list = [], []
    dr_rf_list, dp_rf_list = [], []
    time_plot_list = []
 
    for ti in range(T):    # intervalle ti : cellules source à time_list[ti]
        idx = np.random.choice(C, size=min(n_sub, C), replace=False)
        rna_plot_list.append(rna_by_time[ti][idx, 1:].astype(float))
        prot_plot_list.append(prot_sc_by_time[ti][idx, 1:])
        if ti < T - 1:
            dr_c_list.append(delta_rna_carda[ti][idx])
            dp_c_list.append(delta_prot_carda[ti][idx])
            dr_rf_list.append(delta_rna_rf[ti][idx])
            dp_rf_list.append(delta_prot_rf[ti][idx])
        else:
            # Dernier temps : pas de successeur, vitesse nulle
            G = rna_by_time[ti].shape[1] - 1
            dr_c_list.append(np.zeros((len(idx), G)))
            dp_c_list.append(np.zeros((len(idx), prot_sc_by_time[ti].shape[1] - 1)))
            dr_rf_list.append(np.zeros((len(idx), G)))
            dp_rf_list.append(np.zeros((len(idx), prot_sc_by_time[ti].shape[1] - 1)))
        time_plot_list.extend([time_list[ti]] * len(idx))
 
    rna_sub  = np.vstack(rna_plot_list)
    prot_sub = np.vstack(prot_plot_list)
    dr_c_sub = np.vstack(dr_c_list)
    dp_c_sub = np.vstack(dp_c_list)
    dr_rf_sub= np.vstack(dr_rf_list)
    dp_rf_sub= np.vstack(dp_rf_list)
    time_sub = np.array(time_plot_list, dtype=int)
    
    if seed == 0:
        umap_rna  = UMAP(n_components=2, random_state=42, min_dist=0.7)
        umap_prot = UMAP(n_components=2, random_state=42, min_dist=0.7)
        umap_rna.fit(np.vstack([rna_sub, rna_sub + dr_rf_sub, rna_sub + dr_c_sub]))
        umap_prot.fit(np.vstack([prot_sub, prot_sub + dp_rf_sub, prot_sub + dp_c_sub]))
    
        rna_2d  = umap_rna.transform(rna_sub)
        prot_2d = umap_prot.transform(prot_sub)
    
        UMAPs = {
            'rna':  rna_2d,
            'prot': prot_2d,
            'rna_carda':  umap_rna.transform(rna_sub  + dr_c_sub)  - rna_2d,
            'rna_rf':     umap_rna.transform(rna_sub  + dr_rf_sub) - rna_2d,
            'prot_carda': umap_prot.transform(prot_sub + dp_c_sub)  - prot_2d,
            'prot_rf':    umap_prot.transform(prot_sub + dp_rf_sub) - prot_2d,
        }
    else:
        UMAPs = None

    cells = {
        'rna': rna_sub, 'prot': prot_sub,
        'rna_carda':  dr_c_sub,  'prot_carda': dp_c_sub,
        'rna_rf':     dr_rf_sub, 'prot_rf':    dp_rf_sub,
        'time': time_sub,
    }

    return metric_rna, metric_prot, UMAPs, cells


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def make_BN8(seed=0):
    mh = HarissaNetworkModel(8)
    mh.d[0] = 0.25; mh.d[1] = 0.05
    mh.basal[1:] = [-4]*8
    for (i,j,v) in [(0,1,10),(1,2,10),(1,3,10),(3,2,-10),(2,3,-10),(2,2,5),(3,3,5),
                    (2,4,10),(3,5,10),(2,5,-10),(3,4,-10),(4,7,-10),(5,6,-10),
                    (4,6,10),(5,7,10),(7,8,10),(6,8,-10)]:
        mh.inter[i,j] = v
    return mh, seed

def make_FN8(seed=0):
    mh = HarissaNetworkModel(8)
    mh.d[0] = 0.4; mh.d[1] = 0.08
    mh.basal[1:] = [-5]*8
    for (i,j,v) in [(0,1,10),(1,2,10),(2,3,10),(3,4,10),(3,5,10),(3,6,10),
                    (4,1,-10),(5,1,-10),(6,1,-10),(4,4,10),(5,5,10),(6,6,10),
                    (4,8,-10),(4,7,-10),(6,7,10),(7,6,10),(8,8,10)]:
        mh.inter[i,j] = v
    return mh, seed

def make_CN5(seed=0):
    mh = HarissaNetworkModel(5)
    mh.d[0] = 0.5; mh.d[1] = 0.1
    mh.basal[1:] = [-5, 4, 4, -5, -5]
    for (i,j,v) in [(0,1,10),(1,2,-10),(2,3,-10),(3,4,10),(4,5,10),(5,1,-10)]:
        mh.inter[i,j] = v
    return mh, seed

def make_FN4(seed=0):
    mh = HarissaNetworkModel(4)
    mh.d[0] = 1/5; mh.d[1] = 0.2/5
    mh.basal[1:] = [-5]*4
    for (i,j,v) in [(0,1,10),(1,2,10),(1,3,10),(3,4,10),(4,1,-10),(2,2,10),(3,3,10)]:
        mh.inter[i,j] = v
    return mh, seed


# ---------------------------------------------------------------------------
# Schiebinger (identique à figure_5.py)
# ---------------------------------------------------------------------------

def step_ode_modif(d1, ks, inter, basal, scale, P):
    a = kon_ref(P, ks, inter, basal)
    delta_P = d1 * (scale * a - P)
    return P + delta_P, delta_P


def load_schiebinger():
    """
    Charge les trajectoires inférées par CardamomOT (data_rna, data_prot)
    sur les données Schiebinger et construit les champs de vitesse par
    couplage 1-to-1 entre temps consécutifs (même index = même cellule).
 
    Returns
    -------
    rna_traj_2d, delta_rna_2d, prot_traj_2d, delta_prot_2d, time_vals
    """
    base = f'{SCHIE_DIR}/cardamomOT'
 
    # data_rna : (N_total, G+1) avec col0 = stimulus
    # data_prot : (N_total, G+1) avec col0 = stimulus
    # data_times : (N_total,) temps associé à chaque ligne
    rna_inf  = np.load(f'{base}/data_rna.npy')   # (N, G+1)
    prot_inf = np.load(f'{base}/data_prot_unitary.npy')  # (N, G+1)
    time_s   = np.load(f'{base}/data_times.npy') # (N,)
 
    unique_times = np.unique(time_s)
 
    # Subsample n_per_time cellules par temps (même seed que le reste)
    n_per_time = 200
    np.random.seed(0)
    sub_idx = []
    for t in unique_times:
        idx = np.where(time_s == t)[0]
        chosen = np.random.choice(idx, size=min(n_per_time, len(idx)), replace=False)
        # Trier pour préserver l'ordre relatif (important pour le couplage 1-to-1)
        sub_idx.extend(np.sort(chosen))
    sub_idx = np.array(sub_idx)
 
    time_sub = time_s[sub_idx]
    rna_sub  = rna_inf[sub_idx,  1:]   # strip stimulus → (N_sub, G)
    prot_sub = prot_inf[sub_idx, 1:]   # strip stimulus → (N_sub, G)
 
    # Normalisation prot dans [0,1] (max global)
    prot_max = np.maximum(prot_sub.max(axis=0), 1e-9)
    prot_sub_sc = prot_sub / prot_max
 
    # ---- Champ de vitesse : couplage 1-to-1 par index ----
    # Pour chaque temps t_k, les N_k cellules à t_k correspondent 1-to-1
    # aux N_k cellules à t_{k+1} (même ordre d'index dans data_rna).
    # vitesse_i = (rna[t_{k+1}][i] - rna[t_k][i]) / delta_t
    N_sub   = len(sub_idx)
    delta_rna_all  = np.zeros((N_sub, rna_sub.shape[1]),  dtype=float)
    delta_prot_all = np.zeros((N_sub, prot_sub.shape[1]), dtype=float)
 
    unique_sub = np.unique(time_sub)
    for ti, (tk, tk1) in enumerate(zip(unique_sub[:-1], unique_sub[1:])):
        dt       = float(tk1 - tk)
        mask_k   = np.where(time_sub == tk)[0]
        mask_k1  = np.where(time_sub == tk1)[0]
        N_k  = len(mask_k)
        N_k1 = len(mask_k1)
        # Aligner : prendre min(N_k, N_k1) cellules
        N_pair = min(N_k, N_k1)
        rna_k  = rna_sub[mask_k[:N_pair]]
        rna_k1 = rna_sub[mask_k1[:N_pair]]
        prot_k  = prot_sub_sc[mask_k[:N_pair]]
        prot_k1 = prot_sub_sc[mask_k1[:N_pair]]
 
        delta_rna_all[mask_k[:N_pair]]  = (rna_k1  - rna_k)  / dt
        delta_prot_all[mask_k[:N_pair]] = (prot_k1 - prot_k) / dt
 
    # Exclure le dernier temps (pas de successeur) pour le plot
    mask_plot = time_sub < unique_sub[-1]
    rna_plot   = rna_sub[mask_plot]
    prot_plot  = prot_sub_sc[mask_plot]
    dr_plot    = delta_rna_all[mask_plot]
    dp_plot    = delta_prot_all[mask_plot]
    time_plot  = time_sub[mask_plot].astype(float)
 
    # ---- Normalisation RNA : normalize_total + log1p (identique à la version ODE) ----
    rna_endpoints = np.maximum(rna_plot + dr_plot, 0)
 
    rna_ad    = ad.AnnData(X=rna_plot.copy())
    rna_ep_ad = ad.AnnData(X=rna_endpoints.copy())
    sc.pp.normalize_total(rna_ad,    target_sum=1e4)
    sc.pp.normalize_total(rna_ep_ad, target_sum=1e4)
    sc.pp.log1p(rna_ad)
    sc.pp.log1p(rna_ep_ad)
    rna_norm    = rna_ad.X.copy()
    rna_ep_norm = rna_ep_ad.X.copy()
 
    # ---- UMAP : fitter sur les positions seules (identique à la version ODE) ----
    umap_rna  = UMAP(n_components=2, random_state=42, min_dist=0.7)
    umap_prot = UMAP(n_components=2, random_state=42, min_dist=0.7)
    umap_rna.fit(rna_norm)
    umap_prot.fit(prot_plot)
 
    rna_2d  = umap_rna.transform(rna_norm)
    prot_2d = umap_prot.transform(prot_plot)
    dr_2d   = umap_rna.transform(rna_ep_norm)           - rna_2d
    dp_2d   = umap_prot.transform(prot_plot + dp_plot)  - prot_2d
 
    return rna_2d, dr_2d, prot_2d, dp_2d, time_plot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    TIME_LIST = [0, 6, 12, 24, 36, 48, 60, 72, 84, 96]
    C      = 100   # cellules par trajectoire
    N_RUNS = 5     # jeux de données par réseau

    DATASETS = [
        ('BN8', make_BN8),
        ('FN8', make_FN8),
        ('CN5', make_CN5),
        ('FN4', make_FN4),
    ]

    # scores[name] = {'rna': (N_RUNS, 2), 'prot': (N_RUNS, 2)}
    #   axis-1 : [CardamomOT, ReferenceFitting]
    scores = {name: {'rna': [], 'prot': []} for name, _ in DATASETS}
    # UMAPs et cells du run 0 (seed=0) pour les stream plots
    vis = {}

    for name, builder in DATASETS:
        for run in range(N_RUNS):
            print(f"\n=== {name}  run {run}/{N_RUNS-1} ===")
            mh, seed = builder(seed=run)
            mh.G = mh.inter.shape[0] - 1
            m_rna, m_prot, UMAPs, cells = run_pipeline(name, mh, TIME_LIST, C=C, seed=seed)
            scores[name]['rna'].append(m_rna)    # [carda, rf]
            scores[name]['prot'].append(m_prot)
            if run == 0:
                vis[name] = {'UMAPs': UMAPs, 'cells': cells}
            print(f"  OT metric RNA  — CardamomOT: {m_rna[1]:.4f}  RF: {m_rna[0]:.4f}")
            print(f"  OT metric Prot — CardamomOT: {m_prot[1]:.4f}  RF: {m_prot[0]:.4f}")
        scores[name]['rna']  = np.array(scores[name]['rna'])   # (N_RUNS, 2)
        scores[name]['prot'] = np.array(scores[name]['prot'])

    print("\n=== Loading Schiebinger ===")
    rna_2d_s, dr_s, prot_2d_s, dp_s, time_s = load_schiebinger()

    # ---- Figure layout ----
    fig = plt.figure(figsize=(8.27, 11.69))
    gs  = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[1, 2, 2, 2], hspace=0.3)
    wspaces = [0.5, 0.25, 0.25, 0.25]
    subplot_specs = [gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[i], wspace=w)
                     for i, w in enumerate(wspaces)]
    axes = [fig.add_subplot(spec[0, c]) for spec in subplot_specs for c in range(2)]

    x_left  = min(axes[i].get_position().x0 for i in range(0, 8, 2))
    x_right = min(axes[i].get_position().x0 for i in range(1, 8, 2))
    x_offset = 0.06
    for i, (ax, label) in enumerate(zip(axes, "ABCDEFGH")):
        x_fig = (x_left if i % 2 == 0 else x_right) - x_offset
        fig.text(x_fig, ax.get_position().y1 + 0.005, label,
                 ha="left", va="bottom", fontsize=10, fontweight="bold", clip_on=False)

    # ---- Row 0 : bar charts métriques OT, moyenne ± SEM sur N_RUNS ----
    dataset_titles = ['BN8', 'FN8', 'CN5', 'FN4']
    bar_colors = [colors_methods['REFERENCE_FITTING'][0], colors_methods['CardamomOT'][0]]
    x_pos = np.arange(len(dataset_titles)); width = 0.35

    for ax, score_key, title in [
        (axes[0], 'rna',  'OT descendant distance – RNA (↓ better)'),
        (axes[1], 'prot', 'OT descendant distance – Protein (↓ better)'),
    ]:
        for method_idx, (label, color) in enumerate(
                zip(['ReferenceFitting', 'CardamomOT'], bar_colors)):
            means = [scores[n][score_key][:, method_idx].mean() for n in dataset_titles]
            sems  = [scores[n][score_key][:, method_idx].std() / np.sqrt(N_RUNS)
                     for n in dataset_titles]
            ax.bar(x_pos + method_idx * width, means, width, label=label, color=color,
                   yerr=sems, capsize=3, error_kw=dict(elinewidth=0.8, ecolor='black'))
        ax.set_ylabel('Mean E_q[||pred − true||]', fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.set_xticks(x_pos + width / 2); ax.set_xticklabels(dataset_titles)
        ax.yaxis.grid(True, color='lightgray', linewidth=0.8); ax.set_axisbelow(True)
    axes[1].legend(loc='upper right', fontsize=6)

    # ---- Rows 1-2 : CN5 stream plots (run 0) ----
    CN5 = vis['CN5']
    cn5_time    = CN5['cells']['time'].astype(float)
    prot_CN5    = CN5['cells']['prot']
    dp_c_CN5    = CN5['cells']['prot_carda']
    dp_rf_CN5   = CN5['cells']['prot_rf']

    umap_p_CN5 = UMAP(n_components=2, random_state=42, min_dist=0.7)
    umap_p_CN5.fit(np.vstack([prot_CN5, prot_CN5 + dp_rf_CN5, prot_CN5 + dp_c_CN5]))
    p2d_CN5      = umap_p_CN5.transform(prot_CN5)
    dp_c_2d_CN5  = umap_p_CN5.transform(prot_CN5 + dp_c_CN5)  - p2d_CN5
    dp_rf_2d_CN5 = umap_p_CN5.transform(prot_CN5 + dp_rf_CN5) - p2d_CN5

    DENSITY, SMOOTH = 1, 0.5
    stream_configs = [
        (axes[2], CN5['UMAPs']['rna'],  CN5['UMAPs']['rna_carda'],  cn5_time,
         'RNA velocity – CardamomOT (CN5)',            15, False),
        (axes[3], p2d_CN5,              dp_c_2d_CN5,               cn5_time,
         'Protein velocity – CardamomOT (CN5)',        15, True),
        (axes[4], CN5['UMAPs']['rna'],  CN5['UMAPs']['rna_rf'],     cn5_time,
         'RNA velocity – ReferenceFitting (CN5)',      15, False),
        (axes[5], p2d_CN5,              dp_rf_2d_CN5,              cn5_time,
         'Protein velocity – ReferenceFitting (CN5)',  15, True),
        (axes[6], rna_2d_s,             dr_s,                      time_s,
         'RNA velocity – CardamomOT (Schiebinger)',              5, False),
        (axes[7], prot_2d_s,            dp_s,                      time_s,
         'Inferred protein velocity – CardamomOT (Schiebinger)', 5, True),
    ]
    for ax, emb, vel, t, title, size, cbar in stream_configs:
        plot_stream(ax, emb, vel, t, title, size=size, add_colorbar=cbar,
                    density=DENSITY, smooth=SMOOTH)

    plt.subplots_adjust(hspace=0.2, wspace=0.25)
    plt.savefig('figure_5_trajectories.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print("\nSaved figure_5_trajectories.png")

    try:
        from PIL import Image
        Image.open('figure_5_trajectories.png').convert('RGB').save(
            'figure_5_trajectories.pdf', 'PDF', resolution=300)
        print("Saved figure_5_trajectories.pdf")
    except ImportError:
        print("(PIL not available – skipping PDF export)")


if __name__ == '__main__':
    main()