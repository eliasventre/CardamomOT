"""
figureS9.py
===========
Variabilité des réseaux GRN inférés par CardamomOT sur 3 jeux de données.

Pour chaque dataset (Semrau, Kameneva, Schiebinger), la pipeline d'inférence
(fit_mixture → fit_network → refine_network_degradations) est exécutée N_RUNS
fois indépendamment. Les matrices inter_simul[ns:, ns:, 0] (bloc gène×gène)
sont comparées par cosine similarity pairwise.

Résultats mis en cache dans :
  <dataset>/cardamomOT/variability/inter_run_XX.npy

Paramètres par dataset depuis run_pipeline.sh :
  Semrau     : full,  mean_forcing=1.0
  Kameneva   : full,  mean_forcing=0.5
  Schiebinger: train, mean_forcing=0.0, stimulus=1.0, prior=1.0,
               force_basins=0.0, temporal_basins=0

Figure (A4 paysage) :
  - Ligne 1 : heatmaps des similarités cosinus pairwise (une par dataset)
  - Ligne 2 : violin + stripplot de comparaison inter-datasets
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

N_RUNS = 5  # nombre de re-fits indépendants (run 0 = inférence existante)

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
        stimulus        = 1.0,
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
# Helpers : inférence
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
    Exécute une inférence complète sur le dataset et retourne model.inter (copie).
    Équivalent de fit_mixture + fit_network + refine_network_degradations.
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

    # ── Étape 1 : mixture ────────────────────────────────────────────────────
    model.fit_mixture(
        adata,
        gene_names      = list(adata.var_names),
        min_components  = 2,
        max_components  = 2,
        max_iter_kinetics = 0,
        verb            = True,
        stimulus_schedule = stim_sched,
    )

    # ── Étape 2 : taux de dégradation (depuis adata.var) ─────────────────────
    ns    = model.n_stimuli
    G_tot = adata.shape[1] + ns
    model.d = np.ones((2, G_tot))
    if 'd1' in adata.var.columns:
        model.d[1, ns:] = adata.var['d1'].values
    if 'd0' in adata.var.columns:
        model.d[0, ns:] = adata.var['d0'].values

    # ── Étape 2b : réseau de référence prior (CSV optionnel) ─────────────────
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
            print(f"  Warning: ref_network.csv non chargé ({e})")

    # ── Étape 3 : réseau ──────────────────────────────────────────────────────
    model.fit_network(adata, intensity_prior=100, verb=True)

    # ── Étape 4 : raffinement des dégradations ────────────────────────────────
    model.refine_network_degradations()

    return model.inter.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Gestion du cache
# ─────────────────────────────────────────────────────────────────────────────

def get_inter_list(cfg, n_runs=N_RUNS):
    """
    Charge les matrices inter depuis le cache (cardamomOT/variability/)
    ou les calcule si absentes.  run_00 = inférence existante (inter_simul.npy).
    Retourne une liste de matrices inter et le nombre de stimuli détecté.
    """
    p       = cfg['path']
    var_dir = os.path.join(p, 'cardamomOT', 'variability')
    os.makedirs(var_dir, exist_ok=True)

    inters = []

    # Run 0 = résultat de l'inférence déjà stockée
    existing = os.path.join(p, 'cardamomOT', 'inter_simul.npy')
    cache_0  = os.path.join(var_dir, 'inter_run_00.npy')
    if os.path.exists(existing) and not os.path.exists(cache_0):
        np.save(cache_0, np.load(existing))
    if os.path.exists(cache_0):
        inters.append(np.load(cache_0))
        start = 1
    else:
        start = 0

    for i in range(start, n_runs):
        cache_i = os.path.join(var_dir, f'inter_run_{i:02d}.npy')
        if os.path.exists(cache_i):
            print(f"  [{cfg['name']}] Run {i:02d} : chargé depuis le cache")
            inters.append(np.load(cache_i))
        else:
            print(f"  [{cfg['name']}] Run {i:02d} : inférence en cours…")
            inter = run_single_fit(cfg)
            np.save(cache_i, inter)
            print(f"  [{cfg['name']}] Run {i:02d} : sauvegardé → {cache_i}")
            inters.append(inter)

    stim_sched = _load_stimulus_schedule(p)
    ns = _detect_n_stimuli(stim_sched)
    return inters, ns


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_cosine(inters, ns):
    """
    Calcule la matrice de cosine similarity pairwise entre tous les runs.
    Retourne (sim_matrix NxN, upper_triangle list).
    """
    vecs = []
    for inter in inters:
        block = inter[ns:, ns:, 0] if inter.ndim == 3 else inter[ns:, ns:]
        v = block.flatten()
        norm = np.linalg.norm(v)
        vecs.append(v / norm if norm > 1e-12 else v)

    n = len(vecs)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = np.dot(vecs[i], vecs[j])

    upper = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
    return sim_matrix, upper


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(datasets=DATASETS, n_runs=N_RUNS, save_path='figureS9.png'):

    all_sims     = {}
    all_matrices = {}

    for cfg in datasets:
        print(f"\n══ {cfg['name']} ══")
        inters, ns = get_inter_list(cfg, n_runs)
        sim_matrix, upper = pairwise_cosine(inters, ns)
        all_sims[cfg['name']]     = upper
        all_matrices[cfg['name']] = sim_matrix
        print(f"  → {len(inters)} runs, mean cosine = {np.mean(upper):.4f} ± {np.std(upper):.4f}")

    n_ds = len(datasets)

    fig = plt.figure(figsize=(11.69, 7.0))   # A4 paysage

    gs_top = gridspec.GridSpec(
        1, n_ds,
        figure=fig,
        left=0.06, right=0.96,
        top=0.88, bottom=0.44,
        wspace=0.40,
    )
    gs_bot = gridspec.GridSpec(
        1, 1,
        figure=fig,
        left=0.30, right=0.70,
        top=0.35, bottom=0.10,
    )

    axes_hm  = [fig.add_subplot(gs_top[0, i]) for i in range(n_ds)]
    ax_comp  = fig.add_subplot(gs_bot[0, 0])

    # ── Heatmaps (une par dataset) ──────────────────────────────────────────
    for i, (cfg, ax) in enumerate(zip(datasets, axes_hm)):
        sim_mat = all_matrices[cfg['name']]
        n       = sim_mat.shape[0]

        im = ax.imshow(sim_mat, vmin=0.0, vmax=1.0, cmap='YlOrRd', aspect='auto')

        labels = [f"R{j}" for j in range(n)]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(cfg['name'], fontsize=9, fontweight='bold', pad=4)

        for ii in range(n):
            for jj in range(n):
                v    = sim_mat[ii, jj]
                col  = 'white' if v > 0.75 else 'black'
                ax.text(jj, ii, f'{v:.2f}', ha='center', va='center',
                        fontsize=5.5, color=col)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5.5)
        cbar.set_label('Cosine similarity', fontsize=6)

        mu  = np.mean(all_sims[cfg['name']])
        std = np.std(all_sims[cfg['name']])
        ax.set_xlabel(f'mean = {mu:.3f}  ±  {std:.3f}', fontsize=6.5, color='#444444')

    # ── Violin de comparaison inter-datasets ────────────────────────────────
    vio_data   = []
    vio_labels = []
    for cfg in datasets:
        vio_data.extend(all_sims[cfg['name']])
        vio_labels.extend([cfg['name']] * len(all_sims[cfg['name']]))

    positions = list(range(n_ds))
    ax_comp.set_xlim(-0.6, n_ds - 0.4)

    for i, (cfg, col) in enumerate(zip(datasets, COLORS)):
        vals  = all_sims[cfg['name']]
        if len(vals) < 2:
            ax_comp.scatter([i], vals, s=25, color=col, zorder=4)
            continue

        parts = ax_comp.violinplot([vals], positions=[i], widths=0.6,
                                    showmedians=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor(col)
            pc.set_alpha(0.65)
            pc.set_edgecolor('none')
        parts['cmedians'].set_color('#333333')
        parts['cmedians'].set_linewidth(1.2)
        for key in ('cmins', 'cmaxes', 'cbars'):
            if key in parts:
                parts[key].set_color('#888888')
                parts[key].set_linewidth(0.8)

        jitter = np.random.default_rng(42 + i).uniform(-0.12, 0.12, size=len(vals))
        ax_comp.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=14, color=col, zorder=5, alpha=0.9, edgecolors='white', linewidths=0.4,
        )

    ax_comp.set_xticks(positions)
    ax_comp.set_xticklabels([d['name'] for d in datasets], fontsize=8)
    ax_comp.set_ylabel('Pairwise cosine similarity', fontsize=8)
    ax_comp.set_ylim(-0.05, 1.05)
    ax_comp.axhline(1.0, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
    ax_comp.axhline(0.0, color='gray', linestyle=':',  linewidth=0.6, alpha=0.4)
    ax_comp.set_title('Distribution inter-datasets', fontsize=8.5, fontweight='bold')
    ax_comp.spines[['top', 'right']].set_visible(False)
    ax_comp.tick_params(axis='y', labelsize=7)

    # ── Titre & étiquettes de panneaux ───────────────────────────────────────
    fig.suptitle(
        'Variabilité des réseaux GRN inférés — similarité cosinus inter-runs',
        fontsize=10, fontweight='bold', y=0.975,
    )
    for ax, lbl in zip(axes_hm, list('ABC')):
        ax.text(-0.18, 1.12, lbl, transform=ax.transAxes,
                fontsize=13, fontweight='bold', color='#111111', va='top')
    ax_comp.text(-0.14, 1.10, 'D', transform=ax_comp.transAxes,
                 fontsize=13, fontweight='bold', color='#111111', va='top')

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"\nSaved {save_path}")
    try:
        from PIL import Image
        pdf_path = save_path.replace('.png', '.pdf')
        Image.open(save_path).convert('RGB').save(pdf_path, 'PDF', resolution=300)
        print(f"Saved {pdf_path}")
    except ImportError:
        print("(PIL non disponible — export PDF ignoré)")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-runs', type=int, default=N_RUNS,
                    help='Nombre de re-fits indépendants par dataset')
    ap.add_argument('--out',    type=str, default='figureS9.png',
                    help='Nom du fichier de sortie')
    args = ap.parse_args()
    make_figure(n_runs=args.n_runs, save_path=args.out)
