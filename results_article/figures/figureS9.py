"""
figureS9.py
===========
Variabilité des réseaux GRN inférés par CardamomOT sur 3 jeux de données.

Pour chaque dataset (Semrau, Kameneva, Schiebinger), la pipeline d'inférence
(fit_mixture → fit_network → refine_network_degradations) est exécutée N_RUNS
fois indépendamment. Les matrices inter_simul[ns:, ns:, 0] (bloc gène×gène)
sont comparées à la référence (run_00 = inter_simul.npy de la pipeline).

Résultats mis en cache dans for_figureS9/<dataset>/.

Paramètres par dataset depuis run_pipeline.sh :
  Semrau     : full,  mean_forcing=1.0
  Kameneva   : full,  mean_forcing=0.5
  Schiebinger: train, mean_forcing=0.0, stimulus=1.0, prior=1.0,
               force_basins=0.0, temporal_basins=0

Figure (A4 paysage, 2 lignes × 3 colonnes) :
  Panneau A : cosine similarity pairwise — violin + stripplot par dataset
  Panneau B : AUPR vs réseau de référence (run_00) pour les top-k gènes
              (k = 10, 20, 30, 40, 50 ; capé à G si G < 50)
              Gènes classés par degré L1 pondéré (|arêtes sortantes| + |entrantes|)
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

def get_inter_list(cfg, n_runs=N_RUNS, cache_root='for_figureS9'):
    """
    Charge les matrices inter depuis le cache (for_figureS9/<dataset>/)
    ou les calcule si absentes.

    Structure du cache :
      for_figureS9/<dataset>/inter_run_00.npy  ← copie de inter_simul.npy (référence)
      for_figureS9/<dataset>/inter_run_01.npy  ← re-fit 1
      for_figureS9/<dataset>/inter_run_02.npy  ← re-fit 2
      …

    Si les fichiers existent déjà, l'inférence est sautée.
    Retourne (liste de matrices inter, n_stimuli).
    """
    p       = cfg['path']
    var_dir = os.path.join(cache_root, cfg['name'])
    os.makedirs(var_dir, exist_ok=True)

    inters = []

    # Run 00 = référence depuis experimental_datasets (inter_simul.npy)
    reference = os.path.join(p, 'cardamomOT', 'inter_simul.npy')
    cache_0   = os.path.join(var_dir, 'inter_run_00.npy')
    if os.path.exists(reference) and not os.path.exists(cache_0):
        np.save(cache_0, np.load(reference))
        print(f"  [{cfg['name']}] Run 00 : référence copiée depuis inter_simul.npy")
    if os.path.exists(cache_0):
        inters.append(np.load(cache_0))
        start = 1
    else:
        print(f"  [{cfg['name']}] Warning: inter_simul.npy introuvable, run 00 ignoré")
        start = 0

    for i in range(start, n_runs):
        cache_i = os.path.join(var_dir, f'inter_run_{i:02d}.npy')
        if os.path.exists(cache_i):
            print(f"  [{cfg['name']}] Run {i:02d} : chargé depuis le cache ({cache_i})")
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
    """Similarities cosinus pairwise. Retourne upper-triangle list."""
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
# AUPR vs top-k gènes
# ─────────────────────────────────────────────────────────────────────────────

K_VALUES = [10, 20, 30, 40, 50]


def _gene_block(inter, ns):
    """Extrait le bloc gène×gène (diagonale mise à 0)."""
    b = inter[ns:, ns:, 0] if inter.ndim == 3 else inter[ns:, ns:]
    b = b.copy()
    np.fill_diagonal(b, 0.0)
    return b


def _l1_degree(block):
    """
    Degré L1 pondéré de chaque gène : somme des |arêtes sortantes| +
    somme des |arêtes entrantes|.  Peut être remplacé par une centrali-
    té spectrale, mais reste simple et très interprétable pour des GRN.
    """
    return np.abs(block).sum(axis=1) + np.abs(block).sum(axis=0)


EDGE_THRESHOLD = 1.0   # seuil pour binariser la référence (|inter| > 1 → arête)


def compute_aupr_vs_topk(inters, ns, k_values=None):
    """
    Pour chaque re-run r (runs 1..N) vs la référence (run 0) :
      1. Sélectionner les top-k gènes par degré L1 dans la référence.
      2. Extraire le sous-bloc k×k de la référence et du re-run.
      3. Vérité terrain binaire : |ref[i,j]| > EDGE_THRESHOLD (défaut 1.0).
         Le seuil 1 correspond à "une arrête existe ou non" dans le réseau
         de référence continu.
      4. Scores : |re-run[i,j]|.
      5. AUPR = average_precision_score.

    La baseline aléatoire = sparsité réelle du sous-bloc de référence
    (fraction d'arêtes à 1), qui varie avec k et le dataset.

    Retourne :
        k_eff      : list[int]   — k effectifs (capés à G)
        aupr_mat   : (n_reruns, len(k_eff)) array, NaN si non défini
        baseline   : (len(k_eff),) array  — AUPR random = sparsité par k
    """
    try:
        from sklearn.metrics import average_precision_score as aps
    except ImportError:
        print("  Warning: sklearn non disponible — AUPR ignoré")
        return [], np.zeros((0, 0))

    if k_values is None:
        k_values = K_VALUES
    if len(inters) < 2:
        return [], np.zeros((0, len(k_values))), np.zeros(len(k_values))

    ref = _gene_block(inters[0], ns)
    G   = ref.shape[0]

    # Classement des gènes par degré L1 dans la référence
    order = np.argsort(_l1_degree(ref))[::-1]

    # k effectifs (dédupliqués, capés à G)
    k_eff = []
    for k in k_values:
        ke = min(k, (G // 10) * 10)
        if ke not in k_eff:
            k_eff.append(ke)

    aupr_mat = np.full((len(inters) - 1, len(k_eff)), np.nan)
    baseline = np.full(len(k_eff), np.nan)   # sparsité réelle par k

    for ki, k in enumerate(k_eff):
        idx     = order[:k]
        ref_sub = ref[np.ix_(idx, idx)].flatten()
        y_true  = (np.abs(ref_sub) > EDGE_THRESHOLD).astype(int)
        # baseline aléatoire = proportion d'arêtes présentes dans la référence
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            baseline[ki] = y_true.mean()

        for r, inter in enumerate(inters[1:]):
            blk     = _gene_block(inter, ns)
            run_sub = blk[np.ix_(idx, idx)].flatten()
            y_score = np.abs(run_sub)

            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue
            aupr_mat[r, ki] = aps(y_true, y_score)

    return k_eff, aupr_mat, baseline


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def _draw_violin(ax, vals, color, pos=0):
    """Violin + stripplot pour une liste de valeurs scalaires."""
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

    # ── Collecte des données ─────────────────────────────────────────────────
    all_cos   = {}   # {name: list of pairwise cosine sims}
    all_aupr  = {}   # {name: (k_eff list, aupr_mat)}

    for cfg in datasets:
        print(f"\n══ {cfg['name']} ══")
        inters, ns = get_inter_list(cfg, n_runs)

        cos = pairwise_cosine(inters, ns)
        all_cos[cfg['name']] = cos
        print(f"  cosine : {np.mean(cos):.4f} ± {np.std(cos):.4f}  ({len(inters)} runs)")

        k_eff, aupr_mat, baseline = compute_aupr_vs_topk(inters, ns, k_values)
        all_aupr[cfg['name']] = (k_eff, aupr_mat, baseline)
        if len(k_eff):
            mu = np.nanmean(aupr_mat, axis=0)
            print(f"  AUPR   : " + "  ".join(f"k={k}: {v:.3f}" for k, v in zip(k_eff, mu)))
            print(f"  random : " + "  ".join(f"k={k}: {v:.3f}" for k, v in zip(k_eff, baseline)))

    # ── Layout (A4 paysage) ──────────────────────────────────────────────────
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

    # ── Panneau A : cosine similarity ────────────────────────────────────────
    for i, (cfg, ax) in enumerate(zip(datasets, axes_A)):
        vals  = all_cos[cfg['name']]
        color = COLORS[i]
        _draw_violin(ax, vals, color)
        mu, sd = np.mean(vals), np.std(vals)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylim(0.75, 1.0)
        ax.set_title(cfg['name'], fontsize=9, fontweight='bold', pad=3)
        ax.set_xlabel(f'μ={mu:.3f}  σ={sd:.3f}', fontsize=6.5, color='#444444')
        ax.set_ylabel('Cosine similarity' if i == 0 else '', fontsize=7)
        ax.tick_params(axis='y', labelsize=6.5)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)

    # ── Panneau B : AUPR vs top-k ─────────────────────────────────────────
    for i, (cfg, ax) in enumerate(zip(datasets, axes_B)):
        color                    = COLORS[i]
        k_eff, aupr_mat, baseline = all_aupr[cfg['name']]

        if len(k_eff) == 0 or aupr_mat.shape[0] == 0:
            ax.text(0.5, 0.5, 'AUPR non disponible',
                    ha='center', va='center', transform=ax.transAxes, fontsize=7)
            ax.axis('off')
            continue

        x = np.arange(len(k_eff))

        # Lignes individuelles (thin, semi-transparent)
        for r in range(aupr_mat.shape[0]):
            row = aupr_mat[r]
            mask = ~np.isnan(row)
            if mask.sum() < 2:
                continue
            ax.plot(x[mask], row[mask],
                    color=color, lw=0.9, alpha=0.35, zorder=2)

        # Moyenne + bande ±1σ
        mu_k  = np.nanmean(aupr_mat, axis=0)
        sd_k  = np.nanstd(aupr_mat,  axis=0)
        valid = ~np.isnan(mu_k)
        ax.plot(x[valid], mu_k[valid],
                color=color, lw=2.0, zorder=4, label='mean')
        ax.fill_between(x[valid],
                        mu_k[valid] - sd_k[valid],
                        mu_k[valid] + sd_k[valid],
                        color=color, alpha=0.18, zorder=3)

        # Baseline aléatoire = sparsité réelle par k (varie avec k)
        bl_valid = ~np.isnan(baseline)
        if bl_valid.any():
            ax.plot(x[bl_valid], baseline[bl_valid],
                    color='#888888', linestyle='--', lw=0.9,
                    alpha=0.8, zorder=3, label='random')

        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_eff], fontsize=7)
        ax.set_xlabel('Top-k gènes (degré L1)', fontsize=7)
        ax.set_ylabel('AUPR vs référence' if i == 0 else '', fontsize=7)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(cfg['name'], fontsize=9, fontweight='bold', pad=3)
        ax.tick_params(axis='y', labelsize=6.5)
        ax.spines[['top', 'right']].set_visible(False)
        if i == 0:
            ax.legend(fontsize=6, frameon=False)

    # ── Étiquettes de panneaux ───────────────────────────────────────────────
    fig.text(0.01, 0.93, 'A', fontsize=13, fontweight='bold', color='#111111', va='top')
    fig.text(0.01, 0.49, 'B', fontsize=13, fontweight='bold', color='#111111', va='top')

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
    ap.add_argument('--n-runs',   type=int, default=N_RUNS,
                    help='Nombre de re-fits indépendants par dataset')
    ap.add_argument('--k-values', type=int, nargs='+', default=K_VALUES,
                    help='Valeurs de top-k pour AUPR (ex: 10 20 30 40 50)')
    ap.add_argument('--out',      type=str, default='figureS9.png',
                    help='Nom du fichier de sortie')
    args = ap.parse_args()
    make_figure(n_runs=args.n_runs, k_values=args.k_values, save_path=args.out)
