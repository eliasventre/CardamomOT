"""
figure7.py  –  v2
===============================
Génère une grande figure (A4 portrait) présentant les résultats de perturbations
géniques sur des simulations de données single-cell.

Structure : 5 lignes × 3 colonnes
  Col 1 : Réseau de régulation – cibles du gène perturbé (taille réduite)
  Col 2 : Barplot des proportions de types cellulaires (4 conditions)
  Col 3 : UMAP × 3 (Reference / Sim WT / Sim perturb)

Perturbations :
  Ligne 1 : KO Zfp42  – ./experimental_datasets/Semrau
  Ligne 2 : OV STMN2  – ./experimental_datasets/Kameneva
  Ligne 3 : KO CHGA   – ./experimental_datasets/Kameneva
  Ligne 4 : OV Zfp42  – ./experimental_datasets/Schiebinger
  Ligne 5 : OV Obox6  – ./experimental_datasets/Schiebinger

Changements v2 (par rapport à v1) :
  1. Réseau : nœuds et polices réduits (node_size 550/380, font_size 5.5)
  2. UMAP   : appris sur la concaténation des 3 datasets (project_on_full=False)
  3. Légendes cell-type par jeu de données (Semrau après ligne 1,
     Kameneva après ligne 3, Schiebinger après ligne 5) + trait gris séparateur
"""

# ──────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────
import sys
sys.path += ['./../../']

import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import networkx as nx
from umap import UMAP

from CardamomOT import train_classifier, predict_cell_types

# ──────────────────────────────────────────────────────────────
# Configuration globale
# ──────────────────────────────────────────────────────────────

STIM  = 1.0
PRIOR = 1.0
LABEL = "cell_type"

# Palette douce inspirée de la figure de référence (style matplotlib par défaut amélioré)
SOFT_COLORS = [
    '#4878CF',   # bleu acier
    '#E87B30',   # orange doux
    '#6DB36D',   # vert sage
    '#C94040',   # rouge brique
    '#8C6DBF',   # violet lavande
    '#57A8A8',   # cyan ardoise
    '#C97BAA',   # rose poudré
    '#8C7B6B',   # brun chaud
    '#9CBF57',   # vert olive
    '#CF9B3A',   # or ocre
]

def _soft_color(i):
    return SOFT_COLORS[i % len(SOFT_COLORS)]

# Description des 5 perturbations.
# `last_in_group` : True = dernière ligne de ce jeu de données
#   → on trace la légende cell-type + le trait séparateur en dessous.
PERTURBATIONS = [
    dict(
        path          = './../../experimental_datasets/Semrau',
        gene          = 'Dnmt3a',
        mode          = 'KO',
        perturb_id    = 'KO_none_OV_Dnmt3a',
        label         = 'OV Dnmt3a\nSemrau',
        dataset_group = 'Semrau',
        last_in_group = True,
    ),
    dict(
        path          = './../../experimental_datasets/Kameneva',
        gene          = 'CHGA',
        mode          = 'KO',
        perturb_id    = 'KO_CHGA_OV_none',
        label         = 'KO CHGA\nKameneva',
        dataset_group = 'Kameneva',
        last_in_group = False,
    ),
    dict(
        path          = './../../experimental_datasets/Kameneva',
        gene          = 'STMN2',
        mode          = 'OV',
        perturb_id    = 'KO_CHGA_OV_STMN2',
        label         = 'KO CHGA-OV STMN2\nKameneva',
        dataset_group = 'Kameneva',
        last_in_group = True,
    ),
    dict(
        path          = './../../experimental_datasets/Schiebinger',
        gene          = 'Zfp42',
        mode          = 'OV',
        perturb_id    = 'KO_none_OV_Zfp42',
        label         = 'OV Zfp42\nSchiebinger',
        dataset_group = 'Schiebinger',
        last_in_group = False,
    ),
    dict(
        path          = './../../experimental_datasets/Schiebinger',
        gene          = 'Obox6',
        mode          = 'OV',
        perturb_id    = 'KO_none_OV_Obox6',
        label         = 'OV Obox6\nSchiebinger',
        dataset_group = 'Schiebinger',
        last_in_group = True,
    ),
]

# ──────────────────────────────────────────────────────────────
# Utilitaires réseau
# ──────────────────────────────────────────────────────────────

def load_grn_matrix(p, ns):
    """Charge la matrice GRN inférée (inter_simul.npy) et renvoie (matrix, gene_names)."""
    # data_full.h5ad si disponible, sinon data_train.h5ad
    full_path = os.path.join(p, 'Data', 'data_full.h5ad')
    train_path = os.path.join(p, 'Data', 'data_train.h5ad')
    adata = sc.read_h5ad(full_path if os.path.exists(full_path) else train_path)
    genes = list(adata.var_names)
    grn_path = os.path.join(p, 'cardamomOT', 'inter_simul.npy')
    grn_mat = np.load(grn_path)[ns:, ns:]
    matrix = grn_mat[:, :, 0] if grn_mat.ndim == 3 else grn_mat
    return matrix, genes


def build_gene_subgraph(matrix, gene_names, gene, top_targets=8):
    """
    Construit un DiGraph centré sur `gene` montrant ses top cibles.
    Retourne (G, max_intensity).
    """
    gene_names = list(gene_names)
    if gene not in gene_names:
        return None, 1.0

    idx = gene_names.index(gene)
    series = pd.Series(matrix[idx, :], index=gene_names)
    series = series.drop(gene, errors='ignore')

    top_idx = series.abs().nlargest(top_targets).index
    max_intensity = series.abs().max() or 1.0

    G = nx.DiGraph()
    for tgt in top_idx:
        w = series[tgt]
        if w != 0:
            G.add_edge(gene, tgt, weight=float(w))
    return G, float(max_intensity)


def draw_gene_subgraph(ax, G, gene, max_intensity):
    if G is None or G.number_of_edges() == 0:
        ax.text(0.5, 0.5, f"{gene}\n(no GRN data)", ha='center', va='center',
                transform=ax.transAxes, fontsize=6, color='gray')
        ax.axis('off')
        return

    # --- Layout : gène central fixé à (0, 0) ---
    fixed_positions = {gene: (0.0, 0.0)}
    pos = nx.spring_layout(
        G,
        pos=fixed_positions,
        fixed=[gene],
        k=2.5,
        iterations=100,
        seed=42
    )

    # --- Rayon minimum autour du centre ---
    min_radius = 0.6
    cx, cy = pos[gene]
    for node in pos:
        if node == gene:
            continue
        dx, dy = pos[node][0] - cx, pos[node][1] - cy
        dist = (dx**2 + dy**2) ** 0.5
        if dist < min_radius:
            scale = min_radius / (dist if dist > 1e-6 else 1e-6)
            pos[node] = (cx + dx * scale, cy + dy * scale)

    # --- Dessin ---
    node_colors = ['#4C9BE8' if n == gene else '#E8E8E8' for n in G.nodes]
    node_sizes  = [550 if n == gene else 380 for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=5.5, ax=ax)
    ax.margins(0.15)

    edges_pos = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    edges_neg = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if d['weight'] < 0]

    width_scale = 3.0
    wp = [abs(w) / max_intensity * width_scale for (_, _, w) in edges_pos]
    wn = [abs(w) / max_intensity * width_scale for (_, _, w) in edges_neg]

    kw_pos = dict(arrows=True, arrowsize=8, ax=ax, connectionstyle='arc3,rad=0.1')
    kw_neg = dict(
        ax=ax, connectionstyle='arc3,rad=0.1',
        arrows=True,
        arrowstyle='-[,widthB=0.8,lengthB=0.0',
        min_target_margin=12.5,
    )
    if edges_pos:
        nx.draw_networkx_edges(G, pos,
                               edgelist=[(u, v) for u, v, _ in edges_pos],
                               edge_color='#2ECC71', width=wp, **kw_pos)
    if edges_neg:
        nx.draw_networkx_edges(G, pos,
                               edgelist=[(u, v) for u, v, _ in edges_neg],
                               edge_color='#E74C3C', width=wn, **kw_neg)

    ax.set_title(f"Targets of {gene}", fontsize=6.5, fontweight='bold', pad=2)
    ax.axis('off')


def get_regulator_rank(matrix, gene_names, gene):
    """Retourne le classement (1-based) du gène comme régulateur (par force sortante)."""
    gene_names = list(gene_names)
    if gene not in gene_names:
        return None
    scores = np.abs(matrix).sum(axis=1)
    order  = np.argsort(scores)[::-1]
    ranked = [gene_names[i] for i in order]
    try:
        return ranked.index(gene) + 1
    except ValueError:
        return None

# ──────────────────────────────────────────────────────────────
# Utilitaires types cellulaires
# ──────────────────────────────────────────────────────────────

def get_color_map(adata_full):
    """Renvoie un dict {cell_type: color} stable pour un dataset, palette douce."""
    if LABEL not in adata_full.obs:
        return {}
    cats = adata_full.obs[LABEL].astype(str).unique().tolist()
    return {cat: _soft_color(i) for i, cat in enumerate(cats)}


def compute_proportions(adatas, labels, color_map):
    """
    Calcule les proportions de types cellulaires pour chaque adata.
    Retourne un DataFrame (colonnes = labels, index = cell_types).
    """
    all_cats = list(color_map.keys())
    prop = {}
    for adata, lbl in zip(adatas, labels):
        if LABEL not in adata.obs:
            prop[lbl] = pd.Series(0.0, index=all_cats)
            continue
        counts = adata.obs[LABEL].astype(str).value_counts()
        total  = counts.sum()
        prop[lbl] = pd.Series(
            {cat: counts.get(cat, 0) / total for cat in all_cats}
        )
    return pd.DataFrame(prop)


from scipy.stats import chi2_contingency

def load_simulation_variance(perturb_id, data_dir='for_figure7_data'):
    """
    Charge les proportions pré-calculées par _for_figure7.py.
    Retourne (props_wt, props_single, props_perturb) ou (None, None, None) si absent.
    Utilisé uniquement pour les barres d'erreur (σ inter-runs).
    """
    path = os.path.join(data_dir, f'{perturb_id}.npz')
    if not os.path.exists(path):
        return None, None, None
    try:
        data          = np.load(path, allow_pickle=True)
        props_wt      = data['props_wt']
        props_perturb = data['props_perturb']
        props_single  = data['props_single'] if 'props_single' in data else None
        return props_wt, props_single, props_perturb
    except Exception:
        return None, None, None


def draw_barplot(ax, prop_df, color_map, show_xlabels='none', sim_stds=None):
    """
    Barplot empilé des proportions de types cellulaires.
    show_xlabels : 'top'  -> labels en haut de l'axe (première ligne uniquement)
                   'none' -> pas de labels (lignes suivantes)
    sim_stds     : dict {col_name: {cell_type: std}} — barres d'erreur sur
                   chaque colonne simulée. Ex :
                   {'Sim WT': {...}, 'Sim single': {...}, 'Sim perturb': {...}}
    """
    x        = np.arange(len(prop_df.columns))
    bottom   = np.zeros(len(prop_df.columns))
    cols_lst = list(prop_df.columns)

    for cat in prop_df.index:
        vals = prop_df.loc[cat].values
        ax.bar(x, vals, bottom=bottom,
               color=color_map.get(cat, '#CCCCCC'), label=cat, width=0.6,
               edgecolor='white', linewidth=0.4)

        if sim_stds is not None:
            for col_name, col_std_dict in sim_stds.items():
                if col_name not in cols_lst:
                    continue
                col_idx = cols_lst.index(col_name)
                std_val = col_std_dict.get(cat, 0.0)
                seg_val = vals[col_idx]
                if std_val > 0 and seg_val > 0.02:
                    mid_y = bottom[col_idx] + seg_val / 2.0
                    ax.errorbar(
                        x[col_idx], mid_y,
                        yerr=std_val,
                        fmt='none', ecolor='#222222', elinewidth=0.9,
                        capsize=2.5, capthick=0.9, zorder=6,
                    )

        bottom += vals

    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Proportion', fontsize=6)
    ax.tick_params(axis='y', labelsize=5.5)
    ax.spines[['top', 'right']].set_visible(False)

    if show_xlabels == 'top':
        ax.xaxis.set_tick_params(labeltop=True, labelbottom=False, pad=9)
        ax.set_xticklabels(prop_df.columns, fontsize=6, rotation=30, ha='left')
    else:
        ax.set_xticklabels([])

# ──────────────────────────────────────────────────────────────
# Utilitaires UMAP  –  project_on_full = False (v2)
# ──────────────────────────────────────────────────────────────

def compute_umaps(adata_full, adata_sim, adata_perturb):
    """
    Apprend l'UMAP sur la concaténation des 3 datasets puis répartit.
    project_on_full = False.
    Pas de re-normalisation (faite en amont sur l'ensemble des gènes) ;
    log1p appliqué sur les counts bruts avant UMAP.
    """
    adatas = [adata_full, adata_sim, adata_perturb]
    names  = ['Reference', 'Sim WT', 'Sim perturb']

    for name, A in zip(names, adatas):
        A.obs_names = [f"{name}_{i}" for i in range(A.n_obs)]
        A.obs['source'] = name

    # No re-normalisation; log1p on raw counts
    for adata in adatas:
        sc.pp.log1p(adata)

    adata_all = ad.concat(adatas, join='inner', label='source', keys=names)

    X_all = adata_all.X.toarray() if scipy.sparse.issparse(adata_all.X) else adata_all.X

    reducer = UMAP(random_state=42, min_dist=0.5)
    coords_all = reducer.fit_transform(X_all)
    adata_all.obsm['X_umap'] = coords_all

    # Répartir dans chaque adata individuel
    for name, A in zip(names, adatas):
        mask = (adata_all.obs['source'] == name).values
        A.obsm['X_umap'] = coords_all[mask]

    return adata_full, adata_sim, adata_perturb


def draw_umap_panel(axes_row, adata_full, adata_sim, adata_perturb, color_map, point_size=3, rotate_x=False, rotate_y=False):
    """Dessine 3 sub-axes UMAP colorés par cell_type (ou time si absent)."""
    triples = [
        (adata_full,    'Reference data'),
        (adata_sim,     'Sim WT'),
        (adata_perturb, 'Sim perturb'),
    ]
    use_celltype = LABEL in adata_full.obs

    for ax, (A, title) in zip(axes_row, triples):
        coords = A.obsm['X_umap']
        if rotate_x:
            coords[:, 0] = -coords[:, 0]
        if rotate_y:
            coords[:, 1] = -coords[:, 1]

        if use_celltype and LABEL in A.obs:
            for cat, col in color_map.items():
                mask = (A.obs[LABEL].astype(str) == cat).values
                if mask.any():
                    ax.scatter(coords[mask, 0], coords[mask, 1],
                               c=[col], s=point_size, linewidths=0, rasterized=True)
        elif 'time' in A.obs:
            tv = pd.to_numeric(A.obs['time'], errors='coerce').values
            ax.scatter(coords[:, 0], coords[:, 1],
                       c=tv, cmap='viridis', s=point_size, linewidths=0, rasterized=True)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=point_size, color='gray', rasterized=True)

        ax.set_title(title, fontsize=6, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

# ──────────────────────────────────────────────────────────────
# Chargement & préparation pour une perturbation
# ──────────────────────────────────────────────────────────────

def load_perturbation_data(cfg):
    p          = cfg['path']
    gene       = cfg['gene']
    perturb_id = cfg['perturb_id']

    # GRN
    ns = 2 if cfg['dataset_group'] == 'Schiebinger' else 1
    grn_matrix, gene_names = load_grn_matrix(p, ns)
    rank  = get_regulator_rank(grn_matrix, gene_names, gene)
    if gene == "Col4a2":
        G_net, max_int = build_gene_subgraph(grn_matrix, gene_names, gene, top_targets=4)
    else:
        G_net, max_int = build_gene_subgraph(grn_matrix, gene_names, gene, top_targets=8)

    # AnnData
    full_path2  = os.path.join(p, 'Data', 'data_full.h5ad')
    train_path2 = os.path.join(p, 'Data', 'data_train.h5ad')
    adata_full    = ad.read_h5ad(full_path2 if cfg['dataset_group'] != 'Schiebinger' else train_path2)
    if cfg['dataset_group'] == 'Kameneva' and LABEL in adata_full.obs:
        adata_full.obs[LABEL] = adata_full.obs[LABEL].astype(str).str.capitalize()
    if cfg['dataset_group'] == 'Semrau' and LABEL in adata_full.obs:
        adata_full.obs[LABEL] = adata_full.obs[LABEL].astype(str).str.replace('_', ' ', regex=False)
    if cfg['dataset_group'] == 'Semrau':
        adata_traj_raw = ad.read_h5ad(os.path.join(p, f'cardamomOT/adata_rna_traj_stim{STIM}_prior{PRIOR}.h5ad'))
        adata_sim_raw = ad.read_h5ad(os.path.join(p, f'cardamomOT/adata_sim_stim{STIM}_prior{PRIOR}.h5ad'))
        adata_perturb  = ad.read_h5ad(os.path.join(p, f'cardamomOT/adata_sim_{perturb_id}_stim{STIM}_prior{PRIOR}.h5ad'))
    elif cfg['dataset_group'] == 'Schiebinger':
        adata_traj_raw = ad.read_h5ad(os.path.join(p, f'cardamomOT/adata_beta_stim{STIM}_prior{PRIOR}.h5ad'))
        adata_sim_raw = ad.read_h5ad(os.path.join(p, f'cardamomOT/adata_sim_stim{STIM}_prior{PRIOR}.h5ad'))
        adata_perturb  = ad.read_h5ad(os.path.join(p, f'cardamomOT/adata_sim_{perturb_id}_stim{STIM}_prior{PRIOR}.h5ad'))
    else:
        adata_traj_raw = ad.read_h5ad(os.path.join(p, f'cardamomOT/adata_rna_traj_stim{0.2}_prior{PRIOR}.h5ad'))
        adata_sim_raw = ad.read_h5ad(os.path.join(p, f'cardamomOT/adata_sim_stim{0.2}_prior{PRIOR}.h5ad'))
        adata_perturb  = ad.read_h5ad(os.path.join(p, f'cardamomOT/adata_sim_{perturb_id}_stim{0.2}_prior{PRIOR}.h5ad'))

    # Classifieur & types cellulaires
    color_map = get_color_map(adata_full)
    clf = train_classifier(adata_full, label_key=LABEL)

    adata_sim_wt = predict_cell_types(adata_sim_raw.copy(), clf, label_key=LABEL)
    adata_traj_wt = predict_cell_types(adata_traj_raw.copy(), clf, label_key=LABEL)

    # Sim single-gene : substituer le gène cible dans la simulation WT
    adata_sim_single = adata_sim_raw.copy()
    if gene in adata_full.var_names:
        idx_gene = np.where(adata_full.var_names == gene)[0][0]
        adata_sim_single.X[:, idx_gene] = adata_perturb.X[:, idx_gene].mean()
    adata_sim_single = predict_cell_types(adata_sim_single, clf, label_key=LABEL)

    adata_perturb_pred = predict_cell_types(adata_perturb.copy(), clf, label_key=LABEL)

    # Proportions (4 conditions) — run unique comme fallback
    prop_df = compute_proportions(
        [adata_traj_wt, adata_sim_wt, adata_sim_single, adata_perturb_pred],
        ['Reference', 'Sim WT', 'Sim single', 'Sim perturb'],
        color_map,
    )

    # Chi2 sur les proportions affichées (Sim WT vs Sim perturb, run unique .h5ad)
    # n_cells=1000 fixe : p-values comparables entre datasets, pas d'inflation
    try:
        n_cells = 100 * len(prop_df.index)
        p_wt   = prop_df['Sim WT'].values.astype(float)
        p_pert = prop_df['Sim perturb'].values.astype(float)
        table  = np.vstack([
            np.round(p_wt   * n_cells).astype(int),
            np.round(p_pert * n_cells).astype(int),
        ])
        chi2_stat, chi2_pval = chi2_contingency(table)[:2]
    except Exception:
        chi2_stat, chi2_pval = None, None

    # Errorbars inter-runs uniquement (pré-calculées par _for_figure7.py)
    props_wt_sims, props_single_sims, props_pert_sims = load_simulation_variance(perturb_id)

    # Fallback Sim single : si props_single absent du npz, découper adata_sim_raw
    # en N_SPLITS tranches non-chevauchantes et appliquer la substitution sur chacune.
    ct_order_single = None
    if props_single_sims is None and gene in adata_full.var_names:
        N_SPLITS  = 10
        all_cats  = list(color_map.keys())
        idx_gene  = int(np.where(adata_full.var_names == gene)[0][0])
        mean_val  = float(adata_perturb.X[:, idx_gene].mean()
                          if not scipy.sparse.issparse(adata_perturb.X)
                          else adata_perturb.X[:, idx_gene].toarray().mean())
        n_raw     = adata_sim_raw.n_obs
        chunk     = max(1, n_raw // N_SPLITS)
        idx_perm  = np.random.default_rng(42).permutation(n_raw)
        split_props = []
        for k in range(N_SPLITS):
            sl = idx_perm[k * chunk : (k + 1) * chunk]
            if len(sl) == 0:
                break
            a_k = adata_sim_raw[sl].copy()
            if scipy.sparse.issparse(a_k.X):
                _X = a_k.X.toarray(); _X[:, idx_gene] = mean_val
                a_k = ad.AnnData(X=_X, var=a_k.var.copy(), obs=a_k.obs.copy())
            else:
                a_k.X = np.array(a_k.X); a_k.X[:, idx_gene] = mean_val
            a_k = predict_cell_types(a_k, clf, label_key=LABEL)
            counts = a_k.obs[LABEL].astype(str).value_counts()
            total  = counts.sum() or 1
            split_props.append([counts.get(ct, 0) / total for ct in all_cats])
        if split_props:
            props_single_sims = np.array(split_props)
            ct_order_single   = all_cats

    def _stds_for(arr, ct_order):
        if arr is None or ct_order is None:
            return None
        s = arr.std(axis=0)
        return {ct: float(s[i]) for i, ct in enumerate(ct_order) if ct in color_map}

    try:
        _npz_data = np.load(os.path.join('for_figure7_data', f'{perturb_id}.npz'),
                             allow_pickle=True)
        ct_order  = list(_npz_data['cell_types'])
    except Exception:
        ct_order  = list(color_map.keys())

    sim_stds = {}
    _w = _stds_for(props_wt_sims,     ct_order)
    if _w: sim_stds['Sim WT']     = _w
    _s = _stds_for(props_single_sims, ct_order_single or ct_order)
    if _s: sim_stds['Sim single'] = _s
    _p = _stds_for(props_pert_sims,   ct_order)
    if _p: sim_stds['Sim perturb'] = _p

    if not sim_stds:
        sim_stds = None

    # UMAP joint sur 3 conditions
    af_umap, as_umap, ap_umap = compute_umaps(
        adata_traj_wt.copy(), adata_sim_wt.copy(), adata_perturb_pred.copy()
    )

    return dict(
        gene       = gene,
        rank       = rank,
        G_net      = G_net,
        max_int    = max_int,
        color_map  = color_map,
        prop_df    = prop_df,
        chi2_stat  = chi2_stat,
        chi2_pval  = chi2_pval,
        sim_stds   = sim_stds,
        af_umap    = af_umap,
        as_umap    = as_umap,
        ap_umap    = ap_umap,
        label      = cfg['label'],
        mode       = cfg['mode'],
    )

# ──────────────────────────────────────────────────────────────
# Assemblage de la figure
# ──────────────────────────────────────────────────────────────

def make_figure(perturbations=PERTURBATIONS, save_path='figure_7.pdf'):
    n_rows = len(perturbations)

    fig = plt.figure(figsize=(8.27, 11.69))   # A4 portrait

    gs_main = gridspec.GridSpec(
        n_rows, 3,
        figure=fig,
        left=0.06, right=0.97,
        top=0.91, bottom=0.04,   # laisse de la place pour les titres de colonnes
        hspace=0.35, wspace=0.18,
        width_ratios=[1.1, 0.85, 2.2],
    )

    # Sub-axes UMAP (3 par ligne)
    umap_subaxes = []
    for row in range(n_rows):
        gs_umap = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs_main[row, 2], wspace=0.05,
        )
        umap_subaxes.append([
            fig.add_subplot(gs_umap[0, 0]),
            fig.add_subplot(gs_umap[0, 1]),
            fig.add_subplot(gs_umap[0, 2]),
        ])

    axes_net = [fig.add_subplot(gs_main[r, 0]) for r in range(n_rows)]
    axes_bar = [fig.add_subplot(gs_main[r, 1]) for r in range(n_rows)]

    # Titres de colonnes (haut de figure)
    for x_pos, title in zip(
        [0.13, 0.42, 0.74],
        ['Regulatory targets', 'Cell-type proportions', 'UMAP trajectories'],
    ):
        fig.text(x_pos, 0.975, title, ha='center', va='top',
                 fontsize=8.5, fontweight='bold', color='#222222')

    # Sous-titres des 3 sous-colonnes UMAP (une seule fois, première ligne)
    for ax, subtitle in zip(umap_subaxes[0], ['Reference data', 'Sim WT', 'Sim perturb']):
        # On surcharge le titre du premier tracé après coup → on le note ici
        ax._umap_subtitle = subtitle

    # ── Chargement et tracé ligne par ligne
    group_color_maps = {}   # {dataset_group: color_map}
    row_data_list    = []   # pour positionner les légendes a posteriori

    for row, cfg in enumerate(perturbations):
        print(f"\n── Ligne {row+1}/{n_rows} : {cfg['label']} ──")

        try:
            data = load_perturbation_data(cfg)
        except FileNotFoundError as e:
            print(f"  SKIP (fichier manquant) : {e}")
            for ax in [axes_net[row], axes_bar[row]] + umap_subaxes[row]:
                ax.text(0.5, 0.5, 'Data not found', ha='center', va='center',
                        transform=ax.transAxes, fontsize=6.5, color='gray')
                ax.axis('off')
            row_data_list.append(None)
            continue

        row_data_list.append(data)
        group_color_maps[cfg['dataset_group']] = data['color_map']

        # Étiquette de ligne (à gauche, rotation 90°)
        sp = gs_main[row, 0].get_position(fig)
        fig.text(
            0.005,
            sp.y0 + sp.height / 2,
            data['label'],
            ha='left', va='center', fontsize=6.5, fontweight='bold',
            rotation=90, color='#333333',
        )

        # Col 1 : réseau
        draw_gene_subgraph(axes_net[row], data['G_net'], data['gene'],
                           data['max_int'])

        # Col 2 : barplot (labels en haut pour la 1ère ligne seulement)
        draw_barplot(axes_bar[row], data['prop_df'], data['color_map'],
                     show_xlabels='top' if row == 0 else 'none',
                     sim_stds=data.get('sim_stds'))

        # --- Affichage test chi2 (poolé sur N_SIMS runs) ---
        if data.get('chi2_pval') is not None:
            pval = data['chi2_pval']
            txt = f"$\\chi^2$ p = {pval:.1e}"

            axes_bar[row].text(
                0.75, 1.02,
                txt,
                transform=axes_bar[row].transAxes,
                ha='center', va='bottom',
                fontsize=5.2,
                color='#222222',
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor='#F8F8F8',
                    edgecolor='#BBBBBB',
                    linewidth=0.6,
                    alpha=0.9
                )
            )

        # Col 3 : UMAPs
        _pt_size = 1 if cfg['dataset_group'] == 'Schiebinger' else 3
        rotate_x = row in []
        rotate_y = row in [2]

        draw_umap_panel(umap_subaxes[row],
                        data['af_umap'], data['as_umap'], data['ap_umap'],
                        data['color_map'],
                        point_size=_pt_size,
                        rotate_x=rotate_x, rotate_y=rotate_y)

        # Sous-titres UMAP uniquement pour la première ligne
        if row == 0:
            for ax, subtitle in zip(umap_subaxes[row], ['Reference data', 'Sim WT', 'Sim perturb']):
                ax.set_title(subtitle, fontsize=6, color='#555555', pad=3)

    # Forcer le rendu pour que get_position() retourne les vraies coordonnées
    fig.canvas.draw()

    # ── Étiquettes de panneaux A–E (coin haut-gauche de chaque ligne)
    panel_labels = list('ABCDE')
    for row, label in enumerate(panel_labels[:n_rows]):
        sp = gs_main[row, 0].get_position(fig)
        fig.text(
            0.01,                        # tout à gauche
            sp.y0 + sp.height,           # haut de la ligne
            label,
            ha='left', va='top',
            fontsize=11, fontweight='bold', color='#111111',
        )

    # ── Légendes par groupe de dataset + traits séparateurs
    # Ordre : légende d'abord (juste sous le contenu), puis trait gris en dessous.
    for row, cfg in enumerate(perturbations):
        if not cfg['last_in_group']:
            continue

        color_map = group_color_maps.get(cfg['dataset_group'], {})
        if not color_map:
            continue

        sp = gs_main[row, 0].get_position(fig)
        y_bottom = sp.y0

        # x-center of the barplot column (col 1)
        sp_bar = gs_main[row, 1].get_position(fig)
        bar_x_center = (sp_bar.x0 + sp_bar.x1) / 2

        # ── 1) Légende des types cellulaires, ancrée au bas de la ligne de contenu
        handles = [mpatches.Patch(color=col, label=cat)
                   for cat, col in color_map.items()]
        legend_y = y_bottom - 0.001
        fig.legend(
            handles,
            [h.get_label() for h in handles],
            loc='upper center',
            ncol=min(len(handles), 8),
            fontsize=5.5,
            frameon=False,
            bbox_to_anchor=(bar_x_center, legend_y),
            title=None,              # pas de titre "Cell types —"
            handlelength=1.0,
            handleheight=0.8,
            columnspacing=0.8,
        )

        # ── 2) Trait gris en dessous de la légende (~1.8% plus bas)
        sep_y = y_bottom - 0.020
        line = mlines.Line2D(
            [0.03, 0.97], [sep_y, sep_y],
            transform=fig.transFigure,
            color='#BBBBBB', linewidth=0.9, linestyle='-',
            zorder=10,
        )
        fig.add_artist(line)

    # ── Légende des arêtes réseau (coin bas-gauche)
    # legend_net = [
    #     mpatches.Patch(color='#2ECC71', label='Activation'),
    #     mpatches.Patch(color='#E74C3C', label='Inhibition'),
    # ]
    # fig.legend(
    #     legend_net, [h.get_label() for h in legend_net],
    #     loc='lower left', fontsize=6, frameon=False,
    #     bbox_to_anchor=(0.01, 0.0),
    #     title='Regulation', title_fontsize=6.5,
    #     handlelength=1.0,
    # )

    # ── Sauvegarde
    plt.savefig('figure_7.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print("Saved figure_7.png")
    try:
        from PIL import Image
        Image.open('figure_7.png').convert('RGB').save('figure_7.pdf', 'PDF', resolution=300)
        print("Saved figure_7.pdf")
    except ImportError:
        print("(PIL not available — skipping PDF export)")
    return fig


# ──────────────────────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    make_figure(
        perturbations=PERTURBATIONS,
        save_path='figure_7.pdf',
    )