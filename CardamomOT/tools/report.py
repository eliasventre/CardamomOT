"""
Final post-analysis PDF report: generative model, GRN top regulators and KO/OV predictions.

Gathers in one multi-page PDF what the utils notebooks do separately
(compare_cell_types, plot_data_to_sim, compare_cell_types_across_KOV, plot_data_to_sim_KOV)
plus a top-regulator view of the inferred network (violin plots + per-regulator subgraphs).
"""
import json
import os
import datetime
import textwrap
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import NearestNeighbors

from .characterize_cell_type import (train_classifier, predict_cell_types, read_model_stages, attach_samples,
                                     predict_out_of_fold, reference_labels, predicted_key)
from .embedding import data_embedding, Embedder, embedding_name
from .velocity import knn_velocity_embedding as _knn_velocity_embedding
from ..config import find_data_file, resolve_cell_type_obs
from ..inputs import input_dir
from ..inference.trajectory import kon_ref_vector
from ..model.base import load_network_array, drop_network_axis, load_state_growth, trajectory_pairs
from ..inference.proliferations import ProliferationMLP, growth_path_states, quadrature, interval_stimulus

A4_LANDSCAPE = (11.69, 8.27)
REF_COLOR = '#555555'
SIM_COLOR = '#C94040'
ACT_COLOR = '#2ECC71'
INH_COLOR = '#E74C3C'
REG_COLOR = '#4C9BE8'
STIM_COLOR = '#F39C12'
TGT_COLOR = '#A569BD'
LABEL_KEY = 'cell_type'
LABEL_PRED = predicted_key(LABEL_KEY)  # out-of-fold predicted labels of the observed cells


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _dense(X):
    return X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)


def _read(path):
    return ad.read_h5ad(path) if os.path.exists(path) else None


def _preprocess(X, norm, log):
    X = _dense(X).astype(float)
    if norm:
        X = X / np.maximum(X.sum(axis=1, keepdims=True), 1e-12) * 1e4
    if log:
        X = np.log1p(X)
    return X


def _cell_type_colors(categories):
    cmap = plt.get_cmap('tab10' if len(categories) <= 10 else 'tab20')
    return {c: cmap(i % cmap.N) for i, c in enumerate(categories)}


def _time_subsample(t, n_max, rng):
    """Indices of a stratified-by-time subsample of at most n_max elements (all if None)."""
    n = len(t)
    if n_max is None or n <= n_max:
        return np.arange(n)
    idx = []
    for tv in np.unique(t):
        w = np.where(t == tv)[0]
        k = max(1, int(round(n_max * len(w) / n)))
        idx.append(rng.choice(w, size=min(k, len(w)), replace=False))
    return np.sort(np.concatenate(idx))


def _proportions(adata, categories, key=LABEL_KEY):
    counts = adata.obs[key].astype(str).value_counts(normalize=True) * 100
    return counts.reindex(categories).fillna(0.0)


def _with_predicted(props, data, categories, name='Data'):
    """Proportion rows {stage: proportions} with 'Data (predicted)' (out-of-fold labels) after the data row."""
    out = {}
    for k, v in props.items():
        out[k] = v
        if k == name and data is not None and LABEL_PRED in data.obs:
            out[f'{name} (predicted)'] = _proportions(data, categories, LABEL_PRED)
    return out


def _proportions_by_time(adata, categories, key=LABEL_KEY):
    df = pd.DataFrame({'time': pd.to_numeric(adata.obs['time']).values,
                       'ct': adata.obs[key].astype(str).values})
    tab = pd.crosstab(df['time'], df['ct'], normalize='index') * 100
    return tab.reindex(columns=categories).fillna(0.0)


# ---------------------------------------------------------------------------
# Generic drawing helpers
# ---------------------------------------------------------------------------

def _panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=11, fontweight='bold', clip_on=False)


def _clean_umap_ax(ax, title):
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)


def _umap_time(ax, coords, times, vmin, vmax, title, bg=None):
    if bg is not None:
        ax.scatter(bg[:, 0], bg[:, 1], c='#E6E6E6', s=3, linewidths=0, rasterized=True)
    sca = ax.scatter(coords[:, 0], coords[:, 1], c=times, cmap='viridis', vmin=vmin, vmax=vmax,
                     s=4, linewidths=0, rasterized=True)
    _clean_umap_ax(ax, title)
    return sca


def _umap_celltype(ax, coords, labels, color_map, title, bg=None, s=4, alpha=1.0):
    if bg is not None:
        ax.scatter(bg[:, 0], bg[:, 1], c='#E6E6E6', s=3, linewidths=0, rasterized=True)
    labels = np.asarray(labels).astype(str)
    for cat, col in color_map.items():
        m = labels == cat
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], color=col, s=s, alpha=alpha, linewidths=0, rasterized=True)
    _clean_umap_ax(ax, title)


def _stacked_bars(ax, prop_df, color_map, title):
    bottom = np.zeros(len(prop_df))
    x = np.arange(len(prop_df))
    for cat in prop_df.columns:
        vals = prop_df[cat].values
        ax.bar(x, vals, bottom=bottom, color=color_map[cat], width=0.75, edgecolor='white', linewidth=0.3)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(prop_df.index, rotation=35, ha='right', fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_ylabel('% of cells', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_title(title, fontsize=9)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def _proportions_over_time(ax, tab_ref, tab_alt, color_map, title, ref_label, alt_label):
    for cat in color_map:
        if cat in tab_ref.columns:
            ax.plot(tab_ref.index, tab_ref[cat], color=color_map[cat], ls='--', lw=1.0, alpha=0.8)
        if cat in tab_alt.columns:
            ax.plot(tab_alt.index, tab_alt[cat], color=color_map[cat], ls='-', lw=1.6, marker='o', ms=2.5)
    ax.set_xlabel('time', fontsize=8)
    ax.set_ylabel('% of cells', fontsize=8)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9, loc='left')
    ax.legend(handles=[Line2D([0], [0], color='k', ls='--', lw=1, label=ref_label),
                       Line2D([0], [0], color='k', ls='-', lw=1.6, label=alt_label)],
              fontsize=6.5, frameon=False, loc='lower right', ncol=2, bbox_to_anchor=(1.0, 0.98))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def _sample_labels(A, idx=None, key='dataset_id'):
    """Sample of the cells of A (idx), or None with fewer than two samples."""
    if A is None or key not in A.obs or A.obs[key].astype(str).nunique() < 2:
        return None
    lab = A.obs[key].astype(str).values
    return lab if idx is None else lab[idx]


def _umap_sample(ax, coords, samples, title, bg=None, s=4):
    """Embedding coloured by sample, with its legend."""
    if bg is not None:
        ax.scatter(bg[:, 0], bg[:, 1], c='#E6E6E6', s=3, linewidths=0, rasterized=True)
    cats = sorted(np.unique(samples))
    cmap = plt.get_cmap('Set1' if len(cats) <= 9 else 'tab20')
    for i, c in enumerate(cats):
        m = samples == c
        ax.scatter(coords[m, 0], coords[m, 1], color=cmap(i % cmap.N), s=s, linewidths=0, alpha=0.7,
                   rasterized=True, label=c)
    _clean_umap_ax(ax, title)
    ax.legend(fontsize=6, frameon=False, markerscale=2.5, loc='best')


def _celltype_legend(fig, color_map, y=0.01):
    handles = [mpatches.Patch(color=c, label=k) for k, c in color_map.items()]
    if handles:
        fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 8),
                   frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, y))


_EMBEDDING = {'name': 'UMAP'}  # name of the embedding method of the report, used in the titles


_PAGE_SUFFIX = ['']  # label of the network condition of the section being drawn (condition views)


def _page_title(fig, title, subtitle=None):
    title = title.replace('UMAP', _EMBEDDING['name'])
    if _PAGE_SUFFIX[0] and _PAGE_SUFFIX[0] not in title:
        title += _PAGE_SUFFIX[0]
    subtitle = subtitle.replace('UMAP', _EMBEDDING['name']) if subtitle else subtitle
    # Long titles shrink to fit the page width; subtitles wrap on at most 2 lines
    size = 14 if len(title) <= 90 else max(9.0, 14 * 90 / len(title))
    fig.suptitle(title, fontsize=size, fontweight='bold', x=0.04, ha='left', y=0.985)
    if subtitle:
        lines = textwrap.wrap(subtitle, 160)
        if len(lines) > 1:
            lines = textwrap.wrap(subtitle, 165, max_lines=2, placeholder=" …")
        fig.text(0.04, 0.955, '\n'.join(lines), fontsize=8.5 if len(lines) == 1 else 7.5, color='#555555',
                 ha='left', va='top', linespacing=1.3)


def _error_page(pdf, title, msg):
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, title)
    fig.text(0.05, 0.85, 'Section unavailable:', fontsize=11, fontweight='bold')
    fig.text(0.05, 0.80, '\n'.join(textwrap.wrap(str(msg), 140)), fontsize=9, va='top', family='monospace')
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------------------
# GRN helpers (adapted from results_article/figures/figureS1-3_elias.ipynb)
# ---------------------------------------------------------------------------

def _mutual_top(M, k):
    """E[i, j]: i -> j is among the k strongest (|w|) targets of i and among the k strongest regulators of j."""
    A = np.abs(M)
    n_r, n_t = A.shape
    out = np.zeros(A.shape, bool)
    inc = np.zeros(A.shape, bool)
    out[np.arange(n_r)[:, None], np.argsort(-A, axis=1)[:, :min(k, n_t)]] = True
    inc[np.argsort(-A, axis=0)[:min(k, n_r)], np.arange(n_t)[None, :]] = True
    return out & inc & (A > 0)


def _regulator_subgraph(matrix, gene_names, gene, top_targets=10, keep=None, incoming=False):
    """Star of `gene`: its targets (or its regulators if incoming) among the edges allowed by keep."""
    idx = gene_names.index(gene)
    w = matrix[:, idx] if incoming else matrix[idx, :]
    if keep is not None:
        w = np.where(keep[:, idx] if incoming else keep[idx, :], w, 0.0)
    series = pd.Series(w, index=gene_names).drop(gene, errors='ignore')
    series = series[series != 0]
    top_idx = series.abs().nlargest(top_targets).index
    G = nx.DiGraph()
    G.add_node(gene)
    for other in top_idx:
        G.add_edge(*((other, gene) if incoming else (gene, other)), weight=float(series[other]))
    return G


def _draw_regulator_subgraph(ax, G, gene, max_intensity, center_color, title, highlight=(), empty='no outgoing edge',
                             scale=1.0):
    # scale: size of nodes, labels and edges (smaller panels)
    if G.number_of_edges() == 0:
        ax.text(0.5, 0.5, f"{gene}\n({empty})", ha='center', va='center',
                transform=ax.transAxes, fontsize=7 * scale, color='gray')
        ax.set_title(title, fontsize=8 * scale, fontweight='bold')
        ax.axis('off')
        return

    # Central regulator fixed at the origin, targets pushed beyond a minimum radius
    pos = nx.spring_layout(G, pos={gene: (0.0, 0.0)}, fixed=[gene], k=2.5, iterations=100, seed=42)
    for node in pos:
        if node == gene:
            continue
        d = np.hypot(*pos[node])
        if d < 0.6:
            pos[node] = pos[node] * (0.6 / max(d, 1e-6))

    node_colors = [center_color if n == gene else '#FAD7A0' if str(n).startswith('Stimulus')
                   else ('#FDEBD0' if n in highlight else '#EDEDED') for n in G.nodes]
    node_sizes = [(900 if n == gene else 520) * scale ** 2 for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax,
                           edgecolors='#999999', linewidths=0.4)
    nx.draw_networkx_labels(G, pos, font_size=6 * max(scale, 0.85), ax=ax)

    e_pos = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    e_neg = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0]
    width = lambda el: [(0.4 + 3.0 * abs(G[u][v]['weight']) / max_intensity) * scale for u, v in el]
    if e_pos:
        nx.draw_networkx_edges(G, pos, edgelist=e_pos, edge_color=ACT_COLOR, width=width(e_pos),
                               arrows=True, arrowsize=9 * scale, connectionstyle='arc3,rad=0.1',
                               min_target_margin=12 * scale, ax=ax)
    if e_neg:
        nx.draw_networkx_edges(G, pos, edgelist=e_neg, edge_color=INH_COLOR, width=width(e_neg),
                               arrows=True, arrowstyle='-[,widthB=0.8,lengthB=0.0',
                               connectionstyle='arc3,rad=0.1', min_target_margin=12 * scale, ax=ax)
    ax.margins(0.18)
    ax.set_title(title, fontsize=8 * scale, fontweight='bold')
    ax.axis('off')


def _labelled_violin(ax, values, names, top_idx, title, ylabel, extra=None, min_gap=None):
    parts = ax.violinplot(values, showmeans=False, showextrema=False, widths=0.7)
    for b in parts['bodies']:
        b.set_facecolor('#BBD3EE'); b.set_edgecolor('#4C7FB8'); b.set_alpha(0.8)
    rng = np.random.default_rng(0)
    ax.scatter(1 + rng.uniform(-0.06, 0.06, len(values)), values, s=4, color='#7F8C8D', alpha=0.5, zorder=2)
    ax.scatter(np.ones(len(top_idx)), values[top_idx], s=14, color=REG_COLOR, edgecolor='k', lw=0.4, zorder=3)

    # Labels on the right, pushed down so that consecutive ones are at least min_gap apart
    items = [(values[i], f"{rank + 1}. {names[i]}", dict(fontsize=6.5), '#999999') for rank, i in enumerate(top_idx)]
    for name, y in (extra or []):
        ax.scatter([1], [y], marker='*', s=90, color=STIM_COLOR, edgecolor='k', lw=0.4, zorder=4)
        items.append((y, name, dict(fontsize=7, color='#B9770E', fontweight='bold'), STIM_COLOR))
    items.sort(key=lambda it: -it[0])
    span = (np.max(values) - np.min(values)) or 1.0
    min_gap = min(0.04, 0.95 / max(len(items), 1)) * span if min_gap is None else min_gap  # all labels in range
    y_lab = None
    for y, text, kw, lc in items:
        y_lab = y if y_lab is None else min(y, y_lab - min_gap)
        ax.annotate(text, xy=(1, y), xytext=(1.3, y_lab), va='center', ha='left', **kw,
                    arrowprops=dict(arrowstyle='-', lw=0.3, color=lc, shrinkA=0, shrinkB=2))

    ax.set_xlim(0.55, 1.9)
    ax.set_xticks([])
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_title(title, fontsize=9.5, fontweight='bold')
    for sp in ('top', 'right', 'bottom'):
        ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

class _ReportData:
    """Loads every pipeline output needed by the report once."""

    def __init__(self, p, split, stim, prior, norm, log, perturbations, n_umap, seed, depth_emb=True,
                 emb_method='umap', classifier_method='random_forest'):
        self.classifier_method = classifier_method
        self.p, self.split, self.stim, self.prior = p, split, stim, prior
        self.norm, self.log = norm, log
        self.depth_emb, self.emb_method, self.emb_name = bool(depth_emb), str(emb_method).lower(), embedding_name(emb_method)
        self.no_ref_depth = []  # stages drawn at the cells' depth without their obs['depth_factor']
        self.rng = np.random.default_rng(seed)
        self.n_umap = n_umap
        tag = f'stim{stim}_prior{prior}'
        cdir = os.path.join(p, 'cardamomOT')
        self.tag = tag

        self.adata_data = ad.read_h5ad(os.path.join(p, 'Data', f'data_{split}.h5ad'))
        from ..inputs import depth_factor_used
        self.use_depth = depth_factor_used(p)
        # Counts / depth factor only if the run modelled the depth: otherwise the model describes the raw counts and
        # data and model draws are both shown raw (one transformation for both)
        if self.depth_emb and not self.use_depth:
            print("[report] cell_depth_for_representation ignored: the run does not use the depth factors "
                  "(use_depth_factor = False), data and model shown as raw counts")
        self.depth_emb = self.depth_emb and self.use_depth
        self.genes = list(self.adata_data.var_names)
        # Genes of each network condition (select_genes_per_condition); None: one gene space
        from ..condition_genes import condition_gene_sets
        self.condition_genes = condition_gene_sets(self.adata_data)
        self.parent, self.samples, self.condition = None, None, None
        # Data = observed cells; Reference = RNA of the trajectories: states drawn at each time among the observed
        # cells (target marginal of the OT), hence the observed composition, with or without proliferation
        # Model stages as the notebooks read them (read_model_stages: samples attached, virtual states left out)
        names = (sorted(self.adata_data.obs['dataset_id'].astype(str).unique()) if 'dataset_id' in self.adata_data.obs else [])
        self.stages = {'Data': self.adata_data.copy(),  # shown as configured (_mrna_view); adata_data stays raw
                       **read_model_stages(p, tag, names)}
        missing = [k for k, v in self.stages.items() if v is None]
        if missing:
            raise FileNotFoundError(f"Missing check_sim_to_data outputs with tag '{tag}' for: {missing}. "
                                    f"Run check_sim_to_data.py with the same --stimulus/--prior.")
        self.prot = {'Trajectories': _read(os.path.join(cdir, f'adata_prot_traj_{tag}.h5ad')),
                     'Simulation': _read(os.path.join(cdir, f'adata_prot_simul_{tag}.h5ad'))}

        self.beta_states = self.stages['NB mixture']  # one row per trajectory state (dynamics)
        self.sim_prolif = bool(self.stages['Simulation'].uns.get('proliferation', False))

        # Perturbations listed in KO_OV_Stim_simulate.txt: (label, description, adata or None)
        self.perturbations, self.perturbed_genes = [], set()
        for label, desc, genes in perturbations:
            self.perturbations.append((label, desc, _read(os.path.join(cdir, f'adata_sim_{label}_{tag}.h5ad'))))
            self.perturbed_genes.update(genes)

        # mRNA as shown (cell_depth_for_representation): before the classifier, so that every stage is in one space
        for k, A in self.stages.items():
            self.view(A, 'observed' if k == 'Data' else 'reference' if k == 'Reference' else 'model', k)
        for label, _, A in self.perturbations:
            self.view(A, 'model', label)

        # Sample of each perturbed simulated cell and protein state (model stages: read_model_stages)
        samples_path = os.path.join(cdir, 'data_samples.npy')
        samples_traj = np.load(samples_path).astype(int) if os.path.exists(samples_path) else None
        for _, _, A in self.perturbations:
            attach_samples(A, samples_traj, names)
        for A in self.prot.values():
            attach_samples(A, samples_traj, names)

        # Network conditions: samples of each condition (network_conditions.json, samples in sorted order)
        self.conditions = {}
        cond_json = os.path.join(cdir, 'network_conditions.json')
        if os.path.exists(cond_json) and names:
            import json
            info_c = json.load(open(cond_json))
            sc, cn = np.asarray(info_c['sample_conditions'], dtype=int), list(info_c['conditions'])
            smp = [str(x) for x in info_c.get('samples', names)]
            if len(cn) > 1 and len(sc) == len(smp):
                self.conditions = {str(c): [smp[k] for k in np.flatnonzero(sc == i)] for i, c in enumerate(cn)}

        # Cell types: one classifier per sample (classifier_method) trained on its observed cells, applied in memory
        # (h5ad files untouched); a sample held out of the inference uses the classifier of its reference sample
        # (perturbation_inference), else the closest name
        self.has_ct = LABEL_KEY in self.adata_data.obs
        if self.has_ct:
            from ..inputs import removed_samples
            self.categories = self.adata_data.obs[LABEL_KEY].astype(str).unique().tolist()
            self.color_map = _cell_type_colors(self.categories)
            clf_kw = dict(assign=removed_samples(p)[1], method=classifier_method)
            clf = train_classifier(self.stages['Data'], label_key=LABEL_KEY, **clf_kw)
            for A in [a for k, a in self.stages.items() if k != 'Data'] + [a for _, _, a in self.perturbations if a is not None]:
                predict_cell_types(A, clf, label_key=LABEL_KEY)
            self.clf = clf
            # Data (predicted): out-of-fold labels of the observed cells, the reference of the classified stages (the
            # annotated labels differ from them by the errors of the classifier); Reference: those of its real cells
            D_ = self.stages['Data']
            D_.obs[LABEL_PRED] = pd.Categorical(predict_out_of_fold(D_, label_key=LABEL_KEY, **clf_kw))
            ref = reference_labels(self.stages['Reference'], D_.obs[LABEL_PRED].astype(str).values)
            if ref is not None:
                self.stages['Reference'].obs[LABEL_KEY] = pd.Categorical(ref)
        else:
            self.categories, self.color_map = [], {}

        # Held-out test cells (infer_test.py): observed data and predictions; samples removed from the
        # inference (remove_from_inference) are validated separately (adata_sim_validation_<sample>_*)
        self.test, self.validation = None, {}
        test_path = os.path.join(p, 'Data', 'data_test.h5ad')
        data_test = ad.read_h5ad(test_path) if os.path.exists(test_path) else None
        if data_test is not None and 'dataset_id' in data_test.obs:
            from ..inputs import removed_samples
            removed, refs = removed_samples(p, present=data_test.obs['dataset_id'].astype(str).unique())
            sid = data_test.obs['dataset_id'].astype(str)
            for r in removed:
                sim = _read(os.path.join(cdir, f'adata_sim_validation_{r}_{tag}.h5ad'))
                if sim is not None:
                    obs_r = self.view(data_test[(sid == r).to_numpy()].copy(), 'observed', f'observed {r}')
                    self.view(sim, 'model', f'validation {r}')
                    if self.has_ct:
                        # Classifier of the reference sample of r (the sample whose cells start the simulation)
                        ref_r = str(sim.uns.get('reference_sample', refs.get(r, ''))) or None
                        predict_cell_types(sim, self.clf, label_key=LABEL_KEY, use=ref_r)
                        if LABEL_KEY not in obs_r.obs:
                            predict_cell_types(obs_r, self.clf, label_key=LABEL_KEY, use=ref_r)
                        # Predicted labels of the observed cells of r (not seen by the classifier)
                        obs_r.obs[LABEL_PRED] = predict_cell_types(obs_r.copy(), self.clf, label_key=LABEL_KEY,
                                                                   use=ref_r).obs[LABEL_KEY].values
                    self.validation[r] = (obs_r, sim, str(sim.uns.get('reference_sample', refs.get(r, ''))))
            if removed:
                data_test = data_test[~sid.isin(removed).to_numpy()].copy()
        if data_test is not None and data_test.n_obs and data_test.obs['time'].nunique() > 1:
            stages = {'Test data': self.view(data_test, 'observed', 'test data'),
                      'NB mixture': self.view(_read(os.path.join(cdir, f'adata_beta_test_{tag}.h5ad')), 'model', 'test NB'),
                      'Network': self.view(_read(os.path.join(cdir, f'adata_theta_test_{tag}.h5ad')), 'model', 'test network'),
                      'Simulation': self.view(_read(os.path.join(cdir, f'adata_sim_test_{tag}.h5ad')), 'model', 'test simulation')}
            if all(v is not None for v in stages.values()):
                self.test = stages
                st_path = os.path.join(cdir, 'data_samples_test.npy')
                names_t = sorted(data_test.obs['dataset_id'].astype(str).unique()) if 'dataset_id' in data_test.obs else []
                for k, A in stages.items():
                    if k != 'Test data':
                        attach_samples(A, np.load(st_path).astype(int) if os.path.exists(st_path) else None, names_t)
                if self.has_ct:
                    for k, A in stages.items():
                        if k != 'Test data' or LABEL_KEY not in A.obs:
                            predict_cell_types(A, self.clf, label_key=LABEL_KEY)
                    # Predicted labels of the test cells (not seen by the classifier): reference of the test stages
                    T_ = stages['Test data']
                    T_.obs[LABEL_PRED] = predict_cell_types(T_.copy(), self.clf, label_key=LABEL_KEY).obs[LABEL_KEY].values

        if self.no_ref_depth:
            print(f"[report] Warning: no obs['depth_factor'] for {sorted(set(self.no_ref_depth))} (shown as drawn): rerun "
                  "check_sim_to_data / check_KOV_to_sim / infer_test for cell_depth_for_representation")
        self._fit_umap()

    def view(self, A, kind, name=''):
        """
        mRNA of A as shown in the report, in place. With cell_depth_for_representation: counts / obs['depth_factor'],
        the same transformation for the observed cells, the reference trajectories (depth of their real cell) and the
        model draws (taken at the depth of the cell they mimic, hence the same noise as the data); draws without
        depth factor (run without use_depth_factor) are shown as drawn. Otherwise the counts as drawn.
        """
        if A is None or not self.depth_emb:
            return A
        if 'depth_factor' in A.obs:
            A.X = _dense(A.X).astype(float) / A.obs['depth_factor'].to_numpy(dtype=float)[:, None]
        elif (kind == 'model' and self.use_depth) or (kind == 'reference' and 'depth_factor' in self.adata_data.obs):
            self.no_ref_depth.append(name)
        return A

    def _subsample(self, A):
        """Stratified-by-time subsample of at most n_umap cells (indices)."""
        return _time_subsample(pd.to_numeric(A.obs['time']).values, self.n_umap, self.rng)

    def _fit_umap(self):
        # One mRNA embedding learned on the observed cells (reference), every model output projected onto it; one
        # protein embedding learned on the protein trajectories (prot_embedding)
        self.sub = {k: self._subsample(A) for k, A in self.stages.items()}
        self.reducer = Embedder(self.emb_method, seed=42).fit(
            _preprocess(self.stages['Data'].X[self.sub['Data']], self.norm, self.log))
        self.umap = {k: (self.reducer.embedding_ if k == 'Data' else
                         self.reducer.transform(_preprocess(A.X[self.sub[k]], self.norm, self.log)))
                     for k, A in self.stages.items()}
        self.prot_reducer = None
        Pt = self.prot.get('Trajectories')
        if Pt is not None:
            self.prot_sub = self._subsample(Pt)
            self.prot_reducer = Embedder(self.emb_method, seed=42).fit(_dense(Pt.X[self.prot_sub]).astype(float))
        # Test cells and predictions projected onto the same embedding
        self.test_sub, self.test_umap = {}, {}
        for k, A in (self.test or {}).items():
            self.test_sub[k] = self._subsample(A)
            self.test_umap[k] = self.reducer.transform(_preprocess(A.X[self.test_sub[k]], self.norm, self.log))
        self.val_umap = {}
        for r, (obs_r, sim, _) in self.validation.items():
            t_obs = np.unique(self.times(obs_r))
            sim_o = sim[np.isin(self.times(sim), t_obs)]
            self.val_umap[r] = [(A, idx, self.reducer.transform(_preprocess(A.X[idx], self.norm, self.log)))
                                for A in (obs_r, sim_o) for idx in [self._subsample(A)]]
        self.pert_sub, self.pert_umap = {}, {}
        for label, _, A in self.perturbations:
            if A is None:
                continue
            self.pert_sub[label] = self._subsample(A)
            self.pert_umap[label] = self.reducer.transform(_preprocess(A.X[self.pert_sub[label]], self.norm, self.log))

    def view_of(self, samples=None):
        """
        Stages, subsamples and embeddings of the cells of samples (one network condition; None = every cell, the
        embeddings of _fit_umap): mRNA embedding learned once on the observed cells of the samples, every stage of
        these samples projected onto it, protein embedding learned once on their trajectories. Cached; None when a
        stage has no cell of the samples.
        """
        if samples is None:
            return dict(stages=self.stages, sub=self.sub, umap=self.umap, prot=self.prot,
                        prot_red=self.prot_reducer, prot_sub=getattr(self, 'prot_sub', None))
        key = tuple(samples)
        cache = self.__dict__.setdefault('_views', {})
        if key not in cache:
            stages = {k: _restrict(A, samples) for k, A in self.stages.items()}
            if any(A is None or A.n_obs == 0 for A in stages.values()):
                cache[key] = None
                return None
            sub = {k: self._subsample(A) for k, A in stages.items()}
            red = Embedder(self.emb_method, seed=42).fit(_preprocess(stages['Data'].X[sub['Data']], self.norm, self.log))
            umap = {k: (red.embedding_ if k == 'Data' else
                        red.transform(_preprocess(A.X[sub[k]], self.norm, self.log))) for k, A in stages.items()}
            prot = {k: _restrict(self.prot.get(k), samples) for k in ('Trajectories', 'Simulation')}
            Pt, prot_red, prot_sub = prot['Trajectories'], None, None
            if Pt is not None and Pt.n_obs and prot['Simulation'] is not None and prot['Simulation'].n_obs:
                prot_sub = self._subsample(Pt)
                prot_red = Embedder(self.emb_method, seed=42).fit(_dense(Pt.X[prot_sub]).astype(float))
            cache[key] = dict(stages=stages, sub=sub, umap=umap, prot=prot, prot_red=prot_red, prot_sub=prot_sub)
        return cache[key]

    def condition_view(self, c, samples):
        """
        Report data restricted to the samples of network condition c and to its genes: every stage, protein,
        perturbation, test and validation AnnData of these samples (columns: the genes of the condition with
        select_genes_per_condition), with their own embeddings (learned on the observed cells of the condition).
        Cached; the pages called on it are those of a run on this condition alone.
        """
        cache = self.__dict__.setdefault('_condition_views', {})
        if c in cache:
            return cache[c]
        import copy
        Rc = copy.copy(self)
        Rc.parent, Rc.samples, Rc.condition = self, [str(x) for x in samples], str(c)
        genes = self.genes if self.condition_genes is None else self.condition_genes.get(str(c), self.genes)
        Rc.gene_cols = np.flatnonzero(np.isin(self.genes, genes))
        Rc.genes = [self.genes[i] for i in Rc.gene_cols]
        cols = Rc.gene_cols

        def restrict(A):
            A = _restrict(A, Rc.samples)
            return None if A is None else A[:, cols].copy()

        Rc.cell_mask = self.adata_data.obs['dataset_id'].astype(str).isin(Rc.samples).to_numpy() \
            if 'dataset_id' in self.adata_data.obs else np.ones(self.adata_data.n_obs, bool)
        Rc.adata_data = restrict(self.adata_data)
        Rc.stages = {k: restrict(A) for k, A in self.stages.items()}
        Rc.prot = {k: restrict(A) for k, A in self.prot.items()}
        Rc.perturbations = [(l, d, restrict(A)) for l, d, A in self.perturbations]
        Rc.test = None
        if self.test:
            test = {k: restrict(A) for k, A in self.test.items()}
            Rc.test = test if all(A is not None and A.n_obs for A in test.values()) else None
        Rc.validation = {}
        for r, (obs_r, sim, ref) in self.validation.items():
            cond_r = (str(obs_r.obs['network_condition'].iloc[0]) if 'network_condition' in obs_r.obs and obs_r.n_obs
                      else None)
            if cond_r == str(c) or (cond_r is None and ref in Rc.samples):
                Rc.validation[r] = (obs_r[:, cols].copy(), sim[:, cols].copy(), ref)
        Rc.conditions = {}
        Rc.__dict__.pop('_views', None)
        Rc._teaser = None
        Rc._fit_umap()
        cache[c] = Rc
        return Rc

    def times(self, A, idx=None):
        t = pd.to_numeric(A.obs['time']).values
        return t if idx is None else t[idx]

    def labels(self, A, idx=None):
        lab = A.obs[LABEL_KEY].astype(str).values
        return lab if idx is None else lab[idx]


def _cover_page(pdf, R, info, perturbations_status):
    fig = plt.figure(figsize=A4_LANDSCAPE)
    fig.text(0.06, 0.88, 'CardamomOT — analysis report', fontsize=24, fontweight='bold')
    fig.text(0.06, 0.835, os.path.abspath(R.p), fontsize=10, color='#555555', family='monospace')
    fig.text(0.06, 0.805, datetime.datetime.now().strftime('Generated on %Y-%m-%d %H:%M'), fontsize=9, color='#777777')

    rows = [('Split', R.split), ('Stimulus penalisation', f'{R.stim}'), ('Prior penalisation', f'{R.prior}'),
            ('Genes', f'{len(R.genes)}'), ('Stimuli', f"{info['n_stimuli']}"),
            ('Observed cells', f'{R.adata_data.n_obs}'),
            ('Timepoints', ', '.join(f'{t:g}' for t in np.sort(np.unique(R.times(R.adata_data))))),
            ('Cell types', ', '.join(R.categories) if R.has_ct else "— (no obs['cell_type'])"),
            ('Network shown', "inter_simul.npy"),
            ('Embeddings', f"{R.emb_name}; mRNA: log1p={R.log}, normalise={R.norm}, cell_depth_for_representation={R.depth_emb} "
                           + ("(counts / depth factor, model draws taken at the depth of the cells they mimic)" if R.depth_emb
                              else "(raw counts, model draws at the depth of the cells)"))]
    y = 0.73
    fig.text(0.06, y + 0.02, 'Run', fontsize=13, fontweight='bold')
    for k, v in rows:
        y -= 0.033
        lines = textwrap.wrap(v, 55) or ['']
        fig.text(0.07, y, k, fontsize=9, fontweight='bold', va='top')
        fig.text(0.25, y, '\n'.join(lines), fontsize=9, va='top', linespacing=1.3)
        y -= 0.022 * (len(lines) - 1)

    x0 = 0.60
    fig.text(x0, 0.75, 'Perturbations (perturbation_simulation)', fontsize=13, fontweight='bold')
    y = 0.72
    if not perturbations_status:
        fig.text(x0 + 0.01, y - 0.03, 'No KO_OV_Stim_simulate.txt / no perturbation listed.', fontsize=9, color='#777777')
    for i, (label, desc, ok) in enumerate(perturbations_status[:22], start=1):
        y -= 0.03
        lines = textwrap.wrap(f'P{i}  {desc}', 62)
        fig.text(x0 + 0.01, y, '✓' if ok else '✗', fontsize=10, color='#27AE60' if ok else '#C0392B',
                 fontweight='bold', va='top')
        fig.text(x0 + 0.035, y, '\n'.join(lines), fontsize=8.5, va='top', linespacing=1.25)
        y -= 0.02 * (len(lines) - 1)
        if not ok:
            fig.text(x0 + 0.035, y - 0.017, 'not simulated: run simulate_network_KOV + check_KOV_to_sim', fontsize=6.5, color='#C0392B')
            y -= 0.012
    if len(perturbations_status) > 22:
        fig.text(x0 + 0.035, y - 0.03, f'… and {len(perturbations_status) - 22} more', fontsize=8)

    fig.text(0.06, 0.25, 'Contents', fontsize=13, fontweight='bold')
    fig.text(0.07, 0.04,
             'Data — UMAP of all the cells of Data/data.h5ad (time, samples), also on the genes of the model with select_genes; teaser: classical OT vs CardamomOT displacement fields (all times, then per interval), '
             'then cell-type transitions and fate genes of each; list of the genes of the model\n'
             '1. Generative model — UMAPs of data, trajectories, NB mixture, network modes and simulation; cell-type proportions; gene-pair correlations; proteins'
             + (' (per network condition, embeddings learned on its cells)' if R.conditions else '') + '\n'
             '2. Gene regulatory network — regulatory power (violin plots), top-20 regulators' + (' + stimulus' if info['show_stim'] else '')
             + ' and top-20 regulated genes (mutual top-10 edges)\n'
             '3. In-silico perturbations — overview across KO/OV, then one page per perturbation\n'
             '4. Proliferation — prior vs learned net rates, population growth, proteins driving growth; protein dilution at the birth rate\n'
             '5. Learned dynamics — mRNA and protein fields (mechanistic velocity, displacement along trajectories), summary on mRNA'
             + ('\n6. Held-out test cells — predictions with the network fixed vs the test data' if R.test else '')
             + ('\n6. Validation — samples removed from the inference, predicted from a reference sample'
                if R.validation else ''),
             fontsize=9, va='bottom', linespacing=1.6)
    pdf.savefig(fig); plt.close(fig)


def _data_umap_page(pdf, R, seed=0, genes=None):
    """UMAP of every cell of Data/data.h5ad, on the whole transcriptome or restricted to genes (gene selection)."""
    E, obs = data_embedding(R.p, seed=seed, depth=R.depth_emb, method=R.emb_method, genes=genes)
    coords = E['emb']
    times = pd.to_numeric(obs['time']).values if 'time' in obs else np.zeros(len(obs))
    samples = obs['dataset_id'].astype(str).values if 'dataset_id' in obs else None
    multi = samples is not None and len(np.unique(samples)) > 1
    ct = obs[LABEL_KEY].astype(str).values if LABEL_KEY in obs else None
    panels = ['time'] + (['sample'] if multi else []) + (['cell type'] if ct is not None else [])
    order = np.random.default_rng(seed).permutation(len(coords))  # no category drawn on top of the others
    size = float(np.clip(30000 / len(coords), 0.3, 6))
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, 'Data — UMAP of all the cells' + ('' if genes is None else ', genes of the gene selection'),
                f"Data/data.h5ad, {len(coords)} cells: log1p(counts" + (" / depth factor)" if E['depth'] else ")")
                + (f", {len(E['hvg'])} highly variable genes" if genes is None
                   else f", the {len(E['hvg'])} genes of the model (select_genes)")
                + f", PCA ({E['pca'].shape[1]} components), "
                + {'umap': 'kNN graph, UMAP', 'pca': 'first 2 components', 'phate': 'PHATE'}[R.emb_method]
                + ". Stored in cardamomOT/" + ('embedding_data.npz.' if genes is None else 'embedding_data_genes.npz.'))
    gs = gridspec.GridSpec(1, len(panels), figure=fig, left=0.03, right=0.95, top=0.86, bottom=0.2, wspace=0.08)
    for j, kind in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        ax.set_aspect('equal', adjustable='datalim')
        if kind == 'time':
            sca = ax.scatter(coords[order, 0], coords[order, 1], c=times[order], cmap='viridis', s=size,
                             linewidths=0, rasterized=True)
            _clean_umap_ax(ax, 'colour = time')
            cb = fig.colorbar(sca, cax=ax.inset_axes([0.15, -0.06, 0.7, 0.025]), orientation='horizontal')
            cb.set_label('time', fontsize=8); cb.ax.tick_params(labelsize=7)
            continue
        lab = samples if kind == 'sample' else ct
        cats = sorted(np.unique(lab))
        cmap = R.color_map if kind == 'cell type' and R.color_map else _cell_type_colors(cats)
        ax.scatter(coords[order, 0], coords[order, 1], c=[cmap.get(c, '#CCCCCC') for c in lab[order]], s=size,
                   linewidths=0, rasterized=True)
        _clean_umap_ax(ax, f'colour = {kind}')
        handles = [mpatches.Patch(color=cmap.get(c, '#CCCCCC'), label=c) for c in cats]
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=min(3, len(cats)),
                  frameon=False, fontsize=6.5)
    pdf.savefig(fig); plt.close(fig)


ROLE_COLORS = {'perturbed': '#D35400', 'query': '#1F77B4', 'driver': '#C2185B', 'entropy': '#2CA02C', 'steiner': '#8E44AD',
               'regulator': '#7F5539'}


def _gene_list_page(pdf, R):
    """Genes of the model, alphabetical, on as few columns as fit the page with the largest font."""
    genes = sorted(R.genes, key=str.upper)
    roles, linked, info, tags = {}, {}, '', {}
    path = os.path.join(R.p, 'cardamomOT', 'gene_selection_report.csv')
    if os.path.exists(path):
        rep = pd.read_csv(path)
        if 'condition' in rep and R.condition is not None:  # selection per condition: rows of this condition
            rep = rep[rep['condition'].astype(str) == R.condition]
        if set(rep['gene'].astype(str)) == set(genes):
            roles = dict(zip(rep['gene'].astype(str), rep['role'].astype(str)))
            if 'connected' in rep:
                linked = dict(zip(rep['gene'].astype(str), rep['connected'].astype(bool)))
                info = (f" Italics: no edge inside the selection (probability ≥ selection_edge_prob, "
                        f"literature-feasible; {sum(not v for v in linked.values())} genes).")
            if 'edge_samples' in rep:
                es = dict(zip(rep['gene'].astype(str), rep['edge_samples'].fillna('').astype(str)))
                smp_names = sorted({x.strip() for v in es.values() for x in v.split(',') if x.strip()})
                tags = {g: ''.join(str(smp_names.index(x.strip()) + 1) for x in v.split(',') if x.strip())
                        for g, v in es.items()}
                info += (" Exponents: samples where the gene has such an edge ("
                         + ', '.join(f'{i + 1} = {n}' for i, n in enumerate(smp_names)) + ").")
    pres_path = os.path.join(R.p, 'cardamomOT', 'selection_preservation.json')
    if os.path.exists(pres_path):
        pr = json.load(open(pres_path))
        if pr.get('n_genes') == len(genes):
            info += (f" Trajectory preservation (classical OT on these genes vs every gene; network "
                     f"{pr.get('network_method', '?')}): velocity cosine {pr['velocity_cosine']:.2f} (random genes "
                     f"{pr['random_velocity_cosine']:.2f}), fate JS distance {pr['fate_js']:.2f} "
                     f"(random {pr['random_fate_js']:.2f})"
                     + (f"; fate prediction R² {pr['fate_r2']:.2f} (time only {pr['fate_r2_time']:.2f}, random "
                        f"{pr['fate_r2_random']:.2f}, every gene {pr.get('fate_r2_all_genes', float('nan')):.2f})"
                        if 'fate_r2' in pr else '') + ".")
    # Largest font such that every gene fits on one page (several pages below 4.5 pt)
    width, height = 0.94 * A4_LANDSCAPE[0] * 72, 0.74 * A4_LANDSCAPE[1] * 72  # points
    char = max(len(g) for g in genes) + 2
    for fs in np.arange(11, 4.4, -0.5):
        n_rows = int(height // (fs * 1.4))
        n_cols = max(1, int(width // (char * 0.7 * fs)))
        if n_rows * n_cols >= len(genes):
            break
    per_page = n_rows * n_cols
    for start in range(0, len(genes), per_page):
        chunk = genes[start:start + per_page]
        cols = min(n_cols, int(np.ceil(len(chunk) / n_rows)))
        rows = int(np.ceil(len(chunk) / cols))  # balanced columns
        fig = plt.figure(figsize=A4_LANDSCAPE)
        _page_title(fig, f'Genes of the model ({len(genes)})' + (f' — {start // per_page + 1}' if len(genes) > per_page else ''),
                    f'Data/data_{R.split}.h5ad, alphabetical order' + (', coloured by selection role.' if roles else '.') + info)
        x0, y0 = 0.03, 0.88
        # Columns spaced by their width (at most the page width / cols), rows by the line height
        dx = min(0.94 / cols, 1.25 * char * 0.7 * fs / 72 / A4_LANDSCAPE[0])
        dy = fs * 1.4 / 72 / A4_LANDSCAPE[1]
        for k, g in enumerate(chunk):
            c, r = divmod(k, rows)
            ok = linked.get(g, True)
            fig.text(x0 + c * dx, y0 - r * dy, g, fontsize=fs, va='top', ha='left',
                     color=ROLE_COLORS.get(roles.get(g), '#222222'), fontstyle='normal' if ok else 'italic')
            if tags.get(g):
                fig.text(x0 + c * dx + len(g) * 0.62 * fs / 72 / A4_LANDSCAPE[0], y0 - r * dy, tags[g],
                         fontsize=0.6 * fs, va='top', ha='left', color='#555555')
        if roles:
            n = pd.Series(list(roles.values())).value_counts()
            handles = [mpatches.Patch(color=col, label=f'{role} ({n.get(role, 0)})') for role, col in ROLE_COLORS.items()
                       if n.get(role, 0)]
            fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=8,
                       bbox_to_anchor=(0.5, 0.01))
        pdf.savefig(fig); plt.close(fig)


def _sankey(ax, flows, heights, cats, color_map, times):
    """
    Cell-type flows across the timepoints: node height = share of the cell type at its time; the link a -> b
    leaves a with width h_a · P(b | a) (fate) and reaches b with width h_b · P(a | b) (ancestry).
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Rectangle
    gap, w = 0.02, 0.012
    n_t = len(times)
    xs = np.linspace(0, 1, n_t)
    y = {}
    for k, t in enumerate(times):
        h = heights.get(t, {})
        tot = sum(h.values())
        n_nodes = sum(v > 0 for v in h.values())
        scale = (1 - gap * max(n_nodes - 1, 0)) / max(tot, 1e-12)
        top = 1.0
        for c in cats:
            if h.get(c, 0) > 0:
                y[(t, c)] = [top, top - h[c] * scale, scale]
                top -= h[c] * scale + gap
                ax.add_patch(Rectangle((xs[k] - w / 2, y[(t, c)][1]), w, h[c] * scale, color=color_map[c], lw=0))
    out_pos = {key: v[0] for key, v in y.items()}
    in_pos = {key: v[0] for key, v in y.items()}
    for k in range(n_t - 1):
        t0, t1 = times[k], times[k + 1]
        F = flows[(flows['t_from'] == t0) & (flows['t_to'] == t1)]
        for a in cats:
            for b in cats:
                row = F[(F['cell_type_from'] == a) & (F['cell_type_to'] == b)]
                if row.empty or (t0, a) not in y or (t1, b) not in y:
                    continue
                wl = heights[t0][a] * float(np.nan_to_num(row['fate_probability'].iloc[0])) * y[(t0, a)][2]
                wr = heights[t1][b] * float(np.nan_to_num(row['ancestor_probability'].iloc[0])) * y[(t1, b)][2]
                if wl < 1e-3 and wr < 1e-3:
                    continue
                x0, x1 = xs[k] + w / 2, xs[k + 1] - w / 2
                yl, yr = out_pos[(t0, a)], in_pos[(t1, b)]
                xm = (x0 + x1) / 2
                verts = [(x0, yl), (xm, yl), (xm, yr), (x1, yr), (x1, yr - wr), (xm, yr - wr), (xm, yl - wl),
                         (x0, yl - wl), (x0, yl)]
                codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4,
                         Path.CURVE4, Path.CLOSEPOLY]
                ax.add_patch(PathPatch(Path(verts, codes), facecolor=color_map[a], alpha=0.45, lw=0))
                out_pos[(t0, a)] -= wl
                in_pos[(t1, b)] -= wr
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([])
    step = max(1, int(np.ceil(n_t / 20)))
    ax.set_xticks(xs[::step]); ax.set_xticklabels([f'{t:g}' for t in times[::step]], fontsize=7)
    ax.set_xlabel('time', fontsize=8)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)


def _gene_table(fig, fg, cats, color_map, top=0.42, bottom=0.03):
    """Per cell type: top-10 genes of its ancestors and of the cell type, common genes in bold red."""
    x_ct, cols, width = 0.03, (0.15, 0.575), 0.41
    fig.text(cols[0], top + 0.012, 'Top-10 genes up in the ancestors (vs ancestors of the other cells)', fontsize=8,
             fontweight='bold')
    fig.text(cols[1], top + 0.012, 'Top-10 genes up in the cell type (vs the other cells)', fontsize=8,
             fontweight='bold')
    dy = min(0.06, (top - bottom) / max(len(cats), 1))
    char = 0.68 / 72 / A4_LANDSCAPE[0]  # width of a character per point of font size (figure fraction)
    rows = []
    for c in cats:
        df = fg[fg['cell_type'] == c]
        ya = list(df.nsmallest(10, 'ancestors_rank')['gene']) if df['ancestors_t'].notna().any() else []
        rows.append((c, ya, list(df.nsmallest(10, 'cell_type_rank')['gene'])))
    # One font size for the table: the largest such that every list fits its column (at most 8 pt)
    longest = max([sum(len(g) + 2 for g in lst) for _, ya, yc in rows for lst in (ya, yc)] + [1])
    fs = float(min(8.0, dy * 160, width / (longest * char)))
    for r, (c, ya, yc) in enumerate(rows):
        common = set(ya) & set(yc)
        yy = top - (r + 0.5) * dy
        fig.text(x_ct, yy, c, fontsize=min(fs + 1, 9), color=color_map[c], fontweight='bold', va='center')
        for x0, lst in zip(cols, (ya, yc)):
            if not lst:
                fig.text(x0, yy, '— (no ancestors: first timepoint only)', fontsize=fs, va='center', color='#888888')
            x = x0
            for g in lst:
                fig.text(x, yy, g, fontsize=fs, va='center', color='#C0392B' if g in common else '#222222',
                         fontweight='bold' if g in common else 'normal')
                x += (len(g) + 2) * char * fs


def _transition_page(pdf, title, subtitle, flows, heights, fg, cats, cmap):
    """Sankey of the cell-type flows across every timepoint, and table of the top-10 fate genes."""
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, title, subtitle)
    ax = fig.add_axes([0.04, 0.5, 0.78, 0.39])
    _sankey(ax, flows, heights, cats, cmap, np.sort(list(heights)))
    fig.legend(handles=[mpatches.Patch(color=cmap[c], label=c) for c in cats], loc='center left',
               bbox_to_anchor=(0.83, 0.7), frameon=False, fontsize=7.5)
    _gene_table(fig, fg, cats, cmap)
    pdf.savefig(fig); plt.close(fig)


def _shares(labels, times, mask):
    """{time: {cell type: share}} of the cells in mask (node heights of the Sankey)."""
    return {t: pd.Series(labels[mask & (times == t)]).value_counts(normalize=True).to_dict()
            for t in np.sort(np.unique(times[mask]))}


def _classical_ot_pages(pdf, R, seed=0):
    cdir = os.path.join(R.p, 'cardamomOT', 'classical_OT')
    tr_path, fg_path = os.path.join(cdir, 'transitions.csv'), os.path.join(cdir, 'fate_genes.csv')
    if not (os.path.exists(tr_path) and os.path.exists(fg_path)):
        raise FileNotFoundError('cardamomOT/classical_OT/transitions.csv or fate_genes.csv not found (run '
                                "run_classical_OT.py; needs obs['cell_type'])")
    E, obs = data_embedding(R.p, seed=seed, depth=R.depth_emb, method=R.emb_method)
    times = pd.to_numeric(obs['time']).values if 'time' in obs else np.zeros(len(obs))
    labels = obs[LABEL_KEY].astype(str).values
    cats = sorted(np.unique(labels))
    cmap = R.color_map if R.color_map and set(cats) <= set(R.color_map) else _cell_type_colors(cats)
    tr, fg = pd.read_csv(tr_path), pd.read_csv(fg_path)
    in_cond = np.ones(len(obs), bool)
    if R.samples is not None:  # condition view: flows pooled over its samples, its cells
        from .classical_ot import pooled_transitions
        sub = tr[tr['sample'].astype(str).isin(R.samples)]
        tr = pd.concat([sub, pooled_transitions(sub).assign(sample='all')], ignore_index=True)
        in_cond = obs['dataset_id'].astype(str).isin(R.samples).to_numpy() if 'dataset_id' in obs else in_cond
    tr['cell_type_from'], tr['cell_type_to'] = tr['cell_type_from'].astype(str), tr['cell_type_to'].astype(str)
    fg['cell_type'] = fg['cell_type'].astype(str)
    if R.samples is not None and 'condition' in fg:  # classical OT of each condition apart: its fate genes
        fg = fg[fg['condition'].astype(str) == str(R.condition)]
    vel = np.load(os.path.join(cdir, 'velocity.npz'), allow_pickle=True)
    fit = (pd.Series(vel['transported'], index=vel['obs_names'].astype(str)).reindex(E['obs_names'])
           .fillna(False).to_numpy(dtype=bool) if 'transported' in vel.files else np.ones(len(obs), bool))
    # Same cells as CardamomOT (data_<split>)? Otherwise the classical OT is from an earlier train/test split
    same = set(E['obs_names'][fit]) == set((R.parent or R).adata_data.obs_names.astype(str))
    fit = fit & in_cond
    warn = '' if same else (' WARNING: not the cells of data_<split> (classical OT from an earlier split: rerun '
                            'run_classical_OT.py).')
    if not same:
        print("[report] Warning: the classical OT was run on other cells than data_<split>: rerun run_classical_OT.py")
    _transition_page(pdf, 'Classical OT — cell-type transitions and fate genes',
                     'Waddington-OT couplings (every gene, train cells, per sample, growth from the net rate), pooled over '
                     + ('the samples' if R.samples is None else f'the samples of condition {R.condition} (fate genes of '
                        + ('the condition)' if 'condition' in fg else 'every sample)'))
                     + '. Sankey: node = share of the cell type; links leave ∝ fate and arrive ∝ ancestor '
                     'probabilities. Table: weighted Welch t, genes of both top 10 in red (classical_OT/fate_genes.csv).' + warn,
                     tr[tr['sample'].astype(str) == 'all'], _shares(labels, times, fit), fg, cats, cmap)


def _cardamom_transition_page(pdf, R):
    """Same page from the final soft couplings of CardamomOT, on its cells and genes (tables written as CSV)."""
    from .velocity import coupling_blocks
    from .classical_ot import dense_blocks, transitions, pooled_transitions, fate_genes
    if not R.has_ct:
        raise ValueError("no obs['cell_type']")
    cdir = os.path.join(R.p, 'cardamomOT')
    cp = os.path.join(cdir, 'couplings.npz')
    if not os.path.exists(cp):
        raise FileNotFoundError('cardamomOT/couplings.npz not found: rerun infer_network_structure.py')
    A = (R.parent or R).adata_data  # rows of the couplings: every cell of data_<split>
    blocks = dense_blocks(coupling_blocks(cp, A.n_obs))
    in_cond = np.ones(A.n_obs, bool)
    if R.samples is not None:  # condition view: its samples' blocks and cells, its genes
        names = sorted(A.obs['dataset_id'].astype(str).unique()) if 'dataset_id' in A.obs else ['0']
        keep = {names.index(x) for x in R.samples if x in names}  # blocks are indexed by sorted dataset_id
        blocks = [b for b in blocks if int(b['sample']) in keep]
        in_cond = R.cell_mask
    labels = A.obs[LABEL_KEY].astype(str).values
    times = pd.to_numeric(A.obs['time']).values
    tr = transitions(blocks, labels)
    tr = pd.concat([tr, pooled_transitions(tr)], ignore_index=True)
    # Fate genes on the model genes: log1p(counts / depth factor with use_depth_factor), as in the inference
    cols = getattr(R, 'gene_cols', None)
    X = _dense(A.X).astype(float) if cols is None else _dense(A.X[:, cols]).astype(float)
    if R.use_depth and 'depth_factor' in A.obs:
        X = X / A.obs['depth_factor'].to_numpy(dtype=float)[:, None]
    fg = fate_genes(np.log1p(X), np.asarray(R.genes), blocks, labels, in_cond)
    suffix = '' if R.condition is None else f'_{R.condition}'
    tr.to_csv(os.path.join(cdir, f'teaser_transitions{suffix}.csv'), index=False)
    fg.to_csv(os.path.join(cdir, f'teaser_fate_genes{suffix}.csv'), index=False)
    cats = sorted(np.unique(labels))
    cmap = R.color_map if R.color_map and set(cats) <= set(R.color_map) else _cell_type_colors(cats)
    _transition_page(pdf, 'Teaser — CardamomOT: cell-type transitions and fate genes',
                     'Final soft couplings of the CardamomOT trajectories (model genes, data_<split> cells), pooled over the '
                     'samples. Sankey: node = share of the cell type; links leave ∝ fate and arrive ∝ ancestor '
                     'probabilities. Table: weighted Welch t, genes of both top 10 in red (cardamomOT/teaser_fate_genes.csv).',
                     tr[tr['sample'].astype(str) == 'all'], _shares(labels, times, in_cond), fg, cats, cmap)


def classical_velocity(p, E):
    """
    Displacements of the classical OT (run_classical_OT.py) in the PCA space of the embedding E: barycentric from
    its couplings (descendants, else ancestors; not divided by Δt), the other cells by Gaussian-kernel regression in
    the PCA space of the transport, within their sample and time. Returns dict(V, origin).
    """
    from .velocity import coupling_blocks, barycentric_velocity, kernel_velocity
    cdir = os.path.join(p, 'cardamomOT', 'classical_OT')
    d = np.load(os.path.join(cdir, 'velocity.npz'), allow_pickle=True)
    if 'Z' not in d.files or not np.array_equal(d['obs_names'].astype(str), E['obs_names'].astype(str)):
        raise ValueError('cardamomOT/classical_OT does not match Data/data.h5ad: rerun run_classical_OT.py')
    V, origin = barycentric_velocity(E['pca'], coupling_blocks(os.path.join(cdir, 'couplings.npz'), len(E['pca'])),
                                     rate=False)
    known = origin != ''
    V = kernel_velocity(d['Z'].astype(float), V, known, groups=d['groups'])
    origin[~known] = 'kernel'
    return dict(V=V, origin=origin)


def cardamom_velocity(p, split, E, use_depth, k=30):
    """
    Displacements of CardamomOT on every cell of Data/data.h5ad, in the PCA space of umap_data.npz: barycentric
    (not divided by Δt) from the final soft couplings (cardamomOT/couplings.npz, rows of data_<split>.h5ad), descendants else
    ancestors; the cells without either (never reached, or not in data_<split>) by Gaussian-kernel regression
    in the space of the model mRNAs (log1p, counts / depth factor with use_depth_factor), within their sample and time.
    Returns (V, origin).
    """
    from .velocity import coupling_blocks, barycentric_velocity, kernel_velocity
    from ..config import harmonize_obs
    cp = os.path.join(p, 'cardamomOT', 'couplings.npz')
    if not os.path.exists(cp):
        raise FileNotFoundError('cardamomOT/couplings.npz not found: rerun infer_network_structure.py')
    names = pd.Index(E['obs_names'])
    ref = ad.read_h5ad(os.path.join(p, 'Data', f'data_{split}.h5ad'))
    pos = names.get_indexer(ref.obs_names.astype(str))
    if (pos < 0).any():
        raise ValueError(f'cells of data_{split}.h5ad absent from Data/data.h5ad: rerun the report UMAP')
    V_ref, origin_ref = barycentric_velocity(E['pca'][pos], coupling_blocks(cp, ref.n_obs), rate=False)
    V = np.full((len(names), E['pca'].shape[1]), np.nan)
    origin = np.full(len(names), '', dtype=object)
    V[pos], origin[pos] = V_ref, origin_ref
    # Model mRNAs of every cell (data_full + data_test hold every cell of data.h5ad)
    parts = [ad.read_h5ad(os.path.join(p, 'Data', f'data_{n}.h5ad')) for n in ('full', 'test', split)
             if os.path.exists(os.path.join(p, 'Data', f'data_{n}.h5ad'))]
    A = ad.concat(parts, merge='first')
    A = A[~A.obs_names.duplicated()].copy()
    harmonize_obs(A)
    X = _dense(A.X).astype(float)
    if use_depth and 'depth_factor' in A.obs:  # space of the inference (use_depth_factor), not of the display
        X = X / A.obs['depth_factor'].to_numpy(dtype=float)[:, None]
    Zm = np.full((len(names), X.shape[1]), np.nan)
    pa = names.get_indexer(A.obs_names.astype(str))
    Zm[pa[pa >= 0]] = np.log1p(X[pa >= 0])
    grp = np.full(len(names), '', dtype=object)
    obs = A.obs.iloc[np.flatnonzero(pa >= 0)]
    grp[pa[pa >= 0]] = [f"{s}|{float(t):g}" for s, t in zip(
        obs['dataset_id'].astype(str) if 'dataset_id' in obs else ['0'] * len(obs), obs['time'])]
    have = np.isfinite(Zm).all(axis=1)
    known = origin != ''
    Vk = kernel_velocity(np.nan_to_num(Zm[have]), V[have], known[have], groups=grp[have], k=k)
    V[have] = Vk
    origin[have & ~known] = 'kernel'
    return V, origin


def _teaser_fields(R, seed=0):
    """Data embedding and coupling displacements of classical OT (if run) and CardamomOT, computed once; for a
    condition view, those of the whole run restricted to the cells of its samples (embedding of every cell)."""
    if getattr(R, '_teaser', None) is not None:
        return R._teaser
    if getattr(R, 'parent', None) is not None:
        F = _teaser_fields(R.parent, seed)
        m = F['obs']['dataset_id'].astype(str).isin(R.samples).to_numpy() if 'dataset_id' in F['obs'] \
            else np.ones(len(F['obs']), bool)
        E = dict(F['E'], emb=F['E']['emb'][m], pca=F['E']['pca'][m], obs_names=F['E']['obs_names'][m])
        # Intervals of the samples of the condition
        iv_samples = {iv: [x for x in ss if x in set(R.samples)] for iv, ss in F['interval_samples'].items()}
        iv_samples = {iv: ss for iv, ss in iv_samples.items() if ss}
        R._teaser = dict(F, E=E, obs=F['obs'][m], labels=None if F['labels'] is None else F['labels'][m],
                         times=F['times'][m], fields=[(n, V[m], o[m]) for n, V, o in F['fields']],
                         cell_samples=F['cell_samples'][m], intervals=sorted(iv_samples), interval_samples=iv_samples)
        return R._teaser
    E, obs = data_embedding(R.p, seed=seed, depth=R.depth_emb, method=R.emb_method)
    labels = obs[LABEL_KEY].astype(str).values if LABEL_KEY in obs else None
    cats = sorted(np.unique(labels)) if labels is not None else []
    cmap = R.color_map if R.color_map and set(cats) <= set(R.color_map) else _cell_type_colors(cats)
    times = pd.to_numeric(obs['time']).values if 'time' in obs else np.zeros(len(obs))
    V_cot, origin = cardamom_velocity(R.p, R.split, E, R.use_depth)
    fields = [('CardamomOT (final soft couplings)', V_cot, origin)]
    if os.path.exists(os.path.join(R.p, 'cardamomOT', 'classical_OT', 'velocity.npz')):
        C = classical_velocity(R.p, E)
        fields.insert(0, ('Classical OT', C['V'], C['origin']))
    # Intervals of the CardamomOT couplings (t_from -> t_to) and the samples having each one (consecutive observed
    # times of each sample; couplings indexed by the sorted dataset_id of data_<split>)
    cp = np.load(os.path.join(R.p, 'cardamomOT', 'couplings.npz'))
    A = R.adata_data
    names = sorted(A.obs['dataset_id'].astype(str).unique()) if 'dataset_id' in A.obs else ['0']
    iv_samples = {}
    for k, a, b in {(int(k), float(a), float(b)) for k, a, b in zip(cp['sample'], cp['t_from'], cp['t_to'])}:
        iv_samples.setdefault((a, b), []).append(names[min(k, len(names) - 1)])
    cell_samples = obs['dataset_id'].astype(str).values if 'dataset_id' in obs else np.full(len(obs), '0')
    R._teaser = dict(E=E, obs=obs, labels=labels, cmap=cmap, times=times, fields=fields, cell_samples=cell_samples,
                     intervals=sorted(iv_samples), interval_samples={iv: sorted(ss) for iv, ss in iv_samples.items()})
    return R._teaser


def _direct(origin):
    return np.isin(origin, ['descendants', 'ancestors'])


def _teaser_page(pdf, R, seed=0):
    from .velocity import knn_velocity_embedding
    F = _teaser_fields(R, seed)
    E, labels, cmap, times, fields = F['E'], F['labels'], F['cmap'], F['times'], F['fields']
    origin = fields[-1][2]
    cos = np.nan
    if len(fields) == 2:
        both = _direct(fields[0][2]) & _direct(origin)
        cos = _weighted_cosine(fields[0][1][both], fields[1][1][both])
    size = float(np.clip(30000 / len(E['emb']), 0.3, 6))
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, 'Teaser — displacement fields of classical OT and CardamomOT',
                f"Displacement to the mean descendant (x(t+1) − x(t), not divided by Δt), all times. Agreement on the "
                f"cells with both direct displacements: weighted cosine {cos:.2f} (PCA space). CardamomOT: final soft "
                f"couplings of the trajectories ({int(_direct(origin).sum())} cells); cells never reached or outside "
                "the training set: Gaussian kernel on the model mRNAs.")
    gs = gridspec.GridSpec(1, len(fields), figure=fig, left=0.03, right=0.97, top=0.86, bottom=0.12, wspace=0.06)
    for j, (title, V, _) in enumerate(fields):
        Vemb = knn_velocity_embedding(E['pca'], np.nan_to_num(V), E['emb'])
        ax = fig.add_subplot(gs[0, j])
        if labels is not None:
            _stream(ax, E['emb'], Vemb, labels, f'{title}, colour = cell type', categorical=cmap, s=size,
                    density=1.2, linewidth=1.1, alpha=0.6, color='#1A1A1A')
        else:
            _stream(ax, E['emb'], Vemb, times, title, s=size, density=1.2, linewidth=1.1, alpha=0.6, color='#1A1A1A')
    if labels is not None:
        _celltype_legend(fig, cmap, y=0.02)
    pdf.savefig(fig); plt.close(fig)


def _teaser_time_pages(pdf, R, seed=0, per_page=6):
    """Displacement fields interval by interval (cells of t_from to their mean descendant at t_to), classical OT
    (top row, if run) vs CardamomOT (bottom row), on the data embedding; per-interval agreement (weighted cosine)."""
    from .velocity import knn_velocity_embedding
    F = _teaser_fields(R, seed)
    E, labels, cmap, times, fields, intervals = (F[k] for k in ('E', 'labels', 'cmap', 'times', 'fields', 'intervals'))
    cell_samples, iv_samples = F['cell_samples'], F['interval_samples']
    emb, pca = E['emb'], E['pca']
    lo, hi = emb.min(axis=0), emb.max(axis=0)
    size = float(np.clip(60000 / len(emb), 0.3, 6))
    for start in range(0, len(intervals), per_page):
        chunk = intervals[start:start + per_page]
        fig = plt.figure(figsize=A4_LANDSCAPE)
        _page_title(fig, 'Teaser — displacement fields over time, classical OT vs CardamomOT',
                    'Cells of each t_from (coloured; other cells in grey) of the samples whose next timepoint is t_to, '
                    'towards their mean descendant at t_to, kNN projection among their cells of t_from and t_to. cos: '
                    'weighted cosine between the two methods on the cells of t_from with both direct displacements '
                    '(PCA space).')
        gs = gridspec.GridSpec(len(fields), len(chunk), figure=fig, left=0.03, right=0.99, top=0.86, bottom=0.1,
                               hspace=0.12, wspace=0.04)
        for j, (t0, t1) in enumerate(chunk):
            in_iv = np.isin(cell_samples, iv_samples[(t0, t1)])  # samples whose interval t0 -> t1 it is
            src = np.flatnonzero(in_iv & (times == t0))
            pool = np.flatnonzero(in_iv & ((times == t0) | (times == t1)))
            is_src = np.isin(pool, src)
            both = (_direct(fields[0][2]) & _direct(fields[-1][2]))[src]
            cos = (_weighted_cosine(fields[0][1][src][both], fields[1][1][src][both])
                   if len(fields) == 2 and both.any() else np.nan)
            for i, (name, V, _) in enumerate(fields):
                ax = fig.add_subplot(gs[i, j])
                ax.scatter(emb[:, 0], emb[:, 1], s=size * 0.5, color='#E5E5E5', linewidths=0, rasterized=True)
                if len(src) < 3:
                    ax.axis('off'); continue
                # Projection with the cells of t_to as neighbours (their own displacement unused)
                Vp = np.zeros((len(pool), pca.shape[1]))
                Vp[is_src] = np.nan_to_num(V[src])
                Vemb = knn_velocity_embedding(pca[pool], Vp, emb[pool])[is_src]
                title = f'{t0:g} → {t1:g}' + (f'  (cos {cos:.2f})' if i == 0 and np.isfinite(cos) else '')
                if labels is not None:
                    _stream(ax, emb[src], Vemb, labels[src], title, categorical=cmap, s=size, density=0.7,
                            linewidth=0.9, alpha=0.8, color='#1A1A1A', n_grid=30)
                else:
                    _stream(ax, emb[src], Vemb, np.full(len(src), t0), title, s=size, density=0.7, linewidth=0.9,
                            alpha=0.8, color='#1A1A1A', n_grid=30)
                ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
                if j == 0:
                    ax.text(-0.04, 0.5, name.split(' (')[0], transform=ax.transAxes, rotation=90, ha='right',
                            va='center', fontsize=9, fontweight='bold')
        if labels is not None:
            _celltype_legend(fig, cmap, y=0.02)
        pdf.savefig(fig); plt.close(fig)


def _restrict(A, samples):
    """Cells of A from the samples (obs['dataset_id']); A itself without samples, None if it has no sample column."""
    if samples is None:
        return A
    if A is None or 'dataset_id' not in A.obs:
        return None
    return A[A.obs['dataset_id'].astype(str).isin(samples).to_numpy()]


def _model_umap_page(pdf, R, samples=None, label=''):
    """UMAPs of the stages of the cells of samples (None = all), on the embedding learned on their observed cells."""
    V = R.view_of(samples)
    if V is None:
        _error_page(pdf, f'1. Generative model — trajectories and simulation{label}',
                    f'no cell of the samples {samples} in a stage (no obs["dataset_id"] or empty)')
        return
    stages, sub, umap = V['stages'], V['sub'], V['umap']
    names = list(stages)
    t_all = np.concatenate([R.times(stages[k], sub[k]) for k in names])
    vmin, vmax = float(t_all.min()), float(t_all.max())

    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, f'1. Generative model — trajectories and simulation{label}',
                ('' if samples is None else f"Samples {', '.join(samples)}. ")
                + f'{R.emb_name} learned on the observed data' + ('' if samples is None else ' of these samples')
                + '; the trajectories (Reference: states drawn at each time among the observed cells), the NB mixture '
                + 'and network-driven modes along them and the full simulation'
                + (' (with proliferation)' if R.sim_prolif else ' (without proliferation)') + ' are projected onto it.')
    smp_data = _sample_labels(stages['Data'], sub['Data'])
    if smp_data is not None and len(np.unique(smp_data)) < 2:
        smp_data = None  # a single sample: no sample panel
    gs = gridspec.GridSpec(2 + (smp_data is not None), len(names), figure=fig, left=0.03, right=0.96, top=0.89,
                           bottom=0.12, hspace=0.12, wspace=0.05)
    sca = None
    if smp_data is not None:  # samples of the observed data
        _umap_sample(fig.add_subplot(gs[2, 0]), umap['Data'], smp_data, 'Data — samples')
    for j, k in enumerate(names):
        A = stages[k]
        sca = _umap_time(fig.add_subplot(gs[0, j]), umap[k], R.times(A, sub[k]), vmin, vmax,
                         'Data (observed)' if k == 'Data' else k)
        ax = fig.add_subplot(gs[1, j])
        if R.has_ct:
            _umap_celltype(ax, umap[k], R.labels(A, sub[k]), R.color_map, '')
        else:
            ax.axis('off')
    cax = fig.add_axes([0.965, 0.55, 0.008, 0.3])
    cb = fig.colorbar(sca, cax=cax); cb.set_label('time', fontsize=8); cb.ax.tick_params(labelsize=7)
    _celltype_legend(fig, R.color_map, y=0.03)
    pdf.savefig(fig); plt.close(fig)


def _mode_shuffle(ref_stage, beta_stage, X, rng):
    """X (reference states) with each gene shuffled among the states of the same sample, time and mode (kon of the
    aligned NB-mixture states, layer 'kon' of check_sim_to_data); None if unavailable or not aligned."""
    if 'kon' not in beta_stage.layers or beta_stage.n_obs != ref_stage.n_obs or X.shape[0] != ref_stage.n_obs:
        return None
    kon = _dense(beta_stage.layers['kon']).astype(float)
    smp = (ref_stage.obs['dataset_id'].astype(str).values if 'dataset_id' in ref_stage.obs
           else np.zeros(ref_stage.n_obs, dtype=str))
    base = np.unique(np.char.add(np.char.add(smp.astype(str), '|'),
                                 pd.to_numeric(ref_stage.obs['time']).values.astype(str)), return_inverse=True)[1]
    out = np.empty_like(X)
    for g in range(X.shape[1]):
        lev = np.unique(np.round(kon[:, g], 6), return_inverse=True)[1]
        key = base * (lev.max() + 1) + lev
        pos = np.argsort(key, kind='stable')
        out[pos, g] = X[np.lexsort((rng.random(len(key)), key)), g]
    return out


def _model_checks_page(pdf, R, samples=None, label=''):
    """Cell-type proportions, gene-pair correlations and proteins of the cells of samples (None = all)."""
    V = R.view_of(samples)
    if V is None:
        _error_page(pdf, f'1. Generative model — quantitative checks{label}',
                    f'no cell of the samples {samples} in a stage (no obs["dataset_id"] or empty)')
        return
    stages = V['stages']
    names = list(stages)
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, f'1. Generative model — quantitative checks{label}',
                ('' if samples is None else f"Samples {', '.join(samples)}. ")
                + 'Reference = RNA of the trajectories (states drawn at each time among the observed cells: the observed '
                + 'composition); simulation ' + ('with' if R.sim_prolif else 'without') + ' proliferation.')
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.06, right=0.97, top=0.88, bottom=0.12, hspace=0.45, wspace=0.3)
    if R.has_ct:
        prop = pd.DataFrame(_with_predicted({k: _proportions(stages[k], R.categories) for k in names},
                                            stages['Data'], R.categories)).T
        ax = fig.add_subplot(gs[0, 0]); _stacked_bars(ax, prop, R.color_map, 'Cell-type proportions'); _panel_label(ax, 'A', -0.2)
        ax = fig.add_subplot(gs[0, 1:])
        _proportions_over_time(ax, _proportions_by_time(stages['Reference'], R.categories),
                               _proportions_by_time(stages['Simulation'], R.categories), R.color_map,
                               'Cell-type proportions over time', 'Reference', 'Simulation')
        _panel_label(ax, 'B', -0.08)

    # Gene-pair correlations, reference vs NB mixture / simulation
    ref = _dense(stages['Reference'].X).astype(float)
    iu = np.triu_indices(ref.shape[1], k=1)

    def _pair_corr(X):
        X = _dense(X).astype(float)
        X = X + 1e-9 * R.rng.standard_normal(X.shape) * (X.std(axis=0) == 0)
        return np.corrcoef(X.T)[iu]

    c_ref = _pair_corr(ref)
    # Ceiling of a model with these modes and genes independent given them: reference shuffled per gene within
    # (sample, time, mode of the state)
    shuffled = _mode_shuffle(stages['Reference'], stages['NB mixture'], ref, R.rng)
    ceiling = None
    if shuffled is not None:
        c_sh = _pair_corr(shuffled)
        ok = np.isfinite(c_ref) & np.isfinite(c_sh)
        ceiling = np.corrcoef(c_ref[ok], c_sh[ok])[0, 1]
    for j, (k, lab) in enumerate([('NB mixture', 'C'), ('Simulation', 'D')]):
        c_k = _pair_corr(stages[k].X)
        ok = np.isfinite(c_ref) & np.isfinite(c_k)
        r = np.corrcoef(c_ref[ok], c_k[ok])[0, 1]
        ax = fig.add_subplot(gs[1, j])
        ax.scatter(c_ref[ok], c_k[ok], s=2, alpha=0.35, color='k', linewidths=0, rasterized=True)
        lim = [min(c_ref[ok].min(), c_k[ok].min()), max(c_ref[ok].max(), c_k[ok].max())]
        ax.plot(lim, lim, 'r--', lw=0.8, alpha=0.6)
        ax.set_xlabel('Reference gene-pair correlation', fontsize=8)
        ax.set_ylabel(f'{k} gene-pair correlation', fontsize=8)
        ax.set_title(f'Gene-pair correlations vs {k}\n(R = {r:.2f}'
                     + ('' if ceiling is None else f'; mode-shuffle ceiling {ceiling:.2f}') + ')', fontsize=9)
        ax.tick_params(labelsize=7)
        _panel_label(ax, lab, -0.2)

    # Proteins: embedding learned once on the trajectories of these cells, simulation projected
    Pt, Ps, red = V['prot'].get('Trajectories'), V['prot'].get('Simulation'), V['prot_red']
    if red is not None and Pt is not None and Ps is not None:
        sub_gs = gs[1, 2].subgridspec(1, 2, wspace=0.05)
        it, is_ = V['prot_sub'], R._subsample(Ps)
        emb_s = red.transform(_dense(Ps.X[is_]).astype(float))
        tt, ts = R.times(Pt, it), R.times(Ps, is_)
        lo, hi = min(tt.min(), ts.min()), max(tt.max(), ts.max())
        ax = fig.add_subplot(sub_gs[0, 0]); _umap_time(ax, red.embedding_, tt, lo, hi, 'Proteins — traj.'); _panel_label(ax, 'E', -0.15)
        _umap_time(fig.add_subplot(sub_gs[0, 1]), emb_s, ts, lo, hi, 'Proteins — sim.')
    if R.has_ct:
        _celltype_legend(fig, R.color_map, y=0.01)
    pdf.savefig(fig); plt.close(fig)


def _w1_per_gene(A, B, times, rng, n_max=2000):
    """Per gene, mean over the timepoints of the 1-D Wasserstein distance between log1p counts of A and B."""
    from scipy.stats import wasserstein_distance
    XA, XB = np.log1p(_dense(A.X).astype(float)), np.log1p(_dense(B.X).astype(float))
    tA, tB = pd.to_numeric(A.obs['time']).values, pd.to_numeric(B.obs['time']).values
    out, n = np.zeros(XA.shape[1]), 0
    for t in times:
        ia, ib = np.flatnonzero(tA == t), np.flatnonzero(tB == t)
        if not len(ia) or not len(ib):
            continue
        ia = rng.choice(ia, min(len(ia), n_max), replace=False)
        ib = rng.choice(ib, min(len(ib), n_max), replace=False)
        out += [wasserstein_distance(XA[ia, g], XB[ib, g]) for g in range(XA.shape[1])]
        n += 1
    return out / max(n, 1)


def _test_pages(pdf, R):
    T = R.test
    names = list(T)
    t_all = np.concatenate([R.times(T[k], R.test_sub[k]) for k in names])
    vmin, vmax = float(t_all.min()), float(t_all.max())
    bg = np.vstack([R.umap[k] for k in R.stages])

    # Page: test data and predictions on the joint embedding (training states in grey)
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '6. Held-out test cells — predictions with the network fixed',
                'Test cells classified into basins with the training mixtures, trajectories inferred with the '
                'training network fixed, then simulated; projected onto the UMAP learned on the observed data (grey: training).')
    smp_test = _sample_labels(T[names[0]], R.test_sub[names[0]])
    gs = gridspec.GridSpec(2 + (smp_test is not None), 4, figure=fig, left=0.04, right=0.96, top=0.89, bottom=0.12,
                           hspace=0.18, wspace=0.06)
    sca = None
    if smp_test is not None:  # samples of the test data
        _umap_sample(fig.add_subplot(gs[2, 0]), R.test_umap[names[0]], smp_test, f'{names[0]} — samples', bg=bg)
    for j, k in enumerate(names):
        A = T[k]
        sca = _umap_time(fig.add_subplot(gs[0, j]), R.test_umap[k], R.times(A, R.test_sub[k]), vmin, vmax, k, bg=bg)
        ax = fig.add_subplot(gs[1, j])
        if R.has_ct:
            _umap_celltype(ax, R.test_umap[k], R.labels(A, R.test_sub[k]), R.color_map, '', bg=bg)
        else:
            ax.axis('off')
    cax = fig.add_axes([0.965, 0.55, 0.008, 0.3])
    cb = fig.colorbar(sca, cax=cax); cb.set_label('time', fontsize=8); cb.ax.tick_params(labelsize=7)
    _celltype_legend(fig, R.color_map, y=0.03)
    pdf.savefig(fig); plt.close(fig)

    # Page: distribution distances (sampling floor, training fit, held-out prediction)
    data_tr, data_te = R.stages['Data'], T['Test data']
    sim_tr, sim_te = R.stages['Simulation'], T['Simulation']
    t_data = np.intersect1d(R.times(data_tr), R.times(data_te))
    t_sim = np.intersect1d(t_data, np.intersect1d(R.times(sim_tr), R.times(sim_te)))
    w_floor = _w1_per_gene(data_tr, data_te, t_data, R.rng)
    w_train = _w1_per_gene(sim_tr, data_tr, t_sim, R.rng)
    w_test = _w1_per_gene(sim_te, data_te, t_sim, R.rng)
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '6. Held-out test cells — quantitative checks',
                'Per gene: 1-D Wasserstein distance between log1p counts, averaged over the common timepoints.')
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.06, right=0.97, top=0.88, bottom=0.12, hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    labels = ['Train vs test\ndata', 'Simulation vs\ntrain data', 'Test simulation\nvs test data']
    ax.boxplot([w_floor, w_train, w_test], widths=0.6, showfliers=False)
    for i, w in enumerate([w_floor, w_train, w_test]):
        ax.scatter(np.full(len(w), i + 1) + R.rng.uniform(-0.15, 0.15, len(w)), w, s=4, color='k', alpha=0.4,
                   linewidths=0, rasterized=True)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('W1 (log1p counts)', fontsize=8); ax.tick_params(axis='y', labelsize=7)
    ax.set_title(f'Medians: {np.median(w_floor):.3f} (sampling) · {np.median(w_train):.3f} (fit) · '
                 f'{np.median(w_test):.3f} (held-out)', fontsize=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    _panel_label(ax, 'A', -0.2)

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(w_train, w_test, s=6, color='k', alpha=0.5, linewidths=0)
    lim = [0, max(w_train.max(), w_test.max()) * 1.05]
    ax.plot(lim, lim, 'r--', lw=0.8, alpha=0.6)
    worst = np.argsort(w_test - w_train)[-5:]
    for g in worst:
        ax.annotate(R.genes[g], (w_train[g], w_test[g]), fontsize=6, xytext=(2, 2), textcoords='offset points')
    ax.set_xlabel('W1 simulation vs train data', fontsize=8); ax.set_ylabel('W1 test simulation vs test data', fontsize=8)
    ax.set_title('Generalisation per gene (labels: largest gaps)', fontsize=9); ax.tick_params(labelsize=7)
    _panel_label(ax, 'B', -0.2)

    # Gene-pair correlations, test data vs test simulation
    iu = np.triu_indices(len(R.genes), k=1)

    def _pair_corr(X):
        X = _dense(X).astype(float)
        X = X + 1e-9 * R.rng.standard_normal(X.shape) * (X.std(axis=0) == 0)
        return np.corrcoef(X.T)[iu]

    c_d, c_s = _pair_corr(data_te.X), _pair_corr(sim_te.X)
    ok = np.isfinite(c_d) & np.isfinite(c_s)
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(c_d[ok], c_s[ok], s=2, alpha=0.35, color='k', linewidths=0, rasterized=True)
    lim = [min(c_d[ok].min(), c_s[ok].min()), max(c_d[ok].max(), c_s[ok].max())]
    ax.plot(lim, lim, 'r--', lw=0.8, alpha=0.6)
    ax.set_xlabel('Test data gene-pair correlation', fontsize=8); ax.set_ylabel('Test simulation', fontsize=8)
    ax.set_title(f'Gene-pair correlations (R = {np.corrcoef(c_d[ok], c_s[ok])[0, 1]:.2f})', fontsize=9)
    ax.tick_params(labelsize=7)
    _panel_label(ax, 'C', -0.2)

    # Mean expression per (gene, time): test data vs test simulation and vs training simulation
    def _means(A, times):
        X, t = _dense(A.X).astype(float), R.times(A)
        return np.concatenate([np.log1p(X[t == tv].mean(axis=0)) for tv in times])
    ax = fig.add_subplot(gs[1, 0])
    m_d = _means(data_te, t_sim)
    for A, col, lab in [(sim_tr, '#999999', 'training simulation'), (sim_te, 'k', 'test simulation')]:
        m = _means(A, t_sim)
        ax.scatter(m_d, m, s=4, color=col, alpha=0.6, linewidths=0, label=f'{lab} (R = {np.corrcoef(m_d, m)[0, 1]:.2f})')
    lim = [0, m_d.max() * 1.05]
    ax.plot(lim, lim, 'r--', lw=0.8, alpha=0.6)
    ax.set_xlabel('Test data log1p(mean) per gene and time', fontsize=8); ax.set_ylabel('Prediction', fontsize=8)
    ax.legend(fontsize=6.5, frameon=False); ax.tick_params(labelsize=7)
    ax.set_title('Mean expression', fontsize=9)
    _panel_label(ax, 'D', -0.2)

    if R.has_ct:
        ax = fig.add_subplot(gs[1, 1:])
        # Test cells labelled by the classifier (not seen in its training), as the simulated ones
        pred_te = LABEL_PRED in data_te.obs
        _proportions_over_time(ax, _proportions_by_time(data_te, R.categories, LABEL_PRED if pred_te else LABEL_KEY),
                               _proportions_by_time(sim_te, R.categories), R.color_map,
                               'Cell-type proportions over time', 'Test data (predicted)' if pred_te else 'Test data',
                               'Test simulation')
        _panel_label(ax, 'E', -0.08)
        _celltype_legend(fig, R.color_map, y=0.01)
    pdf.savefig(fig); plt.close(fig)


def _validation_page(pdf, R, r):
    """Sample removed from the inference: simulation from its reference sample's first-timepoint states with its
    own schedule vs its observed cells (baseline: the reference sample's data at the same times)."""
    obs_r, sim, ref = R.validation[r]
    t_obs = np.unique(R.times(obs_r))
    t_last = float(t_obs.max())
    ref_data = None
    if 'dataset_id' in R.adata_data.obs:
        D_ = R.stages['Data']
        m = (D_.obs['dataset_id'].astype(str) == ref).to_numpy() & np.isin(R.times(D_), t_obs)
        ref_data = D_[m] if m.any() else None
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, f'6. Validation — sample {r} (removed from the inference)',
                f'Simulated from the first-timepoint training states of {ref} with the stimulus_test_schedule of {r}; '
                f'compared to the observed cells of {r} at {", ".join(f"{t:g}" for t in t_obs)}. '
                f'Baseline: data of {ref} at the same times.')
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.06, right=0.97, top=0.87, bottom=0.12, hspace=0.45, wspace=0.3)
    bg = np.vstack([R.umap[k] for k in R.stages])
    for j, ((A, idx, coords), lab) in enumerate(zip(R.val_umap[r], [f'Observed {r}', 'Prediction'])):
        ax = fig.add_subplot(gs[0, j])
        if R.has_ct:
            _umap_celltype(ax, coords, R.labels(A, idx), R.color_map, lab, bg=bg)
        else:
            _umap_time(ax, coords, R.times(A, idx), float(t_obs.min()), t_last, lab, bg=bg)
        _panel_label(ax, 'AB'[j], -0.05)

    # Mean expression per (gene, observed time): prediction and baseline vs observed
    def _means(A):
        X, t = _dense(A.X).astype(float), R.times(A)
        return np.concatenate([np.log1p(X[t == tv].mean(axis=0)) if np.any(t == tv) else np.full(X.shape[1], np.nan)
                               for tv in t_obs])
    ax = fig.add_subplot(gs[0, 2])
    m_o = _means(obs_r)
    for A, col, lab in [(ref_data, '#999999', f'data of {ref}'), (sim, 'k', 'prediction')]:
        if A is None:
            continue
        mm = _means(A)
        ok = np.isfinite(mm) & np.isfinite(m_o)
        if ok.sum() > 2:
            ax.scatter(m_o[ok], mm[ok], s=5, color=col, alpha=0.6, linewidths=0,
                       label=f'{lab} (R = {np.corrcoef(m_o[ok], mm[ok])[0, 1]:.2f})')
    lim = [0, np.nanmax(m_o) * 1.05]
    ax.plot(lim, lim, 'r--', lw=0.8, alpha=0.6)
    ax.set_xlabel(f'Observed {r}: log1p(mean) per gene and time', fontsize=8); ax.set_ylabel('Prediction / baseline', fontsize=8)
    ax.legend(fontsize=6.5, frameon=False); ax.tick_params(labelsize=7); ax.set_title('Mean expression', fontsize=9)
    _panel_label(ax, 'C', -0.2)

    # Per-gene W1 vs observed: prediction and baseline; genes where the prediction beats the baseline
    ax = fig.add_subplot(gs[1, 0])
    w_sim = _w1_per_gene(sim, obs_r, t_obs, R.rng)
    groups, labels = [w_sim], ['Prediction']
    if ref_data is not None:
        t_com = np.intersect1d(t_obs, np.unique(R.times(ref_data)))
        w_ref = _w1_per_gene(ref_data, obs_r, t_com, R.rng)
        groups, labels = [w_ref, w_sim], [f'Data of {ref}', 'Prediction']
    ax.boxplot(groups, widths=0.6, showfliers=False)
    for i, w in enumerate(groups):
        ax.scatter(np.full(len(w), i + 1) + R.rng.uniform(-0.15, 0.15, len(w)), w, s=4, color='k', alpha=0.4,
                   linewidths=0, rasterized=True)
    ax.set_xticks(range(1, len(groups) + 1)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(f'W1 vs observed {r} (log1p)', fontsize=8); ax.tick_params(axis='y', labelsize=7)
    title = 'Medians: ' + ' · '.join(f'{np.median(w):.3f}' for w in groups)
    if len(groups) == 2:
        title += f'\nprediction closer for {np.mean(w_sim < w_ref) * 100:.0f}% of genes'
    ax.set_title(title, fontsize=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    _panel_label(ax, 'D', -0.2)

    # Cell-type proportions at the last observed time
    if R.has_ct:
        ax = fig.add_subplot(gs[1, 1])
        # Observed cells: annotated and predicted labels (the prediction is labelled by the same classifier)
        bars = [(f'Observed {r}', obs_r, LABEL_KEY)]
        if LABEL_PRED in obs_r.obs:
            bars.append((f'Observed {r} (predicted)', obs_r, LABEL_PRED))
        bars.append(('Prediction', sim, LABEL_KEY))
        if ref_data is not None:
            bars.append((f'Data {ref}', ref_data, LABEL_KEY))
            if LABEL_PRED in ref_data.obs:
                bars.append((f'Data {ref} (predicted)', ref_data, LABEL_PRED))
        for i, (lab, A, key) in enumerate(bars):
            t = R.times(A)
            all_labs = A.obs[key].astype(str).values
            labs = all_labs[np.isclose(t, t_last)] if np.any(np.isclose(t, t_last)) else all_labs
            bottom = 0.0
            for cat in R.categories:
                v = 100 * np.mean(labs == cat) if len(labs) else 0
                ax.bar(i, v, bottom=bottom, color=R.color_map[cat], width=0.6)
                bottom += v
        ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=6.5, rotation=25, ha='right')
        ax.set_ylabel('% of cells', fontsize=8); ax.set_title(f'Cell types at t = {t_last:g}', fontsize=9)
        ax.tick_params(labelsize=7)
        _panel_label(ax, 'E', -0.2)
        _celltype_legend(fig, R.color_map, y=0.01)

    # Population growth of the prediction (branching simulation)
    if 'log_population' in sim.uns and 'simulated_times' in sim.uns:
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(np.asarray(sim.uns['simulated_times']), np.exp(np.asarray(sim.uns['log_population'])), 'k-o', ms=3)
        ax.set_xlabel('time', fontsize=8); ax.set_ylabel('relative population size', fontsize=8)
        ax.set_title('Predicted population growth', fontsize=9); ax.tick_params(labelsize=7)
        _panel_label(ax, 'F', -0.2)
    pdf.savefig(fig); plt.close(fig)


def _depth_page(pdf, R, diag, per_group):
    """Per-cell depth diagnostic (estimate_cell_depth.py) and, on the selected genes, its effect."""
    fig = plt.figure(figsize=A4_LANDSCAPE)
    # Depth model of the mixture fit (infer_mixture), else the default
    mix_json = os.path.join(R.p, 'cardamomOT', 'mixture_depth.json')
    model = 'mean'
    if os.path.exists(mix_json):
        import json
        model = json.load(open(mix_json)).get('depth_model', model)
    nb = 'NB(k s_i, c)' if model == 'burst_frequency' else 'NB(k, c / s_i)'
    _page_title(fig, '0. Per-cell sequencing depth',
                'Diagnostic on the whole transcriptome within homogeneous groups (sample, time, cell type); '
                f'counts are modelled as {nb} (depth model \'{model}\') when the depth factor s_i is used.')
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.06, right=0.97, top=0.88, bottom=0.1, hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(gs[:, 0]); ax.axis('off')
    if not diag.get('estimated', False):
        lines = [('Depth', 'not estimated'), ('Reason', diag.get('reason', ''))]
    else:
        decision = ('applied' if diag['applied'] else
                    'recommended, not allowed' if diag['recommended'] else 'not needed')
        lines = [('Source', os.path.basename(diag['source'])), ('Method', diag['method']), ('Decision', decision),
                 ('Used by the model', 'yes' if R.use_depth else 'no (use_depth_factor = False)'),
                 ('Spread of s (q95/q05)', f"x{diag['depth_spread_q95_q05']:.2f}"),
                 ('Median CV of depth', f"{diag['median_cv']:.2f}"),
                 ('Extrinsic noise (median of groups)', f"{diag.get('median_extrinsic_noise', np.nan):.2f}"),
                 ('Correlation due to depth', f"{diag['corr_share_depth'] * 100:.0f}% "
                                              f"({diag['corr_raw']:.2f} -> {diag['corr_without_depth']:.2f})"),
                 ('Gene-depth correlation', f"{diag['median_gene_depth_corr']:.2f} (median)"),
                 ('NB modes closer than spread', f"{diag['share_modes_closer_than_spread'] * 100:.0f}% of the genes"),
                 ('Extrinsic noise phi', f"{diag['extrinsic_phi']:.2f}")]
        if 'doublet_depth_corr' in diag:
            lines.append(('Doublet-depth correlation', f"{diag['doublet_depth_corr']:.2f}"))
    dbl = diag.get('predicted_doublets')
    if dbl:
        txt = f"{dbl['n']} cells ({dbl['fraction'] * 100:.1f}%) still present"
        if dbl.get('by_cell_type'):
            txt += ' — ' + ', '.join(f"{c} {v * 100:.1f}%" for c, v in dbl['by_cell_type'].items())
        lines.append(('Predicted doublets', txt + (' (remove upstream)' if dbl['n'] else '')))
    y = 0.95
    for k, v in lines:
        ax.text(0.0, y, k, fontsize=8.5, fontweight='bold', transform=ax.transAxes)
        wrapped = textwrap.wrap(v, 42) or ['']
        ax.text(0.0, y - 0.035, '\n'.join(wrapped), fontsize=8, va='top', transform=ax.transAxes)
        y -= 0.06 + 0.03 * len(wrapped)
    if per_group is not None and len(per_group):
        ax = fig.add_subplot(gs[0, 1:])
        x = np.arange(len(per_group))
        ax.vlines(x, per_group.s_q05, per_group.s_q95, color='k', lw=2)
        ax.axhline(1, color='r', lw=0.8, ls='--')
        ax.set_yscale('log'); ax.set_xticks(x)
        ax.set_xticklabels(per_group.group, rotation=45, ha='right', fontsize=6.5)
        ax.set_ylabel('depth factor s (q05-q95)', fontsize=8); ax.tick_params(axis='y', labelsize=7)
        ax.set_title('Depth factor within each group', fontsize=9)
        _panel_label(ax, 'A', -0.06)
        if 'extrinsic_noise' in per_group:
            # Extrinsic noise Var[c]/E[c]^2 of each group (Fang & Pachter), with its number of Poisson genes
            ax = fig.add_subplot(gs[1, 2])
            ax.bar(x, per_group.extrinsic_noise, color='#4C72B0')
            for xi, (v, n) in enumerate(zip(per_group.extrinsic_noise, per_group.n_poisson_genes)):
                if np.isfinite(v):
                    ax.text(xi, v, str(int(n)), ha='center', va='bottom', fontsize=5.5, rotation=90)
            ax.set_xticks(x); ax.set_xticklabels(per_group.group, rotation=45, ha='right', fontsize=6)
            ax.set_ylabel('extrinsic noise Var[c]/E[c]^2', fontsize=8); ax.tick_params(axis='y', labelsize=7)
            ax.set_title('Extrinsic noise per group (labels: Poisson genes)', fontsize=9)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            _panel_label(ax, 'C', -0.2)
    # Selected genes: within-group correlation with / without the depth factor
    if 'depth_factor' in R.adata_data.obs:
        A = R.adata_data
        X = _dense(A.X).astype(float)
        z = np.log(A.obs['depth_factor'].values.astype(float))
        keys = [k for k in ('dataset_id', 'time', LABEL_KEY) if k in A.obs]
        g = A.obs[keys].astype(str).agg('|'.join, axis=1).values
        raw, res, w = [], [], []
        for k in np.unique(g):
            m = g == k
            if m.sum() < 100:
                continue
            L = np.log1p(X[m]); L -= L.mean(axis=0); zz = z[m] - z[m].mean()
            Rm = L - np.outer(zz, (zz @ L) / max(zz @ zz, 1e-12))
            for M, acc in ((L, raw), (Rm, res)):
                ok = M.std(axis=0) > 0
                C = np.corrcoef(M[:, ok].T); acc.append(np.mean(C[np.triu_indices(C.shape[0], 1)]))
            w.append(m.sum())
        if w:
            ax = fig.add_subplot(gs[1, 1])
            ax.bar([0, 1], [np.average(raw, weights=w), np.average(res, weights=w)], color=['#999999', 'k'])
            ax.set_xticks([0, 1]); ax.set_xticklabels(['raw counts', 'depth removed'], fontsize=8)
            ax.set_ylabel('mean within-group gene-pair correlation', fontsize=8); ax.tick_params(labelsize=7)
            ax.set_title(f'Selected genes ({X.shape[1]})', fontsize=9)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            _panel_label(ax, 'B', -0.2)
    pdf.savefig(fig); plt.close(fig)


def _identity_page(pdf, R, idt):
    """Cell-type identity kept by the NB mixture (identity_mixture.csv of check_mixture_to_data), against the limit
    of each classifier on the real cells and the best a model with these modes and independent genes can do."""
    versions = [('data (cross-validated)', '#4C72B0', 'Data (cross-validated)'),
                ('data permuted within modes', '#8FBBD9', 'Data permuted within modes'),
                ('model draws', '#DD8452', 'NB mixture draws')]
    versions = [v for v in versions if v[0] in set(idt['version'])]
    cts = sorted(idt['cell_type'].unique())
    obs = idt[idt.version == 'observed'].set_index('cell_type')['share'].reindex(cts)
    clfs = [c for c in ('random forest', 'logistic regression') if c in set(idt['classifier'])]
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '1. Generative model — cell-type identity in the NB mixture',
                'One draw per real cell, per-sample classifiers trained on the other cells. Cross-validated data = '
                'classifier limit; permuted within modes (same sample, time, mode) = best model with these modes. '
                'Gap data → permuted: modes too coarse; permuted → draws: NB fit.')
    gs = gridspec.GridSpec(len(clfs), 2, figure=fig, left=0.06, right=0.98, top=0.84, bottom=0.12, hspace=0.45,
                           wspace=0.15)
    x = np.arange(len(cts))
    w = 0.8 / (len(versions) + 1)
    for i, c in enumerate(clfs):
        sub = idt[idt.classifier == c]
        for j, (what, ylab) in enumerate([('share', 'share of the cells'), ('recall', 'recall (true type kept)')]):
            ax = fig.add_subplot(gs[i, j])
            bars = ([('observed', obs.values, '#BBBBBB', 'Observed')] if what == 'share' else [])
            bars += [(v, sub[sub.version == v].set_index('cell_type')[what].reindex(cts).values, col, lab)
                     for v, col, lab in versions]
            for k, (_, vals, col, lab) in enumerate(bars):
                ax.bar(x + (k - (len(bars) - 1) / 2) * w, vals, w, color=col, label=lab)
            ax.set_xticks(x); ax.set_xticklabels(cts, rotation=30, ha='right', fontsize=7)
            ax.tick_params(axis='y', labelsize=7); ax.set_ylabel(ylab, fontsize=8)
            if what == 'recall':
                ax.set_ylim(0, 1)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.set_title(f'{c} — {"proportions" if what == "share" else "recall per cell type"}', fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False, ncol=2)
    pdf.savefig(fig); plt.close(fig)


def _integration_page(pdf, R, rep, sample_key='dataset_id'):
    """Mode means of each sample vs global ones, and UMAPs of raw vs integrated counts by sample."""
    A = R.adata_data
    fig = plt.figure(figsize=A4_LANDSCAPE)
    pct = rep.groupby('sample')['integrated'].mean() * 100
    integ = bool(float(rep['integrate_samples'].iloc[0])) if 'integrate_samples' in rep else True
    refs = rep['reference'].iloc[0] if 'reference' in rep else '?'
    _page_title(fig, f'1. Generative model — per-sample mixtures (integrate_samples = {integ})',
                f'Mixture fitted per sample; counts projected mode by mode on the reference sample {refs} '
                '(quantile matching). Genes projected: '
                + ', '.join(f'sample {s}: {v:.0f}%' for s, v in pct.items()) + '.')
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.06, right=0.97, top=0.86, bottom=0.14, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    samples = list(pct.index)
    cols = {s: plt.get_cmap('Set1')(i % 9) for i, s in enumerate(samples)}
    mode_cols = sorted(c for c in rep.columns if c.endswith('_sample'))
    for s in samples:
        r = rep[rep['sample'] == s]
        for mc in mode_cols:
            x = np.log10(r[mc.replace('_sample', '_global')].values + 1e-3)
            y = np.log10(r[mc].values + 1e-3)
            integ = r['integrated'].values
            ax.scatter(x[integ], y[integ], s=10, color=cols[s], alpha=0.7, linewidths=0)
            ax.scatter(x[~integ], y[~integ], s=12, facecolors='none', edgecolors=cols[s], linewidths=0.6)
    lim = ax.get_xlim()
    ax.plot(lim, lim, 'k--', lw=0.6)
    ax.set_xlabel('log10 reference mode mean', fontsize=8); ax.set_ylabel('log10 sample mode mean', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(handles=[Line2D([0], [0], marker='o', ls='', color=cols[s], label=f'sample {s}') for s in samples]
              + [Line2D([0], [0], marker='o', ls='', color='k', label='projected'),
                 Line2D([0], [0], marker='o', ls='', mfc='none', color='k', label='kept raw')],
              fontsize=6.5, frameon=False)
    ax.set_title('Mode means per sample (all modes)', fontsize=9)

    if 'counts_raw' in A.layers and sample_key in A.obs and R.condition_genes is None:  # common gene space only
        idx = _time_subsample(pd.to_numeric(A.obs['time']).values, R.n_umap, R.rng)
        lab = A.obs[sample_key].astype(str).values[idx]
        s_i = (A.obs['depth_factor'].to_numpy(dtype=float)[idx, None]
               if (R.depth_emb and 'depth_factor' in A.obs) else 1.0)
        for j, (X, title) in enumerate([(A.layers['counts_raw'], 'Raw counts'), (A.X, 'Integrated counts')]):
            E = Embedder(R.emb_method, seed=42).fit_transform(_preprocess(_dense(X[idx]) / s_i, R.norm, R.log))
            ax = fig.add_subplot(gs[0, j + 1])
            for s in samples:
                m = lab == str(s)
                ax.scatter(E[m, 0], E[m, 1], s=4, color=cols[s], linewidths=0, alpha=0.6, rasterized=True)
            _clean_umap_ax(ax, f'{title}, colour = sample')
    pdf.savefig(fig); plt.close(fig)


def _grn_pages(pdf, R, matrix, ns, show_stim, top_n=20, top_targets=10, label=''):
    # label: network condition shown in the titles ('' = common network)
    G_tot = matrix.shape[0]
    stim_names = ['Stimulus'] if ns == 1 else [f'Stimulus {s + 1}' for s in range(ns)]
    names = stim_names + R.genes
    M = matrix - np.diag(np.diag(matrix))
    genes_idx = np.arange(ns, G_tot)

    out_power = np.log1p(np.abs(M).sum(axis=1))
    in_power = np.log1p(np.abs(M[ns:, :]).sum(axis=0))  # incoming from genes only
    g_out, g_in = out_power[genes_idx], in_power[genes_idx]
    g_names = [names[i] for i in genes_idx]
    top_out = np.argsort(g_out)[::-1][:top_n]
    top_in = np.argsort(g_in)[::-1][:top_n]

    # Page: violin plots
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, f'2. Gene regulatory network{label} — regulatory power',
                f"Self-interactions excluded. Top {top_n} genes labelled"
                + (f"; stimulus shown as a star (stimulus = {R.stim} ≥ 0.5)." if show_stim else
                   f"; stimulus not shown (stimulus = {R.stim} < 0.5)."))
    gs = gridspec.GridSpec(1, 4, figure=fig, left=0.06, right=0.98, top=0.86, bottom=0.05, wspace=0.5,
                           width_ratios=[1, 1, 0.95, 0.95])
    extra = [(stim_names[s], out_power[s]) for s in range(ns)] if show_stim else None
    ax = fig.add_subplot(gs[0, 0])
    _labelled_violin(ax, g_out, g_names, top_out, 'Outgoing regulation (regulators)',
                     'log(1 + Σ|outgoing weights|)', extra=extra)
    _panel_label(ax, 'A', -0.25)
    ax = fig.add_subplot(gs[0, 1])
    _labelled_violin(ax, g_in, g_names, top_in, 'Incoming regulation (targets)', 'log(1 + Σ|incoming weights|)')
    _panel_label(ax, 'B', -0.25)

    # Tables: activation / inhibition balance of the top regulators (outgoing) and regulated genes (incoming, from genes)
    def power_table(ax, rows, head, title):
        tab = ax.table(cellText=rows, colLabels=['#', head, 'Σ|w|', 'act.', 'inh.'],
                       loc='upper center', cellLoc='center', colWidths=[0.1, 0.34, 0.2, 0.16, 0.16])
        tab.auto_set_font_size(False); tab.set_fontsize(6.5); tab.scale(1, 1.05)
        for (r, c), cell in tab.get_celld().items():
            cell.set_linewidth(0.3)
            if r == 0:
                cell.set_facecolor('#E8EEF7'); cell.set_text_props(fontweight='bold')
        ax.set_title(title, fontsize=9.5, fontweight='bold')

    counts = lambda w: [f'{np.abs(w).sum():.2f}', f'{(w > 0).sum()}', f'{(w < 0).sum()}']
    rows = [[f'{r + 1}', g_names[i], *counts(np.delete(M[genes_idx[i]], genes_idx[i]))] for r, i in enumerate(top_out)]
    if show_stim:
        rows += [['★', stim_names[s], *counts(M[s, ns:])] for s in range(ns)]
    ax = fig.add_subplot(gs[0, 2]); ax.axis('off'); _panel_label(ax, 'C', -0.05, 1.0)
    power_table(ax, rows, 'Regulator', f'Top {top_n} regulators')
    rows = [[f'{r + 1}', g_names[i], *counts(np.delete(M[ns:, genes_idx[i]], genes_idx[i] - ns))]
            for r, i in enumerate(top_in)]
    ax = fig.add_subplot(gs[0, 3]); ax.axis('off'); _panel_label(ax, 'D', -0.05, 1.0)
    power_table(ax, rows, 'Target', f'Top {top_n} regulated genes')
    pdf.savefig(fig); plt.close(fig)

    # Edges drawn: mutual top-k (among the k strongest targets of the regulator and regulators of the target)
    keep = _mutual_top(M, top_targets)
    if not show_stim:
        keep[:ns] = False
    max_intensity = float(np.abs(M).max() or 1.0)
    perturbed = R.perturbed_genes
    rule = (f'Edges shown only if among the {top_targets} strongest (|w|) targets of the regulator AND the '
            f'{top_targets} strongest regulators of the target; edge width ∝ |weight| (global scale). '
            'Perturbed genes (KO/OV) are highlighted in beige.')

    def star_pages(panels, title, incoming, center_label, center_color):
        # Small subgraphs, 4 x 6 per page (top 20 + stimuli on one page)
        n_r, n_c = 4, 6
        for start in range(0, len(panels), n_r * n_c):
            chunk = panels[start:start + n_r * n_c]
            fig = plt.figure(figsize=A4_LANDSCAPE)
            _page_title(fig, title, rule)
            gs = gridspec.GridSpec(n_r, n_c, figure=fig, left=0.01, right=0.99, top=0.9, bottom=0.06, hspace=0.25,
                                   wspace=0.05)
            for k, (i, col, sub_title) in enumerate(chunk):
                ax = fig.add_subplot(gs[k // n_c, k % n_c])
                G = _regulator_subgraph(M, names, names[i], top_targets, keep, incoming)
                _draw_regulator_subgraph(ax, G, names[i], max_intensity, col, sub_title, highlight=perturbed,
                                         empty='no major incoming edge' if incoming else 'no major outgoing edge',
                                         scale=0.7)
            fig.legend(handles=[Line2D([0], [0], color=ACT_COLOR, lw=2, label='Activation'),
                                Line2D([0], [0], color=INH_COLOR, lw=2, label='Inhibition'),
                                mpatches.Patch(color=center_color, label=center_label)]
                       + ([mpatches.Patch(color=STIM_COLOR, label='Stimulus')] if show_stim else []),
                       loc='lower center', ncol=4, frameon=False, fontsize=8)
            pdf.savefig(fig); plt.close(fig)

    # Top regulators (stimulus first if shown) and their main targets
    panels = [(s, STIM_COLOR, f'{stim_names[s]}') for s in range(ns)] if show_stim else []
    panels += [(genes_idx[i], REG_COLOR, f'#{r + 1} {g_names[i]}') for r, i in enumerate(top_out)]
    star_pages(panels, f'2. Gene regulatory network{label} — top regulators and their main targets', False, 'Top regulator',
               REG_COLOR)
    # Top regulated genes and their main regulators
    panels = [(genes_idx[i], TGT_COLOR, f'#{r + 1} {g_names[i]}') for r, i in enumerate(top_in)]
    star_pages(panels, f'2. Gene regulatory network{label} — top regulated genes and their main regulators', True,
               'Top regulated gene', TGT_COLOR)


def _edge_vectors(nets, ns, show_stim):
    """Signed edge weights of each network as vectors over the same edges: regulators = genes (+ stimuli if shown),
    targets = genes, self-regulations (often artefacts) excluded."""
    G_tot = next(iter(nets.values())).shape[0]
    rows = np.arange(0 if show_stim else ns, G_tot)
    cols = np.arange(ns, G_tot)
    mask = rows[:, None] != cols[None, :]
    return {k: np.asarray(M, dtype=float)[np.ix_(rows, cols)][mask] for k, M in nets.items()}


def network_similarity(nets, ns, show_stim=True, tol=1e-10):
    """
    Pairwise similarity of the condition networks (self-regulations excluded): cosine of the signed weights, and
    Jaccard index of the signed edge sets (an edge is shared if non-zero with the same sign in both). Returns
    (DataFrame of the pairs, {name: edge vector}).
    """
    vec = _edge_vectors(nets, ns, show_stim)
    names = list(vec)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = vec[a], vec[b]
            sx, sy = np.sign(x) * (np.abs(x) > tol), np.sign(y) * (np.abs(y) > tol)
            union = np.sum((sx != 0) | (sy != 0))
            rows.append(dict(network_a=a, network_b=b,
                             cosine=float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-300)),
                             jaccard=float(np.sum((sx == sy) & (sx != 0)) / union) if union else np.nan,
                             n_edges_a=int(np.sum(sx != 0)), n_edges_b=int(np.sum(sy != 0)),
                             n_shared=int(np.sum((sx == sy) & (sx != 0)))))
    return pd.DataFrame(rows), vec


def _top_sets(v, k, tol=1e-10):
    """Signed identities (index * sign) of the k strongest non-zero edges of v."""
    nz = np.flatnonzero(np.abs(v) > tol)
    top = nz[np.argsort(-np.abs(v[nz]))[:k]]
    return set((top + 1) * np.sign(v[top]).astype(int))


def _network_similarity_page(pdf, R, nets, ns, show_stim):
    """Cosine and Jaccard of the condition networks, and agreement of their strongest edges (self-loops excluded)."""
    nets = {k.replace(' — condition ', '').replace(' — sample ', 'sample ').strip() or 'common': v for k, v in nets.items()}
    sim, vec = network_similarity(nets, ns, show_stim)
    sim.to_csv(os.path.join(R.p, 'cardamomOT', f'network_similarity_{R.tag}.csv'), index=False)
    names = list(nets)
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '2. Gene regulatory network — similarity between network conditions',
                'Self-regulations (gene X → gene X) excluded' + ('' if show_stim else '; stimulus edges not included')
                + '. Cosine of the signed weights; Jaccard of the signed edge sets (shared = non-zero with the same sign). '
                'Curves: k strongest edges of each network (|w|).')
    gs = gridspec.GridSpec(1, 4, figure=fig, left=0.05, right=0.98, top=0.82, bottom=0.14, wspace=0.45,
                           width_ratios=[1, 1, 1.25, 1.25])
    for j, (metric, title) in enumerate([('cosine', 'Cosine similarity'), ('jaccard', 'Jaccard index (signed edges)')]):
        M = np.eye(len(names))
        for _, r in sim.iterrows():
            a, b = names.index(r['network_a']), names.index(r['network_b'])
            M[a, b] = M[b, a] = r[metric]
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(M, cmap='viridis', vmin=0, vmax=1)
        for (a, b), v in np.ndenumerate(M):
            ax.text(b, a, f'{v:.2f}', ha='center', va='center', fontsize=8, color='white' if v < 0.6 else 'black')
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=7)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.set_title(title, fontsize=9)
        _panel_label(ax, 'AB'[j], -0.22, 1.12)
    n_max = max(int(np.sum(np.abs(v) > 1e-10)) for v in vec.values())
    ks = np.unique(np.geomspace(5, max(n_max, 6), 30).astype(int))
    cmap = plt.get_cmap('tab10')
    ax_j, ax_r = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])
    c = 0
    for i, a in enumerate(names):
        for b in names:
            if a == b:
                continue
            ta = [_top_sets(vec[a], k) for k in ks]
            tb = [_top_sets(vec[b], k) for k in ks]
            # Recall of a's k strongest edges among b's non-zero edges (same sign)
            sb = _top_sets(vec[b], len(vec[b]))
            ax_r.plot(ks, [len(x & sb) / max(len(x), 1) for x in ta], color=cmap(c % 10), lw=1.2, label=f'{a} in {b}')
            if names.index(b) > i:
                ax_j.plot(ks, [len(x & y) / max(len(x | y), 1) for x, y in zip(ta, tb)], color=cmap(c % 10), lw=1.2,
                          label=f'{a} vs {b}')
            c += 1
    for ax, ylab, lab in ((ax_j, 'Jaccard of the k strongest edges', 'C'),
                          (ax_r, 'recall: share of the k strongest edges\nof a present in b (same sign)', 'D')):
        ax.set_xscale('log'); ax.set_xlabel('k (strongest edges)', fontsize=8); ax.set_ylabel(ylab, fontsize=8)
        ax.set_ylim(0, 1); ax.tick_params(labelsize=7); ax.legend(fontsize=6, frameon=False)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        _panel_label(ax, lab, -0.25)
    pdf.savefig(fig); plt.close(fig)


def _condition_differences_page(pdf, R, diff, pen=None, n_max=40):
    """Edges whose value differs between network conditions (network_differences_simul.csv), largest spread first."""
    conds = [c for c in diff.columns if c not in ('regulator', 'target', 'network', 'sign_change')]
    diff = diff[diff['regulator'].astype(str) != diff['target'].astype(str)]  # no self-regulation
    fig = plt.figure(figsize=A4_LANDSCAPE)
    if len(diff):
        diff = diff.assign(spread=diff[conds].max(axis=1) - diff[conds].min(axis=1)).sort_values('spread', ascending=False)
    _page_title(fig, '2. Gene regulatory network — differences between network conditions',
                f'{len(diff)} edges differ between the conditions ({int(diff["sign_change"].sum()) if len(diff) else 0} '
                f'change sign); the {min(n_max, len(diff))} largest differences. Each condition network = shared network '
                f'+ deviations penalised by network_condition_pen = {pen}.')
    if len(diff):
        top = diff.head(n_max)
        ax = fig.add_axes([0.25, 0.08, 0.7, 0.76])
        y = np.arange(len(top))[::-1]
        cols = plt.get_cmap('Set1')
        for k, c in enumerate(conds):
            ax.scatter(top[c].values, y, s=18, color=cols(k % 9), label=c, zorder=3)
        for yy, (_, r) in zip(y, top.iterrows()):
            ax.plot([r[conds].min(), r[conds].max()], [yy, yy], color='#bbbbbb', lw=1, zorder=1)
        ax.axvline(0, color='k', lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['regulator']} → {r['target']}" + (' *' if r['sign_change'] else '')
                            for _, r in top.iterrows()], fontsize=6.5)
        ax.set_xlabel('Interaction (* = sign change)', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        ax.legend(fontsize=7, frameon=False, loc='lower right')
    else:
        fig.text(0.5, 0.5, 'No edge differs between the conditions (fully fused networks).', ha='center', fontsize=11)
    pdf.savefig(fig); plt.close(fig)


def _perturbation_pages(pdf, R):
    done = [(l, d, A) for l, d, A in R.perturbations if A is not None]
    if not done:
        _error_page(pdf, '3. In-silico perturbations',
                    'No simulated perturbation found (perturbation_simulation sheet absent/empty, or '
                    'simulate_network_KOV.py + check_KOV_to_sim.py not run with this --stimulus/--prior).')
        return
    sim = R.stages['Simulation']
    t_sim = R.times(sim)
    t_last = np.max(t_sim)
    # Short ids P1, P2... (as on the cover page) for the overview panels
    pid = {l: f'P{i}' for i, (l, _, _) in enumerate(R.perturbations, start=1)}
    X_sim = _dense(sim.X).astype(float)

    # Population sizes of the branching simulations (proliferation MLP and RATE effects)
    pops = {pid[l]: np.asarray(A.uns['log_population'], dtype=float) for l, _, A in done if 'log_population' in A.uns}
    if pops and 'log_population' in sim.uns:
        pops = {'WT': np.asarray(sim.uns['log_population'], dtype=float), **pops}

    # Overview page across all perturbations
    if R.has_ct:
        fig = plt.figure(figsize=A4_LANDSCAPE)
        _page_title(fig, '3. In-silico perturbations — overview',
                    f'Cell types predicted by per-sample classifiers ({R.classifier_method}) trained on the observed data.'
                    + (' Population sizes from the branching simulation (net rates + RATE effects).'
                       if pops else ''))
        gs = gridspec.GridSpec(1, 3 if pops else 2, figure=fig, left=0.06, right=0.97, top=0.86, bottom=0.36,
                               wspace=0.55, width_ratios=[1.1, 1.1, 1] if pops else None)
        prop = pd.DataFrame(_with_predicted({'Data': _proportions(R.stages['Data'], R.categories),
                                             'Reference': _proportions(R.stages['Reference'], R.categories),
                                             'WT': _proportions(sim, R.categories),
                                             **{pid[l]: _proportions(A, R.categories) for l, _, A in done}},
                                            R.stages['Data'], R.categories)).T
        ax = fig.add_subplot(gs[0, 0]); _stacked_bars(ax, prop, R.color_map, 'Cell-type proportions (all times)')
        _panel_label(ax, 'A', -0.12)

        # Shift vs WT at the last simulated time point
        wt_last = _proportions(sim[t_sim == t_last], R.categories)
        delta = pd.DataFrame({pid[l]: _proportions(A[R.times(A) == t_last], R.categories) - wt_last
                              for l, _, A in done}).T
        ax = fig.add_subplot(gs[0, 1])
        vmax = max(float(np.abs(delta.values).max()), 1.0)
        im = ax.imshow(delta.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(R.categories))); ax.set_xticklabels(R.categories, rotation=35, ha='right', fontsize=7)
        ax.set_yticks(range(len(delta))); ax.set_yticklabels(delta.index, fontsize=7)
        for (a, b), v in np.ndenumerate(delta.values):
            ax.text(b, a, f'{v:+.0f}', ha='center', va='center', fontsize=6.5,
                    color='white' if abs(v) > 0.6 * vmax else 'black')
        cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label('Δ % vs WT', fontsize=8); cb.ax.tick_params(labelsize=7)
        ax.set_title(f'Cell-type shift vs WT simulation at t = {t_last:g}', fontsize=9)
        _panel_label(ax, 'B', -0.15)
        if pops:
            ax = fig.add_subplot(gs[0, 2])
            t_u = np.sort(np.unique(t_sim))
            cmap_p = plt.get_cmap('tab10')
            for i, (name, lp) in enumerate(pops.items()):
                if len(lp) != len(t_u):
                    continue
                wt = name == 'WT'
                ax.plot(t_u, np.exp(lp), color='k' if wt else cmap_p(i % 10), lw=2 if wt else 1.2,
                        marker='o', ms=2.5, label=name)
                ax.annotate(name, (t_u[-1], np.exp(lp[-1])), xytext=(3, 0), textcoords='offset points',
                            fontsize=6, va='center', color='k' if wt else cmap_p(i % 10))
            ax.set_yscale('log'); ax.set_xlabel('time', fontsize=8)
            ax.set_ylabel('population size (relative to t0)', fontsize=8); ax.tick_params(labelsize=7)
            ax.set_title('Population size', fontsize=9)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            _panel_label(ax, 'C', -0.2)
        # Key of the perturbation ids
        key_lines = [f'{pid[l]}  {d}' for l, d, _ in done]
        half = (len(key_lines) + 1) // 2
        for j, chunk in enumerate((key_lines[:half], key_lines[half:])):
            fig.text(0.06 + 0.47 * j, 0.25, '\n'.join(textwrap.shorten(x, 95) for x in chunk), fontsize=6.5,
                     va='top', family='monospace', linespacing=1.35)
        _celltype_legend(fig, R.color_map, y=0.02)
        pdf.savefig(fig); plt.close(fig)

    t_all = np.concatenate([R.times(R.stages['Data'], R.sub['Data']), R.times(sim, R.sub['Simulation'])])
    vmin, vmax = float(t_all.min()), float(t_all.max())
    bg = R.umap['Data']

    for label, desc, A in done:
        fig = plt.figure(figsize=A4_LANDSCAPE)
        pop_txt = ''
        if pid[label] in pops and 'WT' in pops:
            pop_txt = (f' Population at t = {t_last:g}: ×{np.exp(pops[pid[label]][-1] - pops["WT"][-1]):.2f} '
                       f'vs WT.')
        _page_title(fig, f'3. Perturbation {pid[label]} — {desc}',
                    f'{label}. UMAP: perturbed simulation projected on the WT embedding (grey = observed data).'
                    + pop_txt)
        gs = gridspec.GridSpec(2, 1, figure=fig, left=0.05, right=0.97, top=0.9, bottom=0.08,
                               hspace=0.3, height_ratios=[1.15, 1])
        smp_data = _sample_labels(R.stages['Data'], R.sub['Data'])
        g_top = gs[0].subgridspec(2, 4 if smp_data is not None else 3, hspace=0.12, wspace=0.05)
        if smp_data is not None:  # samples of the observed data
            _umap_sample(fig.add_subplot(g_top[0, 3]), R.umap['Data'], smp_data, 'Data — samples')
        cols = [('Data (observed)', R.umap['Data'], R.stages['Data'], R.sub['Data'], None),
                ('Simulation WT', R.umap['Simulation'], sim, R.sub['Simulation'], bg),
                (f'{pid[label]} (perturbed)', R.pert_umap[label], A, R.pert_sub[label], bg)]
        sca = None
        for j, (name, emb, B, idx, back) in enumerate(cols):
            sca = _umap_time(fig.add_subplot(g_top[0, j]), emb, R.times(B, idx), vmin, vmax, name, bg=back)
            ax = fig.add_subplot(g_top[1, j])
            if R.has_ct:
                _umap_celltype(ax, emb, R.labels(B, idx), R.color_map, '', bg=back)
            else:
                ax.axis('off')
        cax = fig.add_axes([0.975, 0.62, 0.007, 0.22])
        cb = fig.colorbar(sca, cax=cax); cb.set_label('time', fontsize=7); cb.ax.tick_params(labelsize=6)

        g_bot = gs[1].subgridspec(1, 3, wspace=0.35, width_ratios=[0.8, 1.2, 1.1])
        if R.has_ct:
            prop = pd.DataFrame(_with_predicted({'Data': _proportions(R.stages['Data'], R.categories),
                                                 'Reference': _proportions(R.stages['Reference'], R.categories),
                                                 'Sim. WT': _proportions(sim, R.categories),
                                                 'Perturbed': _proportions(A, R.categories)},
                                                R.stages['Data'], R.categories)).T
            _stacked_bars(fig.add_subplot(g_bot[0, 0]), prop, R.color_map, 'Cell-type proportions')
            _proportions_over_time(fig.add_subplot(g_bot[0, 1]), _proportions_by_time(sim, R.categories),
                                   _proportions_by_time(A, R.categories), R.color_map,
                                   'Cell types over time', 'WT', 'Perturbed')

        # Mean-expression log2 fold change vs WT at the last time point
        X_p = _dense(A.X).astype(float)
        t_p = R.times(A)
        lfc = np.log2((X_p[t_p == t_last].mean(axis=0) + 1) / (X_sim[t_sim == t_last].mean(axis=0) + 1))
        order = np.argsort(np.abs(lfc))[::-1][:15][::-1]
        ax = fig.add_subplot(g_bot[0, 2])
        ax.barh(range(len(order)), lfc[order], color=[ACT_COLOR if v > 0 else INH_COLOR for v in lfc[order]])
        ax.set_yticks(range(len(order))); ax.set_yticklabels([R.genes[i] for i in order], fontsize=6.5)
        ax.axvline(0, color='k', lw=0.5)
        ax.set_xlabel('log2 FC (mean, perturbed / WT)', fontsize=8); ax.tick_params(axis='x', labelsize=7)
        ax.set_title(f'Most affected genes at t = {t_last:g}', fontsize=9)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if R.has_ct:
            _celltype_legend(fig, R.color_map, y=0.0)
        pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------------------
# Learned dynamics: proliferation and velocity fields
# ---------------------------------------------------------------------------

class _Dynamics:
    """Trajectory states, network and growth estimates of the run (one row per trajectory state)."""

    def __init__(self, R):
        cdir = os.path.join(R.p, 'cardamomOT')
        load = lambda name: np.load(os.path.join(cdir, name)) if os.path.exists(os.path.join(cdir, name)) else None
        rna = load('data_rna.npy')
        prot = load('data_prot.npy')  # trajectories of fit_network (the only saved ones)
        self.times = load('data_times.npy')
        if rna is None or prot is None or self.times is None:
            raise FileNotFoundError('trajectory files (data_rna / data_prot / data_times) not found')
        self.ns = rna.shape[1] - len(R.genes)
        self.prot_full = prot
        self.P = prot[:, self.ns:]
        samples = load('data_samples.npy')
        self.samples = samples.astype(int) if samples is not None else np.zeros(len(self.times), dtype=int)
        self.tu = np.sort(np.unique(self.times))
        self.N = len(self.times) // len(self.tu)
        # Observed states (virtual ones: timepoints their sample misses)
        valid = load('data_traj_valid.npy')
        self.valid = (valid.astype(bool) if valid is not None and len(valid) == len(self.times)
                      else np.ones(len(self.times), bool))

        # mRNA along trajectories: NB mixture sample of each state (as in figure 5), else trajectory counts
        beta = R.beta_states
        self.M = (_dense(beta.X).astype(float) if beta.n_obs == len(self.times) else rna[:, self.ns:].astype(float))
        # Reference mRNA along the trajectories (counts of the real cells, at the reference depth if used)
        self.rna_ref = rna[:, self.ns:].astype(float)

        self.d = load('degradations.npy')
        self.a = load('mixture_parameters.npy')
        self.basal, self.inter = (None if x is None else drop_network_axis(x)
                                  for x in (load('basal_simul.npy'), load('inter_simul.npy')))
        from ..inference.sigma import load_sigma_spec_grn
        self.spec = load_sigma_spec_grn(cdir)  # families of the GRN sigma (model_learning_grn run), None = logistic

        # Real cell behind each state, its prior rate and cell type
        self.real_idx = load('data_traj_real_idx.npy')
        if self.real_idx is not None and len(self.real_idx) != len(self.times):
            self.real_idx = None
        obs = R.adata_data.obs
        self.prior_cells = obs['proliferation_net_rate'].values.astype(float) if 'proliferation_net_rate' in obs else None
        self.prior = self._per_state(self.prior_cells)
        self.ct = self._per_state(obs[LABEL_KEY].astype(str).values, fill='') if R.has_ct else None

        # Growth: OT pass (rate over the next interval) and proliferation MLP R(P)
        # Net growth of the states used after the inference: R_opt of the growth pass, or the prior rates
        self.recomputed = bool(getattr(R, 'use_recomputed_proliferation', True))
        self.R_opt = load_state_growth(cdir, self.recomputed)
        # Label of these rates: growth OT pass (R_opt), or the prior rates (use_recomputed_proliferation = False)
        self.growth_label = 'growth OT pass' if self.recomputed else 'prior rates (used)'
        self.growth_sym = 'R_opt' if self.recomputed else 'R_prior'
        if self.R_opt is not None and len(self.R_opt) != len(self.times):
            self.R_opt = None
        # Proliferation MLP: one, or one per network condition (ConditionProliferation, rows routed by sample)
        from ..inference.proliferations import load_proliferation, ConditionProliferation
        self.mlp = load_proliferation(cdir)
        self.per_condition_mlp = isinstance(self.mlp, ConditionProliferation)
        # Stimuli of the interval starting at each state (as in training and simulation)
        self.stim_state = interval_stimulus(prot, self.times, self.ns) if self.ns > 0 else np.zeros((len(self.times), 0))
        self.R_mlp = self.mlp.predict(self.P, self.stim_state, samples=self.samples) if self.mlp is not None else None
        # The effects of the inference stimuli on the rates (RATEk) are learned by the MLP from its stimulus inputs
        diag_path = os.path.join(cdir, 'prolif_network_diagnostics.json')
        self.mlp_diag = json.load(open(diag_path)) if (self.mlp is not None and os.path.exists(diag_path)) else None
        # Learned rate per state: MLP if trained, else the OT pass estimate
        self.R_learned = self.R_mlp if self.R_mlp is not None else self.R_opt
        self.learned_label = 'MLP R(P)' if self.R_mlp is not None else self.growth_label

        # Birth rates per state (dilution of the proteins): prior of the real cell, its regression on the state (MLP), and the
        # one of the simulations (MLP if the simulation used it, else the prior); refitted d1 per interval
        rate = lambda c: obs[c].to_numpy(dtype=float) if c in obs else None
        net = rate('proliferation_net_rate')
        b_cells = rate('proliferation_birth_rate')
        if b_cells is None and net is not None:
            b_cells = np.maximum(net, 0.0)
        d_cells = rate('proliferation_death_rate')
        if d_cells is None and net is not None:  # as the model: birth - net (max(-net, 0) without birth)
            d_cells = np.maximum(b_cells - net, 0.0)
        self.birth_prior, self.death_prior = self._per_state(b_cells), self._per_state(d_cells)
        self.birth_mlp = self.death_mlp = None
        if self.mlp is not None:  # two heads: regressed prior rates; one head: max(R, 0), max(-R, 0)
            self.birth_mlp = self.mlp.predict_birth(self.P, self.stim_state, samples=self.samples)
            self.death_mlp = self.mlp.predict_death(self.P, self.stim_state, samples=self.samples)
        # The simulations dilute at the MLP birth whenever an MLP is present (with or without branching)
        self.birth_sim = None
        if R.protein_dilution:
            self.birth_sim = self.birth_mlp if self.birth_mlp is not None else self.birth_prior
        self.birth_sim_label = ('MLP birth (prior regressed on the state)' if self.birth_mlp is not None
                                else "prior birth (obs['proliferation_birth_rate'])")
        self.d_t = load('degradations_temporal.npy')
        self.gene_cols = None  # all genes (for_condition: those of the condition)

    def for_condition(self, Rc):
        """Dynamics of the trajectories of the samples of the condition view Rc (its slots, its genes, its MLP), real
        cells indexed in Rc.adata_data."""
        import copy
        D = copy.copy(self)
        names = sorted(Rc.parent.adata_data.obs['dataset_id'].astype(str).unique()) \
            if 'dataset_id' in Rc.parent.adata_data.obs else ['0']
        s_idx = [names.index(x) for x in Rc.samples if x in names]
        T, N = len(self.tu), self.N
        slots = np.flatnonzero(np.isin(self.samples.reshape(T, N)[0], s_idx))
        rows = (np.arange(T)[:, None] * N + slots[None, :]).ravel()
        D.N = len(slots)
        take = lambda v: None if v is None else np.asarray(v)[rows]
        for attr in ('times', 'samples', 'valid', 'prot_full', 'P', 'M', 'rna_ref', 'prior', 'ct', 'R_opt', 'R_mlp',
                     'R_learned', 'birth_prior', 'death_prior', 'birth_mlp', 'death_mlp', 'birth_sim',
                     'stim_state'):
            setattr(D, attr, take(getattr(self, attr)))
        cols = np.asarray(Rc.gene_cols)
        D.gene_cols = cols
        D.P, D.M, D.rna_ref = D.P[:, cols], D.M[:, cols], D.rna_ref[:, cols]
        full = np.concatenate([np.arange(self.ns), self.ns + cols])
        D.d = None if self.d is None else self.d[..., full]
        if self.d_t is not None:
            dt = self.d_t[:, :, s_idx[0]] if self.d_t.ndim == 4 else self.d_t  # kinetics of the condition
            D.d_t = dt[..., full]
        # Real cells re-indexed in Rc.adata_data (-1: other condition)
        pos = np.full(len(Rc.cell_mask), -1)
        pos[Rc.cell_mask] = np.arange(int(Rc.cell_mask.sum()))
        D.real_idx = None if self.real_idx is None else np.where(self.real_idx[rows] >= 0,
                                                                 pos[np.maximum(self.real_idx[rows], 0)], -1)
        D.prior_cells = None if self.prior_cells is None else self.prior_cells[Rc.cell_mask]
        if self.per_condition_mlp:  # the condition's own MLP, on its genes (as D.P), and its diagnostics
            c = int(self.mlp.sample_cond[s_idx[0]])
            D.mlp, D.per_condition_mlp = self.mlp.nets[c], False
            name = str(self.mlp.names[c]) if self.mlp.names else str(c)
            D.mlp_diag = self.mlp_diag.get(name) if isinstance(self.mlp_diag, dict) else None
        return D

    def _per_state(self, per_cell, fill=np.nan):
        if per_cell is None or self.real_idx is None:
            return None
        out = np.full(len(self.real_idx), fill, dtype=per_cell.dtype if fill == '' else float)
        ok = self.real_idx >= 0
        out[ok] = per_cell[self.real_idx[ok]]
        return out

    def per_cell(self, per_state, n_cells):
        """Mean over the trajectory states behind each real cell (NaN if never reached)."""
        out = np.full(n_cells, np.nan)
        if per_state is None or self.real_idx is None:
            return out
        ok = (self.real_idx >= 0) & np.isfinite(per_state)
        s = np.bincount(self.real_idx[ok], weights=per_state[ok], minlength=n_cells)
        c = np.bincount(self.real_idx[ok], minlength=n_cells)
        out[c > 0] = s[c > 0] / c[c > 0]
        return out

    def kon(self):
        """Burst frequency of each state given the network, and its mRNA scale k1/c (per-sample basal and mixture)."""
        out = np.zeros_like(self.prot_full, dtype=float)
        scale = np.zeros_like(self.prot_full, dtype=float)
        for s in np.unique(self.samples):
            m = self.samples == s
            a = self.a[min(s, self.a.shape[0] - 1)] if self.a.ndim == 3 else self.a
            k1 = np.max(a[:-1], axis=0)
            b = self.basal[min(s, self.basal.shape[0] - 1)] if self.basal.ndim == 2 else self.basal
            i = self.inter[min(s, self.inter.shape[0] - 1)] if self.inter.ndim == 3 else self.inter
            out[m] = kon_ref_vector(self.prot_full[m].astype(float), (a[:-1] / k1).T, i, b, self.spec)
            scale[m] = k1 / a[-1]
        out, scale = out[:, self.ns:], scale[:, self.ns:]
        if self.gene_cols is not None:  # genes of the condition (for_condition)
            out, scale = out[:, self.gene_cols], scale[:, self.gene_cols]
        return out, scale

    def trajectory_velocity(self, X, rate=True):
        """(x_b − x_a)/(t_b − t_a) along each trajectory slot, from each observed time a to the next observed time b
        of its sample, the displacement x_b − x_a if not rate; NaN at the last observed time and the virtual states."""
        T, N = len(self.tu), self.N
        Y = X.reshape(T, N, -1)
        V = np.full_like(Y, np.nan, dtype=float)
        for a, b, slots in trajectory_pairs(self.valid, T):
            V[a, slots] = (Y[b, slots] - Y[a, slots]) / ((self.tu[b] - self.tu[a]) if rate else 1.0)
        return V.reshape(T * N, -1)


def _per_time_scale(v_meca, v_traj, times):
    """Rescale mechanistic velocities so that their norm matches the trajectory ones at each time (figure 5)."""
    out = v_meca.copy()
    alphas = {}
    for t in np.unique(times):
        m = (times == t) & np.all(np.isfinite(v_traj), axis=1)
        nm = np.linalg.norm(v_meca[m])
        if m.any() and nm > 0:
            alphas[t] = np.linalg.norm(v_traj[m]) / nm
    default = np.mean(list(alphas.values())) if alphas else 1.0
    for t in np.unique(times):
        out[times == t] *= alphas.get(t, default)
    return out


def _stream(ax, E, Vemb, c, title, cmap='viridis', vmin=None, vmax=None, n_grid=40, categorical=None,
            s=5, density=1.1, linewidth=0.7, alpha=None, color='#8B1A1A'):
    """Streamlines of the embedded velocities, smoothed on a grid (Gaussian kernel over neighbours)."""
    if categorical is not None:
        _umap_celltype(ax, E, c, categorical, title, s=s, alpha=1.0 if alpha is None else alpha)
        sca = None
    else:
        sca = ax.scatter(E[:, 0], E[:, 1], c=c, cmap=cmap, vmin=vmin, vmax=vmax, s=s, alpha=0.5 if alpha is None else alpha,
                         linewidths=0, rasterized=True)
        _clean_umap_ax(ax, title)
    lo, hi = E.min(axis=0), E.max(axis=0)
    xs, ys = np.linspace(lo[0], hi[0], n_grid), np.linspace(lo[1], hi[1], n_grid)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    dist, ind = NearestNeighbors(n_neighbors=min(30, len(E))).fit(E).kneighbors(grid)
    step = np.mean([xs[1] - xs[0], ys[1] - ys[0]])
    w = np.exp(-0.5 * (dist / step) ** 2)
    mass = w.sum(axis=1)
    Vg = (Vemb[ind] * w[..., None]).sum(axis=1) / np.maximum(mass, 1e-12)[:, None]
    Vg[mass < 0.1 * np.percentile(mass, 99)] = np.nan
    U, W = Vg[:, 0].reshape(gx.shape), Vg[:, 1].reshape(gx.shape)
    ax.streamplot(xs, ys, U, W, color=color, density=density, linewidth=linewidth, arrowsize=0.7 + 0.4 * (linewidth > 1))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    return sca


def _weighted_cosine(A, B):
    """Cosine similarity per state, averaged with weights ∝ |a|·|b| (figure 5)."""
    ok = np.all(np.isfinite(A), axis=1) & np.all(np.isfinite(B), axis=1)
    A, B = A[ok], B[ok]
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    w = na * nb
    if w.sum() == 0:
        return np.nan
    return float(np.sum((A * B).sum(axis=1) / np.maximum(w, 1e-12) * w) / w.sum())


def _proliferation_pages(pdf, R, D):
    if D.R_learned is None:
        raise FileNotFoundError('no growth estimate (data_R_opt.npy / data_R_prior.npy from infer_network_structure)')
    has_prior = D.prior_cells is not None
    n_cells = R.adata_data.n_obs
    learned_cells = D.per_cell(D.R_learned, n_cells)
    sub, emb = R.sub['Data'], R.umap['Data']
    prior_label = "prior (obs['proliferation_net_rate'])" if has_prior else 'prior: uniform (none provided)'

    # Page: maps and per-cell-type comparison
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '4. Proliferation — prior vs learned net rate (birth − death)',
                f"Learned = {D.learned_label}, mean over the trajectory states behind each observed cell "
                "(grey = cell not reached). Rates per time unit of the data.")
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.05, right=0.95, top=0.88, bottom=0.08, hspace=0.35, wspace=0.25)
    vals = np.concatenate([learned_cells[np.isfinite(learned_cells)]]
                          + ([D.prior_cells[np.isfinite(D.prior_cells)]] if has_prior else []))
    lo, hi = (np.percentile(vals, [2, 98]) if vals.size else (0, 1))

    def _rate_map(ax, v, title, cmap, vmin, vmax):
        ax.scatter(emb[:, 0], emb[:, 1], c='#DDDDDD', s=4, linewidths=0, rasterized=True)
        vs = v[sub]
        ok = np.isfinite(vs)
        sca = ax.scatter(emb[ok, 0], emb[ok, 1], c=vs[ok], cmap=cmap, vmin=vmin, vmax=vmax, s=4,
                         linewidths=0, rasterized=True)
        _clean_umap_ax(ax, title)
        cb = fig.colorbar(sca, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=6)

    if has_prior:
        _rate_map(fig.add_subplot(gs[0, 0]), D.prior_cells, 'Prior net rate', 'viridis', lo, hi)
    else:
        ax = fig.add_subplot(gs[0, 0]); ax.axis('off')
        ax.text(0.5, 0.5, 'No prior provided:\nuniform net rate\n(only relative rates are learned)',
                ha='center', va='center', fontsize=9, color='#555555', transform=ax.transAxes)
    _rate_map(fig.add_subplot(gs[0, 1]), learned_cells, f'Learned net rate ({D.learned_label})', 'viridis', lo, hi)
    diff = learned_cells - (D.prior_cells if has_prior else np.nanmean(learned_cells))
    dm = np.nanpercentile(np.abs(diff), 98) if np.isfinite(diff).any() else 1.0
    _rate_map(fig.add_subplot(gs[0, 2]), diff,
              'Learned − prior' if has_prior else 'Learned − mean', 'RdBu_r', -dm, dm)

    # Cell types of the proliferation anchors: cell_type_proliferation, else cell_type_transition, else cell_type
    pkey = resolve_cell_type_obs(R.adata_data, 'proliferation')
    if pkey is not None:
        p_cells = R.adata_data.obs[pkey].astype(str).values
        p_cats = R.categories if pkey == LABEL_KEY else sorted(set(p_cells))
        p_cmap = R.color_map if pkey == LABEL_KEY else _cell_type_colors(p_cats)
        p_states = D.ct if pkey == LABEL_KEY else D._per_state(p_cells, fill='')
    else:
        p_cells = p_cats = p_cmap = p_states = None

    if pkey is not None:
        cats = p_cats
        ct_cells = p_cells
        ax = fig.add_subplot(gs[1, :2])
        pos = np.arange(len(cats))
        for off, v, col, lab in [(-0.18, D.prior_cells, '#9AA5B1', prior_label), (0.18, learned_cells, REG_COLOR, 'learned')]:
            if v is None:
                continue
            data = [v[(ct_cells == c) & np.isfinite(v)] for c in cats]
            keep = [i for i, x in enumerate(data) if len(x) > 1]
            if keep:
                parts = ax.violinplot([data[i] for i in keep], positions=pos[keep] + off, widths=0.32,
                                      showmeans=True, showextrema=False)
                for b in parts['bodies']:
                    b.set_facecolor(col); b.set_alpha(0.7)
                parts['cmeans'].set_color('k')
            ax.plot([], [], color=col, lw=6, label=lab)
        ax.set_xticks(pos); ax.set_xticklabels(cats, rotation=25, ha='right', fontsize=7)
        ax.set_ylabel('net rate', fontsize=8); ax.tick_params(axis='y', labelsize=7)
        ax.axhline(0, color='k', lw=0.4)
        ax.legend(fontsize=7, frameon=False, loc='upper right')
        ax.set_title(f'Net proliferation rate per cell type ({pkey})', fontsize=9)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

        # Table: anchor rate (sheet proliferation_rates) vs prior and learned means
        anchors = {}
        pr_path = find_data_file(input_dir(R.p), 'proliferation_rates')
        if pr_path is not None:
            tab = pd.read_csv(pr_path, sep=None, engine='python', header=None)
            anchors = {str(k): float(v) for k, v in zip(tab.iloc[:, 0], tab.iloc[:, 1])}
        fmt = lambda x: f'{x:.3g}' if np.isfinite(x) else '—'
        mean_of = lambda v, c: np.nanmean(v[ct_cells == c]) if v is not None and np.isfinite(v[ct_cells == c]).any() \
            else np.nan
        # Anchors are rates without stimulus: add the mean effect of the inference stimuli on the cells of each type,
        # as in the prior and in the MLP (trained with the stimulus)
        obs_p = R.adata_data.obs
        stim_eff = (obs_p['proliferation_net_rate'].to_numpy(dtype=float)
                    - obs_p['proliferation_net_rate_base'].to_numpy(dtype=float)
                    if {'proliferation_net_rate', 'proliferation_net_rate_base'} <= set(obs_p.columns) else None)
        anchor_of = lambda c: anchors.get(c, np.nan) + (0.0 if stim_eff is None else np.nan_to_num(mean_of(stim_eff, c)))
        rows = [[c, fmt(anchor_of(c)),
                 fmt(mean_of(D.prior_cells, c)) if has_prior else '—', fmt(mean_of(learned_cells, c)),
                 fmt(mean_of(learned_cells, c) - mean_of(D.prior_cells, c)) if has_prior else '—']
                for c in cats]
        ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
        t = ax.table(cellText=rows, colLabels=['Cell type', 'Anchor' + (' + stimulus' if stim_eff is not None else ''),
                                               'Prior', 'Learned', 'Δ'],
                     loc='upper center', cellLoc='center', colWidths=[0.30, 0.28, 0.15, 0.15, 0.14])
        t.auto_set_font_size(False); t.set_fontsize(7); t.scale(1, 1.3)
        for (r, c), cell in t.get_celld().items():
            cell.set_linewidth(0.3)
            if r == 0:
                cell.set_facecolor('#E8EEF7'); cell.set_text_props(fontweight='bold')
        ax.set_title(f'Mean net rate per cell type ({pkey}; Δ = learned − prior)', fontsize=9)
        # Can R(P) separate these types at all? MLP fit on its target, and separability of the types from the proteins
        notes = []
        if D.R_mlp is not None and D.R_opt is not None:
            ok = np.isfinite(D.R_opt) & np.isfinite(D.R_learned)
            if ok.sum() > 2 and np.var(D.R_opt[ok]) > 0:
                r2 = 1 - np.mean((D.R_learned[ok] - D.R_opt[ok]) ** 2) / np.var(D.R_opt[ok])
                notes.append(f"MLP fit of its target ({D.growth_sym}, per state): R² = {r2:.2f}")
        if p_states is not None and len(set(p_states) - {''}) > 1:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            lab = np.asarray(p_states)
            idx = np.flatnonzero(lab != '')
            idx = R.rng.choice(idx, min(len(idx), 5000), replace=False)
            if min(np.bincount(np.unique(lab[idx], return_inverse=True)[1])) >= 3:
                acc = cross_val_score(LogisticRegression(max_iter=2000, class_weight='balanced'), D.P[idx], lab[idx],
                                      cv=3, scoring='balanced_accuracy').mean()
                notes.append(f"types separated from the model's proteins with balanced accuracy {acc:.2f} "
                             f"(logistic regression, 3-fold; chance {1 / len(set(lab[idx])):.2f}): R(P) cannot learn "
                             "rates that differ between types it cannot tell apart")
        if notes:
            ax.text(0.0, 0.05, textwrap.fill('. '.join(notes) + '.', 60), transform=ax.transAxes, fontsize=7,
                    va='bottom', ha='left', color='#444444')
    pdf.savefig(fig); plt.close(fig)

    # Page: population growth, rates over time, genes driving R
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '4. Proliferation — population growth and drivers',
                'Relative population size from the mean growth factor per interval; '
                'its absolute level follows the prior (anchored on the sheet population_sizes by fit_population_anchors, if filled).')
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.06, right=0.97, top=0.86, bottom=0.12, wspace=0.35)
    T, N, dt = len(D.tu), D.N, np.diff(D.tu)
    ax = fig.add_subplot(gs[0, 0])
    curves = []
    if D.prior is not None:
        pr = D.prior.reshape(T, N)[:-1]
        curves.append(('prior', '#9AA5B1', '--', np.nanmean(np.exp(pr * dt[:, None]), axis=1)))
    if D.R_opt is not None:
        ro = D.R_opt.reshape(T, N)[:-1]
        curves.append((D.growth_label, '#E67E22', '-', np.nanmean(np.exp(ro * dt[:, None]), axis=1)))
    if D.mlp is not None:
        states, _ = growth_path_states(D.prot_full, D.times, ns=D.ns, n_nodes=5)
        if D.gene_cols is not None:  # genes of the condition (inputs of its MLP)
            states = states[..., D.gene_cols]
        _, w = quadrature(5)
        S = D.prot_full[:T * N, :D.ns].reshape(T, N, D.ns)[1:, None]       # stimuli of each interval
        integ = (D.mlp.predict(states, S) * w[None, :, None]).sum(axis=1) * dt[:, None]
        curves.append(('MLP R(P) along paths', REG_COLOR, '-', np.mean(np.exp(integ), axis=1)))
    for z, (lab, col, ls, g) in enumerate(curves):
        # Prior drawn last: the OT pass is anchored on it (same mean growth)
        ax.plot(D.tu, np.concatenate([[0], np.cumsum(np.log(g))]), color=col, ls=ls, marker='o', ms=3, label=lab,
                lw=1.6 if lab == 'prior' else 1.2, zorder=10 if lab == 'prior' else z)
    ax.set_xlabel('time', fontsize=8); ax.set_ylabel('log population size (relative to t0)', fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=7, frameon=False)
    ax.set_title('Population growth', fontsize=9)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    if p_states is not None:
        for c, col in p_cmap.items():
            m = p_states == c
            if not m.any():
                continue
            mean_t = [np.nanmean(D.R_learned[m & (D.times == t)]) if (m & (D.times == t)).any() else np.nan for t in D.tu]
            ax.plot(D.tu, mean_t, color=col, marker='o', ms=2.5, lw=1.4)
            if D.prior is not None:
                ax.plot(D.tu, [np.nanmean(D.prior[m & (D.times == t)]) if (m & (D.times == t)).any() else np.nan
                               for t in D.tu], color=col, ls='--', lw=0.9, alpha=0.8)
        ax.legend(handles=[Line2D([0], [0], color='k', lw=1.4, label='learned'),
                           Line2D([0], [0], color='k', ls='--', lw=0.9, label='prior')],
                  fontsize=7, frameon=False)
    else:
        mean_t = [np.nanmean(D.R_learned[D.times == t]) for t in D.tu]
        ax.plot(D.tu, mean_t, color=REG_COLOR, marker='o', ms=2.5)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('time', fontsize=8); ax.set_ylabel('mean net rate', fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title('Net rate over time' + (f' per cell type ({pkey})' if p_states is not None else ''), fontsize=9)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    if D.mlp is not None:
        import torch
        k = int(D.mlp.n_stim)
        X = torch.tensor(np.hstack([D.stim_state[:, :k], D.P]), dtype=torch.float32, requires_grad=True)
        D.mlp(X).sum().backward()
        imp = X.grad.numpy()[:, k:].mean(axis=0) * D.P.std(axis=0)
        order = np.argsort(np.abs(imp))[::-1][:15][::-1]
        ax.barh(range(len(order)), imp[order], color=[ACT_COLOR if v > 0 else INH_COLOR for v in imp[order]])
        ax.set_yticks(range(len(order))); ax.set_yticklabels([R.genes[i] for i in order], fontsize=6.5)
        ax.axvline(0, color='k', lw=0.5)
        ax.set_xlabel('mean ∂R/∂P × sd(P)', fontsize=8); ax.tick_params(axis='x', labelsize=7)
        ax.set_title('Proteins driving the learned net rate', fontsize=9)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No proliferation MLP\n(simulate_with_proliferation = True,\nthen infer_network_simul)',
                ha='center', va='center', fontsize=9, color='#555555', transform=ax.transAxes)
    if p_states is not None:
        _celltype_legend(fig, p_cmap, y=0.01)
    pdf.savefig(fig); plt.close(fig)

    # Page: cell-type composition of the trajectories and of the simulation, and MLP fit
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '4. Proliferation — cell-type composition and MLP fit',
                'Trajectory states are drawn at each time among the observed cells (target marginal of the OT): they '
                'follow the observed composition, which a simulation with proliferation (branching) should reproduce.'
                + ((' Branching at the net rate of the proliferation MLP.' if D.mlp is not None else
                    ' Branching at the net growth of the nearest trajectory states (proliferation_mlp = False).')
                   if R.sim_prolif else ' The simulation has no proliferation.'))
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.06, right=0.97, top=0.86, bottom=0.14, wspace=0.35)
    t_obs = pd.to_numeric(R.adata_data.obs['time']).values
    sim = R.stages['Simulation']
    sim_lab = R.labels(sim) if (pkey == LABEL_KEY and R.has_ct) else None
    t_sim = R.times(sim)

    def _props(labels, times, weights=None):
        # Cell-type proportions (%) per time, optionally weighted
        out = np.full((len(D.tu), len(p_cats)), np.nan)
        for i, t in enumerate(D.tu):
            m = (times == t) & (labels != '')
            if not m.any():
                continue
            w = np.ones(m.sum()) if weights is None else weights[m]
            out[i] = [100 * w[labels[m] == c].sum() / w.sum() for c in p_cats]
        return out

    ax = fig.add_subplot(gs[0, 0])
    if p_states is not None:
        # Trajectories: annotated labels of their real cells (as the data); simulation: labels of the classifier, compared
        # with the out-of-fold predicted labels of the data
        props = {'data': _props(p_cells, t_obs), 'trajectories': _props(p_states, D.times)}
        if sim_lab is not None:
            obs_d = R.stages['Data'].obs  # rows of R.adata_data
            if LABEL_PRED in obs_d:
                props['data (predicted)'] = _props(obs_d[LABEL_PRED].astype(str).values, t_obs)
            props['simulation'] = _props(sim_lab, t_sim)
        styles = {'data': ('-', 2.0), 'data (predicted)': ('-', 0.8), 'trajectories': (':', 1.2),
                  'simulation': ('-.', 1.0)}
        for j, c in enumerate(p_cats):
            for name, P in props.items():
                ls, lw = styles[name]
                ax.plot(D.tu, P[:, j], color=p_cmap[c], ls=ls, lw=lw)
        ax.legend(handles=[Line2D([0], [0], color='k', ls=styles[n][0], lw=styles[n][1], label=n) for n in props],
                  fontsize=6.5, frameon=False)
        ax.set_ylim(0, 100); ax.set_xlabel('time', fontsize=8); ax.set_ylabel('% of cells', fontsize=8)
        ax.tick_params(labelsize=7); ax.set_title(f'Cell-type proportions ({pkey})', fontsize=9)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

        # Distance to the data: total variation between proportion vectors, per time
        ax = fig.add_subplot(gs[0, 1])
        for name, P in props.items():
            if name.startswith('data'):
                continue
            ref_p = props['data (predicted)'] if (name == 'simulation' and 'data (predicted)' in props) else props['data']
            tv = 0.5 * np.nansum(np.abs(P - ref_p), axis=1)
            tv[np.all(np.isnan(P), axis=1)] = np.nan
            vs = ' vs predicted data' if ref_p is not props['data'] else ''
            ax.plot(D.tu, tv, ls=styles[name][0], marker='o', ms=3, lw=1.3, label=f'{name}{vs} (mean {np.nanmean(tv):.1f})',
                    color={'trajectories': '#9AA5B1', 'simulation': REG_COLOR}[name])
        ax.set_xlabel('time', fontsize=8); ax.set_ylabel('distance to data (total variation, %)', fontsize=8)
        ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, frameon=False)
        ax.set_title('Cell-type composition vs observed data', fontsize=9)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Needs cell types of the trajectory states',
                ha='center', va='center', fontsize=9, color='#555555', transform=ax.transAxes)

    ax = fig.add_subplot(gs[0, 2])
    if D.mlp_diag is not None:
        d = D.mlp_diag
        x = np.arange(len(d['growth_target']))
        ax.bar(x - 0.18, d['growth_target'], width=0.36, color='#E67E22', label=D.growth_label)
        ax.bar(x + 0.18, d['growth_mlp'], width=0.36, color=REG_COLOR, label='MLP along paths')
        ax.set_xticks(x); ax.set_xticklabels([f"{t:g}" for t in d['interval_start']], fontsize=7)
        ax.axhline(0, color='k', lw=0.4)
        ax.set_xlabel('interval start', fontsize=8); ax.set_ylabel('log population growth', fontsize=8)
        ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, frameon=False, loc='upper left')
        fmt = lambda v: f'{v:.2f}' if v is not None and np.isfinite(v) else '—'
        ax.set_title(f"MLP fit — R² of path log gains: held-out {fmt(d['r2_val'])}, train {fmt(d['r2_train'])}\n"
                     f"({d['n_val_paths']} held-out paths, best epoch {d['best_epoch']}, "
                     f"stimulus input {'yes' if d.get('with_stimulus') else 'no'})", fontsize=8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No MLP fit diagnostics\n(rerun infer_network_simul\nwith proliferation)',
                ha='center', va='center', fontsize=9, color='#555555', transform=ax.transAxes)
    if p_states is not None:
        _celltype_legend(fig, p_cmap, y=0.01)
    pdf.savefig(fig); plt.close(fig)


def _dilution_page(pdf, R, D):
    """Protein dilution at the birth rate b: effective half-lives, fraction of the way to equilibrium per interval,
    birth rates per cell type (prior vs learned)."""
    if D.birth_sim is None:
        raise ValueError('protein_dilution = False, or no birth rate (obs proliferation_birth_rate / net rate)')
    ns = D.ns
    var = R.adata_data.var
    d1_lit = var['d1'].to_numpy(dtype=float) if 'd1' in var else D.d[1, ns:]
    # Literature in the units of the refined rate: d1_eff = d0 d1 / (d0 + d1) for the ODE and the protein-only PDMP
    eff = getattr(R, 'simulation_model', 'ode') != 'sde2'
    if eff:
        d0_lit = var['d0'].to_numpy(dtype=float) if 'd0' in var else D.d[0, ns:]
        d1_lit = d0_lit * d1_lit / (d0_lit + d1_lit)
    # Temporal d1 (mean over the samples with kinetics per network condition)
    d1_tm = None if D.d_t is None else (D.d_t[:, 1] if D.d_t.ndim == 3 else np.nanmean(D.d_t[:, 1], axis=1))
    d1_fit = np.nanmean(d1_tm[:, ns:], axis=0) if d1_tm is not None else D.d[1, ns:]
    b_states = np.nan_to_num(D.birth_sim)
    b_mean = float(np.mean(b_states))
    hl_lit, hl_eff = np.log(2) / d1_lit, np.log(2) / (d1_fit + b_mean)
    share = b_mean / (d1_fit + b_mean)

    # mRNA (mrna_dilution): relaxing at d0 + b, d0 = literature turnover minus the growth of its source cell lines
    mrna = bool(getattr(R, 'mrna_dilution', False)) and 'd0' in var and D.d is not None
    mrna_txt = ' mRNA too (mrna_dilution): relaxation at d0 + b, same levels.' if mrna else ''

    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '4. Proliferation — protein dilution',
                f'Proteins diluted at the birth rate b of each cell: dP/dt = d1 u − (d1 + b) P (d1 pure degradation, '
                f'refitted with the dilution). b of the simulations: {D.birth_sim_label}, mean {b_mean:.4f} h^-1.'
                + mrna_txt)
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.07, right=0.97, top=0.86, bottom=0.1, hspace=0.45, wspace=0.35)

    # A: effective vs literature half-lives per gene
    ax = fig.add_subplot(gs[:, 0])
    sca = ax.scatter(hl_lit, hl_eff, c=share, cmap='viridis', vmin=0, vmax=1, s=12, linewidths=0)
    lim = [min(hl_lit.min(), hl_eff.min()) * 0.8, max(hl_lit.max(), hl_eff.max()) * 1.2]
    ax.plot(lim, lim, 'k--', lw=0.6)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim)
    d1_name = 'd1_eff' if eff else 'd1'
    ax.set_xlabel(f'literature half-life ln2/{d1_name} (h)', fontsize=8)
    ax.set_ylabel(f'effective half-life ln2/({d1_name} refit + mean b) (h)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title('Protein half-lives per gene', fontsize=9)
    cb = fig.colorbar(sca, ax=ax, fraction=0.05, pad=0.02); cb.set_label('share of dilution b/(d1 + b)', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    _panel_label(ax, 'A', -0.2)

    # B: median fraction of the way to the equilibrium over each interval, without / with dilution
    ax = fig.add_subplot(gs[0, 1:])
    dts = np.diff(D.tu)
    T, N = len(D.tu), D.N
    b_int = b_states.reshape(T, N)[:-1].mean(axis=1)
    d1_int = d1_tm[:, ns:] if d1_tm is not None and len(d1_tm) == len(dts) else np.tile(d1_fit, (len(dts), 1))
    f_lit = [np.median(1 - np.exp(-d1_lit * dt)) for dt in dts]
    f_dil = [np.median(1 - np.exp(-(d1_int[k] + b_int[k]) * dt)) for k, dt in enumerate(dts)]
    x = np.arange(len(dts))
    ax.bar(x - 0.2, f_lit, 0.4, color='#AAAAAA', label='literature d1, no dilution')
    ax.bar(x + 0.2, f_dil, 0.4, color='#4C9BE8', label='refitted d1 + b (simulations)')
    ax.set_xticks(x); ax.set_xticklabels([f'{a:g}→{b:g}' for a, b in zip(D.tu[:-1], D.tu[1:])], fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel('median fraction to equilibrium', fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title('How far the proteins can move over each interval: 1 − exp(−(d1 + b) Δt)', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    _panel_label(ax, 'B', -0.08)

    # C: birth rates per cell type, prior vs regressed on the states (MLP)
    ax = fig.add_subplot(gs[1, 1])
    key = 'cell_type_proliferation' if 'cell_type_proliferation' in R.adata_data.obs else LABEL_KEY
    labels = D._per_state(R.adata_data.obs[key].astype(str).values, fill='') if key in R.adata_data.obs else None
    if labels is not None:
        cats = [c for c in sorted(set(labels)) if c]
        pos = np.arange(len(cats))
        for j, (vals, col, lab) in enumerate([(D.birth_prior, '#AAAAAA', 'prior'), (D.birth_mlp, '#4C9BE8', 'MLP')]):
            if vals is None:
                continue
            data = [np.nan_to_num(vals[labels == c]) for c in cats]
            parts = ax.violinplot(data, positions=pos + (j - 0.5) * 0.35, widths=0.32, showextrema=False, showmeans=True)
            for bd in parts['bodies']:
                bd.set_facecolor(col); bd.set_alpha(0.7)
            ax.plot([], [], color=col, lw=6, label=lab)
        ax.set_xticks(pos); ax.set_xticklabels(cats, rotation=30, ha='right', fontsize=7)
        ax.legend(fontsize=7, frameon=False)
    ax.set_ylabel('birth rate b (h$^{-1}$)', fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title(f'Birth rate per {key}', fontsize=9)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    _panel_label(ax, 'C', -0.2)

    # D: summary
    ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
    rows = [[f'median literature {d1_name} (h^-1)', f'{np.median(d1_lit):.4f}'],
            [f'median refitted {d1_name} (h^-1)', f'{np.median(d1_fit):.4f}'],
            ['mean birth b (h^-1)', f'{b_mean:.4f}'],
            ['median half-life, literature (h)', f'{np.median(hl_lit):.1f}'],
            ['median effective half-life (h)', f'{np.median(hl_eff):.1f}'],
            ['median share of dilution', f'{np.median(share):.2f}']]
    if D.death_prior is not None:
        rows.append(['mean death, prior (h^-1)', f'{np.nanmean(D.death_prior):.4f}'])
    if D.birth_mlp is not None:
        rows += [['mean birth, MLP (h^-1)', f'{np.mean(D.birth_mlp):.4f}'],
                 ['mean death, MLP (h^-1)', f'{np.mean(D.death_mlp):.4f}']]
    if mrna:
        d0_tab, d0_pure = var['d0'].to_numpy(dtype=float), np.maximum(D.d[0, ns:], 1e-12)
        mu = R.adata_data.uns.get('degradation_rates', {}).get('mrna_growth_rate')
        rows += [['median mRNA half-life, literature (h)', f'{np.log(2) / np.median(d0_tab):.2f}'],
                 ['growth of its source cell lines (h^-1)', '—' if mu is None else f'{float(mu):.4f}'],
                 ['median mRNA half-life, pure decay d0 (h)', f'{np.log(2) / np.median(d0_pure):.2f}'],
                 ['median effective mRNA half-life (h)', f'{np.log(2) / np.median(d0_pure + b_mean):.2f}'],
                 ['median share of dilution, mRNA', f'{np.median(b_mean / (d0_pure + b_mean)):.2f}']]
    tab = ax.table(cellText=rows, loc='upper center', cellLoc='left', colWidths=[0.75, 0.25])
    tab.auto_set_font_size(False); tab.set_fontsize(7.5); tab.scale(1, 1.4)
    for cell in tab.get_celld().values():
        cell.set_linewidth(0.3)
    _panel_label(ax, 'D', -0.05, 1.0)
    pdf.savefig(fig); plt.close(fig)


def _velocity_pages(pdf, R, D, k_top=8):
    if D.d is None or D.a is None or D.basal is None or D.inter is None:
        raise FileNotFoundError('network/degradation files needed for the mechanistic velocities')
    idx = np.arange(len(D.times))  # every trajectory state
    t = D.times[idx]
    kon, scale_m = D.kon()
    d0, d1 = D.d[0, D.ns:], D.d[1, D.ns:]

    # Reference mRNA (inferred at the reference depth if the run used depth factors), shown at the reference
    # depth with cell_depth_for_representation, else at the depth of the real cells (as the NB states D.M)
    M_ref = D.rna_ref
    depth = R.adata_data.obs['depth_factor'].to_numpy(dtype=float) if 'depth_factor' in R.adata_data.obs else None
    s_state = (np.where(D.real_idx >= 0, depth[np.maximum(D.real_idx, 0)], 1.0)[:, None]
               if depth is not None and D.real_idx is not None else 1.0)
    if R.depth_emb and not R.use_depth:
        M_ref = M_ref / s_state
    elif not R.depth_emb and R.use_depth:
        M_ref = M_ref * s_state
    # mRNA: mechanistic velocity on the NB-sampled counts (top-k genes kept), and along the reference trajectories
    # Trajectories: rate for the scale of the mechanistic field and the agreement, displacement drawn
    vM_traj = D.trajectory_velocity(M_ref)
    dM_traj = D.trajectory_velocity(M_ref, rate=False)
    s_m = s_state if (R.use_depth and not R.depth_emb) else 1.0  # NB states drawn at the cells' depth
    vM_meca = _per_time_scale(d0 * (scale_m * kon * s_m - D.M), vM_traj, D.times)
    top = np.argpartition(np.abs(vM_meca), -min(k_top, vM_meca.shape[1]), axis=1)[:, -min(k_top, vM_meca.shape[1]):]
    mask = np.zeros_like(vM_meca, dtype=bool)
    mask[np.arange(len(mask))[:, None], top] = True
    vM_meca = np.where(mask, vM_meca, 0.0)
    # Proteins: mechanistic velocity d1·kon − (d1 + b)·P (dilution at the birth rate b of the simulations) and
    # along trajectories
    vP_traj = D.trajectory_velocity(D.P)
    dP_traj = D.trajectory_velocity(D.P, rate=False)
    dil = 0.0 if D.birth_sim is None else np.nan_to_num(D.birth_sim)[:, None] * D.P
    vP_meca = _per_time_scale(d1 * (kon - D.P) - dil, vP_traj, D.times)

    growth = D.R_learned[idx] if D.R_learned is not None else None
    for name, X, v_meca, v_traj, d_traj, to_space, red in [
        ('mRNA', np.log1p(M_ref[idx]), vM_meca[idx], vM_traj[idx], dM_traj[idx], lambda v: v / (1 + M_ref[idx]),
         R.reducer),
        ('Proteins', D.P[idx], vP_meca[idx], vP_traj[idx], dP_traj[idx], lambda v: v, R.prot_reducer),
    ]:
        # Projected onto the embedding learned once on the reference (observed mRNA / protein trajectories)
        E = (red.transform(_preprocess(M_ref[idx], R.norm, R.log) if name == 'mRNA' else X) if red is not None
             else Embedder(R.emb_method, seed=42).fit_transform(X))
        V_meca = _knn_velocity_embedding(X, to_space(v_meca), E)
        V_traj = _knn_velocity_embedding(X, np.nan_to_num(to_space(d_traj)), E)
        cos = _weighted_cosine(v_meca, v_traj)

        fig = plt.figure(figsize=A4_LANDSCAPE)
        _page_title(fig, f'5. Learned dynamics — {name} velocity and displacement fields',
                    f'All {len(idx)} trajectory states projected onto the UMAP learned on the '
                    + ((f"observed mRNA (log1p, {'reference depth' if R.depth_emb else 'raw counts'})") if name == 'mRNA'
                       else 'protein trajectories (raw levels)') + '; kNN-transition projection of the fields'
                    + f'. Mechanistic vs trajectory velocity agreement (weighted cosine, gene space): {cos:.2f}.')
        gs = gridspec.GridSpec(1, 3, figure=fig, left=0.03, right=0.95, top=0.86, bottom=0.12, wspace=0.12)
        meca_title = ('Mechanistic (noisy, figure 5): d0·(k·kon(P) − M)' if name == 'mRNA'
                      else ('Mechanistic: d1·(kon(P) − P) − b·P' if D.birth_sim is not None
                            else 'Mechanistic: d1·(kon(P) − P)'))
        vmin, vmax = float(t.min()), float(t.max())
        sca = _stream(fig.add_subplot(gs[0, 0]), E, V_meca, t, meca_title, vmin=vmin, vmax=vmax)
        _stream(fig.add_subplot(gs[0, 1]), E, V_traj, t, 'Trajectories: displacement x(t+1) − x(t)', vmin=vmin, vmax=vmax)
        cax = fig.add_axes([0.04, 0.07, 0.55, 0.018])
        cb = fig.colorbar(sca, cax=cax, orientation='horizontal'); cb.set_label('time', fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax = fig.add_subplot(gs[0, 2])
        if growth is not None and np.isfinite(growth).any():
            lo, hi = np.nanpercentile(growth, [2, 98])
            sca2 = _stream(ax, E, V_traj, growth, f'Displacements, colour = learned net rate ({D.learned_label})',
                           cmap='magma', vmin=lo, vmax=hi)
            cb2 = fig.colorbar(sca2, ax=ax, fraction=0.045, pad=0.02); cb2.ax.tick_params(labelsize=6)
        elif D.ct is not None:
            _stream(ax, E, V_traj, D.ct[idx], 'Displacements, colour = cell type', categorical=R.color_map)
        else:
            ax.axis('off')
        pdf.savefig(fig); plt.close(fig)
        if name == 'mRNA':
            summary = (E, V_traj)  # summary pages drawn at the end of the section
    E_m, V_m = summary
    # One summary page per cell-type annotation used by CardamomOT (selection, proliferation, transitions)
    obs = R.adata_data.obs
    for key in ('cell_type', 'cell_type_proliferation', 'cell_type_transition'):
        if key == 'cell_type':
            labels, cmap_ct = (D.ct[idx] if D.ct is not None else None), R.color_map
        elif key in obs:
            per_state = D._per_state(obs[key].astype(str).values, fill='')
            if per_state is None:
                continue
            labels = per_state[idx]
            cmap_ct = _cell_type_colors(sorted(set(obs[key].astype(str))))
        else:
            continue
        if key == 'cell_type' or labels is not None:
            _presentation_page(pdf, R, D, idx, E_m, V_m, t, growth, labels, cmap_ct, key)


def _presentation_page(pdf, R, D, idx, E, V_traj, t, growth, labels=None, color_map=None, key='cell_type'):
    """Summary figure for talks, on the mRNA UMAP: states by time (no field), mRNA displacement along the
    inferred trajectories over the cell types, and final learned net rates (no field)."""
    sz = float(np.clip(20000 / len(E), 0.5, 10))  # point size for every state
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '5. Learned dynamics — summary' + ('' if key == 'cell_type' else f' ({key})'),
                f'All {len(idx)} trajectory states on the UMAP learned on the observed mRNA (log1p, '
                f"{'reference depth' if R.depth_emb else 'raw counts'}); displacement along the inferred "
                f'trajectories x(t+1) − x(t); background of the field: {key}.')
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.03, right=0.95, top=0.86, bottom=0.14, wspace=0.12)
    lo, hi = E.min(axis=0), E.max(axis=0)  # same extent on the three panels
    ax = fig.add_subplot(gs[0, 0])
    sca = ax.scatter(E[:, 0], E[:, 1], c=t, cmap='viridis', s=sz, linewidths=0, rasterized=True)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    _clean_umap_ax(ax, 'Time')
    cb = fig.colorbar(sca, cax=ax.inset_axes([0.0, -0.08, 1.0, 0.03]), orientation='horizontal')
    cb.set_label('time', fontsize=8); cb.ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[0, 1])
    if labels is not None:
        _stream(ax, E, V_traj, labels, f'mRNA displacement along trajectories, {key}', categorical=color_map,
                s=sz, density=0.8, linewidth=1.5, alpha=0.55, color='#1A1A1A')
    else:
        _stream(ax, E, V_traj, t, 'mRNA displacement along trajectories', s=sz, density=0.8, linewidth=1.5,
                alpha=0.55, color="#1A1A1A")
    ax = fig.add_subplot(gs[0, 2])
    if growth is not None and np.isfinite(growth).any():
        ok = np.isfinite(growth)
        glo, ghi = np.nanpercentile(growth, [2, 98])
        ax.scatter(E[~ok, 0], E[~ok, 1], c='#DDDDDD', s=sz, linewidths=0, rasterized=True)
        sca = ax.scatter(E[ok, 0], E[ok, 1], c=growth[ok], cmap='magma', vmin=glo, vmax=ghi, s=sz, linewidths=0,
                         rasterized=True)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        _clean_umap_ax(ax, f'Net proliferation rate ({D.learned_label})')
        cb = fig.colorbar(sca, cax=ax.inset_axes([0.0, -0.08, 1.0, 0.03]), orientation='horizontal')
        cb.set_label('net rate (h$^{-1}$)', fontsize=8); cb.ax.tick_params(labelsize=7)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No net proliferation rate', ha='center', va='center', fontsize=9, color='#555555',
                transform=ax.transAxes)
    if labels is not None:
        _celltype_legend(fig, color_map, y=0.01)
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(p, split, stim, prior, perturbations=(), out_path=None,
                    normtransform=False, logtransform=True, n_umap=4000, top_regulators=20,
                    top_targets=10, seed=0, depth_embeddings=True, embedding_method_visualization='umap',
                    protein_dilution=True, classifier_method='random_forest', select_genes=False,
                    use_recomputed_proliferation=True, simulation_model='ode', mrna_dilution=False):
    """Write the final multi-page PDF report of a CardamomOT run.

    Parameters
    ----------
    p : str
        Project directory (containing Data/ and cardamomOT/).
    split : str
        Data split used for the run (``"full"`` or ``"train"``).
    stim, prior : float
        Stimulus / prior penalisation values of the run (suffix ``stim{stim}_prior{prior}``).
        The stimulus regulator is shown only if ``stim >= 0.5``.
    perturbations : iterable of (label, description, genes)
        KO/OV conditions from the ``perturbation_simulation`` sheet (label as in file names,
        genes = perturbed gene names, highlighted in the GRN subgraphs).
    out_path : str, optional
        Output PDF path (default: ``<p>/CardamomOT_report_stim{stim}_prior{prior}.pdf``).
    normtransform, logtransform : bool
        Preprocessing applied before the embeddings.
    depth_embeddings : bool
        cell_depth_for_representation: mRNA shown as counts / depth factor, for the observed cells and for the model
        draws (taken at the depth of the cells they mimic); False: raw counts and model draws at the depth of the cells.
    embedding_method_visualization : str
        'umap', 'pca' or 'phate' (every 2-D embedding of the report).
    classifier_method : str
        Cell types of the model outputs: 'random_forest' or 'logistic' (per-sample classifiers on the data).
    select_genes : bool
        Genes of the model chosen by the gene selection: the UMAP of all the cells is also drawn on these genes.
    simulation_model : str
        Model of the refined rates: 'ode' or 'sde1' (protein rate d1_eff = d0 d1 / (d0 + d1)), 'sde2' (explicit mRNA, d1).
    mrna_dilution : bool
        mRNA diluted at the birth rate too (with protein_dilution): mRNA half-lives on the dilution page.
    n_umap : int or None
        Max cells per dataset used in UMAPs (stratified by time); None = all.
    top_regulators, top_targets : int
        Number of regulators (and regulated genes) drawn; an edge is drawn only if it is among the
        top_targets strongest targets of its regulator and the top_targets strongest regulators of its target.

    Returns
    -------
    str
        Path of the written PDF.
    """
    if out_path is None:
        out_path = os.path.join(p, f'CardamomOT_report_stim{stim}_prior{prior}.pdf')
    matplotlib.rcParams['pdf.fonttype'] = 42

    _EMBEDDING['name'] = embedding_name(embedding_method_visualization)
    R = _ReportData(p, split, stim, prior, normtransform, logtransform, list(perturbations), n_umap, seed,
                    depth_embeddings, embedding_method_visualization, classifier_method)
    R.protein_dilution = bool(protein_dilution)
    R.mrna_dilution = bool(mrna_dilution) and R.protein_dilution
    R.simulation_model = str(simulation_model)
    R.use_recomputed_proliferation = bool(use_recomputed_proliferation)

    inter = load_network_array(os.path.join(p, 'cardamomOT', 'inter_simul.npy'))
    # Network conditions: one network per condition (inter per sample, condition of each sample in network_conditions.json)
    cond_networks, cond_pen = {'': inter}, None
    cond_json = os.path.join(p, 'cardamomOT', 'network_conditions.json')
    if inter.ndim == 3 and os.path.exists(cond_json):
        import json
        info_c = json.load(open(cond_json))
        sc, cond_pen = np.asarray(info_c['sample_conditions']), info_c.get('network_condition_pen')
        cond_networks = {f' — condition {c}': inter[int(np.flatnonzero(sc == k)[0])]
                         for k, c in enumerate(info_c['conditions'])}
    elif inter.ndim == 3:
        cond_networks = {f' — sample {k}': inter[k] for k in range(inter.shape[0])}
    matrix = next(iter(cond_networks.values()))
    ns = matrix.shape[0] - len(R.genes)
    show_stim = stim >= 0.5
    info = dict(n_stimuli=ns, show_stim=show_stim)
    status = [(l, d, A is not None) for l, d, A in R.perturbations]

    # Trajectory/growth data shared by sections 4 and 5, loaded once on first use
    _dyn = []
    def dyn():
        if not _dyn:
            _dyn.append(_Dynamics(R))
        return _dyn[0]

    with PdfPages(out_path) as pdf:
        _cover_page(pdf, R, info, status)
        rep_path = os.path.join(p, 'cardamomOT', 'integration_report.csv')
        # Network conditions: every section below that depends on the cells of a condition is drawn condition by
        # condition, on its own view (its samples, its genes with select_genes_per_condition, its embeddings)
        conds = R.conditions
        view = lambda c: R.condition_view(c, conds[c])
        per_condition = lambda title, fn: [(f'{title} — condition {c}', (lambda c=c: fn(view(c))), c) for c in conds] \
            if conds else [(title, lambda: fn(R), None)]
        sections = [('Data — UMAP of all the cells', lambda: _data_umap_page(pdf, R, seed), None)]
        if select_genes and R.condition_genes is None:  # one gene space: the cells of every condition on it
            sections.append(('Data — UMAP of all the cells, genes of the gene selection',
                             lambda: _data_umap_page(pdf, R, seed, genes=R.genes), None))
        sections += per_condition('Teaser — displacement fields', lambda V: _teaser_page(pdf, V, seed))
        sections += per_condition('Teaser — displacement fields over time', lambda V: _teaser_time_pages(pdf, V, seed))
        sections += per_condition('Classical OT — cell-type transitions', lambda V: _classical_ot_pages(pdf, V, seed))
        sections += per_condition('Teaser — CardamomOT cell-type transitions', lambda V: _cardamom_transition_page(pdf, V))
        if R.condition_genes is None:
            sections.append(('Genes of the model', lambda: _gene_list_page(pdf, R), None))
        else:
            sections += per_condition('Genes of the model', lambda V: _gene_list_page(pdf, V))
        depth_json = os.path.join(p, 'cardamomOT', 'depth_diagnostic.json')
        if os.path.exists(depth_json):
            import json
            depth_csv = os.path.join(p, 'cardamomOT', 'depth_diagnostic.csv')
            sections.append(('0. Per-cell sequencing depth', lambda: _depth_page(
                pdf, R, json.load(open(depth_json)), pd.read_csv(depth_csv) if os.path.exists(depth_csv) else None), None))
        # Generative model: embeddings learned on the cells of each condition, UMAPs, proportions and reproduction
        sections += per_condition('1. Generative model', lambda V: _model_umap_page(pdf, V))
        sections += per_condition('1. Generative model — quantitative checks', lambda V: _model_checks_page(pdf, V))
        idt_path = os.path.join(p, 'cardamomOT', 'identity_mixture.csv')
        if os.path.exists(idt_path):
            sections.append(('1. Cell-type identity in the mixture',
                             lambda: _identity_page(pdf, R, pd.read_csv(idt_path)), None))
        if os.path.exists(rep_path):
            sections.append(('1. Sample integration', lambda: _integration_page(pdf, R, pd.read_csv(rep_path)), None))
        for label, mat in cond_networks.items():
            sections.append((f'2. Gene regulatory network{label}', lambda label=label, mat=mat: _grn_pages(
                pdf, R, mat, ns, show_stim, top_regulators, top_targets, label=label), None))
        if len(cond_networks) > 1:
            sections.append(('2. Similarity between network conditions',
                             lambda: _network_similarity_page(pdf, R, cond_networks, ns, show_stim), None))
        diff_csv = os.path.join(p, 'cardamomOT', 'network_differences_simul.csv')
        if len(cond_networks) > 1 and os.path.exists(diff_csv):
            sections.append(('2. Differences between network conditions',
                             lambda: _condition_differences_page(pdf, R, pd.read_csv(diff_csv), cond_pen), None))
        # Perturbations, proliferation, dilution, velocity fields, test and validation: per condition
        dyn_of = lambda V: dyn() if V.parent is None else dyn().for_condition(V)
        sections += per_condition('3. In-silico perturbations', lambda V: _perturbation_pages(pdf, V))
        sections += per_condition('4. Proliferation', lambda V: _proliferation_pages(pdf, V, dyn_of(V)))
        sections += per_condition('4. Protein dilution', lambda V: _dilution_page(pdf, V, dyn_of(V)))
        sections += per_condition('5. Learned dynamics — velocity fields', lambda V: _velocity_pages(pdf, V, dyn_of(V)))
        if R.test:
            sections += per_condition('6. Held-out test cells', lambda V: _test_pages(pdf, V) if V.test else None)
        if conds:
            for c in conds:
                for r in view(c).validation:
                    sections.append((f'6. Validation of sample {r} — condition {c}',
                                     lambda r=r, c=c: _validation_page(pdf, view(c), r), c))
        else:
            for r in R.validation:
                sections.append((f'6. Validation of sample {r}', lambda r=r: _validation_page(pdf, R, r), None))
        for title, fn, c in sections:
            _PAGE_SUFFIX[0] = '' if c is None else f' — condition {c}'
            try:
                fn()
            except Exception as e:  # keep the rest of the report if one section fails
                plt.close('all')
                print(f"[report] Warning: section '{title}' failed: {e!r}")
                _error_page(pdf, title, repr(e))
        _PAGE_SUFFIX[0] = ''
        d = pdf.infodict()
        d['Title'] = f'CardamomOT report — {os.path.basename(os.path.normpath(p))}'
        d['Author'] = 'CardamomOT'
    return out_path
