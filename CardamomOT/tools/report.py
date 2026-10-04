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
from umap import UMAP
from sklearn.neighbors import NearestNeighbors

from .characterize_cell_type import train_classifier, predict_cell_types
from ..config import find_data_file, resolve_cell_type_obs
from ..inputs import input_dir
from ..inference.trajectory import kon_ref_vector
from ..inference.proliferations import ProliferationMLP, growth_path_states, quadrature, interval_stimulus

A4_LANDSCAPE = (11.69, 8.27)
REF_COLOR = '#555555'
SIM_COLOR = '#C94040'
ACT_COLOR = '#2ECC71'
INH_COLOR = '#E74C3C'
REG_COLOR = '#4C9BE8'
STIM_COLOR = '#F39C12'
LABEL_KEY = 'cell_type'


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


def _proportions(adata, categories):
    counts = adata.obs[LABEL_KEY].astype(str).value_counts(normalize=True) * 100
    return counts.reindex(categories).fillna(0.0)


def _proportions_by_time(adata, categories):
    df = pd.DataFrame({'time': pd.to_numeric(adata.obs['time']).values,
                       'ct': adata.obs[LABEL_KEY].astype(str).values})
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


def _celltype_legend(fig, color_map, y=0.01):
    handles = [mpatches.Patch(color=c, label=k) for k, c in color_map.items()]
    if handles:
        fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 8),
                   frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, y))


def _page_title(fig, title, subtitle=None):
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

def _regulator_subgraph(matrix, gene_names, gene, top_targets=8):
    idx = gene_names.index(gene)
    series = pd.Series(matrix[idx, :], index=gene_names).drop(gene, errors='ignore')
    series = series[series != 0]
    top_idx = series.abs().nlargest(top_targets).index
    G = nx.DiGraph()
    G.add_node(gene)
    for tgt in top_idx:
        G.add_edge(gene, tgt, weight=float(series[tgt]))
    return G


def _draw_regulator_subgraph(ax, G, gene, max_intensity, center_color, title, highlight=()):
    if G.number_of_edges() == 0:
        ax.text(0.5, 0.5, f"{gene}\n(no outgoing edge)", ha='center', va='center',
                transform=ax.transAxes, fontsize=7, color='gray')
        ax.set_title(title, fontsize=8, fontweight='bold')
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

    node_colors = [center_color if n == gene else ('#FDEBD0' if n in highlight else '#EDEDED') for n in G.nodes]
    node_sizes = [900 if n == gene else 520 for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax,
                           edgecolors='#999999', linewidths=0.4)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

    e_pos = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    e_neg = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0]
    width = lambda el: [0.4 + 3.0 * abs(G[u][v]['weight']) / max_intensity for u, v in el]
    if e_pos:
        nx.draw_networkx_edges(G, pos, edgelist=e_pos, edge_color=ACT_COLOR, width=width(e_pos),
                               arrows=True, arrowsize=9, connectionstyle='arc3,rad=0.1',
                               min_target_margin=12, ax=ax)
    if e_neg:
        nx.draw_networkx_edges(G, pos, edgelist=e_neg, edge_color=INH_COLOR, width=width(e_neg),
                               arrows=True, arrowstyle='-[,widthB=0.8,lengthB=0.0',
                               connectionstyle='arc3,rad=0.1', min_target_margin=12, ax=ax)
    ax.margins(0.18)
    ax.set_title(title, fontsize=8, fontweight='bold')
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
    min_gap = 0.04 * span if min_gap is None else min_gap
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

    def __init__(self, p, split, stim, prior, norm, log, perturbations, n_umap, seed):
        self.p, self.split, self.stim, self.prior = p, split, stim, prior
        self.norm, self.log = norm, log
        self.rng = np.random.default_rng(seed)
        self.n_umap = n_umap
        tag = f'stim{stim}_prior{prior}'
        cdir = os.path.join(p, 'cardamomOT')
        self.tag = tag

        self.adata_data = ad.read_h5ad(os.path.join(p, 'Data', f'data_{split}.h5ad'))
        from ..inputs import depth_factor_used
        self.use_depth = depth_factor_used(p)
        self.genes = list(self.adata_data.var_names)
        # Data = observed cells; Reference = RNA of the trajectories (one state per ancestor: without
        # proliferation), growth-weighted below if the simulation used the proliferation MLP
        self.stages = {
            'Data': self.adata_data,
            'Reference': _read(os.path.join(cdir, f'adata_rna_traj_{tag}.h5ad')),
            'NB mixture': _read(os.path.join(cdir, f'adata_beta_{tag}.h5ad')),
            'Network': _read(os.path.join(cdir, f'adata_theta_{tag}.h5ad')),
            'Simulation': _read(os.path.join(cdir, f'adata_sim_{tag}.h5ad')),
        }
        missing = [k for k, v in self.stages.items() if v is None]
        if missing:
            raise FileNotFoundError(f"Missing check_sim_to_data outputs with tag '{tag}' for: {missing}. "
                                    f"Run check_sim_to_data.py with the same --stimulus/--prior.")
        self.prot = {'Trajectories': _read(os.path.join(cdir, f'adata_prot_traj_{tag}.h5ad')),
                     'Simulation': _read(os.path.join(cdir, f'adata_prot_simul_{tag}.h5ad'))}

        # Simulation with proliferation: trajectory stages replaced by their growth-weighted version
        # (check_sim_to_data), the raw states being kept for the dynamics (one row per state)
        self.beta_states = self.stages['NB mixture']
        self.growth_ref = False
        if bool(self.stages['Simulation'].uns.get('proliferation', False)):
            growth = {k: _read(os.path.join(cdir, f'adata_{n}_growth_{tag}.h5ad'))
                      for k, n in (('Reference', 'rna_traj'), ('NB mixture', 'beta'), ('Network', 'theta'),
                                   ('Trajectories', 'prot_traj'))}
            if all(v is not None for v in growth.values()):
                self.stages['Reference'] = growth['Reference']
                self.stages['NB mixture'], self.stages['Network'] = growth['NB mixture'], growth['Network']
                self.prot['Trajectories'] = growth['Trajectories']
                self.growth_ref = True
            else:
                print("[report] Warning: simulation with proliferation but no growth-weighted trajectories "
                      "(rerun check_sim_to_data.py); raw trajectories shown")

        # Perturbations listed in KO_OV_Stim_simulate.txt: (label, description, adata or None)
        self.perturbations, self.perturbed_genes = [], set()
        for label, desc, genes in perturbations:
            self.perturbations.append((label, desc, _read(os.path.join(cdir, f'adata_sim_{label}_{tag}.h5ad'))))
            self.perturbed_genes.update(genes)

        # Cell types: classifier trained on observed data, applied in memory (h5ad files untouched)
        self.has_ct = LABEL_KEY in self.adata_data.obs
        if self.has_ct:
            self.categories = self.adata_data.obs[LABEL_KEY].astype(str).unique().tolist()
            self.color_map = _cell_type_colors(self.categories)
            clf = train_classifier(self.adata_data, label_key=LABEL_KEY)
            for A in [a for k, a in self.stages.items() if k != 'Data'] + [a for _, _, a in self.perturbations if a is not None]:
                predict_cell_types(A, clf, label_key=LABEL_KEY)
            self.clf = clf
        else:
            self.categories, self.color_map = [], {}

        # Held-out test cells (infer_test.py with split=train): observed data and predictions
        self.test = None
        test_path = os.path.join(p, 'Data', 'data_test.h5ad')
        if split == 'train' and os.path.exists(test_path):
            stages = {'Test data': ad.read_h5ad(test_path),
                      'NB mixture': _read(os.path.join(cdir, f'adata_beta_test_{tag}.h5ad')),
                      'Network': _read(os.path.join(cdir, f'adata_theta_test_{tag}.h5ad')),
                      'Simulation': _read(os.path.join(cdir, f'adata_sim_test_{tag}.h5ad'))}
            if all(v is not None for v in stages.values()):
                self.test = stages
                if self.has_ct:
                    for k, A in stages.items():
                        if k != 'Test data' or LABEL_KEY not in A.obs:
                            predict_cell_types(A, self.clf, label_key=LABEL_KEY)

        self._fit_umap()

    def _subsample(self, A):
        """Stratified-by-time subsample of at most n_umap cells (indices)."""
        return _time_subsample(pd.to_numeric(A.obs['time']).values, self.n_umap, self.rng)

    def _fit_umap(self):
        # One joint WT embedding; perturbations are projected onto it for comparability
        self.sub = {k: self._subsample(A) for k, A in self.stages.items()}
        X = np.vstack([_preprocess(A.X[self.sub[k]], self.norm, self.log) for k, A in self.stages.items()])
        self.reducer = UMAP(random_state=42, min_dist=0.7).fit(X)
        self.umap, start = {}, 0
        for k in self.stages:
            n = len(self.sub[k])
            self.umap[k] = self.reducer.embedding_[start:start + n]
            start += n
        # Test cells and predictions projected onto the same embedding
        self.test_sub, self.test_umap = {}, {}
        for k, A in (self.test or {}).items():
            self.test_sub[k] = self._subsample(A)
            self.test_umap[k] = self.reducer.transform(_preprocess(A.X[self.test_sub[k]], self.norm, self.log))
        self.pert_sub, self.pert_umap = {}, {}
        for label, _, A in self.perturbations:
            if A is None:
                continue
            self.pert_sub[label] = self._subsample(A)
            self.pert_umap[label] = self.reducer.transform(_preprocess(A.X[self.pert_sub[label]], self.norm, self.log))

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
            ('Network shown', f"inter_simul.npy, network #{info['net_index']} / {info['n_networks']}"),
            ('UMAP preprocessing', f"normalise={R.norm}, log1p={R.log}")]
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

    fig.text(0.06, 0.22, 'Contents', fontsize=13, fontweight='bold')
    fig.text(0.07, 0.04,
             '1. Generative model — UMAPs of data, trajectories, NB mixture, network modes and simulation; cell-type proportions; gene-pair correlations; proteins\n'
             '2. Gene regulatory network — regulatory power (violin plots) and top-10 regulators' + (' + stimulus' if info['show_stim'] else '') + '\n'
             '3. In-silico perturbations — overview across KO/OV, then one page per perturbation\n'
             '4. Proliferation — prior vs learned net rates, population growth, proteins driving growth\n'
             '5. Learned dynamics — mRNA and protein velocity fields (mechanistic and along trajectories), summary on mRNA'
             + ('\n6. Held-out test cells — predictions with the network fixed vs the test data' if R.test else ''),
             fontsize=9, va='bottom', linespacing=1.6)
    pdf.savefig(fig); plt.close(fig)


def _model_pages(pdf, R):
    names = list(R.stages)
    t_all = np.concatenate([R.times(R.stages[k], R.sub[k]) for k in names])
    vmin, vmax = float(t_all.min()), float(t_all.max())

    # Page: UMAPs by time and cell type
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '1. Generative model — trajectories and simulation',
                'Joint UMAP of the observed data, the trajectories (Reference: one state per ancestor, i.e. '
                + ('growth-weighted by exp ∫R_opt, as the simulation has proliferation)' if R.growth_ref
                   else 'without proliferation, as the simulation)')
                + ', NB mixture and network-driven modes along them, and the full simulation.')
    gs = gridspec.GridSpec(2, len(names), figure=fig, left=0.03, right=0.96, top=0.89, bottom=0.12, hspace=0.12,
                           wspace=0.05)
    sca = None
    for j, k in enumerate(names):
        A = R.stages[k]
        sca = _umap_time(fig.add_subplot(gs[0, j]), R.umap[k], R.times(A, R.sub[k]), vmin, vmax,
                         'Data (observed)' if k == 'Data' else k)
        ax = fig.add_subplot(gs[1, j])
        if R.has_ct:
            _umap_celltype(ax, R.umap[k], R.labels(A, R.sub[k]), R.color_map, '')
        else:
            ax.axis('off')
    cax = fig.add_axes([0.965, 0.55, 0.008, 0.3])
    cb = fig.colorbar(sca, cax=cax); cb.set_label('time', fontsize=8); cb.ax.tick_params(labelsize=7)
    _celltype_legend(fig, R.color_map, y=0.03)
    pdf.savefig(fig); plt.close(fig)

    # Page: cell-type proportions, gene-pair correlations, proteins
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '1. Generative model — quantitative checks',
                'Reference = RNA of the trajectories equivalent to the simulation '
                + ('(growth-weighted: simulation with proliferation).' if R.growth_ref else '(without proliferation).'))
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.06, right=0.97, top=0.88, bottom=0.12, hspace=0.45, wspace=0.3)
    if R.has_ct:
        prop = pd.DataFrame({k: _proportions(R.stages[k], R.categories) for k in names}).T
        ax = fig.add_subplot(gs[0, 0]); _stacked_bars(ax, prop, R.color_map, 'Cell-type proportions'); _panel_label(ax, 'A', -0.2)
        ax = fig.add_subplot(gs[0, 1:])
        _proportions_over_time(ax, _proportions_by_time(R.stages['Reference'], R.categories),
                               _proportions_by_time(R.stages['Simulation'], R.categories), R.color_map,
                               'Cell-type proportions over time', 'Reference', 'Simulation')
        _panel_label(ax, 'B', -0.08)

    # Gene-pair correlations, reference vs NB mixture / simulation
    ref = _dense(R.stages['Reference'].X).astype(float)
    iu = np.triu_indices(ref.shape[1], k=1)

    def _pair_corr(X):
        X = _dense(X).astype(float)
        X = X + 1e-9 * R.rng.standard_normal(X.shape) * (X.std(axis=0) == 0)
        return np.corrcoef(X.T)[iu]

    c_ref = _pair_corr(ref)
    for j, (k, lab) in enumerate([('NB mixture', 'C'), ('Simulation', 'D')]):
        c_k = _pair_corr(R.stages[k].X)
        ok = np.isfinite(c_ref) & np.isfinite(c_k)
        r = np.corrcoef(c_ref[ok], c_k[ok])[0, 1]
        ax = fig.add_subplot(gs[1, j])
        ax.scatter(c_ref[ok], c_k[ok], s=2, alpha=0.35, color='k', linewidths=0, rasterized=True)
        lim = [min(c_ref[ok].min(), c_k[ok].min()), max(c_ref[ok].max(), c_k[ok].max())]
        ax.plot(lim, lim, 'r--', lw=0.8, alpha=0.6)
        ax.set_xlabel('Reference gene-pair correlation', fontsize=8)
        ax.set_ylabel(f'{k} gene-pair correlation', fontsize=8)
        ax.set_title(f'Gene-pair correlations vs {k}\n(R = {r:.2f})', fontsize=9)
        ax.tick_params(labelsize=7)
        _panel_label(ax, lab, -0.2)

    # Proteins: UMAP fitted on trajectories, simulation projected
    if all(v is not None for v in R.prot.values()):
        sub_gs = gs[1, 2].subgridspec(1, 2, wspace=0.05)
        Pt, Ps = R.prot['Trajectories'], R.prot['Simulation']
        it, is_ = R._subsample(Pt), R._subsample(Ps)
        red = UMAP(random_state=42, min_dist=0.7).fit(_dense(Pt.X[it]))
        emb_s = red.transform(_dense(Ps.X[is_]))
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
                'training network fixed, then simulated; projected onto the joint UMAP (grey: training).')
    gs = gridspec.GridSpec(2, 4, figure=fig, left=0.04, right=0.96, top=0.89, bottom=0.12, hspace=0.18, wspace=0.06)
    sca = None
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
    data_tr, data_te = R.adata_data, T['Test data']
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
        _proportions_over_time(ax, _proportions_by_time(data_te, R.categories),
                               _proportions_by_time(sim_te, R.categories), R.color_map,
                               'Cell-type proportions over time', 'Test data', 'Test simulation')
        _panel_label(ax, 'E', -0.08)
        _celltype_legend(fig, R.color_map, y=0.01)
    pdf.savefig(fig); plt.close(fig)


def _depth_page(pdf, R, diag, per_group):
    """Per-cell depth diagnostic (estimate_cell_depth.py) and, on the selected genes, its effect."""
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '0. Per-cell sequencing depth',
                'Diagnostic on the whole transcriptome within homogeneous groups (sample, time, cell type); '
                'counts are modelled as NB(k, c / s_i) when the depth factor s_i is applied.')
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


def _integration_page(pdf, R, rep, sample_key='dataset_id'):
    """Mode means of each sample vs global ones, and UMAPs of raw vs integrated counts by sample."""
    A = R.adata_data
    fig = plt.figure(figsize=A4_LANDSCAPE)
    pct = rep.groupby('sample')['integrated'].mean() * 100
    lam = float(rep['integrate_samples'].iloc[0]) if 'integrate_samples' in rep else 1.0
    _page_title(fig, f'1. Generative model — per-sample mixtures (integrate_samples = {lam:g})',
                'Mixture fitted per sample, parameters pushed towards the common target by integrate_samples '
                '(counts quantile-matched accordingly). Genes with sample-specific parameters: '
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
    ax.set_xlabel('log10 global mode mean', fontsize=8); ax.set_ylabel('log10 sample mode mean', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(handles=[Line2D([0], [0], marker='o', ls='', color=cols[s], label=f'sample {s}') for s in samples]
              + [Line2D([0], [0], marker='o', ls='', color='k', label='integrated'),
                 Line2D([0], [0], marker='o', ls='', mfc='none', color='k', label='kept raw')],
              fontsize=6.5, frameon=False)
    ax.set_title('Mode means per sample (all modes)', fontsize=9)

    if 'counts_raw' in A.layers and sample_key in A.obs:
        idx = _time_subsample(pd.to_numeric(A.obs['time']).values, R.n_umap, R.rng)
        lab = A.obs[sample_key].astype(str).values[idx]
        for j, (X, title) in enumerate([(A.layers['counts_raw'], 'Raw counts'), (A.X, 'Integrated counts')]):
            E = UMAP(random_state=42, min_dist=0.7).fit_transform(_preprocess(X[idx], R.norm, R.log))
            ax = fig.add_subplot(gs[0, j + 1])
            for s in samples:
                m = lab == str(s)
                ax.scatter(E[m, 0], E[m, 1], s=4, color=cols[s], linewidths=0, alpha=0.6, rasterized=True)
            _clean_umap_ax(ax, f'{title}, colour = sample')
    pdf.savefig(fig); plt.close(fig)


def _grn_pages(pdf, R, matrix, ns, show_stim, top_n=10, top_targets=8):
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
    _page_title(fig, '2. Gene regulatory network — regulatory power',
                f"Self-interactions excluded. Top {top_n} genes labelled"
                + (f"; stimulus shown as a star (stimulus = {R.stim} ≥ 0.5)." if show_stim else
                   f"; stimulus not shown (stimulus = {R.stim} < 0.5)."))
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.07, right=0.97, top=0.86, bottom=0.08, wspace=0.55,
                           width_ratios=[1, 1, 1.1])
    extra = [(stim_names[s], out_power[s]) for s in range(ns)] if show_stim else None
    ax = fig.add_subplot(gs[0, 0])
    _labelled_violin(ax, g_out, g_names, top_out, 'Outgoing regulation (regulators)',
                     'log(1 + Σ|outgoing weights|)', extra=extra)
    _panel_label(ax, 'A', -0.25)
    ax = fig.add_subplot(gs[0, 1])
    _labelled_violin(ax, g_in, g_names, top_in, 'Incoming regulation (targets)', 'log(1 + Σ|incoming weights|)')
    _panel_label(ax, 'B', -0.25)

    # Table: activation / inhibition balance of top regulators
    ax = fig.add_subplot(gs[0, 2]); ax.axis('off'); _panel_label(ax, 'C', -0.05, 1.0)
    rows = []
    for r, i in enumerate(top_out):
        gi = genes_idx[i]
        w = np.delete(M[gi], gi)
        rows.append([f'{r + 1}', g_names[i], f'{np.abs(w).sum():.2f}', f'{(w > 0).sum()}', f'{(w < 0).sum()}'])
    if show_stim:
        for s in range(ns):
            w = M[s, ns:]
            rows.append(['★', stim_names[s], f'{np.abs(w).sum():.2f}', f'{(w > 0).sum()}', f'{(w < 0).sum()}'])
    tab = ax.table(cellText=rows, colLabels=['#', 'Regulator', 'Σ|w|', '# act.', '# inh.'],
                   loc='upper center', cellLoc='center', colWidths=[0.08, 0.32, 0.2, 0.17, 0.17])
    tab.auto_set_font_size(False); tab.set_fontsize(7.5); tab.scale(1, 1.35)
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.3)
        if r == 0:
            cell.set_facecolor('#E8EEF7'); cell.set_text_props(fontweight='bold')
    ax.set_title('Top regulators', fontsize=9.5, fontweight='bold')
    pdf.savefig(fig); plt.close(fig)

    # Pages: per-regulator subgraphs (stimulus first if shown), 3 x 4 per page
    panels = [(s, STIM_COLOR, f'{stim_names[s]}') for s in range(ns)] if show_stim else []
    panels += [(genes_idx[i], REG_COLOR, f'#{r + 1} {g_names[i]}') for r, i in enumerate(top_out)]
    max_intensity = float(np.abs(M).max() or 1.0)
    perturbed = R.perturbed_genes
    for start in range(0, len(panels), 12):
        chunk = panels[start:start + 12]
        fig = plt.figure(figsize=A4_LANDSCAPE)
        _page_title(fig, '2. Gene regulatory network — top regulators and their main targets',
                    f'Top {top_targets} targets per regulator; edge width ∝ |weight| (global scale). '
                    'Perturbed genes (KO/OV) are highlighted in beige.')
        gs = gridspec.GridSpec(3, 4, figure=fig, left=0.02, right=0.98, top=0.9, bottom=0.07, hspace=0.25, wspace=0.08)
        for k, (i, col, title) in enumerate(chunk):
            ax = fig.add_subplot(gs[k // 4, k % 4])
            G = _regulator_subgraph(M, names, names[i], top_targets)
            _draw_regulator_subgraph(ax, G, names[i], max_intensity, col, title, highlight=perturbed)
        fig.legend(handles=[Line2D([0], [0], color=ACT_COLOR, lw=2, label='Activation'),
                            Line2D([0], [0], color=INH_COLOR, lw=2, label='Inhibition'),
                            mpatches.Patch(color=REG_COLOR, label='Top regulator')]
                   + ([mpatches.Patch(color=STIM_COLOR, label='Stimulus')] if show_stim else []),
                   loc='lower center', ncol=4, frameon=False, fontsize=8)
        pdf.savefig(fig); plt.close(fig)


def _perturbation_pages(pdf, R):
    done = [(l, d, A) for l, d, A in R.perturbations if A is not None]
    if not done:
        _error_page(pdf, '3. In-silico perturbations',
                    'No simulated perturbation found (Data/KO_OV_Stim_simulate.txt absent/empty, or '
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
                    'Cell types predicted by a random forest trained on the observed data.'
                    + (' Population sizes from the branching simulation (proliferation MLP + RATE effects).'
                       if pops else ''))
        gs = gridspec.GridSpec(1, 3 if pops else 2, figure=fig, left=0.06, right=0.97, top=0.86, bottom=0.36,
                               wspace=0.55, width_ratios=[1.1, 1.1, 1] if pops else None)
        prop = pd.DataFrame({'Data': _proportions(R.stages['Data'], R.categories),
                             'Reference': _proportions(R.stages['Reference'], R.categories),
                             'WT': _proportions(sim, R.categories),
                             **{pid[l]: _proportions(A, R.categories) for l, _, A in done}}).T
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
        g_top = gs[0].subgridspec(2, 3, hspace=0.12, wspace=0.05)
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
            prop = pd.DataFrame({'Data': _proportions(R.stages['Data'], R.categories),
                                 'Reference': _proportions(R.stages['Reference'], R.categories),
                                 'Sim. WT': _proportions(sim, R.categories),
                                 'Perturbed': _proportions(A, R.categories)}).T
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
        prot = load('data_prot_forsimul.npy')
        if prot is None:
            prot = load('data_prot.npy')
        self.times = load('data_times.npy')
        if rna is None or prot is None or self.times is None:
            raise FileNotFoundError('trajectory files (data_rna / data_prot[_forsimul] / data_times) not found')
        self.ns = rna.shape[1] - len(R.genes)
        self.prot_full = prot
        self.P = prot[:, self.ns:]
        samples = load('data_samples.npy')
        self.samples = samples.astype(int) if samples is not None else np.zeros(len(self.times), dtype=int)
        self.tu = np.sort(np.unique(self.times))
        self.N = len(self.times) // len(self.tu)

        # mRNA along trajectories: NB mixture sample of each state (as in figure 5), else trajectory counts
        beta = R.beta_states
        self.M = (_dense(beta.X).astype(float) if beta.n_obs == len(self.times) else rna[:, self.ns:].astype(float))

        self.d = load('degradations.npy')
        self.a = load('mixture_parameters.npy')
        self.basal = load('basal_simul.npy')
        self.inter = load('inter_simul.npy')

        # Real cell behind each state, its prior rate and cell type
        self.real_idx = load('data_traj_real_idx.npy')
        if self.real_idx is not None and len(self.real_idx) != len(self.times):
            self.real_idx = None
        obs = R.adata_data.obs
        self.prior_cells = obs['proliferation_net_rate'].values.astype(float) if 'proliferation_net_rate' in obs else None
        self.prior = self._per_state(self.prior_cells)
        self.ct = self._per_state(obs[LABEL_KEY].astype(str).values, fill='') if R.has_ct else None

        # Growth: OT pass (rate over the next interval) and proliferation MLP R(P)
        self.R_opt = load('data_R_opt.npy')
        if self.R_opt is not None and len(self.R_opt) != len(self.times):
            self.R_opt = None
        self.mlp = None
        pt, npt = os.path.join(cdir, 'prolif_network.pt'), load('prolif_network_n_proteins.npy')
        if os.path.exists(pt) and npt is not None:
            import torch
            self.mlp = ProliferationMLP(int(np.ravel(npt)[0]))
            self.mlp.load_state_dict(torch.load(pt, map_location='cpu', weights_only=True), strict=False)
            self.mlp.eval()
        # Stimuli of the interval starting at each state (as in training and simulation)
        self.stim_state = interval_stimulus(prot, self.times, self.ns) if self.ns > 0 else np.zeros((len(self.times), 0))
        self.R_mlp = self.mlp.predict(self.P, self.stim_state) if self.mlp is not None else None
        # Part of the inference stimuli (perturbation_inference, RATEk): not in the MLP, added back with the
        # inference schedule (rate of each state over the interval it starts, as for R_opt)
        self.R_stim = None
        stim_pkl = os.path.join(cdir, 'stimulus_rates.pkl')
        if self.R_mlp is not None and self.real_idx is not None and os.path.exists(stim_pkl):
            import pickle
            from ..stimulus_rates import schedule_values
            srm = pickle.load(open(stim_pkl, 'rb'))
            Xd = _dense(R.adata_data.X).astype(float)
            if R.use_depth and 'depth_factor' in R.adata_data.obs:
                Xd = Xd / R.adata_data.obs['depth_factor'].to_numpy(dtype=float)[:, None]
            S = srm.effect(Xd[self.real_idx]).reshape(len(self.tu), self.N, -1)
            U = schedule_values(R.p, self.tu, srm.n_stimuli)[:, :srm.n_stimuli]
            off = np.zeros((len(self.tu), self.N))
            off[:-1] = np.einsum('tnk,tk->tn', (S[:-1] + S[1:]) / 2, U[1:])
            self.R_stim = off.ravel()
            self.R_mlp = self.R_mlp + self.R_stim
        diag_path = os.path.join(cdir, 'prolif_network_diagnostics.json')
        self.mlp_diag = json.load(open(diag_path)) if (self.mlp is not None and os.path.exists(diag_path)) else None
        # Cumulative log mass of each state along its path (growth OT pass), for the growth-weighted trajectories
        self.L_growth = None
        if self.R_opt is not None:
            gain = np.nan_to_num(self.R_opt.reshape(len(self.tu), self.N)[:-1] * np.diff(self.tu)[:, None])
            self.L_growth = np.vstack([np.zeros((1, self.N)), np.cumsum(gain, axis=0)]).ravel()
        # Learned rate per state: MLP if trained, else the OT pass estimate
        self.R_learned = self.R_mlp if self.R_mlp is not None else self.R_opt
        self.learned_label = 'MLP R(P)' if self.R_mlp is not None else 'growth OT pass'

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
            b = self.basal[min(s, self.basal.shape[0] - 1)] if self.basal.ndim == 3 else self.basal
            out[m] = kon_ref_vector(self.prot_full[m].astype(float), (a[:-1] / k1).T, self.inter, b)
            scale[m] = k1 / a[-1]
        return out[:, self.ns:], scale[:, self.ns:]

    def trajectory_velocity(self, X):
        """(x_{t+1} − x_t)/Δt along each trajectory slot; NaN at the last time."""
        T, N = len(self.tu), self.N
        Y = X.reshape(T, N, -1)
        V = np.full_like(Y, np.nan, dtype=float)
        V[:-1] = (Y[1:] - Y[:-1]) / np.diff(self.tu)[:, None, None]
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


def _knn_velocity_embedding(X, V, E, k=30, scale=10.0):
    """
    scVelo-style projection: transition probabilities to the kNN of each cell from the
    cosine between its velocity and the displacements to its neighbours, then expected
    unit displacement in the embedding minus the uniform-transition one.
    """
    k = min(k, len(X) - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    dX = X[idx] - X[:, None, :]
    nv = np.linalg.norm(V, axis=1)
    cos = (dX * V[:, None, :]).sum(-1) / (np.linalg.norm(dX, axis=-1) * nv[:, None] + 1e-12)
    P = np.exp(scale * cos)
    P /= P.sum(axis=1, keepdims=True)
    dE = E[idx] - E[:, None, :]
    dE /= np.linalg.norm(dE, axis=-1, keepdims=True) + 1e-12
    Vemb = (P[..., None] * dE).sum(axis=1) - dE.mean(axis=1)
    Vemb[~(nv > 0) | ~np.isfinite(nv)] = 0.0
    return Vemb


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
        raise FileNotFoundError('no growth estimate (data_R_opt.npy from infer_network_structure)')
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

        # Table: anchor rate (Data/proliferation_rates) vs prior and learned means
        anchors = {}
        pr_path = find_data_file(input_dir(R.p), 'proliferation_rates')
        if pr_path is not None:
            tab = pd.read_csv(pr_path, sep=None, engine='python', header=None)
            anchors = {str(k): float(v) for k, v in zip(tab.iloc[:, 0], tab.iloc[:, 1])}
        # Effects of the inference stimuli on the cell types (anchors are then rates without stimulus)
        from ..stimulus_rates import load_effects, split_effects
        stim_ct = {}
        for k, (ct, _) in split_effects(load_effects(R.p), cats).items():
            for c, dlt in ct.items():
                stim_ct[c] = stim_ct.get(c, '') + f' {dlt:+.3g}·u{k}'
        fmt = lambda x: f'{x:.3g}' if np.isfinite(x) else '—'
        rows = [[c, fmt(anchors.get(c, np.nan)) + stim_ct.get(c, ''),
                 fmt(np.nanmean(D.prior_cells[ct_cells == c])) if has_prior else '—',
                 fmt(np.nanmean(learned_cells[ct_cells == c])) if np.isfinite(learned_cells[ct_cells == c]).any() else '—']
                for c in cats]
        ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
        t = ax.table(cellText=rows, colLabels=['Cell type', 'Anchor' + (' (+ stimulus)' if stim_ct else ''), 'Prior',
                                               'Learned'],
                     loc='upper center', cellLoc='center', colWidths=[0.34, 0.30, 0.18, 0.18])
        t.auto_set_font_size(False); t.set_fontsize(7); t.scale(1, 1.3)
        for (r, c), cell in t.get_celld().items():
            cell.set_linewidth(0.3)
            if r == 0:
                cell.set_facecolor('#E8EEF7'); cell.set_text_props(fontweight='bold')
        ax.set_title(f'Mean net rate per cell type ({pkey})', fontsize=9)
    pdf.savefig(fig); plt.close(fig)

    # Page: population growth, rates over time, genes driving R
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '4. Proliferation — population growth and drivers',
                'Relative population size from the mean growth factor per interval; '
                'unless Data/population_sizes is given, its absolute level follows the prior.')
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.06, right=0.97, top=0.86, bottom=0.12, wspace=0.35)
    T, N, dt = len(D.tu), D.N, np.diff(D.tu)
    ax = fig.add_subplot(gs[0, 0])
    curves = []
    if D.prior is not None:
        pr = D.prior.reshape(T, N)[:-1]
        curves.append(('prior', '#9AA5B1', '--', np.nanmean(np.exp(pr * dt[:, None]), axis=1)))
    if D.R_opt is not None:
        ro = D.R_opt.reshape(T, N)[:-1]
        curves.append(('growth OT pass', '#E67E22', '-', np.nanmean(np.exp(ro * dt[:, None]), axis=1)))
    if D.mlp is not None:
        states, _ = growth_path_states(D.prot_full, D.times, ns=D.ns, n_nodes=5)
        _, w = quadrature(5)
        S = D.prot_full[:T * N, :D.ns].reshape(T, N, D.ns)[1:, None]       # stimuli of each interval
        integ = (D.mlp.predict(states, S) * w[None, :, None]).sum(axis=1) * dt[:, None]
        if D.R_stim is not None:  # part of the inference stimuli over each interval
            integ = integ + D.R_stim.reshape(T, N)[:-1] * dt[:, None]
        curves.append(('MLP R(P) along paths', REG_COLOR, '-', np.mean(np.exp(integ), axis=1)))
    for z, (lab, col, ls, g) in enumerate(curves):
        # Prior drawn last: the OT pass is anchored on it (same mean growth) unless population sizes are given
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

    # Page: growth-weighted trajectories (what a simulation with proliferation must reproduce) and MLP fit
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '4. Proliferation — growth-weighted trajectories and MLP fit',
                'The OT selection keeps one descendant per ancestor (trajectories without proliferation); weighting '
                'each path by its expansion exp(∫R_opt) should bring them back to the observed data.'
                + (' The simulation used the proliferation MLP.' if R.growth_ref else ''))
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
    if p_states is not None and D.L_growth is not None:
        w_growth = np.zeros_like(D.L_growth)
        for t in D.tu:
            m = D.times == t
            w_growth[m] = np.exp(D.L_growth[m] - D.L_growth[m].max())
        props = {'data': _props(p_cells, t_obs), 'trajectories': _props(p_states, D.times),
                 'growth-weighted': _props(p_states, D.times, w_growth)}
        if sim_lab is not None:
            props['simulation'] = _props(sim_lab, t_sim)
        styles = {'data': ('-', 2.0), 'trajectories': (':', 1.2), 'growth-weighted': ('--', 1.2),
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
            if name == 'data':
                continue
            tv = 0.5 * np.nansum(np.abs(P - props['data']), axis=1)
            tv[np.all(np.isnan(P), axis=1)] = np.nan
            ax.plot(D.tu, tv, ls=styles[name][0], marker='o', ms=3, lw=1.3, label=f'{name} (mean {np.nanmean(tv):.1f})',
                    color={'trajectories': '#9AA5B1', 'growth-weighted': '#E67E22', 'simulation': REG_COLOR}[name])
        ax.set_xlabel('time', fontsize=8); ax.set_ylabel('distance to data (total variation, %)', fontsize=8)
        ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, frameon=False)
        ax.set_title('Cell-type composition vs observed data', fontsize=9)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Needs cell types and the growth OT pass (data_R_opt.npy)',
                ha='center', va='center', fontsize=9, color='#555555', transform=ax.transAxes)

    ax = fig.add_subplot(gs[0, 2])
    if D.mlp_diag is not None:
        d = D.mlp_diag
        x = np.arange(len(d['growth_target']))
        ax.bar(x - 0.18, d['growth_target'], width=0.36, color='#E67E22', label='growth OT pass')
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


def _velocity_pages(pdf, R, D, n_cells=2000, k_top=8):
    if D.d is None or D.a is None or D.basal is None or D.inter is None:
        raise FileNotFoundError('network/degradation files needed for the mechanistic velocities')
    idx = _time_subsample(D.times, n_cells, R.rng)
    t = D.times[idx]
    kon, scale_m = D.kon()
    d0, d1 = D.d[0, D.ns:], D.d[1, D.ns:]

    # mRNA: mechanistic velocity on the NB-sampled counts (top-k genes kept), and along trajectories
    vM_traj = D.trajectory_velocity(D.M)
    vM_meca = _per_time_scale(d0 * (scale_m * kon - D.M), vM_traj, D.times)
    top = np.argpartition(np.abs(vM_meca), -min(k_top, vM_meca.shape[1]), axis=1)[:, -min(k_top, vM_meca.shape[1]):]
    mask = np.zeros_like(vM_meca, dtype=bool)
    mask[np.arange(len(mask))[:, None], top] = True
    vM_meca = np.where(mask, vM_meca, 0.0)
    # Proteins: mechanistic velocity d1·(kon − P) and along trajectories
    vP_traj = D.trajectory_velocity(D.P)
    vP_meca = _per_time_scale(d1 * (kon - D.P), vP_traj, D.times)

    growth = D.R_learned[idx] if D.R_learned is not None else None
    for name, X, v_meca, v_traj, to_space in [
        ('mRNA', np.log1p(D.M[idx]), vM_meca[idx], vM_traj[idx], lambda v: v / (1 + D.M[idx])),
        ('Proteins', D.P[idx], vP_meca[idx], vP_traj[idx], lambda v: v),
    ]:
        E = UMAP(random_state=42, min_dist=0.7).fit_transform(X)
        V_meca = _knn_velocity_embedding(X, to_space(v_meca), E)
        V_traj = _knn_velocity_embedding(X, np.nan_to_num(to_space(v_traj)), E)
        cos = _weighted_cosine(v_meca, v_traj)

        fig = plt.figure(figsize=A4_LANDSCAPE)
        _page_title(fig, f'5. Learned dynamics — {name} velocity fields',
                    f'{len(idx)} trajectory states; kNN-transition projection on a UMAP of '
                    + ('log1p NB-sampled mRNA' if name == 'mRNA' else 'protein levels')
                    + f'. Mechanistic vs trajectory velocity agreement (weighted cosine, gene space): {cos:.2f}.')
        gs = gridspec.GridSpec(1, 3, figure=fig, left=0.03, right=0.95, top=0.86, bottom=0.12, wspace=0.12)
        meca_title = ('Mechanistic (noisy, figure 5): d0·(k·kon(P) − M)' if name == 'mRNA'
                      else 'Mechanistic: d1·(kon(P) − P)')
        vmin, vmax = float(t.min()), float(t.max())
        sca = _stream(fig.add_subplot(gs[0, 0]), E, V_meca, t, meca_title, vmin=vmin, vmax=vmax)
        _stream(fig.add_subplot(gs[0, 1]), E, V_traj, t, 'Trajectories: (x(t+1) − x(t)) / Δt', vmin=vmin, vmax=vmax)
        cax = fig.add_axes([0.04, 0.07, 0.55, 0.018])
        cb = fig.colorbar(sca, cax=cax, orientation='horizontal'); cb.set_label('time', fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax = fig.add_subplot(gs[0, 2])
        if growth is not None and np.isfinite(growth).any():
            lo, hi = np.nanpercentile(growth, [2, 98])
            sca2 = _stream(ax, E, V_traj, growth, f'Trajectories, colour = learned net rate ({D.learned_label})',
                           cmap='magma', vmin=lo, vmax=hi)
            cb2 = fig.colorbar(sca2, ax=ax, fraction=0.045, pad=0.02); cb2.ax.tick_params(labelsize=6)
        elif D.ct is not None:
            _stream(ax, E, V_traj, D.ct[idx], 'Trajectories, colour = cell type', categorical=R.color_map)
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
    """Summary figure for talks, on the mRNA UMAP: states by time (no field), mRNA velocity along the
    inferred trajectories over the cell types, and final learned net rates (no field)."""
    fig = plt.figure(figsize=A4_LANDSCAPE)
    _page_title(fig, '5. Learned dynamics — summary' + ('' if key == 'cell_type' else f' ({key})'),
                f'{len(idx)} trajectory states on the UMAP of log1p NB-sampled mRNA; velocity along the inferred '
                f'trajectories (x(t+1) − x(t)) / Δt; background of the velocity field: {key}.')
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.03, right=0.95, top=0.86, bottom=0.14, wspace=0.12)
    lo, hi = E.min(axis=0), E.max(axis=0)  # same extent on the three panels
    ax = fig.add_subplot(gs[0, 0])
    sca = ax.scatter(E[:, 0], E[:, 1], c=t, cmap='viridis', s=10, linewidths=0, rasterized=True)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    _clean_umap_ax(ax, 'Time')
    cb = fig.colorbar(sca, cax=ax.inset_axes([0.0, -0.08, 1.0, 0.03]), orientation='horizontal')
    cb.set_label('time', fontsize=8); cb.ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[0, 1])
    if labels is not None:
        _stream(ax, E, V_traj, labels, f'mRNA velocity along trajectories, {key}', categorical=color_map,
                s=10, density=0.8, linewidth=1.5, alpha=0.55, color='#1A1A1A')
    else:
        _stream(ax, E, V_traj, t, 'mRNA velocity along trajectories', s=10, density=0.8, linewidth=1.5,
                alpha=0.55, color="#1A1A1A")
    ax = fig.add_subplot(gs[0, 2])
    if growth is not None and np.isfinite(growth).any():
        ok = np.isfinite(growth)
        glo, ghi = np.nanpercentile(growth, [2, 98])
        ax.scatter(E[~ok, 0], E[~ok, 1], c='#DDDDDD', s=10, linewidths=0, rasterized=True)
        sca = ax.scatter(E[ok, 0], E[ok, 1], c=growth[ok], cmap='magma', vmin=glo, vmax=ghi, s=10, linewidths=0,
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

def generate_report(p, split, stim, prior, perturbations=(), out_path=None, net_index=0,
                    normtransform=False, logtransform=True, n_umap=4000, top_regulators=10,
                    top_targets=8, seed=0):
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
        KO/OV conditions from ``Data/KO_OV_Stim_simulate.txt`` (label as in file names,
        genes = perturbed gene names, highlighted in the GRN subgraphs).
    out_path : str, optional
        Output PDF path (default: ``<p>/CardamomOT_report_stim{stim}_prior{prior}.pdf``).
    net_index : int
        Network index along the last axis of ``inter_simul.npy``.
    normtransform, logtransform : bool
        Preprocessing applied before UMAP.
    n_umap : int or None
        Max cells per dataset used in UMAPs (stratified by time); None = all.
    top_regulators, top_targets : int
        Number of regulators drawn and of targets per regulator subgraph.

    Returns
    -------
    str
        Path of the written PDF.
    """
    if out_path is None:
        out_path = os.path.join(p, f'CardamomOT_report_stim{stim}_prior{prior}.pdf')
    matplotlib.rcParams['pdf.fonttype'] = 42

    R = _ReportData(p, split, stim, prior, normtransform, logtransform, list(perturbations), n_umap, seed)

    inter = np.load(os.path.join(p, 'cardamomOT', 'inter_simul.npy'))
    n_networks = inter.shape[2] if inter.ndim == 3 else 1
    matrix = inter[:, :, net_index] if inter.ndim == 3 else inter
    ns = matrix.shape[0] - len(R.genes)
    show_stim = stim >= 0.5
    info = dict(n_stimuli=ns, net_index=net_index, n_networks=n_networks, show_stim=show_stim)
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
        sections = []
        depth_json = os.path.join(p, 'cardamomOT', 'depth_diagnostic.json')
        if os.path.exists(depth_json):
            import json
            depth_csv = os.path.join(p, 'cardamomOT', 'depth_diagnostic.csv')
            sections.append(('0. Per-cell sequencing depth', lambda: _depth_page(
                pdf, R, json.load(open(depth_json)), pd.read_csv(depth_csv) if os.path.exists(depth_csv) else None)))
        sections.append(('1. Generative model', lambda: _model_pages(pdf, R)))
        if os.path.exists(rep_path):
            sections.append(('1. Sample integration', lambda: _integration_page(pdf, R, pd.read_csv(rep_path))))
        sections += [('2. Gene regulatory network',
                      lambda: _grn_pages(pdf, R, matrix, ns, show_stim, top_regulators, top_targets)),
                     ('3. In-silico perturbations', lambda: _perturbation_pages(pdf, R)),
                     ('4. Proliferation', lambda: _proliferation_pages(pdf, R, dyn())),
                     ('5. Learned dynamics — velocity fields', lambda: _velocity_pages(pdf, R, dyn()))]
        if R.test:
            sections.append(('6. Held-out test cells', lambda: _test_pages(pdf, R)))
        for title, fn in sections:
            try:
                fn()
            except Exception as e:  # keep the rest of the report if one section fails
                plt.close('all')
                print(f"[report] Warning: section '{title}' failed: {e!r}")
                _error_page(pdf, title, repr(e))
        d = pdf.infodict()
        d['Title'] = f'CardamomOT report — {os.path.basename(os.path.normpath(p))}'
        d['Author'] = 'CardamomOT'
    return out_path
