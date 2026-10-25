"""
Gene selection from the whole transcriptome (select_genes.py, select_genes = True; train cells only):

1. terminal genes among the high-entropy genes of O. Gandrillon's workflow (gandrillon.py: KD and BUB-entropy
   changes between consecutive timepoints, which carry the time information), each chosen with a round robin over
   the cell types (each against the rest, Wilcoxon on log-normalised counts; significantly DE genes first) so that
   all cell types are covered (no cell type: by decreasing entropy change):
   - at most n_query genes of genes_queries (gene_lists sheet);
   - fate drivers of the classical OT (round robin over the fates);
   - at least n_entropy entropy genes (at least 2 per cell type);
2. a coarse global network on the highly variable genes (+ terminals), built by a network method
   (global_networks: OTVelo-Corr built in, or a custom method of the project);
With several samples (obs['dataset_id']), the selection runs before the integration, so every dynamic
signal is computed within each sample: entropy changes, NB basins and Wilcoxon groups per sample (a gene
informative in one sample is a candidate: union), highly variable genes with the sample as batch, and one
network per sample turned into edge probabilities against its own nulls, combined with weights
cells x transitions (consensus of the shared network); literature and Steiner selection on the combination.
3. a budgeted prize-collecting Steiner selection (steiner_selection): a set of genes, not a tree, in which the
   terminals are anchored to the stimuli by the most probable paths (anchoring = noisy-OR over every edge of the
   selection, so that several paths add up and loops are allowed), then grown by the genes worth the most
   (regulations brought to the selected genes + anchoring brought to the terminals), each gene costing closure_min,
   the remaining budget going to further terminals of the round robins.
"""
import re
import heapq
import numpy as np
import pandas as pd
import scipy.sparse

from .gandrillon import gandrillon_scores, gandrillon_genes
from .mixture import group_init
from .global_networks import build_global_network

# Mitochondrial, ribosomal and (by name) non-coding genes, excluded from the gene universe
_EXCLUDE = re.compile(
    r'^(MT-|mt-|Mt-)'                          # mitochondrial
    r'|^(RP[SL]\d|Rp[sl]\d|MRP[SL]\d|Mrp[sl]\d)'  # ribosomal proteins
    r'|^(LINC|Linc|LOC|MIR|Mir|SNOR|Snor|SCARNA|ENSG|ENSMUSG|Gm\d)'  # non-coding / unannotated
    r'|-AS\d*$|-DT$|-IT\d*$|Rik$|\.\d+$'
)


def gene_universe_mask(var_names, var=None):
    """True for genes kept in the universe: protein-coding if var has a biotype column, else by name."""
    names = np.asarray(var_names).astype(str)
    keep = np.array([not _EXCLUDE.search(g) for g in names])
    if var is not None:
        for col in ('gene_type', 'gene_biotype', 'biotype', 'feature_type'):
            if col in var.columns:
                keep &= var[col].astype(str).str.contains('protein_coding|Gene Expression').values
                break
    return keep


def _dense(X):
    return X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)


def de_groups(adata, genes, cell_type_key=None, sample_key='dataset_id', depth=None):
    """
    Wilcoxon scores (scanpy rank_genes_groups, log-normalised counts) of the genes for each cell type
    (cell_type_key) against the rest, within each sample (obs[sample_key]; sample differences are not taken
    for dynamics). No timepoint groups: the entropy changes of the candidates already carry the time.
    depth: per-cell depth factors (log1p of the counts at the reference depth) or None (total-count normalisation).
    Returns a DataFrame genes x groups ('<key>=<c>', prefixed by '<sample>|' with several samples) of
    scores and one of adjusted p-values (empty without cell_type_key).
    """
    import scanpy as sc
    full = sc.AnnData(X=_dense(adata[:, genes].X).astype(np.float32), obs=adata.obs.copy())
    full.var_names = list(genes)
    if depth is None:
        sc.pp.normalize_total(full, target_sum=1e4)
    else:
        full.X = full.X / np.asarray(depth, dtype=np.float32)[:, None]
    sc.pp.log1p(full)
    smp = full.obs[sample_key].astype(str).values if sample_key in full.obs else np.zeros(full.n_obs).astype(str)
    multi = len(np.unique(smp)) > 1
    scores, padj = {}, {}
    keys = [(cell_type_key, cell_type_key)] if cell_type_key else []
    for sample in np.unique(smp):
        sub = full[smp == sample].copy()
        prefix = f'{sample}|' if multi else ''
        for key, label in keys:
            groups = sub.obs[key].astype(str)
            if groups.nunique() < 2 or groups.value_counts().min() < 2:
                continue
            sub.obs['_g'] = pd.Categorical(groups)
            sc.tl.rank_genes_groups(sub, '_g', method='wilcoxon', n_genes=len(genes))
            res = sub.uns['rank_genes_groups']
            for gr in sub.obs['_g'].cat.categories:
                s = pd.Series(res['scores'][gr], index=res['names'][gr]).reindex(genes)
                p = pd.Series(res['pvals_adj'][gr], index=res['names'][gr]).reindex(genes)
                scores[f'{prefix}{label}={gr}'], padj[f'{prefix}{label}={gr}'] = s.values, p.values
    return pd.DataFrame(scores, index=genes), pd.DataFrame(padj, index=genes)


def nb_init_separation(X, times, cell_types=None, n_cells=1024, seuil=0.01, seed=0, depth=None, dispersion_max=None):
    """
    Separation of the extreme modes found by the initialization of the NB mixture (group_init,
    as in infer_mixture: one mode per time, and per cell type, with a common rate; the most
    diverse labelling kept), per gene (column of X), on at most n_cells cells per timepoint (None = all).
    A gene whose initialization gives no distinct basins is unlikely to get them from the CEM.
    depth: per-cell depth factors (NB(k, c / s_i)) or None. dispersion_max: cap of the rate c, as in the mixture.
    The cells of each timepoint are drawn in proportion to the cell types, as the sub-sample of the mixture.
    """
    from .sampling import stratified_choice
    rng = np.random.default_rng(seed)
    ct = None if cell_types is None else np.asarray(cell_types)
    sub = np.concatenate([stratified_choice(idx, len(idx) if n_cells is None else n_cells,
                                            None if ct is None else ct[idx], rng=rng)
                          for idx in (np.flatnonzero(times == t) for t in np.unique(times))])
    Xs = np.rint(X[sub]).astype(int)
    groups = [times[sub], None if cell_types is None else np.asarray(cell_types)[sub]]

    # Serial: a few ms per gene, less than the transfer overhead of parallel tasks
    return _separation_block(Xs, groups, seuil, None if depth is None else np.asarray(depth, dtype=float)[sub],
                             dispersion_max)


def nb_separation_samples(X, times, cell_types=None, samples=None, n_cells=1024, seuil=0.01, seed=0, depth=None,
                          dispersion_max=None):
    """
    nb_init_separation within each sample (before integration, a shift between samples is not a basin),
    largest value over the samples: a gene with distinct basins in one sample is kept (per-sample mixtures).
    """
    if samples is None or len(np.unique(samples)) == 1:
        return nb_init_separation(X, times, cell_types, n_cells, seuil, seed, depth, dispersion_max)
    out = np.zeros(X.shape[1])
    for s in np.unique(samples):
        m = np.asarray(samples) == s
        ct = None if cell_types is None else np.asarray(cell_types)[m]
        if len(np.unique(times[m])) < 2 and (ct is None or len(np.unique(ct)) < 2):
            continue
        out = np.maximum(out, nb_init_separation(X[m], times[m], ct, n_cells, seuil, seed,
                                                 None if depth is None else np.asarray(depth)[m], dispersion_max))
    return out


def _separation_block(Xb, groups, seuil, s=None, dispersion_max=None):
    out = np.zeros(Xb.shape[1])
    for j in range(Xb.shape[1]):
        x = Xb[:, j]
        if x.min() < x.max():
            res = group_init(x, groups, seuil, s, c_max=dispersion_max)
            out[j] = 0.0 if res is None else res[3]
    return out


def round_robin(candidates, scores, n, taken=(), padj=None, alpha=0.05):
    """
    Up to n candidates, each group in turn taking its best-scored candidate (higher Wilcoxon score =
    more expressed in the group) not yet taken; with padj, the candidates significantly DE in the group
    (padj < alpha, score > 0) come first. Returns [(gene, group)].
    """
    if n <= 0 or not len(candidates):
        return []
    taken = set(taken)
    ranks = {}
    for gr in scores.columns:
        s = scores.loc[candidates, gr]
        de = (padj.loc[candidates, gr] < alpha) & (s > 0) if padj is not None else pd.Series(False, index=s.index)
        ranks[gr] = list(pd.DataFrame({'de': de, 's': s}).sort_values(['de', 's'], ascending=False).index)
    out = []
    while len(out) < n:
        added = False
        for gr, ranked in ranks.items():
            while ranked and ranked[0] in taken:
                ranked.pop(0)
            if ranked and len(out) < n:
                g = ranked.pop(0)
                taken.add(g)
                out.append((g, gr))
                added = True
        if not added:
            break
    # Without groups (no cell type): candidates in their given order (decreasing entropy change)
    for g in candidates:
        if len(out) >= n:
            break
        if g not in taken and not scores.shape[1]:
            taken.add(g)
            out.append((g, ''))
    return out


def _fdr_probabilities(real, null):
    """1 - q for |real| against the |null| distribution (q: monotone tail FDR, pi0 = 1); 0 where real = 0."""
    r, n = np.abs(np.ravel(real)), np.sort(np.abs(np.ravel(null)))
    order = np.argsort(-r)
    rs = r[order]
    S_real = np.arange(1, len(rs) + 1) / len(rs)
    S_null = (len(n) - np.searchsorted(n, rs, side='left')) / len(n)
    q = np.minimum.accumulate(np.minimum(S_null / S_real, 1.0)[::-1])[::-1]
    w = np.empty_like(r)
    w[order] = 1.0 - q
    w[r == 0] = 0.0
    return w.reshape(np.shape(real))


def edge_probabilities(C, C_null, ns, C_null_stim=None):
    """
    Probability of each edge being real, w = 1 - q, q being the FDR of its |C| against the network
    built by the same method on permuted data: gene-to-gene edges against the pooled null gene
    edges of C_null, each stimulus row against its own null row of C_null_stim (default C_null).
    """
    C_null_stim = C_null if C_null_stim is None else C_null_stim
    G = C.shape[0]
    W = np.zeros((G, G))
    off = ~np.eye(G - ns, dtype=bool)
    block = np.zeros((G - ns, G - ns))
    block[off] = _fdr_probabilities(C[ns:, ns:][off], C_null[ns:, ns:][off])
    W[ns:, ns:] = block
    for s in range(ns):
        W[s, ns:] = _fdr_probabilities(C[s, ns:], C_null_stim[s, ns:])
    return W


def permuted_counts(adata, within, seed=0, sample_key='dataset_id'):
    """
    Copy of adata with each gene permuted independently across cells: within each (sample, time)
    group (keeps every gene's temporal marginals, removes cell-to-cell dependences) or across
    all cells (also removes temporal trends).
    """
    X = _dense(adata.X).copy()
    rng = np.random.default_rng(seed)
    if within:
        t = adata.obs['time'].values
        smp = adata.obs[sample_key].values if sample_key in adata.obs else np.zeros(adata.n_obs)
        groups = [np.flatnonzero((t == a) & (smp == b)) for a in np.unique(t) for b in np.unique(smp)]
    else:
        groups = [np.arange(adata.n_obs)]
    for idx in groups:
        if len(idx) > 1:
            for g in range(X.shape[1]):
                X[idx, g] = X[rng.permutation(idx), g]
    out = adata.copy()
    out.X = X
    return out


def literature_probabilities(W, ns, F, covered, weight=1.0, pseudo=1.0):
    """
    Edge probabilities combined with literature feasibility F (gene block) by Bayes' rule on the
    odds: odds' = odds * LR^weight, LR = P(F | edge) / P(F | no edge) estimated on the covered
    pairs with W as soft labels (pseudo-counts: pseudo pairs). Uncovered pairs and stimulus rows
    are unchanged; w = 0 stays 0 (the literature reweights, it does not create edges).
    Returns (W', (LR+, LR-)).
    """
    G = W.shape[0] - ns
    w = np.clip(W[ns:, ns:], 0.0, 1.0 - 1e-6)
    cov = covered & ~np.eye(G, dtype=bool)
    wc, fc = w[cov], F[cov]
    p1 = ((wc * fc).sum() + pseudo * fc.mean()) / (wc.sum() + pseudo)
    p0 = (((1 - wc) * fc).sum() + pseudo * fc.mean()) / ((1 - wc).sum() + pseudo)
    lr_pos, lr_neg = p1 / max(p0, 1e-12), (1 - p1) / max(1 - p0, 1e-12)
    lr = np.where(cov, np.where(F, lr_pos, lr_neg), 1.0) ** weight
    odds = w / (1 - w) * lr
    out = W.copy()
    out[ns:, ns:] = np.where(w > 0, odds / (1 + odds), 0.0)
    return out, (lr_pos, lr_neg)


def steiner_selection(W, roots, required, optional, budget, k_in=10, k_stim=10, w_min=0.05, edge_prior=0.9, C=None,
                      closure_min=0.5, W_parts=None):
    """
    Budgeted prize-collecting Steiner selection on edge probabilities W: a set of genes, not a tree, every edge
    between selected genes counting.

    Graph: edges i -> j with W >= w_min, the k_in most probable regulators of each gene and the k_stim most probable
    targets of each root (ties in W broken by |C| when C is given); a terminal left without incoming edge keeps its
    best one (probability floored at 1e-3: a last resort). Each edge is real with probability p = W * edge_prior
    (edge_prior < 1: every extra step costs something, so that among equally probable paths the shortest wins).
    Anchoring of a selected gene: probability that it is reached from the roots through the selection, noisy-OR over
    its selected regulators, a_j = 1 - prod_i (1 - p_ij a_i) (fixed point; several paths add up, loops allowed).
    1. Required terminals: all selected; the most probable path to each terminal not yet anchored is added, cheapest
       first, starting from any anchored gene (path probability times its anchoring), within budget.
    2. Prize-collecting growth: each gene outside the selection is worth the expected number of real regulations it
       brings to the selected genes, sum_j W_kj (W >= w_min; a missing regulator, even of a gene that has others,
       biases an inference restricted to the selection), plus the anchoring it brings to the terminals,
       sum_t a_k p_kt (1 - a_t) (new or redundant paths from the stimuli); the best gene is added, repeatedly, while
       it is worth at least closure_min (the cost of a gene).
    3. Optional terminals in order (terminal + its most probable path) while the budget allows.
    W_parts (list of per-sample W, combination 'any'): balanced growth, each step serving the sample whose regulation
    of the selection comes least from inside it (closure_fraction), with the regulation term of that sample.
    Returns the selected nodes (in order of addition), the best anchoring regulator and the anchoring of each anchored
    node, the set of optional terminals added and the list of the genes of step 2.
    """
    n = W.shape[0]
    roots = list(roots)
    Wk = np.where(W >= w_min, W, 0.0)
    # Ranking score: probability, then raw strength as tie-breaker
    rank = Wk + (1e-6 * np.abs(C) / max(np.abs(C).max(), 1e-12) if C is not None else 0.0) * (Wk > 0)
    for r in roots:
        Wk[r, np.argsort(-rank[r])[k_stim:]] = 0.0
        rank[r] *= Wk[r] > 0
    # Every terminal keeps at least its best incoming edge (even below w_min), so a path can reach it
    for j in set(required) | set(optional):
        if not (Wk[:, j] > 0).any():
            col = W[:, j] + (1e-6 * np.abs(C[:, j]) if C is not None else 0.0)
            col[j] = -np.inf
            i = int(np.argmax(col))
            if W[i, j] > 0 or (C is not None and C[i, j] != 0):
                Wk[i, j] = max(W[i, j], 1e-3)
                rank[i, j] = Wk[i, j]
    # Probabilities of the kept edges (the k_in most probable regulators of each gene), no self-loop
    P = np.zeros((n, n))
    for j in range(n):
        top = [i for i in np.argsort(-rank[:, j])[:k_in] if Wk[i, j] > 0 and i != j]
        P[top, j] = np.minimum(Wk[top, j] * edge_prior, 1.0 - 1e-9)
    adj = [[(int(j), -np.log(P[i, j])) for j in np.flatnonzero(P[i])] for i in range(n)]
    parts = [W] if W_parts is None else list(W_parts)
    Wgs = []
    for Wp in parts:
        Wg = np.where(Wp >= w_min, Wp, 0.0)
        np.fill_diagonal(Wg, 0.0)
        Wgs.append(Wg)

    selected = list(dict.fromkeys(required))
    chosen = set(selected)
    terminals = list(selected)
    extra, closure = set(), []

    def anchoring():
        # Noisy-OR fixed point on the roots and the selection (least fixed point, from a = 0 off the roots)
        nodes = roots + [v for v in selected if v not in set(roots)]
        sub = P[np.ix_(nodes, nodes)]
        is_root = np.arange(len(nodes)) < len(roots)
        a_sub = is_root.astype(float)
        for _ in range(200):
            new = np.where(is_root, 1.0, -np.expm1(np.log1p(-sub * a_sub[:, None]).sum(axis=0)))
            done = np.max(np.abs(new - a_sub)) < 1e-8
            a_sub = new
            if done:
                break
        a = np.zeros(n)
        a[nodes] = a_sub
        return a

    def paths(a):
        # Most probable paths from the anchored genes (distance -log(path probability x anchoring of its start))
        dist = np.full(n, np.inf)
        prev = np.full(n, -1)
        src = np.flatnonzero(a > 0)
        dist[src] = -np.log(a[src])
        heap = [(dist[s_], int(s_)) for s_ in src]
        heapq.heapify(heap)
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                if d + w < dist[v]:
                    dist[v], prev[v] = d + w, u
                    heapq.heappush(heap, (d + w, v))
        return dist, prev

    def new_nodes(t, prev):
        # Genes of the path to t not selected yet
        path, u = [], t
        while prev[u] != -1:
            path.append(u)
            u = prev[u]
        return [v for v in reversed(path) if v not in chosen]

    def add(nodes):
        for v in nodes:
            if v not in chosen:
                chosen.add(v)
                selected.append(v)

    # 1. Required terminals not yet anchored, cheapest path first
    while True:
        a = anchoring()
        dist, prev = paths(a)
        pending = sorted((t for t in terminals if a[t] <= 0 and np.isfinite(dist[t])), key=lambda t: dist[t])
        path = next((p for p in (new_nodes(t, prev) for t in pending) if p and len(chosen) + len(p) <= budget), None)
        if path is None:
            break
        add(path)
    # 2. Prize-collecting growth: regulation explained in the selection + anchoring brought to the terminals
    excluded = np.zeros(n, bool)
    excluded[roots] = True
    while len(chosen) < budget:
        a = anchoring()
        inside = roots + [v for v in selected if v not in set(roots)]
        sel = sorted(chosen)
        excluded[sel] = True
        # Anchoring gain of each candidate k: its anchoring through the selection, times its edges to the terminals
        a_k = -np.expm1(np.log1p(-P[inside] * a[inside][:, None]).sum(axis=0))
        term = [t for t in terminals if t in chosen]
        anchor_gain = a_k * (P[:, term] @ (1.0 - a[term])) if term else np.zeros(n)
        order = np.argsort([closure_fraction(Wp, sel, roots, w_min) for Wp in parts]) if len(parts) > 1 else [0]
        k = None
        for i in order:
            gain = Wgs[i][:, sel].sum(axis=1) + anchor_gain
            gain[excluded] = -np.inf
            c = int(np.argmax(gain))
            if gain[c] >= closure_min:
                k = c
                break
        if k is None:
            break
        add([k])
        closure.append(k)
    # 3. Optional terminals in order, while the budget allows
    for t in optional:
        if len(chosen) >= budget:
            break
        if t in chosen:
            continue
        dist, prev = paths(anchoring())
        path = new_nodes(t, prev) if np.isfinite(dist[t]) else None
        if path is not None and len(chosen) + len(path) <= budget:
            add(path)
            terminals.append(t)
            extra.add(t)
    # Best anchoring regulator and anchoring of each anchored node
    a = anchoring()
    inside = roots + [v for v in selected if v not in set(roots)]
    parent, anchor = {}, {}
    for v in selected:
        if a[v] > 0 and v not in set(roots):
            contrib = P[inside, v] * a[inside]
            parent[v] = int(inside[int(np.argmax(contrib))])
            anchor[v] = float(a[v])
    return selected, parent, anchor, extra, closure


def selection_edges(W, node_names, sel, ns, p_min=0.6, F=None, covered=None):
    """
    Edges of the global network inside the selection (stimuli included as regulators): i -> j with
    W >= p_min and, with the literature (F, covered: gene blocks), feasible within literature_depth;
    pairs the literature does not cover are kept, as in the prior, and marked '*'.
    Returns {gene: (regulators, targets)}, lists of (name, w, covered) by decreasing w.
    """
    pos = {g: i for i, g in enumerate(node_names)}
    src = list(range(ns)) + [pos[g] for g in sel]
    tgt = [pos[g] for g in sel]
    E = W[np.ix_(src, tgt)] >= p_min
    cov = np.ones(E.shape, bool)
    if F is not None:
        g_src, g_tgt = np.array([max(i - ns, 0) for i in src]), np.array(tgt) - ns
        cov[ns:] = covered[np.ix_(g_src[ns:], g_tgt)]
        E[ns:] &= F[np.ix_(g_src[ns:], g_tgt)] | ~cov[ns:]
    E[ns + np.arange(len(sel)), np.arange(len(sel))] = False
    out = {g: ([], []) for g in sel}
    for a, b in zip(*np.nonzero(E)):
        u, v, w = node_names[src[a]], sel[b], float(W[src[a], tgt[b]])
        out[v][0].append((u, w, bool(cov[a, b])))
        if a >= ns:
            out[u][1].append((v, w, bool(cov[a, b])))
    for lists in out.values():
        for lst in lists:
            lst.sort(key=lambda x: -x[1])
    return out


def closure_fraction(W, sel, roots, w_min=0.05):
    """Share of the probable regulation (W >= w_min) entering the selected genes that comes from the selection (or the stimuli)."""
    Wg = np.where(W >= w_min, W, 0.0)
    inside = list(sel) + list(roots)
    total = Wg[:, sel].sum()
    return float(Wg[np.ix_(inside, sel)].sum() / total) if total > 0 else 1.0


def select_genes(adata, queries, num_max_genes, n_query=20, n_entropy=10, stim=None, cell_type_key=None,
                 n_hvg=3000, n_top_entropy=200, n_cells_entropy=1000, network_method='otvelo_granger',
                 network_params=None, project_path=None, k_in=10, k_stim=10, min_edge_prob=0.05,
                 edge_prior=0.9, null_network='hybrid', closure_min=0.5, literature=True, literature_depth=3,
                 literature_weight=1.0, literature_resources='extended', max_free_params=None,
                 min_entropy_change=0.1, min_nb_separation=0.15, n_cells_mixture=1024, seuil_mixture=0.01,
                 dispersion_max=None,
                 n_cells_network=1000, sample_key='dataset_id', forced_genes=None, stimulus_targets=None,
                 drivers=None, n_driver=0,
                 species=None, use_depth_factor=False, report_edge_prob=0.6, entropy_preselection=True,
                 sample_combination='consensus', seed=0, verb=True):
    """
    Full selection on raw counts (see module docstring). n_query + n_entropy must leave budget for
    the paths (< num_max_genes). The network is turned into edge probabilities against the same
    method run on cells permuted gene by gene (edge_probabilities). null_network: 'hybrid' (gene
    edges against a within-(sample, time) permutation: evidence beyond shared temporal trends,
    which CardamomOT explains by the stimulus and basals; stimulus edges against an all-cell
    permutation), 'within_time' or 'all_cells' for both. The Steiner selection anchors the terminals to the
    stimuli by the most probable paths and grows by prize (steiner_selection); remaining budget is filled with
    further terminals of the round robins.
    literature: edge probabilities reweighted by OmniPath feasibility (paths of at most
    literature_depth edges ending with a TF -> target edge, literature_probabilities; species
    detected from the gene names if None), and literature prior of the selection returned.
    Variability floor, applied to every gene (queries included) before anything else: the largest
    BUB-entropy change between consecutive timepoints (Gandrillon MDE, bits) must reach
    min_entropy_change, and the NB-mixture initialization (nb_init_separation, n_cells_mixture
    cells per time, seuil_mixture, rate at most dispersion_max as in the mixture) must separate its extreme modes by
    min_nb_separation.
    entropy_preselection: queries, drivers and the genes of the network (Steiner paths, growth) are taken
    among the entropy candidates (Gandrillon KD & MDE, top n_top_entropy per transition) and the forced genes only.
    stim: (n_times, n_stimuli) schedule on the sorted timepoints, or {sample: schedule} (None: default).
    drivers: [(gene, 'fate=<type>')] fate drivers of the classical OT (round robin over the fates,
    classical_ot.driver_order): the first n_driver are required terminals (role 'driver', after the queries,
    before the entropy genes), subject to the universe and the variability floor; the others fill the budget.
    forced_genes: genes always selected (required terminals, exempt from the variability floor),
    e.g. those perturbed in KO_OV_Stim_simulate.txt. stimulus_targets: one gene list per stimulus
    (read_stimulus_targets): stimulus edges are restricted to its listed genes, possible interactions only
    (they do not enter the gene universe; no constraint if
    none is in the data).
    max_free_params (with the literature, for a hard prior): the gene budget is the largest one
    whose selection has at most this many free network parameters (free_parameters of its
    literature prior); num_max_genes is then ignored.
    The report lists, for every selected gene, its regulators (is_regulated_by) and targets (regulates)
    inside the selection: edges with probability >= report_edge_prob, literature-feasible (selection_edges), and,
    with several samples, the samples in which it has such an edge (edge_samples).
    sample_combination (several samples): 'consensus' = edge probabilities averaged over the samples (weights cells x
    transitions; an edge must be supported in several samples, for a network shared by them), 'any' = probabilistic
    OR 1 - prod(1 - W_s) (an edge supported in one sample is a candidate, for independent condition networks), with a
    closure balanced over the samples (steiner_selection).
    Returns (selected gene names, report DataFrame, dict(C, C_null, W, genes) or None,
    literature prior of the selection (G x G, selection order) or None).
    """
    import scanpy as sc
    n_driver = n_driver if drivers else 0
    if n_query + n_entropy + n_driver >= num_max_genes and not max_free_params:
        raise ValueError(f"n_query_genes + n_entropy_genes + n_driver_genes ({n_query} + {n_entropy} + {n_driver}) "
                         f"must be lower than "
                         f"num_max_genes ({num_max_genes}) to leave budget for the paths from the stimulus")
    names = np.asarray(adata.var_names).astype(str)
    name_set = set(names)
    missing = [g for g in queries if g not in name_set]
    if missing and verb:
        print(f"[gene_selection] {len(missing)} query genes absent from the data: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    queries = [g for g in dict.fromkeys(queries) if g in name_set]
    drivers = [(g, grp) for g, grp in (drivers or []) if g in name_set]
    forced = [g for g in dict.fromkeys(forced_genes or []) if g in name_set]
    missing_f = [g for g in dict.fromkeys(forced_genes or []) if g not in name_set]
    if missing_f and verb:
        print(f"[gene_selection] Warning: perturbed genes absent from the data: {missing_f}")
    queries = list(dict.fromkeys(forced + queries))  # forced genes enter the universe as queries

    # Universe: protein-coding, no mito/ribo, expressed in at least 3 cells (queries always kept)
    Xa = adata.X  # kept sparse: whole-transcriptome dense copies do not fit large datasets
    expressed = np.asarray((Xa > 0).sum(axis=0)).ravel() >= 3
    universe = (gene_universe_mask(names, adata.var) & expressed) | np.isin(names, queries)
    ad_u = adata[:, universe]
    u_names = names[universe]
    times = adata.obs['time'].values.astype(float) if 'time' in adata.obs else np.zeros(adata.n_obs)
    tu = np.sort(np.unique(times))
    # Samples (obs[sample_key]): selection before the integration, so dynamics are scored within each sample
    smp = adata.obs[sample_key].astype(str).values if sample_key in adata.obs else np.full(adata.n_obs, '0')
    samples = list(np.unique(smp))
    multi = len(samples) > 1
    dyn = [s_ for s_ in samples if len(np.unique(times[smp == s_])) >= 2]  # samples with a dynamics
    # Gene scores of the whole universe (entropy, KD, variability) on at most n_cells_entropy cells per (time, sample)
    rng0 = np.random.default_rng(seed)
    sub = np.sort(np.concatenate([rng0.choice(i, min(n_cells_entropy, len(i)), replace=False)
                                  for i in (np.flatnonzero((times == t) & (smp == s_)) for s_ in samples for t in tu)
                                  if len(i)]))
    # Depth factors (estimate_cell_depth.py): scores on counts at the reference depth
    depth = (np.asarray(adata.obs['depth_factor'].values, dtype=float)
             if (use_depth_factor and 'depth_factor' in adata.obs) else None)
    X = _dense(Xa[sub][:, universe]).astype(np.float32)
    if depth is not None:
        X = X / depth[sub, None].astype(np.float32)
    t_sub, s_sub = times[sub], smp[sub]
    cts = adata.obs[cell_type_key].astype(str).values if cell_type_key else None
    if verb:
        print(f"[gene_selection] Universe: {len(u_names)} genes (protein-coding, no mito/ribo, >= 3 cells); "
              f"gene scores on {len(sub)} cells (<= {n_cells_entropy} per time"
              + (f" and sample; {len(samples)} samples scored separately" if multi else "") + ")"
              + ("; counts at the reference depth (obs['depth_factor'])" if depth is not None else ""))

    # Variability floor: Gandrillon entropy change, then distinct basins at the NB-mixture initialization
    kd = mde = None
    eligible = np.ones(len(u_names), bool)
    if dyn:
        # Transitions of every sample stacked: a gene changing in one sample is a candidate (union)
        parts = [gandrillon_scores(X[s_sub == s_], t_sub[s_sub == s_], n_cells=n_cells_entropy, rng=seed) for s_ in dyn]
        kd, mde = np.vstack([q[0] for q in parts]), np.vstack([q[1] for q in parts])
        eligible &= mde.max(axis=0) >= min_entropy_change
    n_mde = int(eligible.sum())
    # Network universe: highly variable genes among those above the entropy floor
    hvg = np.zeros(len(u_names), bool)
    if dyn:
        tmp = sc.AnnData(X=np.log1p(X[:, eligible]))
        tmp.obs['sample'] = s_sub
        # Several samples: variability within each sample (batch_key), not between them
        sc.pp.highly_variable_genes(tmp, n_top_genes=min(n_hvg, int(eligible.sum())), flavor='seurat',
                                    batch_key='sample' if multi else None)
        hvg[np.flatnonzero(eligible)[tmp.var['highly_variable'].values]] = True
    nb_check = bool(dyn) or (cts is not None and len(np.unique(cts)) > 1)
    if nb_check:
        # NB basins checked here on the candidate terminals only; the genes the Steiner selection or the
        # closure would add are checked on demand (most network genes are never selected)
        ent0 = set(u_names[gandrillon_genes(kd, mde, top_n=n_top_entropy)]) if kd is not None else set()
        cand = np.flatnonzero(eligible & np.isin(u_names, list(ent0) + list(queries) + [g for g, _ in drivers]))
        d = nb_separation_samples(_dense(Xa[:, np.flatnonzero(universe)[cand]]), times, cts, smp if multi else None,
                                  n_cells_mixture, seuil_mixture, seed, depth, dispersion_max)
        eligible[cand[d < min_nb_separation]] = False
    # Forced genes stay selectable whatever their variability
    below = [g for g in forced if g in set(u_names[~eligible])]
    eligible[np.isin(u_names, forced)] = True
    if below and verb:
        print(f"[gene_selection] Warning: perturbed genes kept although below the variability floor: {below}")
    dropped_q = [g for g in queries if g in set(u_names[~eligible])]
    if verb:
        print(f"[gene_selection] Variability floor: {len(u_names) - n_mde} genes with entropy change < "
              f"{min_entropy_change} bit, {n_mde - int(eligible.sum())} candidate terminals without distinct NB "
              f"basins at initialization (separation < {min_nb_separation}); {int(eligible.sum())} genes kept")
        if dropped_q:
            print(f"[gene_selection] {len(dropped_q)} query genes below the floor, dropped: "
                  f"{dropped_q[:15]}{' ...' if len(dropped_q) > 15 else ''}")
    queries = [g for g in queries if g not in set(dropped_q)]
    n_drv = len(drivers)
    drivers = [(g, grp) for g, grp in drivers if g in set(u_names[eligible]) and g not in set(queries)]
    if verb and n_drv:
        print(f"[gene_selection] Fate drivers (classical OT): {len(drivers)} of {n_drv} in the universe and above the "
              f"variability floor")
    keep_u = eligible
    universe[np.flatnonzero(universe)[~keep_u]] = False
    ad_u = adata[:, universe]
    u_names = u_names[keep_u]
    kd = kd[:, keep_u] if kd is not None else None
    mde = mde[:, keep_u] if mde is not None else None
    hvg = hvg[keep_u]

    # Entropy candidates (Gandrillon) and DE status of all candidates
    ent_candidates = []
    if kd is not None:
        ent_idx = gandrillon_genes(kd, mde, top_n=n_top_entropy)
        # Highest entropy change first (largest MDE over the transitions)
        ent_candidates = [u_names[i] for i in ent_idx[np.argsort(-mde[:, ent_idx].max(axis=0), kind='stable')]]
        if verb:
            print(f"[gene_selection] Entropy genes (KD top {n_top_entropy} & MDE top {n_top_entropy} per transition): "
                  f"{len(ent_candidates)}")
    # Entropy pre-selection: queries, drivers and every gene of the network (hence of the Steiner selection and of the
    # closure) among the entropy candidates only; the perturbed genes are always kept
    allowed = None
    if entropy_preselection and ent_candidates:
        allowed = set(ent_candidates) | set(forced)
        out_q = [g for g in queries if g not in allowed]
        out_d = [g for g, _ in drivers if g not in allowed]
        queries = [g for g in queries if g in allowed]
        drivers = [(g, grp) for g, grp in drivers if g in allowed]
        if verb:
            print(f"[gene_selection] Entropy pre-selection ({len(ent_candidates)} genes + perturbed ones): "
                  f"{len(queries)} queries kept, {len(out_q)} dropped{(' ' + str(out_q[:10])) if out_q else ''}; "
                  f"{len(drivers)} fate drivers kept, {len(out_d)} dropped")
    pool = list(dict.fromkeys(queries + [g for g, _ in drivers] + ent_candidates))
    scores, padj = de_groups(ad_u, pool, cell_type_key, sample_key, depth) if pool else (pd.DataFrame(), pd.DataFrame())

    # At least two entropy genes per cell type (within each sample)
    if 2 * scores.shape[1] > n_entropy:
        if verb:
            print(f"[gene_selection] n_entropy_genes raised from {n_entropy} to twice the number of cell-type groups "
                  f"({2 * scores.shape[1]}): two entropy genes per cell type")
        n_entropy = 2 * scores.shape[1]
    # Round robins over the cell types (DE genes first): the first n_query / n_entropy are required, the rest fill
    # the budget; queries ordered by entropy change too
    rank_e = {g: i for i, g in enumerate(ent_candidates)}
    queries = sorted(queries, key=lambda g: rank_e.get(g, len(rank_e)))
    q_all = round_robin(queries, scores, len(queries), padj=padj)
    e_all = round_robin([g for g in ent_candidates if g not in queries], scores, len(ent_candidates), padj=padj)
    # Fate drivers (classical OT, already in round robin over the fates), then entropy genes
    q_req = q_all[:n_query]
    taken = set(forced) | {g for g, _ in q_req}
    d_all = [x for x in drivers if x[0] not in taken]
    d_req = d_all[:n_driver]
    taken |= {g for g, _ in d_req}
    e_all = [x for x in e_all if x[0] not in {g for g, _ in d_req}]
    e_req = [x for x in e_all if x[0] not in taken][:n_entropy]
    required = list(dict.fromkeys(forced + [g for g, _ in q_req] + [g for g, _ in d_req] + [g for g, _ in e_req]))
    group = dict(q_all + e_all + d_all)
    # Role of a gene in several lists: perturbed > query > driver > entropy
    role = {**{g: 'entropy' for g, _ in e_all}, **{g: 'driver' for g, _ in d_all}, **{g: 'query' for g, _ in q_all},
            **{g: 'perturbed' for g in forced}}
    # Budget left: further queries, drivers and entropy genes in turn
    rest = [[g for g, _ in lst if g not in required] for lst in (q_all, d_all, e_all)]
    optional = list(dict.fromkeys(g for i in range(max(map(len, rest), default=0)) for lst in rest if i < len(lst)
                                  for g in [lst[i]]))
    if verb:
        print(f"[gene_selection] Required terminals: {len(q_req)}/{len(queries)} queries, {len(d_req)} fate drivers, "
              f"{len(e_req)} entropy genes (round robin over {scores.shape[1]} cell-type groups)")
        if len(e_req) < n_entropy:
            print(f"[gene_selection] Warning: only {len(e_req)} entropy genes available (n_entropy_genes = {n_entropy})")

    net, sel, rows, lg, edges = None, list(required), {}, None, {}
    edge_samples = {}  # gene -> samples in which it has an edge inside the selection
    if not dyn:
        if verb:
            print("[gene_selection] Single timepoint: no network, required terminals only")
    else:
        # Network universe: highly variable genes + all candidate terminals
        # Listed stimulus targets join the network universe, so that the constraint can apply
        # Listed stimulus targets are possible interactions only: they do not enter the network universe
        keep = hvg | np.isin(u_names, pool)
        if allowed is not None:
            keep = np.isin(u_names, list(allowed))  # entropy pre-selection: the network stays among them
        net_genes = list(u_names[keep])
        # One network per sample with its own nulls (at most n_cells_network cells per time), turned into edge
        # probabilities, then combined with weights cells x transitions (consensus of a shared network)
        rng1 = np.random.default_rng(seed)
        parts, weights = [], []
        for s_ in dyn:
            cells = np.sort(np.concatenate([rng1.choice(i, min(n_cells_network, len(i)), replace=False)
                                            for i in (np.flatnonzero((times == t) & (smp == s_)) for t in tu) if len(i)]))
            ad_net = adata[cells, net_genes].copy()
            if depth is not None:  # network on counts at the reference depth
                ad_net.X = _dense(ad_net.X).astype(np.float32) / depth[cells, None].astype(np.float32)
            ts = np.sort(np.unique(times[cells]))
            st = _sample_stimulus(stim, s_, len(tu))
            st = None if st is None else st[np.searchsorted(tu, ts)]
            if verb:
                print(f"[gene_selection] Global network '{network_method}' on {len(net_genes)} genes and {len(cells)} "
                      f"cells (<= {n_cells_network} per time" + (f", sample {s_}" if multi else "")
                      + f"), and its null network(s) ({null_network})")
            run = lambda ad_: build_global_network(network_method, ad_, st, seed=seed, params=network_params,
                                                   project_path=project_path)
            C_s = run(ad_net)
            C_time = run(permuted_counts(ad_net, True, seed)) if null_network in ('hybrid', 'within_time') else None
            C_all = run(permuted_counts(ad_net, False, seed)) if null_network in ('hybrid', 'all_cells') else None
            C_null_s = C_time if C_time is not None else C_all
            C_null_stim_s = C_all if C_all is not None else C_time
            ns = C_s.shape[0] - len(net_genes)
            parts.append((C_s, C_null_s, C_null_stim_s, edge_probabilities(C_s, C_null_s, ns, C_null_stim_s)))
            weights.append(float(np.sum(smp == s_)) * (len(ts) - 1))
        w = np.array(weights) / np.sum(weights)
        C, C_null, C_null_stim, W = (sum(wi * q[k] for wi, q in zip(w, parts)) for k in range(4))
        any_mode = multi and len(parts) > 1 and sample_combination == 'any'
        if any_mode:  # probabilistic OR: an edge supported in one sample is a candidate
            W = 1.0 - np.prod([1.0 - q[3] for q in parts], axis=0)
        if verb and multi:
            print(f"[gene_selection] Edge probabilities combined over the samples ({sample_combination}"
                  + ("" if any_mode else ", weights cells x transitions") + "): "
                  + ', '.join(f'{s_} {wi:.2f}' for s_, wi in zip(dyn, w)))
        node_names = [f'Stimulus{"" if ns == 1 else " " + str(i + 1)}' for i in range(ns)] + net_genes
        idx = {g: ns + i for i, g in enumerate(net_genes)}
        net = dict(C=C, C_null=C_null, C_null_stim=C_null_stim, W=W, genes=np.array(node_names))
        if multi:
            net.update(W_samples=np.stack([q[3] for q in parts]), samples=np.array(dyn), sample_weights=w)
        lg = _literature(names, species, literature, literature_resources, verb)
        if lg is not None:
            # Feasibility with any intermediate (the selection is not known yet)
            reg, tgt = lg.coverage(net_genes)
            F = lg.feasibility(net_genes, literature_depth)
            covered = reg[:, None] & tgt[None, :]
            W, lr = literature_probabilities(net['W'], ns, F, covered, literature_weight)
            net.update(W=W, W_data=net['W'], F=F)
            if multi:  # same literature reweighting of each sample (closure balance, edges per sample)
                net['W_samples'] = np.stack([literature_probabilities(Ws, ns, F, covered, literature_weight)[0]
                                             for Ws in net['W_samples']])
            if verb:
                print(f"[gene_selection] Literature (depth {literature_depth}): {reg.mean() * 100:.0f}% of the genes "
                      f"covered as regulators, {tgt.mean() * 100:.0f}% as targets, {F[reg][:, tgt].mean() * 100:.1f}% "
                      f"of the covered pairs feasible; LR {lr[0]:.2f} (feasible) / {lr[1]:.2f} (infeasible)")
        if verb:
            off = ~np.eye(len(net_genes), dtype=bool)
            print(f"[gene_selection] Edge probabilities: {np.mean(W[ns:, ns:][off] >= 0.5) * 100:.2f}% of gene edges "
                  f"with w >= 0.5, {np.mean(W[ns:, ns:][off] >= min_edge_prob) * 100:.2f}% with w >= {min_edge_prob}")
        # Variability floor of the other network genes, checked when the tree or the closure picks them:
        # a gene failing it is removed from the graph and the selection is redone
        checked = {idx[g] for g in pool if g in idx}
        rejected = []
        Wm, Cm = W.copy(), C.copy()
        Wsm = net['W_samples'].copy() if multi and 'W_samples' in net else None
        # Possible targets of the stimuli: other stimulus edges removed from the graph
        if stimulus_targets:
            from ..config import stimulus_target_mask
            mask = stimulus_target_mask(stimulus_targets, net_genes, ns)
            # A stimulus whose listed targets are in the data but none in the network has no possible target here
            data_up = {str(g).upper() for g in names}
            for s_ in range(ns):
                col = {str(g).upper() for g in stimulus_targets[s_]} if s_ < len(stimulus_targets) else set()
                if col & data_up and not ({str(g).upper() for g in net_genes} & col):
                    mask = np.ones((ns, len(net_genes)), bool) if mask is None else mask
                    mask[s_] = False
                    if verb:
                        print(f"[gene_selection] stimulus {s_}: no listed target among the network genes, no stimulus edge")
            if mask is not None:
                Wm[:ns, ns:] *= mask
                Cm[:ns, ns:] *= mask
                if Wsm is not None:
                    Wsm[:, :ns, ns:] *= mask

        def steiner(B):
            while True:
                res = steiner_selection(
                    Wm, list(range(ns)), [idx[g] for g in required], [idx[g] for g in optional], B,
                    k_in=k_in, k_stim=k_stim, w_min=min_edge_prob, edge_prior=edge_prior, C=Cm,
                    closure_min=closure_min, W_parts=list(Wsm) if any_mode else None)
                new = [v for v in res[0] if v >= ns and v not in checked] if nb_check else []
                if not new:
                    return res
                d = nb_separation_samples(_dense(adata[:, [node_names[v] for v in new]].X), times, cts,
                                          smp if multi else None, n_cells_mixture, seuil_mixture, seed, depth,
                                          dispersion_max)
                checked.update(new)
                bad = [v for v, dv in zip(new, d) if dv < min_nb_separation]
                if not bad:
                    return res
                rejected.extend(bad)
                Wm[bad, :], Wm[:, bad], Cm[bad, :], Cm[:, bad] = 0, 0, 0, 0
                if Wsm is not None:
                    Wsm[:, bad, :], Wsm[:, :, bad] = 0, 0
        if max_free_params and lg is None and verb:
            print(f"[gene_selection] Warning: no literature, parameter budget ignored (budget {num_max_genes} genes)")
        if max_free_params and lg is not None:
            num_max_genes, res = _parameter_budget(steiner, lg, node_names, ns, max_free_params, len(required),
                                                   len(net_genes), literature_depth, verb)
        else:
            res = steiner(num_max_genes)
        chosen, parent, anchor, extra, closure = res
        if verb and nb_check:
            print(f"[gene_selection] Variability floor of the genes added by the tree or the closure: "
                  f"{len(checked) - len({idx[g] for g in pool if g in idx})} checked, {len(rejected)} rejected")
        sel = [node_names[v] for v in chosen]
        for v, u in parent.items():
            rows[node_names[v]] = dict(anchor_parent=node_names[u], anchoring=anchor[v], edge_prob=float(W[u, v]))
        chosen_set = list(chosen)
        for v in closure:
            role[node_names[v]] = 'regulator'
        # Every probable and literature-feasible edge inside the selection (Wm: stimulus-target constraint)
        edges = selection_edges(Wm, node_names, sel, ns, report_edge_prob,
                                *((F, covered) if lg is not None else (None, None)))
        # Samples in which each gene has such an edge (several samples)
        if Wsm is not None:
            for s_, Ws in zip(dyn, Wsm):
                e_s = selection_edges(Ws, node_names, sel, ns, report_edge_prob,
                                      *((F, covered) if lg is not None else (None, None)))
                for g, (rg, tg) in e_s.items():
                    if rg or tg:
                        edge_samples.setdefault(g, []).append(str(s_))
        terminals_idx = [idx[g] for g in required]
        if verb:
            if Wsm is not None:
                print("[gene_selection] Regulation of the selection coming from inside it, per sample: "
                      + ', '.join(f'{s_} {closure_fraction(Ws, list(chosen), list(range(ns)), min_edge_prob) * 100:.0f}%'
                                  for s_, Ws in zip(dyn, Wsm)))
            path_nodes = [v for v in chosen if v not in set(closure)]
            print(f"[gene_selection] Regulation of the selection coming from inside it (w >= {min_edge_prob}): "
                  f"{closure_fraction(W, terminals_idx, list(range(ns)), min_edge_prob) * 100:.0f}% for the required "
                  f"terminals alone, {closure_fraction(W, chosen_set, list(range(ns)), min_edge_prob) * 100:.0f}% for the "
                  f"final selection ({len(closure)} closure regulators)")
    if lg is None and not dyn:
        lg = _literature(names, species, literature, literature_resources, verb)
    prior = None
    st0 = _sample_stimulus(stim, samples[0], len(tu))
    n_stim = 0 if st0 is None else st0.shape[1]
    if lg is not None:
        # Prior of the final selection: intermediates outside it (observed ones are network chains)
        prior = lg.prior(sel, literature_depth)
        if verb:
            off = ~np.eye(len(sel), dtype=bool)
            print(f"[gene_selection] Literature prior of the selection: {np.mean(prior[off] > 0) * 100:.0f}% of the "
                  f"edges allowed ({int((prior[off] > 0).sum())} free interactions), {np.mean(prior[off] == 1) * 100:.0f}% "
                  f"with weight 1 (feasible through unobserved intermediates, or not covered); "
                  f"{free_parameters(prior, n_stim)} free network parameters")
    fmt = lambda lst: ', '.join(f"{u} ({w:.2f}{'' if c else '*'})" for u, w, c in lst)
    report = pd.DataFrame([dict(gene=g, role=role.get(g, 'steiner'), group=group.get(g, ''),
                                required=g in required, **rows.get(g, {})) for g in sel],
                          columns=['gene', 'role', 'group', 'required', 'anchor_parent', 'anchoring', 'edge_prob'])
    report['anchored'] = report['anchor_parent'].notna()
    reg_by, regs = ([edges.get(g, ([], []))[k] for g in report['gene']] for k in (0, 1))
    report['n_regulators'] = [len(x) for x in reg_by]
    report['is_regulated_by'] = [fmt(x) for x in reg_by]
    report['n_targets'] = [len(x) for x in regs]
    report['regulates'] = [fmt(x) for x in regs]
    report['connected'] = (report['n_regulators'] + report['n_targets']) > 0
    if multi:
        report['edge_samples'] = [', '.join(edge_samples.get(g, [])) for g in report['gene']]
    if len(pool):
        sig = (padj < 0.05) & (scores > 0)
        report['DE_groups'] = [', '.join(sig.columns[sig.loc[g].values]) if g in sig.index else '' for g in report['gene']]
    if verb:
        r = report['role']
        print(f"[gene_selection] Selected {len(sel)} genes (budget {num_max_genes}): {(r == 'query').sum()} queries, "
              f"{(r == 'driver').sum()} fate drivers, "
              f"{(r == 'entropy').sum()} entropy genes, {(r == 'steiner').sum()} Steiner genes, "
              f"{(r == 'regulator').sum()} growth genes (regulators, redundant paths) "
              f"({int((~report['required'] & r.isin(['query', 'entropy', 'driver'])).sum())} extra terminals); "
              f"{int((~report['anchored'] & report['required']).sum())} required terminals not anchored to the stimuli")
        if net is not None:
            print(f"[gene_selection] Edges inside the selection (w >= {report_edge_prob}"
                  + (", literature-feasible" if lg is not None else "") + f"): {int(report['n_regulators'].sum())}; "
                  f"{int((~report['connected']).sum())} of the {len(sel)} genes without any (is_regulated_by / regulates "
                  f"of cardamomOT/gene_selection_report.csv)")
    return sel, report, net, prior


def _sample_stimulus(stim, sample, n_times):
    """(n_times, n_stimuli) stimulus schedule of a sample: stim is shared (array) or per sample ({sample: array})."""
    if stim is None:
        return None
    if isinstance(stim, dict):
        default = stim.get(None)
        st = stim.get(str(sample), default)
        return None if st is None else np.asarray(st, dtype=float).reshape(n_times, -1)
    return np.asarray(stim, dtype=float).reshape(n_times, -1)


def free_parameters(prior, ns):
    """Free network parameters under a hard prior: allowed gene -> gene entries (self-regulation
    included) + the stimulus -> gene edges."""
    return int((prior > 0).sum() + ns * prior.shape[0])


def _parameter_budget(steiner, lg, node_names, ns, max_free_params, n_required, n_genes, depth, verb):
    """
    Largest gene budget whose Steiner selection has at most max_free_params free parameters under
    its literature prior (bisection; the count grows roughly as budget^2 x allowed share).
    Returns (budget, steiner result).
    """
    cache = {}

    def run(B):
        if B not in cache:
            res = steiner(B)
            sel = [node_names[v] for v in res[0]]
            cache[B] = (free_parameters(lg.prior(sel, depth), ns), len(sel), res)
        return cache[B]

    lo, hi = n_required + 1, min(n_genes, int(np.ceil(np.sqrt(max_free_params))) * 3)
    if run(lo)[0] > max_free_params:
        if verb:
            print(f"[gene_selection] Warning: the {n_required} required terminals already exceed "
                  f"max_free_params = {max_free_params} ({run(lo)[0]} free parameters)")
        return lo, run(lo)[2]
    if run(hi)[0] <= max_free_params:
        lo = hi = run(hi)[1]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        n_free, n_sel, _ = run(mid)
        if n_free <= max_free_params and n_sel == mid:
            lo = mid
        elif n_free <= max_free_params:  # no candidate left: the selection stopped growing
            lo = hi = n_sel
        else:
            hi = mid
    n_free, n_sel, res = run(lo)
    if verb:
        print(f"[gene_selection] Parameter budget: {n_sel} genes for {n_free} free network parameters "
              f"(max_free_params = {max_free_params})")
    return n_sel, res


def _literature(names, species, use, resources, verb):
    """Literature graph of the species (detected from the gene names if None), None if unused or unavailable."""
    if not use:
        return None
    from .literature import literature_graph
    if species is None:
        from .halflife_db import detect_species
        species = detect_species(list(names))[0]
    try:
        return literature_graph(species, resources)
    except Exception as e:  # offline, OmniPath down
        if verb:
            print(f"[gene_selection] Warning: literature unavailable ({e}); selection on the data only")
        return None


def proliferation_markers(adata, n, use_depth=False, anchors=None, exclude=(), label_key='cell_type_proliferation',
                          cell_type_keys=('cell_type_proliferation', 'cell_type'), group_keys=('dataset_id', 'time'),
                          max_cells=20000, max_corr=0.8, n_pool=60, seed=0, verb=True):
    """
    n genes always selected so that the proliferation MLP R(P) can tell the proliferation states of the cells apart:
    ceil(n / 2) that best separate the labels of obs[label_key] (one indicator per label, taken in turn), the others
    that best predict the prior net rate (obs['proliferation_net_rate']; without it, the indicator of the most
    proliferative cell type: highest rate in anchors {cell type: rate}, else a name containing 'prolif', of the first
    of cell_type_keys in obs). All n go to the available target when only one applies. Net-rate markers alone are
    mostly cell-cycle genes, which follow the labels (e.g. quiescent / proliferative) less well. Genes are ranked by
    |correlation| of log1p(counts, / depth factor with use_depth) with each target within (sample, time) groups (no
    batch or time trend), then kept greedily when correlated below max_corr with those already kept (non-redundant).
    exclude: genes already forced. Mitochondrial and ribosomal genes are left out. Returns (genes, description) —
    ([], reason) when nothing applies.
    """
    import scipy.sparse
    if n <= 0:
        return [], 'n_proliferation_markers = 0'
    obs = adata.obs
    rng = np.random.default_rng(seed)
    # Targets: label indicators, and the net rate (or the most proliferative type)
    labels = []
    if label_key in obs:
        lab = obs[label_key].astype(str).to_numpy()
        cats = sorted(set(lab))
        if len(cats) >= 2:
            labels = [(f"{label_key}={c}", (lab == c).astype(float)) for c in (cats if len(cats) > 2 else cats[:1])]
    rate, what_rate = None, ''
    if 'proliferation_net_rate' in obs:
        r = obs['proliferation_net_rate'].to_numpy(dtype=float)
        if np.isfinite(r).all() and np.nanstd(r) > 0:
            rate, what_rate = r, 'net rate'
    if rate is None:
        key = next((k for k in cell_type_keys if k in obs), None)
        if key is not None:
            types = obs[key].astype(str).to_numpy()
            cats = sorted(set(types))
            rates = {c: anchors[c] for c in cats if anchors and c in anchors}
            best = max(rates, key=rates.get) if rates else next((c for c in cats if 'prolif' in c.lower()), None)
            if best is not None and len(cats) >= 2:
                rate, what_rate = (types == best).astype(float), f"{key}={best} (most proliferative)"
    if not labels and rate is None:
        return [], f"no obs['{label_key}'] labels, obs['proliferation_net_rate'] nor proliferative cell type"
    n_lab = 0 if not labels else (n if rate is None else (n + 1) // 2)
    names = np.asarray(adata.var_names).astype(str)
    up = np.char.upper(names)
    ok_gene = ~(np.char.startswith(up, 'MT-') | np.char.startswith(up, 'RPL') | np.char.startswith(up, 'RPS')
                | np.isin(names, list(exclude)))
    rows = np.arange(adata.n_obs) if adata.n_obs <= max_cells else np.sort(rng.choice(adata.n_obs, max_cells, replace=False))
    X = adata.X[rows]
    X = scipy.sparse.csr_matrix(X) if not scipy.sparse.issparse(X) else X.tocsr()
    X = X.astype(float)
    if use_depth and 'depth_factor' in obs:
        X = scipy.sparse.diags(1.0 / obs['depth_factor'].to_numpy(dtype=float)[rows]) @ X
    X = X.log1p()
    keys = [k for k in group_keys if k in obs]
    grp = (np.unique(np.array(['|'.join(map(str, v)) for v in zip(*[obs[k].to_numpy()[rows] for k in keys])]),
                     return_inverse=True)[1] if keys else np.zeros(len(rows), dtype=int))
    G = scipy.sparse.csr_matrix((np.ones(len(rows)), (grp, np.arange(len(rows)))))
    n_g = np.asarray(G.sum(axis=1)).ravel()
    S = np.asarray((G @ X).todense())
    var = np.asarray(X.multiply(X).sum(axis=0)).ravel() - (S ** 2 / n_g[:, None]).sum(axis=0)

    def correlations(target):
        # Within-group centring: covariance with the centred target, within-group variances
        t = target[rows]
        t_c = t - (G @ t / n_g)[grp]
        corr = np.asarray(X.T @ t_c).ravel() / np.sqrt(np.maximum(var, 1e-12) * max(float(t_c @ t_c), 1e-12))
        corr[~ok_gene] = 0.0
        return corr

    targets = [(nm, correlations(t)) for nm, t in labels] + ([(what_rate, correlations(rate))] if rate is not None else [])
    pools = [(nm, corr, [g for g in np.argsort(-np.abs(corr))[:n_pool] if abs(corr[g]) > 0]) for nm, corr in targets]
    union = sorted({g for _, _, pool in pools for g in pool})
    if not union:
        return [], 'no gene correlated with the proliferation labels or rate'
    col = {g: j for j, g in enumerate(union)}
    Z = np.asarray(X[:, union].todense())
    Z = Z - (G @ Z / n_g[:, None])[grp]
    Z /= np.maximum(Z.std(axis=0), 1e-12)
    kept, why = [], []

    def take(k):
        # Next non-redundant gene of the pool of target k (None if exhausted)
        nm, corr, pool = pools[k]
        while pool:
            g = pool.pop(0)
            if g not in kept and all(abs(float(Z[:, col[g]] @ Z[:, col[h]]) / len(rows)) < max_corr for h in kept):
                kept.append(g)
                why.append(f"{names[g]} ({nm} {corr[g]:+.2f})")
                return g
        return None

    n_lab_t = len(labels)
    k, misses = 0, 0
    while len(kept) < n_lab and misses < n_lab_t:  # label targets in turn
        misses = 0 if take(k % n_lab_t) is not None else misses + 1
        k += 1
    while len(kept) < n and any(pool for _, _, pool in pools):  # net rate, then any target left
        if take(len(pools) - 1) is None:
            for kk in range(len(pools)):
                if len(kept) < n:
                    take(kk)
    genes = [names[g] for g in kept]
    what = (f"labels of obs['{label_key}']" if labels else '') + (' + ' if labels and rate is not None else '') + \
        (f"prior {what_rate}" if rate is not None else '')
    if verb:
        print(f"[gene_selection] Proliferation markers ({what}, within (sample, time), non-redundant): " + ', '.join(why))
    return genes, what
