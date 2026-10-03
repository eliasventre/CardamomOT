"""
Gene selection from the whole transcriptome (select_genes_and_split.py, change=1):

1. terminal genes, each chosen with a round robin over the groups (each timepoint and each cell
   type against the rest, Wilcoxon on log-normalised counts) so that all groups are covered:
   - at most n_query genes of Data/genes_queries.txt;
   - at least n_entropy genes among the entropy-selected genes of O. Gandrillon's workflow
     (gandrillon.py: KD and BUB-entropy changes between consecutive timepoints);
2. a coarse global network on the highly variable genes (+ terminals), built by a network method
   (global_networks: OTVelo-Corr built in, or a custom method of the project);
3. a directed Steiner tree rooted at the stimuli reaching the terminals, the remaining budget
   (num_max_genes - terminals) going to intermediate genes on the paths (a path may go through
   several terminals), built greedily by shortest paths on costs -log(|C| / max|C|).
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


def de_groups(adata, genes, cell_type_key=None):
    """
    Wilcoxon scores (scanpy rank_genes_groups, log-normalised counts) of the genes for each group
    against the rest: each timepoint, and each cell type if cell_type_key is given.
    Returns a DataFrame genes x groups ('time=<t>', '<key>=<c>') of scores and one of adjusted p-values.
    """
    import scanpy as sc
    sub = sc.AnnData(X=_dense(adata[:, genes].X).astype(np.float32), obs=adata.obs.copy())
    sub.var_names = list(genes)
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    scores, padj = {}, {}
    keys = [('time', 'time')] + ([(cell_type_key, cell_type_key)] if cell_type_key else [])
    for key, label in keys:
        groups = sub.obs[key].astype(str)
        if groups.nunique() < 2:
            continue
        sub.obs['_g'] = pd.Categorical(groups)
        sc.tl.rank_genes_groups(sub, '_g', method='wilcoxon', n_genes=len(genes))
        res = sub.uns['rank_genes_groups']
        for gr in sub.obs['_g'].cat.categories:
            s = pd.Series(res['scores'][gr], index=res['names'][gr]).reindex(genes)
            p = pd.Series(res['pvals_adj'][gr], index=res['names'][gr]).reindex(genes)
            scores[f'{label}={gr}'], padj[f'{label}={gr}'] = s.values, p.values
    return pd.DataFrame(scores, index=genes), pd.DataFrame(padj, index=genes)


def nb_init_separation(X, times, cell_types=None, n_cells=1024, seuil=0.01, seed=0, depth=None):
    """
    Separation of the extreme modes found by the initialization of the NB mixture (group_init,
    as in infer_mixture: one mode per time, and per cell type, with a common rate; the most
    diverse labelling kept), per gene (column of X), on at most n_cells cells per timepoint.
    A gene whose initialization gives no distinct basins is unlikely to get them from the CEM.
    depth: per-cell depth factors (NB(k, c / s_i)) or None.
    """
    rng = np.random.default_rng(seed)
    sub = np.concatenate([rng.choice(np.flatnonzero(times == t), min(n_cells, int(np.sum(times == t))), replace=False)
                          for t in np.unique(times)])
    Xs = np.rint(X[sub]).astype(int)
    groups = [times[sub], None if cell_types is None else np.asarray(cell_types)[sub]]

    # Serial: a few ms per gene, less than the transfer overhead of parallel tasks
    return _separation_block(Xs, groups, seuil, None if depth is None else np.asarray(depth, dtype=float)[sub])


def _separation_block(Xb, groups, seuil, s=None):
    out = np.zeros(Xb.shape[1])
    for j in range(Xb.shape[1]):
        x = Xb[:, j]
        if x.min() < x.max():
            res = group_init(x, groups, seuil, s)
            out[j] = 0.0 if res is None else res[3]
    return out


def round_robin(candidates, scores, n, taken=()):
    """
    Up to n candidates, each group in turn taking its best-scored candidate (higher Wilcoxon score =
    more expressed in the group) not yet taken. Returns [(gene, group)].
    """
    if n <= 0 or not len(candidates):
        return []
    taken = set(taken)
    ranks = {gr: [g for g in scores.loc[candidates, gr].sort_values(ascending=False).index] for gr in scores.columns}
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
    # Without groups (single time and cell type): candidates in their given order
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
                      closure_min=0.5):
    """
    Budgeted directed Steiner tree on edge probabilities W (greedy shortest-path heuristic, path
    cost = -log of the product of the edge probabilities, i.e. most probable paths).

    Edges i -> j with W >= w_min, the k_in most probable regulators of each gene and the k_stim
    most probable targets of each root; each edge costs -log(W * edge_prior): edge_prior < 1 makes
    every extra step cost something, so that among equally probable paths the shortest is taken.
    Ties in W (many edges at w = 1) are broken by |C| when C is given. A terminal left without
    incoming edge keeps its best one (probability floored at 1e-3: costly, used as a last resort). All required terminals are selected; the paths to them
    are added while the total number of selected genes stays within budget. Then the regulation
    of the selection is closed: the gene bringing the most probable regulation to the selected genes
    (sum of its W >= w_min towards them) is added, repeatedly, while it brings at least closure_min
    (missing regulators bias an inference restricted to the selection). Finally the optional
    terminals are tried in order (terminal + its path) until the budget is full.
    Returns the selected nodes (in order of addition), the parent and path cost of each connected
    node, the set of optional terminals added and the list of closure regulators.
    """
    n = W.shape[0]
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
    adj = [[] for _ in range(n)]
    for j in range(n):
        col = Wk[:, j]
        for i in np.argsort(-rank[:, j])[:k_in]:
            if col[i] > 0 and i != j:
                adj[i].append((j, -np.log(col[i] * edge_prior)))

    def dijkstra_from(sources):
        dist = np.full(n, np.inf)
        prev = np.full(n, -1)
        heap = [(0.0, s) for s in sources]
        for s in sources:
            dist[s] = 0.0
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                if d + w < dist[v]:
                    dist[v], prev[v] = d + w, u
                    heapq.heappush(heap, (d + w, v))
        return dist, prev

    selected = list(dict.fromkeys(required))
    chosen = set(selected)
    in_tree, parent, cost, extra = set(roots), {}, {}, set()

    def attach(t, dist, prev):
        # Path from the tree to t; returns its new nodes (not yet selected) or None
        path, u = [], t
        while u not in in_tree:
            path.append(u)
            u = prev[u]
        return path

    def commit(path, dist, prev):
        for v in reversed(path):
            in_tree.add(v)
            parent[v] = prev[v]
            cost[v] = float(dist[v])
            if v not in chosen:
                chosen.add(v)
                selected.append(v)

    # 1. Required terminals, cheapest first
    pending = set(selected)
    while pending:
        dist, prev = dijkstra_from(list(in_tree))
        done = False
        for t in sorted((t for t in pending if np.isfinite(dist[t])), key=lambda x: dist[x]):
            path = attach(t, dist, prev)
            if len(chosen) + sum(v not in chosen for v in path) <= budget:
                commit(path, dist, prev)
                pending -= in_tree
                done = True
                break
        if not done:
            break
    # 2. Regulatory closure of the selection
    Wg = np.where(W >= w_min, W, 0.0)
    Wg[list(roots), :] = 0.0
    score = Wg[:, sorted(chosen)].sum(axis=1)
    score[sorted(chosen)] = -np.inf
    score[list(roots)] = -np.inf
    closure = []
    while len(chosen) < budget:
        k = int(np.argmax(score))
        if score[k] < closure_min:
            break
        chosen.add(k)
        selected.append(k)
        closure.append(k)
        score += Wg[:, k]
        score[k] = -np.inf
    # 3. Optional terminals in order, while the budget allows
    for t in optional:
        if len(chosen) >= budget:
            break
        if t in chosen:
            continue
        dist, prev = dijkstra_from(list(in_tree))
        if not np.isfinite(dist[t]):
            continue
        path = attach(t, dist, prev)
        if len(chosen) + sum(v not in chosen for v in path) <= budget:
            commit(path, dist, prev)
            extra.add(t)
    return selected, parent, cost, extra, closure


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
                 n_cells_network=1000, sample_key='dataset_id', forced_genes=None, stimulus_targets=None,
                 species=None, use_depth_factor=False, seed=0, verb=True):
    """
    Full selection on raw counts (see module docstring). n_query + n_entropy must leave budget for
    the paths (< num_max_genes). The network is turned into edge probabilities against the same
    method run on cells permuted gene by gene (edge_probabilities). null_network: 'hybrid' (gene
    edges against a within-(sample, time) permutation: evidence beyond shared temporal trends,
    which CardamomOT explains by the stimulus and basals; stimulus edges against an all-cell
    permutation), 'within_time' or 'all_cells' for both. The Steiner tree follows the most probable
    paths; remaining budget is filled with further terminals of the round robins.
    literature: edge probabilities reweighted by OmniPath feasibility (paths of at most
    literature_depth edges ending with a TF -> target edge, literature_probabilities; species
    detected from the gene names if None), and literature prior of the selection returned.
    Variability floor, applied to every gene (queries included) before anything else: the largest
    BUB-entropy change between consecutive timepoints (Gandrillon MDE, bits) must reach
    min_entropy_change, and the NB-mixture initialization (nb_init_separation, n_cells_mixture
    cells per time, seuil_mixture) must separate its extreme modes by min_nb_separation.
    forced_genes: genes always selected (required terminals, exempt from the variability floor),
    e.g. those perturbed in KO_OV_Stim_simulate.txt. stimulus_targets: one gene list per stimulus
    (read_stimulus_targets): stimulus edges are restricted to its listed genes (no constraint if
    none is in the data).
    max_free_params (with the literature, for a hard prior): the gene budget is the largest one
    whose selection has at most this many free network parameters (free_parameters of its
    literature prior); num_max_genes is then ignored.
    Returns (selected gene names, report DataFrame, dict(C, C_null, W, genes) or None,
    literature prior of the selection (G x G, selection order) or None).
    """
    import scanpy as sc
    if n_query + n_entropy >= num_max_genes and not max_free_params:
        raise ValueError(f"n_query_genes + n_entropy_genes ({n_query} + {n_entropy}) must be lower than "
                         f"num_max_genes ({num_max_genes}) to leave budget for the paths from the stimulus")
    names = np.asarray(adata.var_names).astype(str)
    name_set = set(names)
    missing = [g for g in queries if g not in name_set]
    if missing and verb:
        print(f"[gene_selection] {len(missing)} query genes absent from the data: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    queries = [g for g in dict.fromkeys(queries) if g in name_set]
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
    # Gene scores of the whole universe (entropy, KD, variability) on at most n_cells_entropy cells per time
    rng0 = np.random.default_rng(seed)
    sub = np.sort(np.concatenate([rng0.choice(np.flatnonzero(times == t), min(n_cells_entropy, int(np.sum(times == t))),
                                              replace=False) for t in tu]))
    # Depth factors (estimate_cell_depth.py): scores on counts at the reference depth
    depth = (np.asarray(adata.obs['depth_factor'].values, dtype=float)
             if (use_depth_factor and 'depth_factor' in adata.obs) else None)
    X = _dense(Xa[sub][:, universe]).astype(np.float32)
    if depth is not None:
        X = X / depth[sub, None].astype(np.float32)
    t_sub = times[sub]
    cts = adata.obs[cell_type_key].astype(str).values if cell_type_key else None
    if verb:
        print(f"[gene_selection] Universe: {len(u_names)} genes (protein-coding, no mito/ribo, >= 3 cells); "
              f"gene scores on {len(sub)} cells (<= {n_cells_entropy} per time)"
              + ("; counts at the reference depth (obs['depth_factor'])" if depth is not None else ""))

    # Variability floor: Gandrillon entropy change, then distinct basins at the NB-mixture initialization
    kd = mde = None
    eligible = np.ones(len(u_names), bool)
    if len(tu) >= 2:
        kd, mde, _ = gandrillon_scores(X, t_sub, n_cells=n_cells_entropy, rng=seed)
        eligible &= mde.max(axis=0) >= min_entropy_change
    n_mde = int(eligible.sum())
    # Network universe: highly variable genes among those above the entropy floor
    hvg = np.zeros(len(u_names), bool)
    if len(tu) >= 2:
        tmp = sc.AnnData(X=np.log1p(X[:, eligible]))
        sc.pp.highly_variable_genes(tmp, n_top_genes=min(n_hvg, int(eligible.sum())), flavor='seurat')
        hvg[np.flatnonzero(eligible)[tmp.var['highly_variable'].values]] = True
    nb_check = len(tu) >= 2 or (cts is not None and len(np.unique(cts)) > 1)
    if nb_check:
        # NB basins checked here on the candidate terminals only; the genes the Steiner tree or the
        # closure would add are checked on demand (most network genes are never selected)
        ent0 = set(u_names[gandrillon_genes(kd, mde, top_n=n_top_entropy)]) if kd is not None else set()
        cand = np.flatnonzero(eligible & np.isin(u_names, list(ent0) + list(queries)))
        d = nb_init_separation(_dense(Xa[:, np.flatnonzero(universe)[cand]]), times, cts, n_cells_mixture,
                               seuil_mixture, seed, depth)
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
    keep_u = eligible
    universe[np.flatnonzero(universe)[~keep_u]] = False
    ad_u = adata[:, universe]
    u_names = u_names[keep_u]
    kd = kd[:, keep_u] if kd is not None else None
    mde = mde[:, keep_u] if mde is not None else None
    hvg = hvg[keep_u]

    # Entropy candidates (Gandrillon) and DE status of all candidates
    ent_candidates = []
    if len(tu) >= 2:
        ent_candidates = [u_names[i] for i in gandrillon_genes(kd, mde, top_n=n_top_entropy)]
        if verb:
            print(f"[gene_selection] Entropy genes (KD top {n_top_entropy} & MDE top {n_top_entropy} per transition): "
                  f"{len(ent_candidates)}")
    pool = list(dict.fromkeys(queries + ent_candidates))
    scores, padj = de_groups(ad_u, pool, cell_type_key) if pool else (pd.DataFrame(), pd.DataFrame())

    # Round robins over groups: the first n_query / n_entropy are required, the rest fill the budget
    q_all = round_robin(queries, scores, len(queries))
    e_all = round_robin([g for g in ent_candidates if g not in queries], scores, len(ent_candidates))
    q_req = q_all[:n_query]
    e_req = [x for x in e_all if x[0] not in {g for g, _ in q_req}][:n_entropy]
    required = list(dict.fromkeys(forced + [g for g, _ in q_req] + [g for g, _ in e_req]))
    group = dict(q_all + e_all)
    role = {**{g: 'query' for g, _ in q_all}, **{g: 'entropy' for g, _ in e_all}, **{g: 'perturbed' for g in forced}}
    q_rest = [g for g, _ in q_all if g not in required]
    e_rest = [g for g, _ in e_all if g not in required]
    optional = [g for pair in zip(q_rest, e_rest) for g in pair] + q_rest[len(e_rest):] + e_rest[len(q_rest):]
    if verb:
        print(f"[gene_selection] Required terminals: {len(q_req)}/{len(queries)} queries, {len(e_req)} entropy genes "
              f"(round robin over {scores.shape[1]} groups)")
        if len(e_req) < n_entropy:
            print(f"[gene_selection] Warning: only {len(e_req)} entropy genes available (n_entropy_genes = {n_entropy})")

    net, sel, rows, lg = None, list(required), {}, None
    if len(tu) < 2:
        if verb:
            print("[gene_selection] Single timepoint: no network, required terminals only")
    else:
        # Network universe: highly variable genes + all candidate terminals
        # Listed stimulus targets join the network universe, so that the constraint can apply
        listed = [g for col in (stimulus_targets or []) for g in col]
        keep = hvg | np.isin(u_names, pool) | np.isin(u_names, listed)
        net_genes = list(u_names[keep])
        # Network and its nulls on at most n_cells_network cells per (time, sample)
        smp = adata.obs[sample_key].values if sample_key in adata.obs else np.zeros(adata.n_obs)
        rng1 = np.random.default_rng(seed)
        cells = np.sort(np.concatenate([
            rng1.choice(i, min(n_cells_network, len(i)), replace=False)
            for i in (np.flatnonzero((times == t) & (smp == s)) for t in tu for s in np.unique(smp)) if len(i)]))
        ad_net = adata[cells, net_genes].copy()
        if depth is not None:  # network on counts at the reference depth
            ad_net.X = _dense(ad_net.X).astype(np.float32) / depth[cells, None].astype(np.float32)
        if verb:
            print(f"[gene_selection] Global network '{network_method}' on {len(net_genes)} genes and {len(cells)} cells "
                  f"(<= {n_cells_network} per time and sample), and its null network(s) ({null_network})")
        run = lambda ad_: build_global_network(network_method, ad_, stim, seed=seed, params=network_params,
                                               project_path=project_path)
        C = run(ad_net)
        C_time = run(permuted_counts(ad_net, True, seed)) if null_network in ('hybrid', 'within_time') else None
        C_all = run(permuted_counts(ad_net, False, seed)) if null_network in ('hybrid', 'all_cells') else None
        C_null = C_time if C_time is not None else C_all
        C_null_stim = C_all if C_all is not None else C_time
        ns = C.shape[0] - len(net_genes)
        W = edge_probabilities(C, C_null, ns, C_null_stim)
        node_names = [f'Stimulus{"" if ns == 1 else " " + str(i + 1)}' for i in range(ns)] + net_genes
        idx = {g: ns + i for i, g in enumerate(net_genes)}
        net = dict(C=C, C_null=C_null, C_null_stim=C_null_stim, W=W, genes=np.array(node_names))
        lg = _literature(names, species, literature, literature_resources, verb)
        if lg is not None:
            # Feasibility with any intermediate (the selection is not known yet)
            reg, tgt = lg.coverage(net_genes)
            F = lg.feasibility(net_genes, literature_depth)
            W, lr = literature_probabilities(net['W'], ns, F, reg[:, None] & tgt[None, :], literature_weight)
            net.update(W=W, W_data=net['W'], F=F)
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
        # Possible targets of the stimuli: other stimulus edges removed from the graph
        if stimulus_targets:
            from ..config import stimulus_target_mask
            mask = stimulus_target_mask(stimulus_targets, net_genes, ns)
            if mask is not None:
                Wm[:ns, ns:] *= mask
                Cm[:ns, ns:] *= mask

        def steiner(B):
            while True:
                res = steiner_selection(
                    Wm, list(range(ns)), [idx[g] for g in required], [idx[g] for g in optional], B,
                    k_in=k_in, k_stim=k_stim, w_min=min_edge_prob, edge_prior=edge_prior, C=Cm,
                    closure_min=closure_min)
                new = [v for v in res[0] if v >= ns and v not in checked] if nb_check else []
                if not new:
                    return res
                d = nb_init_separation(_dense(adata[:, [node_names[v] for v in new]].X), times, cts, n_cells_mixture,
                                       seuil_mixture, seed, depth)
                checked.update(new)
                bad = [v for v, dv in zip(new, d) if dv < min_nb_separation]
                if not bad:
                    return res
                rejected.extend(bad)
                Wm[bad, :], Wm[:, bad], Cm[bad, :], Cm[:, bad] = 0, 0, 0, 0
        if max_free_params and lg is None and verb:
            print(f"[gene_selection] Warning: no literature, parameter budget ignored (budget {num_max_genes} genes)")
        if max_free_params and lg is not None:
            num_max_genes, res = _parameter_budget(steiner, lg, node_names, ns, max_free_params, len(required),
                                                   len(net_genes), literature_depth, verb)
        else:
            res = steiner(num_max_genes)
        chosen, parent, cost, extra, closure = res
        if verb and nb_check:
            print(f"[gene_selection] Variability floor of the genes added by the tree or the closure: "
                  f"{len(checked) - len({idx[g] for g in pool if g in idx})} checked, {len(rejected)} rejected")
        sel = [node_names[v] for v in chosen]
        for v, u in parent.items():
            rows[node_names[v]] = dict(parent=node_names[u], cost=cost[v], edge_prob=float(W[u, v]))
        chosen_set = list(chosen)
        for v in closure:
            top = [u for u in np.argsort(-W[v, chosen_set]) if W[v, chosen_set[u]] >= min_edge_prob][:3]
            rows[node_names[v]] = dict(regulates=', '.join(node_names[chosen_set[u]] for u in top))
            role[node_names[v]] = 'regulator'
        terminals_idx = [idx[g] for g in required]
        if verb:
            path_nodes = [v for v in chosen if v not in set(closure)]
            print(f"[gene_selection] Regulation of the selection coming from inside it (w >= {min_edge_prob}): "
                  f"{closure_fraction(W, terminals_idx, list(range(ns)), min_edge_prob) * 100:.0f}% for the required "
                  f"terminals alone, {closure_fraction(W, chosen_set, list(range(ns)), min_edge_prob) * 100:.0f}% for the "
                  f"final selection ({len(closure)} closure regulators)")
    if lg is None and len(tu) < 2:
        lg = _literature(names, species, literature, literature_resources, verb)
    prior = None
    n_stim = 0 if stim is None else np.asarray(stim).reshape(len(tu), -1).shape[1]
    if lg is not None:
        # Prior of the final selection: intermediates outside it (observed ones are network chains)
        prior = lg.prior(sel, literature_depth)
        if verb:
            off = ~np.eye(len(sel), dtype=bool)
            print(f"[gene_selection] Literature prior of the selection: {np.mean(prior[off] > 0) * 100:.0f}% of the "
                  f"edges allowed ({int((prior[off] > 0).sum())} free interactions), {np.mean(prior[off] == 1) * 100:.0f}% "
                  f"with weight 1 (feasible through unobserved intermediates, or not covered); "
                  f"{free_parameters(prior, n_stim)} free network parameters")
    report = pd.DataFrame([dict(gene=g, role=role.get(g, 'steiner'), group=group.get(g, ''),
                                required=g in required, **rows.get(g, {})) for g in sel],
                          columns=['gene', 'role', 'group', 'required', 'parent', 'cost', 'edge_prob', 'regulates'])
    report['connected'] = report['parent'].notna()
    if len(pool):
        sig = (padj < 0.05) & (scores > 0)
        report['DE_groups'] = [', '.join(sig.columns[sig.loc[g].values]) if g in sig.index else '' for g in report['gene']]
    if verb:
        r = report['role']
        print(f"[gene_selection] Selected {len(sel)} genes (budget {num_max_genes}): {(r == 'query').sum()} queries, "
              f"{(r == 'entropy').sum()} entropy genes, {(r == 'steiner').sum()} Steiner genes, "
              f"{(r == 'regulator').sum()} closure regulators "
              f"({int((~report['required'] & r.isin(['query', 'entropy'])).sum())} extra terminals); "
              f"{int((~report['connected'] & report['required']).sum())} required terminals not connected")
    return sel, report, net, prior


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
