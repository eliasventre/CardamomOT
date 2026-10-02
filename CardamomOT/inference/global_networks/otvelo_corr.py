"""
OTVelo-Corr global network (Zhao et al., PLOS Comput Biol 2024; reimplemented from
github.com/sandstede-lab/OT-Velocity, identical to it up to machine precision): entropic fused
Gromov-Wasserstein couplings between consecutive timepoints on log(x+1) counts, gene velocities by
finite differences through the couplings, and time-lagged correlation of the velocities
C = sum_t V_t^T T_t V_{t+1} / (T-1) (signed, directed: rows regulate columns). Stimuli enter as
source-only pseudo-genes whose expression is the stimulus schedule.
"""
import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph

DEFAULTS = dict(n_cells=500, n_pcs=50, eps=1e-2, alpha=0.5)


def _graph_distances(X, n_neighbors):
    """Geodesic distances on the kNN graph (as OTVelo/SCOT), infinite ones capped at the max."""
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance', metric='euclidean', include_self=True)
    D = dijkstra(graph, directed=False)
    finite = D[np.isfinite(D)]
    D[~np.isfinite(D)] = finite.max() if finite.size else 0
    return D


def otvelo_couplings(X_list, eps=1e-2, alpha=0.5):
    """Entropic fused Gromov-Wasserstein coupling between consecutive timepoints (X_list: cells x dims)."""
    from ot.gromov import entropic_fused_gromov_wasserstein
    import ot
    Ts = []
    for X1, X2 in zip(X_list[:-1], X_list[1:]):
        M = ot.dist(X1, X2)
        M = M / max(M.max(), 1e-12)
        k = max(1, min(int(0.2 * len(X1)), int(0.2 * len(X2)), 50))
        D1, D2 = _graph_distances(X1, k), _graph_distances(X2, k)
        a = alpha
        if D1.max() == 0 or D2.max() == 0:
            a = 0
        else:
            D1, D2 = D1 / D1.max(), D2 / D2.max()
        T = entropic_fused_gromov_wasserstein(M, D1, D2, epsilon=eps, alpha=a)
        if not np.all(np.isfinite(T)) or abs(T.sum() - 1) > 1e-3:
            T = np.ones(T.shape) / T.size
        Ts.append(np.asarray(T))
    return Ts


def otvelo_velocities(Y_list, Ts, times):
    """
    Velocities (cells x genes) at each timepoint: forward difference at the first, backward at the
    last, Δt-weighted average of both in between, mapping cells through the normalised couplings.
    """
    K = len(Y_list)
    dt = np.diff(times)
    fwd = [Ts[k] / Ts[k].sum(axis=1, keepdims=True) @ Y_list[k + 1] - Y_list[k] for k in range(K - 1)]
    bwd = [Y_list[k + 1] - Ts[k].T / Ts[k].sum(axis=0)[:, None] @ Y_list[k] for k in range(K - 1)]
    V = [fwd[0] / dt[0]]
    for k in range(1, K - 1):
        w_f, w_b = dt[k - 1] / (dt[k] + dt[k - 1]), dt[k] / (dt[k] + dt[k - 1])
        V.append(w_f * fwd[k] / dt[k] + w_b * bwd[k - 1] / dt[k - 1])
    V.append(bwd[-1] / dt[-1])
    return V


def otvelo_corr(V, Ts, n_sources_only=0):
    """
    Time-lagged correlation of the velocities through the couplings, velocities scaled to unit
    standard deviation per gene; the first n_sources_only columns (stimuli) are never targets.
    """
    allV = np.vstack(V)
    sd = np.sqrt(np.var(allV, axis=0))
    sd[sd == 0] = 1
    Vn = [v / sd for v in V]
    C = sum(Vn[k].T @ Ts[k] @ Vn[k + 1] for k in range(len(Ts))) / len(Ts)
    np.fill_diagonal(C, 0)
    C[:, :n_sources_only] = 0
    return C


def otvelo_network(X_log, times, stim=None, n_cells=500, n_pcs=50, eps=1e-2, alpha=0.5, rng=None):
    """
    OTVelo-Corr network on log counts X_log (cells x genes) with timepoints `times`.
    stim : (n_times, n_stimuli) schedule values per sorted timepoint, prepended as pseudo-genes.
    Couplings use the first n_pcs principal components if fewer than the genes (0 = no PCA).
    Returns C ((n_stimuli + G) x (n_stimuli + G)).
    """
    rng = np.random.default_rng(rng)
    tu = np.sort(np.unique(times))
    idx = [np.flatnonzero(times == t) for t in tu]
    idx = [rng.choice(i, n_cells, replace=False) if len(i) > n_cells else i for i in idx]
    Y = [X_log[i] for i in idx]
    Z = Y
    if n_pcs and n_pcs < X_log.shape[1]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_pcs, random_state=0).fit(np.vstack(Y))
        Z = [pca.transform(y) for y in Y]
    Ts = otvelo_couplings(Z, eps=eps, alpha=alpha)
    ns = 0
    if stim is not None:
        stim = np.asarray(stim, dtype=float).reshape(len(tu), -1)
        ns = stim.shape[1]
        Y = [np.hstack([np.tile(stim[k], (len(y), 1)), y]) for k, y in enumerate(Y)]
    V = otvelo_velocities(Y, Ts, tu)
    return otvelo_corr(V, Ts, n_sources_only=ns)


def build_network(adata, stim, seed=0, sample_key='dataset_id', n_cells=500, n_pcs=50, eps=1e-2, alpha=0.5):
    """
    Global network interface (see global_networks.load_network_method): OTVelo-Corr on log(x+1)
    raw counts of adata (cells x genes, obs['time']), one network per sample (obs[sample_key])
    averaged with cell weights. stim: (n_times, n_stimuli) schedule per sorted timepoint.
    Returns C ((n_stimuli + G) x (n_stimuli + G)), stimuli first, rows regulate columns.
    """
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.asarray(adata.X)
    X = np.log1p(X.astype(np.float32))
    times = adata.obs['time'].values.astype(float)
    tu = np.sort(np.unique(times))
    samples = adata.obs[sample_key].values if sample_key in adata.obs else np.zeros(adata.n_obs)
    C, w_tot = 0, 0
    for s in np.unique(samples):
        m = samples == s
        ts = np.sort(np.unique(times[m]))
        if len(ts) < 2:
            continue
        stim_s = None if stim is None else np.asarray(stim, dtype=float).reshape(len(tu), -1)[np.searchsorted(tu, ts)]
        C = C + m.sum() * otvelo_network(X[m], times[m], stim=stim_s, n_cells=n_cells, n_pcs=n_pcs,
                                         eps=eps, alpha=alpha, rng=seed)
        w_tot += m.sum()
    if w_tot == 0:
        raise ValueError("OTVelo-Corr needs at least two timepoints in a sample")
    return C / w_tot
