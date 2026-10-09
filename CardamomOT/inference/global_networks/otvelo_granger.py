"""
OTVelo-Granger global network (Zhao et al., PLOS Comput Biol 2024, eq. 2.7-2.8): for each pair of
consecutive timepoints, an elastic net regresses the velocities of each target gene at t+1 on the
velocities at t of its regulators, mapped onto the cells of t+1 through the coupling; the
coefficients are summed over the intervals. Sparser than OTVelo-Corr, it removes indirect and
confounded correlations. To scale to thousands of genes, the regressors of each target are its
k_candidates strongest OTVelo-Corr regulators (stimuli always included).

Penalty calibrated for networks of any size: regressors (velocities averaged through the coupling, which
shrink as the coupling gets diffuse) and responses are scaled to unit RMS per interval (not centred: the
mean velocity is the stimulus effect), so that en_alpha acts as a correlation threshold; the stimuli are
penalised stim_weight times less (they play the role of the intercept); en_alpha = 'auto' keeps the
paper's value (1, calibrated on ~10 genes) up to 50 genes, then min(1, sqrt(50 / G)).
"""
import warnings
import numpy as np
import scipy.sparse
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet

from .otvelo_corr import otvelo_couplings, otvelo_velocities

# Paper defaults (Zhao et al. 2024): alpha = 0.5, eps = 0.01, lambda (en_alpha) = 1 for ~10 genes, r (l1_ratio) = 0.5
DEFAULTS = dict(n_cells=500, n_pcs=50, eps=1e-2, alpha=0.5, k_candidates=50, en_alpha='auto', l1_ratio=0.5,
                scale=True, stim_weight=10.0)


def auto_alpha(n_genes, g_ref=50):
    """Elastic-net penalty for n_genes: the paper's 1 up to g_ref genes, then sqrt(g_ref / n_genes)."""
    return float(min(1.0, np.sqrt(g_ref / max(n_genes, 1))))


def _rms(M):
    """Columns scaled to unit root mean square (not centred)."""
    s = np.sqrt((M ** 2).mean(axis=0))
    s[s < 1e-12] = 1.0
    return M / s


def _fit_target(j, cand, Xs, Ys, en_alpha, l1_ratio):
    """Coefficients of the candidate regulators of target j, summed over the intervals."""
    coef = np.zeros(len(cand))
    for X, Y in zip(Xs, Ys):
        model = ElasticNet(alpha=en_alpha, l1_ratio=l1_ratio, fit_intercept=False)
        with warnings.catch_warnings():
            # Constant (zero-velocity) targets give a spurious warning with a null duality gap
            warnings.simplefilter('ignore', ConvergenceWarning)
            model.fit(X[:, cand], Y[:, j])
        coef += model.coef_
    return coef


def otvelo_granger_network(X_log, times, stim=None, n_cells=500, n_pcs=50, eps=1e-2, alpha=0.5,
                           k_candidates=50, en_alpha='auto', l1_ratio=0.5, scale=True, stim_weight=10.0, rng=None):
    """OTVelo-Granger on log counts X_log (cells x genes); returns C ((n_stimuli + G) x (n_stimuli + G))."""
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
    return granger_network(Y, Ts, tu, stim, k_candidates, en_alpha, l1_ratio, scale, stim_weight)


def granger_network(Y, Ts, tu, stim=None, k_candidates=50, en_alpha='auto', l1_ratio=0.5, scale=True,
                    stim_weight=10.0):
    """
    Granger regression of the velocities through given couplings: Y[k] log counts of the cells at tu[k],
    Ts[k] coupling between the cells of tu[k] and tu[k + 1]. Returns C ((n_stimuli + G) x (n_stimuli + G)).
    """
    if en_alpha == 'auto':
        en_alpha = auto_alpha(Y[0].shape[1])
    ns = 0
    if stim is not None:
        stim = np.asarray(stim, dtype=float).reshape(len(tu), -1)
        ns = stim.shape[1]
        Y = [np.hstack([np.tile(stim[k], (len(y), 1)), y]) for k, y in enumerate(Y)]
    V = otvelo_velocities(Y, Ts, tu)
    sd = np.sqrt(np.var(np.vstack(V), axis=0))
    sd[sd == 0] = 1
    Vn = [v / sd for v in V]
    G = Vn[0].shape[1]

    # Regressors: velocities at t mapped onto the cells of t+1; responses: velocities at t+1
    Xs = [(Ts[k] / Ts[k].sum(axis=0, keepdims=True)).T @ Vn[k] for k in range(len(Ts))]
    Ys = [Vn[k + 1] for k in range(len(Ts))]
    if scale:
        Xs, Ys = [_rms(x) for x in Xs], [_rms(y) for y in Ys]
    # Stimuli penalised stim_weight times less (scaled column, coefficient rescaled below)
    Xs = [np.hstack([x[:, :ns] * stim_weight, x[:, ns:]]) for x in Xs]
    # Candidate regulators: strongest lagged correlations (OTVelo-Corr), plus the stimuli and the
    # target's own past (Granger: controls its autocorrelation; self-coefficient dropped after)
    Ccorr = sum(Vn[k].T @ Ts[k] @ Vn[k + 1] for k in range(len(Ts))) / len(Ts)
    np.fill_diagonal(Ccorr, 0)
    k = min(k_candidates, G - 1)
    cands = []
    for j in range(ns, G):
        top = np.argsort(-np.abs(Ccorr[:, j]))[:k]
        cands.append(np.unique(np.concatenate([np.arange(ns), [j], top])))
    coefs = Parallel(n_jobs=-1)(delayed(_fit_target)(j, c, Xs, Ys, en_alpha, l1_ratio)
                                for j, c in zip(range(ns, G), cands))
    C = np.zeros((G, G))
    for j, c, w in zip(range(ns, G), cands, coefs):
        C[c, j] = w
    C[:ns] *= stim_weight
    np.fill_diagonal(C, 0)
    return C


def build_network(adata, stim, seed=0, sample_key='dataset_id', n_cells=500, n_pcs=50, eps=1e-2,
                  alpha=0.5, k_candidates=50, en_alpha='auto', l1_ratio=0.5, scale=True, stim_weight=10.0):
    """
    Global network interface (see global_networks): OTVelo-Granger on log(x+1) raw counts, one
    network per sample (obs[sample_key]) averaged with cell weights. Returns C, stimuli first.
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
        C = C + m.sum() * otvelo_granger_network(X[m], times[m], stim=stim_s, n_cells=n_cells, n_pcs=n_pcs,
                                                 eps=eps, alpha=alpha, k_candidates=k_candidates,
                                                 en_alpha=en_alpha, l1_ratio=l1_ratio, scale=scale,
                                                 stim_weight=stim_weight, rng=seed)
        w_tot += m.sum()
    if w_tot == 0:
        raise ValueError("OTVelo-Granger needs at least two timepoints in a sample")
    return C / w_tot
