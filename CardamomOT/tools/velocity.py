"""
Cell velocities from transport couplings, shared by run_classical_OT.py and the report.

- couplings: sparse blocks (source cell, target cell, mass) between consecutive timepoints, rows of the
  dataset they were computed on (CardamomOT: data_<split>.h5ad, saved by infer_network_structure.py);
- barycentric velocities: (mean of the descendants − x) / Δt, or (x − mean of the ancestors) / Δt for the
  cells without descendants (last time, cells left out by the trajectories); displacements without / Δt (rate=False);
- Gaussian-kernel regression of the velocities for the cells without any (not in the training set, or
  never reached by the transport), within their timepoint;
- kNN-transition projection of velocities onto a 2-D embedding (scVelo style).
"""
import numpy as np
import scipy.sparse
from sklearn.neighbors import NearestNeighbors


def save_couplings(path, blocks):
    """Write the coupling blocks ({src, tgt, w, sample, t_from, t_to}) as one sparse table."""
    n = [len(b['w']) for b in blocks]
    np.savez_compressed(path,
                        src=np.concatenate([b['src'] for b in blocks]).astype(np.int64),
                        tgt=np.concatenate([b['tgt'] for b in blocks]).astype(np.int64),
                        w=np.concatenate([b['w'] for b in blocks]).astype(np.float32),
                        sample=np.repeat([b['sample'] for b in blocks], n).astype(np.int32),
                        t_from=np.repeat([b['t_from'] for b in blocks], n).astype(float),
                        t_to=np.repeat([b['t_to'] for b in blocks], n).astype(float))


def coupling_blocks(path_or_dict, n_cells):
    """{(sample, t_from, t_to): (n_cells x n_cells) CSR coupling}, duplicated entries summed."""
    d = dict(np.load(path_or_dict)) if isinstance(path_or_dict, str) else path_or_dict
    out = {}
    keys = np.stack([d['sample'], d['t_from'], d['t_to']], axis=1)
    for key in np.unique(keys, axis=0):
        m = (keys == key).all(axis=1)
        P = scipy.sparse.coo_matrix((d['w'][m].astype(float), (d['src'][m], d['tgt'][m])), shape=(n_cells, n_cells))
        out[(int(key[0]), float(key[1]), float(key[2]))] = P.tocsr()
    return out


def barycentric_velocity(X, blocks, rate=True):
    """
    Velocity of each cell (rows of X) from the couplings: descendants when the cell is a source, else
    ancestors when it is a target; NaN without either; the displacement (not divided by Δt) if not rate.
    Returns (V, origin) with origin 'descendants', 'ancestors' or ''.
    """
    V = np.full(X.shape, np.nan)
    origin = np.full(len(X), '', dtype=object)
    for direction in ('descendants', 'ancestors'):
        for (_, t0, t1), P in blocks.items():
            dt = (t1 - t0) if rate else 1.0
            M = P if direction == 'descendants' else P.T.tocsr()
            mass = np.asarray(M.sum(axis=1)).ravel()
            rows = np.flatnonzero((mass > 0) & (origin == ''))
            if not len(rows):
                continue
            mean = (M[rows] @ X) / mass[rows, None]
            V[rows] = (mean - X[rows]) / dt if direction == 'descendants' else (X[rows] - mean) / dt
            origin[rows] = direction
    return V, origin


def kernel_velocity(Z, V, known, groups=None, k=30):
    """
    Velocities of the cells with known = False: Gaussian-kernel average over their k nearest known cells
    in Z (bandwidth: distance to the k-th neighbour of each query), within their group (e.g. timepoint)
    when it has known cells.
    """
    V = V.copy()
    groups = np.zeros(len(Z)) if groups is None else np.asarray(groups)
    for g in np.unique(groups):
        q = np.flatnonzero((groups == g) & ~known)
        ref = np.flatnonzero((groups == g) & known)
        if not len(ref):
            ref = np.flatnonzero(known)
        if not len(q) or not len(ref):
            continue
        kk = min(k, len(ref))
        dist, ind = NearestNeighbors(n_neighbors=kk).fit(Z[ref]).kneighbors(Z[q])
        h = np.maximum(dist[:, -1:], 1e-12)
        w = np.exp(-0.5 * (dist / h) ** 2)
        V[q] = (w[..., None] * V[ref][ind]).sum(axis=1) / w.sum(axis=1, keepdims=True)
    return V


def knn_velocity_embedding(X, V, E, k=30, scale=10.0, chunk=4000):
    """
    scVelo-style projection: transition probabilities to the kNN of each cell from the cosine between its
    velocity and the displacements to its neighbours, then expected unit displacement in the embedding
    minus the uniform-transition one (computed by chunks of cells).
    """
    k = min(k, len(X) - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    Vemb = np.zeros((len(X), E.shape[1]))
    for a in range(0, len(X), chunk):
        sl = slice(a, a + chunk)
        dX = X[idx[sl]] - X[sl][:, None, :]
        nv = np.linalg.norm(V[sl], axis=1)
        cos = (dX * V[sl][:, None, :]).sum(-1) / (np.linalg.norm(dX, axis=-1) * nv[:, None] + 1e-12)
        P = np.exp(scale * cos)
        P /= P.sum(axis=1, keepdims=True)
        dE = E[idx[sl]] - E[sl][:, None, :]
        dE /= np.linalg.norm(dE, axis=-1, keepdims=True) + 1e-12
        out = (P[..., None] * dE).sum(axis=1) - dE.mean(axis=1)
        out[~(nv > 0) | ~np.isfinite(nv)] = 0.0
        Vemb[sl] = out
    return Vemb
