"""
2-D embeddings of the report (embedding_method_visualization: 'umap', 'pca' or 'phate') and the embedding of every cell of
Data/data.h5ad, computed once and cached in cardamomOT/embedding_data.npz with its PCA, so that the velocity
fields of every method are drawn on the same embedding.

mRNA shown as log1p(counts / obs['depth_factor']) with cell_depth_for_representation (if the factor exists),
else log1p(counts).
"""
import os
import numpy as np
import scipy.sparse
import anndata as ad
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

EMBEDDING_NAMES = {'umap': 'UMAP', 'pca': 'PCA', 'phate': 'PHATE'}


def embedding_name(method):
    return EMBEDDING_NAMES.get(str(method).lower(), str(method).upper())


class Embedder:
    """
    2-D embedding fitted on X (fit), then other points placed on it (transform): native for UMAP and PCA, by
    a kNN extension for PHATE (inverse-distance mean of the coordinates of the k nearest fitted points).
    """

    def __init__(self, method='umap', seed=42, k=10):
        self.method, self.seed, self.k = str(method).lower(), seed, k
        if self.method not in EMBEDDING_NAMES:
            raise ValueError(f"embedding_method_visualization must be one of {sorted(EMBEDDING_NAMES)} (got {method!r})")

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if self.method == 'umap':
            from umap import UMAP
            self.model = UMAP(random_state=self.seed, min_dist=0.7).fit(X)
            self.embedding_ = self.model.embedding_
        elif self.method == 'pca':
            from sklearn.decomposition import PCA
            self.model = PCA(n_components=2, random_state=self.seed).fit(X)
            self.embedding_ = self.model.transform(X)
        else:
            self.embedding_ = _phate(X, self.seed)
            self.X_fit, self.nn = X, NearestNeighbors(n_neighbors=min(self.k, len(X))).fit(X)
        return self

    def fit_transform(self, X):
        return self.fit(X).embedding_

    def transform(self, Y):
        Y = np.asarray(Y, dtype=float)
        if self.method in ('umap', 'pca'):
            return self.model.transform(Y)
        dist, ind = self.nn.kneighbors(Y)
        w = 1.0 / np.maximum(dist, 1e-12)
        return (w[..., None] * self.embedding_[ind]).sum(axis=1) / w.sum(axis=1, keepdims=True)


def _phate(X, seed):
    try:
        import phate
    except ImportError as e:
        raise ImportError("embedding_method_visualization = 'phate' needs the phate package (pip install phate)") from e
    import warnings
    with warnings.catch_warnings():  # duplicated states (trajectories) only trigger numerical warnings
        warnings.simplefilter('ignore', RuntimeWarning)
        return phate.PHATE(n_components=2, random_state=seed, verbose=0, n_jobs=-1).fit_transform(X)


def log_normalised(A, use_depth=True):
    """log1p of the counts / obs['depth_factor'] if use_depth (use_depth_factor) and computed, else of the counts
    normalised per cell (normalize_total) (CSR, float32)."""
    X = scipy.sparse.csr_matrix(A.X, dtype=np.float32)
    if use_depth and 'depth_factor' in A.obs:
        X = scipy.sparse.diags(1.0 / A.obs['depth_factor'].astype(float).values).astype(np.float32) @ X
    else:
        tot = np.asarray(X.sum(axis=1)).ravel()
        X = scipy.sparse.diags((np.median(tot) / np.maximum(tot, 1e-12)).astype(np.float32)) @ X
    X = X.tocsr()
    X.data = np.log1p(X.data)
    return X


def log_view(A, depth=True):
    """log1p(counts / depth factor) if depth and obs['depth_factor'] exists, else log1p(counts) (CSR, float32)."""
    X = scipy.sparse.csr_matrix(A.X, dtype=np.float32)
    if depth and 'depth_factor' in A.obs:
        X = (scipy.sparse.diags(1.0 / A.obs['depth_factor'].astype(float).values).astype(np.float32) @ X).tocsr()
    X.data = np.log1p(X.data)
    return X


def data_embedding(p, n_hvg=2000, n_pcs=50, n_neighbors=15, seed=0, depth=True, method='umap'):
    """
    Embedding of every cell of Data/data.h5ad: log1p(counts / depth factor if depth and computed, else counts),
    n_hvg highly variable genes (seurat), PCA, then UMAP (kNN graph), the first 2 principal components, or
    PHATE (on the PCA). Cached in cardamomOT/embedding_data.npz (cell names, embedding 'emb', PCA, HVG with
    their mean and PCA loadings), reused while the cells, depth and method are the same.
    Returns (dict of arrays, obs of the cells).
    """
    from ..config import ensure_raw_counts, harmonize_obs
    method = str(method).lower()
    data_path = os.path.join(p, 'Data', 'data.h5ad')
    cache = os.path.join(p, 'cardamomOT', 'embedding_data.npz')
    obs = ad.read_h5ad(data_path, backed='r').obs.copy()
    use_depth = bool(depth and 'depth_factor' in obs)
    if os.path.exists(cache):
        E = dict(np.load(cache, allow_pickle=True))
        if (len(E['obs_names']) == len(obs) and (E['obs_names'] == obs.index.values.astype(str)).all()
                and bool(E['depth']) == use_depth and str(E['method']) == method):
            tmp = ad.AnnData(obs=obs)
            harmonize_obs(tmp)
            return E, tmp.obs
    A = ensure_raw_counts(ad.read_h5ad(data_path), data_path)
    harmonize_obs(A)
    print(f"[embedding] {embedding_name(method)} of the {A.n_obs} cells of Data/data.h5ad (cached in {cache})")
    A.X = log_view(A, use_depth)
    batch = 'dataset_id' if 'dataset_id' in A.obs and A.obs['dataset_id'].nunique() > 1 else None
    sc.pp.highly_variable_genes(A, n_top_genes=min(n_hvg, A.n_vars), flavor='seurat', batch_key=batch)
    A = A[:, A.var['highly_variable'].values].copy()
    sc.pp.pca(A, n_comps=min(n_pcs, A.n_vars - 1, A.n_obs - 1), random_state=seed)
    if method == 'umap':
        sc.pp.neighbors(A, n_neighbors=n_neighbors, random_state=seed)
        sc.tl.umap(A, random_state=seed)
        emb = A.obsm['X_umap']
    elif method == 'pca':
        emb = A.obsm['X_pca'][:, :2]
    else:
        emb = _phate(A.obsm['X_pca'], seed)
    E = dict(obs_names=A.obs_names.values.astype(str), emb=np.asarray(emb), pca=A.obsm['X_pca'],
             hvg=A.var_names.values.astype(str), hvg_mean=np.asarray(A.X.mean(axis=0)).ravel(),
             pcs=A.varm['PCs'], depth=use_depth, method=method)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **E)
    old = os.path.join(p, 'cardamomOT', 'umap_data.npz')  # cache of an earlier version
    if os.path.exists(old):
        os.remove(old)
    return E, A.obs
