"""
Literature feasibility of the edges of a CardamomOT network (OmniPath).

An edge A -> B of CardamomOT means that protein A changes the transcription of gene B, possibly
through unobserved intermediates. It is feasible when the literature graph holds a path
A -> ... -> B of at most `depth` edges whose intermediates are not observed (not in `blocked`)
and whose last edge is transcriptional (CollecTRI TF -> target); the other edges may be
post-translational (OmniPath signalling) or transcriptional. Only feasibility is kept (no sign;
no length weight through unobserved intermediates, which make the effect look direct; paths
through observed genes are down-weighted, see prior). Genes the literature does not cover (no
TF -> gene edge for a target, absent for a regulator) get a neutral prior.
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse

CACHE_DIR = os.environ.get('CARDAMOMOT_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'cardamomot'))


def _split_complexes(df):
    """One row per (source member, target member): OmniPath complexes are 'A_B' gene symbols."""
    df = df.assign(source=df['source'].str.split('_'), target=df['target'].str.split('_'))
    return df.explode('source').explode('target')


# Post-translational ('ppi') and transcriptional ('tf') OmniPath resources; 'extended' adds less
# curated ones (fewer false negatives, for the prior where a missing edge is the costly error)
RESOURCES = dict(
    core=dict(ppi=[('OmniPath', {})], tf=[('CollecTRI', {})]),
    extended=dict(ppi=[('OmniPath', {}), ('PathwayExtra', {}), ('KinaseExtra', {}), ('LigRecExtra', {})],
                  tf=[('CollecTRI', {}), ('Dorothea', dict(dorothea_levels=['A', 'B', 'C', 'D'])), ('TFtarget', {})]),
)


def load_literature(species='human', resources='extended', cache_dir=CACHE_DIR):
    """
    Directed literature edges (source, target, kind) with upper-case gene symbols: kind 'ppi' for
    post-translational interactions, 'tf' for TF -> target (RESOURCES[resources]). Downloaded
    once and cached in cache_dir.
    """
    path = os.path.join(cache_dir, f'literature_{species}_{resources}.csv.gz')
    if os.path.exists(path):
        return pd.read_csv(path)
    import omnipath as op
    parts = []
    for kind, sources in RESOURCES[resources].items():
        for name, kw in sources:
            d = getattr(op.interactions, name).get(genesymbols=True, organism=species, **kw)
            if kind == 'ppi':
                d = d[d['is_directed'].astype(bool)]
            d = pd.DataFrame(dict(source=d['source_genesymbol'].astype(str).str.upper(),
                                  target=d['target_genesymbol'].astype(str).str.upper(), kind=kind))
            parts.append(_split_complexes(d))
    df = pd.concat(parts).drop_duplicates()
    df = df[df['source'] != df['target']].reset_index(drop=True)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(path, index=False)
    return df


class LiteratureGraph:
    """Sparse literature graph: A_all (all directed edges) and A_tf (transcriptional edges)."""

    def __init__(self, edges):
        nodes = pd.Index(pd.unique(pd.concat([edges['source'], edges['target']])))
        self.nodes, self.index = nodes, {g: i for i, g in enumerate(nodes)}
        n = len(nodes)
        src, tgt = nodes.get_indexer(edges['source']), nodes.get_indexer(edges['target'])
        tf = (edges['kind'] == 'tf').values
        mat = lambda m: scipy.sparse.csr_matrix((np.ones(m.sum(), np.float32), (src[m], tgt[m])), shape=(n, n))
        self.A_all = (mat(np.ones(len(edges), bool)) > 0).astype(np.float32)
        self.A_tf = (mat(tf) > 0).astype(np.float32)
        self.is_target = np.asarray(self.A_tf.sum(axis=0)).ravel() > 0

    def coverage(self, genes):
        """(regulator covered, target covered) per gene: in the graph / has a TF -> gene edge."""
        idx = np.array([self.index.get(str(g).upper(), -1) for g in genes])
        return idx >= 0, (idx >= 0) & self.is_target[np.maximum(idx, 0)]

    def observed_hops(self, genes, depth=3, observed=(), chunk=512):
        """
        K[i, j] = minimal number of observed intermediates (in `observed`, gene names) on a path
        genes[i] -> ... -> genes[j] of at most depth edges whose last edge is transcriptional;
        -1 without such a path.
        """
        n, L = len(self.nodes), max(depth - 1, 0)
        idx = np.array([self.index.get(str(g).upper(), -1) for g in genes])
        ok = np.flatnonzero(idx >= 0)
        obs = np.zeros(n, bool)
        obs[[self.index[str(g).upper()] for g in observed if str(g).upper() in self.index]] = True
        A_allT, A_tfT = self.A_all.T.tocsr(), self.A_tf.T.tocsr()
        K = np.full((len(genes), len(genes)), -1, int)
        for c0 in range(0, len(ok), chunk):
            rows = ok[c0:c0 + chunk]
            # reach[c]: nodes reached within the steps done with exactly c observed intermediates
            # (states dominated by fewer observed intermediates dropped); the source has c = 0
            reach = [np.zeros((len(rows), n), bool) for _ in range(L + 1)]
            reach[0][np.arange(len(rows)), idx[rows]] = True
            front = [r.copy() for r in reach]
            for _ in range(L):
                new = [np.zeros((len(rows), n), bool) for _ in range(L + 1)]
                for c in range(L + 1):
                    if front[c].any():
                        nxt = (A_allT @ front[c].T.astype(np.float32)).T > 0
                        new[c] |= nxt & ~obs
                        if c < L:
                            new[c + 1] |= nxt & obs
                seen = np.zeros((len(rows), n), bool)
                for c in range(L + 1):
                    seen |= reach[c]
                    front[c] = new[c] & ~seen
                    reach[c] |= front[c]
                    seen |= front[c]
            Kc = np.full((len(rows), len(ok)), -1, int)
            for c in range(L, -1, -1):
                hit = ((A_tfT @ reach[c].T.astype(np.float32)).T > 0)[:, idx[ok]]
                Kc[hit] = c
            K[rows[:, None], ok[None, :]] = Kc
        np.fill_diagonal(K, -1)
        return K

    def feasibility(self, genes, depth=3, blocked=()):
        """F[i, j] = True when a path genes[i] -> ... -> genes[j] of at most depth edges, last edge
        transcriptional, avoids the blocked genes as intermediates."""
        return self.observed_hops(genes, depth, blocked) == 0 if len(blocked) else self.observed_hops(genes, depth) >= 0

    def prior(self, genes, depth=3, observed=None):
        """
        Prior weights (G x G) of CardamomOT's ref_network: 1 for edges feasible through unobserved
        intermediates only (they collapse into a direct effect) and for pairs the literature does
        not cover; 1 / (k + 1) when every path goes through k >= 1 observed genes (the network may
        still need a direct edge, e.g. a hidden branch); 0 for covered pairs without any path.
        observed: genes treated as observed (default: genes).
        """
        reg, tgt = self.coverage(genes)
        K = self.observed_hops(genes, depth, genes if observed is None else observed)
        P = np.where(K >= 0, 1.0 / (np.maximum(K, 0) + 1), 0.0)
        P = np.where(reg[:, None] & tgt[None, :], P, 1.0)
        np.fill_diagonal(P, 1.0)
        return P


@lru_cache(maxsize=4)
def literature_graph(species='human', resources='extended', cache_dir=CACHE_DIR):
    return LiteratureGraph(load_literature(species, resources, cache_dir))
