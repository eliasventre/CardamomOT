r"""
Option 3 — Mechanistic Palantir (``PLAN.md`` §5.1).

Pseudotime = shortest-path distance in a GRN-weighted kNN graph, refined by waypoints, as in
Palantir (Setty et al. 2019) but with the diffusion distances replaced by mechanistic transition
costs:

1. **graph space**: the burst modes :math:`k^{\mathrm{mix}}_c` of the cells (from the NB mixture),
   standardised and weighted by the separation of the two modes of each gene — no diffusion map;
2. **edge weights**: one of ``costs.EDGE_COSTS`` (mechanistic, plausibility, geometric), computed
   for the k-NN edges only;
3. **Dijkstra from the root population** → :math:`\tau^{(0)}`;
4. **waypoints** by max-min sampling, then the Palantir consensus
   :math:`\tau_c = \sum_w \hat w_c\,[d(c_0,w) + d(w,c)]`, :math:`\hat w_c \propto 1/d(c,w)`,
   iterated ``n_consensus`` times (waypoints inconsistent with their own root distance dropped);
5. :math:`\tau \leftarrow \tau/\max\tau` (dividing by the max, **not** ranking: the relative
   distances — hence the density read by the growth pass — are preserved, ``PLAN.md`` §6.2).

The graph is symmetrised by default (the mean of the two directed costs per edge): Palantir's
distances are symmetric, and the directed kNN graph may leave cells unreachable (counted in the
diagnostics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

from ..costs import (CostContext, EDGE_COSTS, edge_cost, source_sigmoid, step_of)
from .common import root_mask
from .features import FEATURE_SPACES, FEATURE_WEIGHTS, feature_matrix

EPS = 1e-16
EDGE_WEIGHT_CHOICES = EDGE_COSTS + ("auto",)


@dataclass
class PalantirResult:
    """Pseudotime and diagnostics of one run of the ordering."""

    tau: np.ndarray
    diagnostics: dict = field(default_factory=dict)


class PalantirLike:
    """Mechanistic Palantir: GRN-weighted kNN graph + Dijkstra + waypoint consensus."""

    def __init__(self, n_neighbors: int = 60, n_waypoints: int = 50,
                 n_waypoints_local: int = 5,
                 edge_weight: str = "geometric", alpha_geom: float = 0.5,
                 alpha_switch: float = 0.5, dtau: Optional[float] = None,
                 n_consensus: int = 3, symmetrise: bool = True,
                 feature_weight: str = "gene_weights", feature: str = "posterior",
                 n_pcs: int = 0, consensus: str = "min", geom_space: str = "expression",
                 waypoint_quantile: float = 0.95, seed: int = 0):
        if edge_weight not in EDGE_WEIGHT_CHOICES:
            raise ValueError(f"edge_weight must be one of {EDGE_WEIGHT_CHOICES}, "
                             f"got {edge_weight!r}")
        if feature_weight not in FEATURE_WEIGHTS:
            raise ValueError(f"feature_weight must be one of {FEATURE_WEIGHTS}, "
                             f"got {feature_weight!r}")
        if feature not in FEATURE_SPACES:
            raise ValueError(f"feature must be one of {FEATURE_SPACES}, got {feature!r}")
        if consensus not in ("min", "mean"):
            raise ValueError(f"consensus must be 'min' or 'mean', got {consensus!r}")
        self.feature = feature
        self.n_neighbors = int(n_neighbors)
        self.n_waypoints = int(n_waypoints)
        self.n_waypoints_local = int(n_waypoints_local)
        self.edge_weight = edge_weight
        self.alpha_geom = float(alpha_geom)
        self.alpha_switch = float(alpha_switch)
        self.dtau = dtau
        self.n_consensus = int(n_consensus)
        self.symmetrise = bool(symmetrise)
        self.feature_weight = feature_weight
        self.n_pcs = int(n_pcs)
        self.consensus = consensus
        self.geom_space = geom_space
        self.waypoint_quantile = float(waypoint_quantile)
        self.seed = seed

    @classmethod
    def from_model(cls, model) -> "PalantirLike":
        """Read the ``palantir_*`` parameters of the model."""
        return cls(
            n_neighbors=getattr(model, "palantir_n_neighbors", 60),
            n_waypoints=getattr(model, "palantir_n_waypoints", 50),
            n_waypoints_local=getattr(model, "palantir_n_waypoints_local", 5),
            edge_weight=getattr(model, "palantir_edge_weight", "geometric"),
            alpha_geom=getattr(model, "palantir_alpha_geom", 0.5),
            alpha_switch=getattr(model, "palantir_alpha_switch", 0.5),
            dtau=getattr(model, "palantir_dtau", None),
            n_consensus=getattr(model, "palantir_n_consensus", 3),
            symmetrise=getattr(model, "palantir_symmetrise", True),
            feature_weight=getattr(model, "palantir_feature_weight", "gene_weights"),
            feature=getattr(model, "palantir_feature", "posterior"),
            n_pcs=getattr(model, "palantir_n_pcs", 0),
            consensus=getattr(model, "palantir_consensus", "min"),
            geom_space=getattr(model, "palantir_geom_space", "expression"),
            seed=getattr(model, "seed", 0) or 0,
        )

    # ------------------------------------------------------------------------------------------
    def run(self, model, roots=None, verb: bool = False) -> PalantirResult:
        """
        Compute the pseudotime of every cell of ``model`` (needs ``model.modes``, ``model.counts``
        and, for the mechanistic weights, ``model.cell_prot``; see
        ``PseudotimeNetworkModel.initialise_cell_states``).

        ``roots``: boolean mask of the root cells (default: cells with ``tau == 0``).
        """
        ns = int(model.n_stimuli)
        N = int(model.modes.shape[0])
        modes = np.asarray(model.modes[:, ns:], dtype=float)
        counts = getattr(model, "counts", None)
        root_mask = self._roots(model, roots, N)
        n_roots = int(root_mask.sum())
        if not n_roots:
            raise ValueError("no root cell: pass roots=... (boolean mask)")

        ctx = CostContext.from_model(model, dtau=self._dtau(model), alpha=self.alpha_switch,
                                     alpha_geom=self.alpha_geom, geom_space=self.geom_space)
        weight_kind = self.resolved_edge_weight(model)
        F = self._features(model)
        k = int(min(self.n_neighbors, N - 1))
        if k < 1:
            raise ValueError("not enough cells for a kNN graph")

        neigh = NearestNeighbors(n_neighbors=k + 1).fit(F).kneighbors(F, return_distance=False)
        src = np.repeat(np.arange(N), k)
        tgt = neigh[:, 1:].ravel()

        weights, rows, cols = self._graph_edges(model, src, tgt, modes, counts, ctx, root_mask, N,
                                               weight_kind)
        graph = csr_matrix((np.maximum(weights, 1e-12), (rows, cols)), shape=(N, N))

        d_root = dijkstra(graph, directed=True, indices=np.flatnonzero(root_mask), min_only=True)
        n_unreached = int(np.sum(~np.isfinite(d_root)))
        d_root = self._fill_unreached(d_root, graph)

        waypoints = self._maxmin_waypoints(F, d_root, root_mask)
        tau, kept = self._consensus(graph, d_root, waypoints)
        tau = np.maximum(tau - np.min(tau[root_mask]), 0.0)
        tau[root_mask] = 0.0          # the root population is the origin, by definition
        spread = float(np.max(tau))
        tau = tau / spread if spread > 0 else tau

        diagnostics = dict(
            method="palantir", edge_weight=weight_kind, consensus=self.consensus,
            feature=self.feature, feature_weight=self.feature_weight, symmetrise=self.symmetrise,
            n_neighbors=k, n_roots=n_roots, n_unreached=n_unreached,
            n_waypoints=len(waypoints), n_waypoints_kept=len(kept),
            n_waypoints_local=self.n_waypoints_local,
            dtau=ctx.dtau, alpha_geom=self.alpha_geom,
            edge_cost_mean=float(np.mean(weights)), edge_cost_max=float(np.max(weights)),
            tau_max=spread,
        )
        if verb:
            print(f"  [palantir/{weight_kind}] {N} cells, k={k}, "
                  f"{len(waypoints)} waypoints ({len(kept)} kept), "
                  f"{n_unreached} unreachable")
        return PalantirResult(tau=tau, diagnostics=diagnostics)

    # ------------------------------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------------------------------
    def resolved_edge_weight(self, model) -> str:
        """
        ``'auto'``: ``'plausibility'`` while the network is not fitted (:math:`\\Theta=0`, where the
        mechanistic weight degenerates to the target mode alone and the bridge only adds noise),
        then ``'mechanistic'`` from the first fitted network on — it then uses :math:`\\Theta` and the
        protein state of the cells (``model.cell_prot``, the trajectory of the last E-step).
        """
        if self.edge_weight != "auto":
            return self.edge_weight
        inter = getattr(model, "inter", None)
        fitted = inter is not None and np.any(np.asarray(inter) != 0)
        return "mechanistic" if fitted else "plausibility"

    def _dtau(self, model):
        """Step of the graph: the bin width when the pseudotime axis has a duration (model time),
        else None (one protein relaxation time, the time unit not being identified yet)."""
        return step_of(model, self.dtau)

    @staticmethod
    def _roots(model, roots, N: int) -> np.ndarray:
        return root_mask(model, roots, N)

    def _features(self, model) -> np.ndarray:
        """
        Space of the kNN graph (``features.feature_matrix``): standardised per gene, weighted by
        the separation of the two modes of each gene, optionally PCA-reduced.
        """
        return feature_matrix(model, self.feature, self.feature_weight, self.n_pcs, self.seed)

    def _graph_edges(self, model, src, tgt, modes, counts, ctx, root_mask, N, weight_kind="mechanistic"):
        """
        (weights, rows, cols) of the graph edges.  With ``symmetrise``, the weight of an edge is the
        mean of the two directions and both directions are written (one kNN pair -> two edges).
        """
        stim = 1.0 - root_mask.astype(float)   # the stimulus is on after the root population

        def costs(s, t):
            kw = dict(rna_i=None if counts is None else counts[s],
                      rna_j=None if counts is None else counts[t],
                      stim=stim[t])
            if weight_kind == "mechanistic":
                if getattr(model, "cell_prot", None) is None:
                    raise ValueError(
                        "mechanistic edge weights need the protein state of the cells: call "
                        "PseudotimeNetworkModel.initialise_cell_states(data) first")
                kw["P0"] = np.asarray(model.cell_prot)[s][:, ctx.ns:]
            elif weight_kind == "plausibility":
                kw["sig_i"] = source_sigmoid(modes[s], ctx, stim=stim[s])
            return edge_cost(weight_kind, modes[s], modes[t], ctx, **kw)

        if not self.symmetrise:
            return costs(src, tgt), src, tgt

        key = np.minimum(src, tgt) * N + np.maximum(src, tgt)      # unordered pairs
        uniq, _ = np.unique(key, return_index=True)
        a, b = uniq // N, uniq % N
        w = 0.5 * (costs(a, b) + costs(b, a))
        rows = np.concatenate([a, b])
        cols = np.concatenate([b, a])
        return np.concatenate([w, w]), rows, cols

    @staticmethod
    def _fill_unreached(d: np.ndarray, graph, n_iter: int = 20) -> np.ndarray:
        """
        Cells without a path from the roots (a component disconnected from them): their distance is
        relaxed from their reached neighbours, ``d_j = min_i (d_i + w_ij)``, a few times.
        """
        d = d.copy()
        if np.all(np.isfinite(d)):
            return d
        src, tgt = graph.nonzero()
        w = np.asarray(graph[src, tgt]).ravel()
        for _ in range(int(n_iter)):
            inf = ~np.isfinite(d)
            if not inf.any():
                break
            cand = np.full_like(d, np.inf)
            with np.errstate(invalid="ignore"):
                np.minimum.at(cand, tgt, d[src] + w)
            d = np.minimum(d, cand)
        if not np.all(np.isfinite(d)):          # still isolated: after the reached cells
            finite = d[np.isfinite(d)]
            d[~np.isfinite(d)] = (finite.max() if finite.size else 1.0) * 1.01 + 1e-9
        return d

    def _maxmin_waypoints(self, F: np.ndarray, d_root: np.ndarray, root_mask: np.ndarray):
        """Max-min sampling in the feature space, starting from the central root cell."""
        N = F.shape[0]
        n_w = int(min(self.n_waypoints, N))
        if n_w < 1:
            return []
        root_idx = np.flatnonzero(root_mask)
        centre = F[root_idx].mean(axis=0)
        start = int(root_idx[np.argmin(np.linalg.norm(F[root_idx] - centre, axis=1))])
        chosen = [start]
        d_min = np.linalg.norm(F - F[start], axis=1)
        for _ in range(1, n_w):
            i = int(np.argmax(d_min))
            if d_min[i] <= EPS:
                break
            chosen.append(i)
            d_min = np.minimum(d_min, np.linalg.norm(F - F[i], axis=1))
        return chosen

    def _consensus(self, graph, d_root: np.ndarray, waypoints):
        """
        Waypoint consensus.  Each waypoint :math:`w` proposes the perspective estimate
        :math:`(d(c_0,w) + d(w,c))/2`, which is the length of a path that is forced through
        :math:`w`: it is >= :math:`d(c_0,c)` (triangle inequality), with equality when :math:`w`
        lies on a shortest path.  Two aggregations:

        * ``'min'`` (default): the tightest perspective, i.e. the best correction of the graph
          shortcuts (a shortcut makes :math:`d(c_0,c)` too small, a forced detour restores it).
          It is monotone and jitter-free, and the far waypoints are ignored automatically;
        * ``'mean'``: Palantir's weighted mean :math:`\\sum_w \\hat w_c [d(c_0,w)+d(w,c)]`,
          :math:`\\hat w_c \\propto 1/d(c,w)`, over the ``n_waypoints_local`` nearest waypoints
          (the global mean is biased for the cells close to the root when the waypoints are
          sparse), iterated while dropping the waypoints inconsistent with their root distance.
        """
        if not waypoints:
            return d_root.copy(), []
        wp = np.asarray(waypoints)
        Dw = np.stack([self._distances(graph, w) for w in waypoints])       # (W, N)
        if self.consensus == "min":
            return (0.5 * (d_root[wp][:, None] + Dw)).min(axis=0), list(waypoints)

        kept = list(range(len(waypoints)))
        tau = self._local_consensus(Dw, d_root, waypoints, kept)
        for _ in range(max(0, self.n_consensus - 1)):
            dev = np.abs(tau[np.asarray(waypoints)[kept]] - d_root[np.asarray(waypoints)[kept]])
            thr = float(np.quantile(dev, self.waypoint_quantile)) if len(kept) > 3 else np.inf
            new = [i for i, e in zip(kept, dev) if e <= thr]
            # never keep less than half of the waypoints (a noisy drop would bias the consensus)
            if len(new) < max(3, len(kept) // 2) or len(new) == len(kept):
                break
            kept = new
            tau = self._local_consensus(Dw, d_root, waypoints, kept)
        return tau, [waypoints[i] for i in kept]

    @staticmethod
    def _distances(graph, w) -> np.ndarray:
        """Shortest-path distances from cell w (unreachable cells filled with the maximum)."""
        d = dijkstra(graph, directed=True, indices=w)
        finite = np.isfinite(d)
        if not finite.all():
            d = np.where(finite, d, d[finite].max() if finite.any() else 0.0)
        return d

    def _local_consensus(self, Dw: np.ndarray, d_root: np.ndarray, waypoints, kept) -> np.ndarray:
        """Weighted mean of ``d(c0,w) + d(w,c)`` over the nearest waypoints of each cell."""
        wp = np.asarray(waypoints)[kept]
        dw = Dw[kept]                                              # (K, N)
        d0 = np.broadcast_to(d_root[wp][:, None], dw.shape)
        k = int(min(max(self.n_waypoints_local, 1), dw.shape[0]))
        idx = np.argpartition(dw, k - 1, axis=0)[:k]               # k nearest waypoints per cell
        dw_s = np.take_along_axis(dw, idx, axis=0)
        d0_s = np.take_along_axis(d0, idx, axis=0)
        floor = 1e-3 * max(float(np.median(d_root)), 1e-9)
        w = 1.0 / np.maximum(dw_s, floor)
        return (w * (d0_s + dw_s)).sum(axis=0) / np.maximum(w.sum(axis=0), EPS)
