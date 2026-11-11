r"""
Option 2 — Path-dependent TSP (``PLAN.md`` §5.3).

The cells are ordered by a **tour** minimised with a windowed 2-opt, where the cost of a step
depends on the *state reached at the previous step* — not only on the two cells:

.. math::

    \mathrm{dist}(c_k, c_{k+1}) = \mathrm{loss}\big(k_{on}(P_{c_k\to c_{k+1}};\Theta),
    k^{\mathrm{mix}}_{c_{k+1}}\big)
    + \alpha\,\mathrm{geom}(c_k, c_{k+1}),

with :math:`P_{c_k}` propagated **sequentially along the tour** from the root population
(``costs.bridge_end``, the same flow as the E-step).  Because the cost is path-dependent, reversing
a segment :math:`(c_{a+1}, \dots, c_b)` changes every edge inside it and requires re-propagating the
proteins from :math:`c_a` — which is why the moves are windowed (``window``).

1. **Initial tour**: the cells sorted by the τ of the previous iteration (warm start).  When there
   is no usable τ yet, another option produces it (``init='palantir'`` or ``'partial_ot'``), because
   a TSP started from scratch has no chance of finding the right direction.
2. **2-opt (first improvement)**: draw ``n_moves`` random windows, reverse
   :math:`(c_{a+1},\dots,c_b)` whenever it shortens :math:`\sum_k \mathrm{dist}(c_k,c_{k+1})`,
   re-propagating the proteins over the window only.
3. **τ = cumulative path length** normalised by its maximum (``use_path_length``), which is the
   pseudo-metric the bins and the growth pass need; ``use_path_length=False`` falls back to the tour
   rank, the comparator that shows what the path length buys.

Known limitations (documented in ``plan.tex`` and ``PLAN.md`` §5.3): the cost is asymmetric and
non-metric, so the 2-opt has no convergence guarantee, and a **single tour cannot represent a
branching trajectory** — the two arms of BN8/CH20 are interleaved along one path.  Option 2 is
therefore expected to be the weakest of the three on trees, and its interest is (a) as an upper
bound on what a pure ordering can do once the direction is known, and (b) as a refinement of the
other two options: it always starts from their τ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..costs import (CostContext, bridge_end, geometric_cost, network_cost, step_of)
from .common import root_mask as _root_mask

EPS = 1e-16
INITS = ("palantir", "partial_ot", "previous")
TSP_EDGE_COSTS = ("mechanistic", "geometric")


@dataclass
class TSPResult:
    """Pseudotime and diagnostics of one run of the ordering."""

    tau: np.ndarray
    diagnostics: dict = field(default_factory=dict)


class PathTSP:
    """Path-dependent TSP refined by a windowed 2-opt (Option 2)."""

    def __init__(self, init: str = "palantir", n_moves: int = 1000, window: int = 40,
                 use_path_length: bool = True, edge_weight: str = "mechanistic",
                 alpha_geom: float = 1.0, alpha_switch: float = 0.5,
                 dtau: Optional[float] = None, geom_space: str = "expression", seed: int = 0):
        if init not in INITS:
            raise ValueError(f"init must be one of {INITS}, got {init!r}")
        if edge_weight not in TSP_EDGE_COSTS:
            raise ValueError(f"edge_weight must be one of {TSP_EDGE_COSTS}, "
                             f"got {edge_weight!r}")
        if int(window) < 2:
            raise ValueError("window must be >= 2")
        if int(n_moves) < 0:
            raise ValueError("n_moves must be >= 0")
        self.init = init
        self.n_moves = int(n_moves)
        self.window = int(window)
        self.use_path_length = bool(use_path_length)
        self.edge_weight = edge_weight
        self.alpha_geom = float(alpha_geom)
        self.alpha_switch = float(alpha_switch)
        self.dtau = dtau
        self.geom_space = geom_space
        self.seed = seed

    @classmethod
    def from_model(cls, model) -> "PathTSP":
        """Read the ``tsp_*`` parameters of the model."""
        return cls(
            init=getattr(model, "tsp_init", "palantir"),
            n_moves=getattr(model, "tsp_n_moves", 1000),
            window=getattr(model, "tsp_window", 40),
            use_path_length=getattr(model, "tsp_use_path_length", True),
            edge_weight=getattr(model, "tsp_edge_weight", "mechanistic"),
            alpha_geom=getattr(model, "tsp_alpha_geom", 1.0),
            alpha_switch=getattr(model, "tsp_alpha_switch", 0.5),
            dtau=getattr(model, "tsp_dtau", None),
            geom_space=getattr(model, "tsp_geom_space", "expression"),
            seed=getattr(model, "seed", 0) or 0,
        )

    # ------------------------------------------------------------------------------------------
    def run(self, model, roots=None, verb: bool = False) -> TSPResult:
        """
        Pseudotime of every cell of ``model`` (needs ``model.modes``, and ``model.cell_prot`` for
        the mechanistic step cost; see ``PseudotimeNetworkModel.initialise_cell_states``).
        """
        ns = int(model.n_stimuli)
        N = int(model.modes.shape[0])
        modes = np.asarray(model.modes[:, ns:], dtype=float)
        counts = getattr(model, "counts", None)
        root_mask = _root_mask(model, roots, N)
        if not root_mask.any():
            raise ValueError("no root cell: pass roots=... (boolean mask)")
        if self.edge_weight == "mechanistic" and getattr(model, "cell_prot", None) is None:
            raise ValueError(
                "the mechanistic step cost needs the protein state of the cells: call "
                "PseudotimeNetworkModel.initialise_cell_states(data) first")

        ctx = CostContext.from_model(model, dtau=self._dtau(model), alpha=self.alpha_switch,
                                     alpha_geom=self.alpha_geom, geom_space=self.geom_space)
        stim = 1.0 - root_mask.astype(float)

        order, init_used = self._initial_order(model, root_mask, modes, counts, ctx, stim)
        P0 = self._start_protein(model, root_mask, modes, ns)
        costs, P_at = self._segment(order, P0, modes, counts, ctx, stim)
        length0 = float(costs.sum())

        n_tried, n_acc = self._two_opt(order, costs, P_at, modes, counts, ctx, stim, root_mask)

        tau = self._tau_from_tour(order, costs, root_mask)
        spread = float(np.max(tau))
        tau = tau / spread if spread > 0 else tau

        diagnostics = dict(
            method="tsp", init=init_used, edge_weight=self.edge_weight,
            use_path_length=self.use_path_length, n_moves=self.n_moves, window=self.window,
            n_moves_tried=n_tried, n_moves_accepted=n_acc, n_roots=int(root_mask.sum()),
            tour_length_start=length0, tour_length_end=float(costs.sum()),
            tour_improvement=(1.0 - float(costs.sum()) / length0) if length0 > 0 else 0.0,
            dtau=ctx.dtau, alpha_geom=self.alpha_geom, tau_max=spread,
        )
        if verb:
            print(f"  [tsp/{self.edge_weight}] {N} cells, init={init_used}, "
                  f"{n_acc}/{n_tried} windows accepted, "
                  f"tour {length0:.3f} -> {diagnostics['tour_length_end']:.3f}")
        return TSPResult(tau=tau, diagnostics=diagnostics)

    # ------------------------------------------------------------------------------------------
    # initial tour
    # ------------------------------------------------------------------------------------------
    def _dtau(self, model):
        return step_of(model, self.dtau)

    def _initial_order(self, model, root_mask, modes, counts, ctx, stim):
        """Cells sorted by the previous τ (warm start), else by the τ of the ``init`` option."""
        tau = getattr(model, "tau", None)
        if self.init == "previous":
            if tau is None:
                raise ValueError("init='previous' needs an existing pseudotime (model.tau)")
            return self._order_from_tau(tau, root_mask), "previous"
        if tau is not None and np.any(np.asarray(tau, dtype=float) > 0):
            return self._order_from_tau(tau, root_mask), "previous"
        init_tau = self._tau_from_init_option(model, root_mask, modes, counts, ctx, stim)
        return self._order_from_tau(init_tau, root_mask), self.init

    @staticmethod
    def _order_from_tau(tau, root_mask) -> np.ndarray:
        """Root cells first (they are all at τ = 0), then the others by increasing τ."""
        tau = np.asarray(tau, dtype=float).ravel()
        roots = np.flatnonzero(root_mask)
        rest = np.flatnonzero(~root_mask)
        return np.concatenate([roots[np.argsort(tau[roots], kind="stable")],
                               rest[np.argsort(tau[rest], kind="stable")]]).astype(int)

    def _tau_from_init_option(self, model, root_mask, modes, counts, ctx, stim):
        """τ of the ``init`` option, computed on the current state of the model."""
        from . import get_option
        return np.asarray(get_option(self.init, model).run(model, roots=root_mask).tau, dtype=float)

    def _start_protein(self, model, root_mask, modes, ns: int) -> np.ndarray:
        """Protein state the tour starts from: the mean of the root population."""
        prot = getattr(model, "cell_prot", None)
        if prot is not None:
            return np.asarray(prot, dtype=float)[root_mask][:, ns:].mean(axis=0)
        return modes[root_mask].mean(axis=0)

    # ------------------------------------------------------------------------------------------
    # the tour
    # ------------------------------------------------------------------------------------------
    def _step(self, i: int, j: int, P, modes, counts, ctx, stim):
        """Cost of the edge ``i -> j`` and the protein state reached at ``j``."""
        mi, mj = modes[i:i + 1], modes[j:j + 1]
        p_next = bridge_end(ctx.d1, P[None, :], mi, mj, ctx.s1, ctx.dtau, ctx.alpha)[0]
        geom = float(geometric_cost(None if counts is None else counts[i:i + 1],
                                    None if counts is None else counts[j:j + 1],
                                    mi, mj, ctx.geom_space)[0])
        if self.edge_weight == "geometric":
            return geom, p_next
        cost = float(network_cost(p_next[None, :], mj, ctx, stim=float(stim[j]))[0])
        return cost + ctx.alpha_geom * geom, p_next

    def _segment(self, seq, P_start, modes, counts, ctx, stim):
        """
        Propagate the proteins along ``seq`` from ``P_start``: the cost of every edge and the
        protein reached at every position (``plist[0] == P_start``).
        """
        n = int(np.size(seq))
        plist = np.empty((n, modes.shape[1]))
        plist[0] = P_start
        costs = np.zeros(max(n - 1, 0))
        for m in range(n - 1):
            costs[m], plist[m + 1] = self._step(int(seq[m]), int(seq[m + 1]), plist[m],
                                                modes, counts, ctx, stim)
        return costs, plist

    def _two_opt(self, order, costs, P_at, modes, counts, ctx, stim, root_mask):
        """
        First-improvement windowed 2-opt.  A move reverses ``order[a+1 : b+1]`` with
        ``b - a <= window``; the roots block is never touched (``a >= n_roots - 1``).
        Returns ``(n_moves_tried, n_moves_accepted)``; ``order`` / ``costs`` / ``P_at`` are updated.
        """
        N = int(order.size)
        n_roots = int(root_mask.sum())
        lo = max(n_roots - 1, 0)
        if self.n_moves <= 0 or N - 2 <= lo:
            return 0, 0
        rng = np.random.default_rng(self.seed)
        n_tried, n_acc = 0, 0
        for _ in range(self.n_moves):
            a = int(rng.integers(lo, N - 2))
            hi = int(min(a + self.window, N - 2))
            if hi <= a + 1:
                continue
            b = int(rng.integers(a + 2, hi + 1))
            n_tried += 1
            seq = np.concatenate(([order[a]], order[a + 1:b + 1][::-1], [order[b + 1]]))
            new_costs, new_p = self._segment(seq, P_at[a], modes, counts, ctx, stim)
            if new_costs.sum() < costs[a:b + 1].sum() - 1e-12:
                order[a + 1:b + 1] = order[a + 1:b + 1][::-1]
                costs[a:b + 1] = new_costs
                P_at[a + 1:b + 2] = new_p[1:]
                n_acc += 1
        return n_tried, n_acc

    def _tau_from_tour(self, order, costs, root_mask) -> np.ndarray:
        """
        τ = cumulative path length along the tour (``use_path_length``), else the tour rank.
        The root population is pinned at 0.
        """
        N = int(order.size)
        if self.use_path_length:
            cum = np.concatenate(([0.0], np.cumsum(costs)))
        else:
            cum = np.arange(N, dtype=float) / max(N - 1, 1)
        tau = np.empty(N)
        tau[order] = cum
        tau[root_mask] = 0.0
        return np.maximum(tau, 0.0)
