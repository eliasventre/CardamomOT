r"""
Option 1 — Partial OT from the initial population (``PLAN.md`` §5.2).

The flow analogue of CardamomOT: instead of a global OT between time points, the cells are ordered
by a sequence of **partial** transport problems, each one moving a fraction of the mass of the
current frontier onto the still unexplained cells.

Per EM iteration, given :math:`\Theta`:

1. :math:`\mathcal{C}_0` = root population, :math:`\tau=0`; :math:`\mathcal{U}` = the other cells.
2. For :math:`k = 0, 1, \dots`:

   1. **frontier** :math:`\mathcal{F}_k \subset \mathcal{C}_k` (the ``frontier_cap`` cells with the
      largest :math:`\tau`, i.e. the leading edge of the wave);
   2. **candidates** :math:`\mathcal{T}_k \subset \mathcal{U}` (:math:`\mathcal{U}` itself with
      ``restrict='all'``, else the ``knn_candidates`` nearest unexplained cells of each frontier
      cell in the burst-mode space, ``features.feature_matrix``);
   3. **cost** :math:`C_k(c,c') = \mathrm{loss}(k_{on}(P_{c\to c'};\Theta), k^{\mathrm{mix}}_{c'})
      + \alpha\,\mathrm{geom}(c,c')`, i.e. ``costs.mechanistic_cost`` — the same PDMP bridge as the
      E-step, over one step :math:`\Delta\tau`;
   4. **partial OT**: :math:`\gamma_k` minimises :math:`\langle C_k,\gamma\rangle
      + \varepsilon\,\mathrm{KL}(\gamma\,\|\,\mu\otimes\nu)` under the relaxed marginals
      :math:`\gamma\mathbf 1 \le \mu`, :math:`\gamma^{\top}\mathbf 1 \le \nu` and
      :math:`\mathbf 1^{\top}\gamma\mathbf 1 = \beta` (``ot.partial.entropic_partial_wasserstein``),
      or with the mass chosen by a Lagrange multiplier instead (``partial='lagrange'``, the
      phantom-column formulation of ``plan.tex``, ``ot.partial.partial_wasserstein_lagrange``).
      :math:`\beta` (``mass_per_step``) is what controls the *width* of the wave: with a uniform
      :math:`\nu` over the candidates, a fraction :math:`\beta` of them receives mass;
   5. **assign**: :math:`\tau_{c'} = \sum_c \gamma_k(c,c')\,(\tau_c+\Delta\tau) /
      \sum_c \gamma_k(c,c')` (a mass-weighted step, so :math:`\tau` stays an absolute coordinate);
      the cells with a column mass above ``min_mass`` join the frontier.
3. **fill**: cells that never received mass (a component disconnected from the roots) get the
   pseudotime of their reached neighbours plus one step.
4. :math:`\tau \leftarrow \tau/\max\tau` (dividing by the max and **not** ranking: the relative
   distances, hence the density read by the growth pass, are preserved — ``PLAN.md`` §6.2).

Diagnostics of a run: number of waves, cells reached per wave, transported mass, mean cost, the
number of times the candidate pool had to be widened (``n_fallback``) and the number of cells left
to the final fill (``n_unreached``) — the two quantities that measure the failure mode of the
method (*no cell left behind*, ``PLAN.md`` §7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import ot
from sklearn.neighbors import NearestNeighbors

from ..costs import EDGE_COSTS, CostContext, edge_cost, source_sigmoid, step_of
from .common import root_mask
from .features import feature_matrix

EPS = 1e-16
RESTRICT = ("knn", "all")
PARTIAL = ("mass", "lagrange")
FRONTIER_MODES = ("front", "all", "random")


@dataclass
class PartialOTResult:
    """Pseudotime and diagnostics of one run of the ordering."""

    tau: np.ndarray
    diagnostics: dict = field(default_factory=dict)


class PartialOT:
    """Waves of partial transport from the root population (Option 1)."""

    def __init__(self, mass_per_step: float = 0.25, max_steps: Optional[int] = None,
                 reg: float = 0.05, alpha_geom: float = 1.0, restrict: str = "knn",
                 knn_candidates: int = 10, phantom_cost: Optional[float] = None,
                 partial: str = "mass", min_mass: float = 1e-3,
                 frontier_cap: Optional[int] = 200, max_candidates: Optional[int] = 400,
                 wave_max: Optional[int] = None, frontier_mode: str = "front",
                 edge_weight: str = "mechanistic", alpha_switch: float = 0.5,
                 dtau: Optional[float] = None, geom_space: str = "expression",
                 feature: str = "posterior", feature_weight: str = "gene_weights",
                 n_pcs: int = 0, num_iter_max: int = 200, seed: int = 0):
        if restrict not in RESTRICT:
            raise ValueError(f"restrict must be one of {RESTRICT}, got {restrict!r}")
        if partial not in PARTIAL:
            raise ValueError(f"partial must be one of {PARTIAL}, got {partial!r}")
        if frontier_mode not in FRONTIER_MODES:
            raise ValueError(f"frontier_mode must be one of {FRONTIER_MODES}, "
                             f"got {frontier_mode!r}")
        if edge_weight not in EDGE_COSTS:
            raise ValueError(f"edge_weight must be one of {EDGE_COSTS}, got {edge_weight!r}")
        if not 0.0 < float(mass_per_step) <= 1.0:
            raise ValueError("mass_per_step must be in (0, 1]")
        self.mass_per_step = float(mass_per_step)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.reg = float(reg)
        self.alpha_geom = float(alpha_geom)
        self.restrict = restrict
        self.knn_candidates = int(knn_candidates)
        self.phantom_cost = phantom_cost
        self.partial = partial
        self.min_mass = float(min_mass)
        self.frontier_cap = frontier_cap
        self.max_candidates = max_candidates
        self.wave_max = wave_max
        self.frontier_mode = frontier_mode
        self.edge_weight = edge_weight
        self.alpha_switch = float(alpha_switch)
        self.dtau = dtau
        self.geom_space = geom_space
        self.feature = feature
        self.feature_weight = feature_weight
        self.n_pcs = int(n_pcs)
        self.num_iter_max = int(num_iter_max)
        self.seed = seed

    @classmethod
    def from_model(cls, model) -> "PartialOT":
        """Read the ``partial_ot_*`` parameters of the model."""
        return cls(
            mass_per_step=getattr(model, "partial_ot_mass_per_step", 0.25),
            max_steps=getattr(model, "partial_ot_max_steps", None),
            reg=getattr(model, "partial_ot_reg", 0.05),
            alpha_geom=getattr(model, "partial_ot_alpha_geom", 1.0),
            restrict=getattr(model, "partial_ot_restrict", "knn"),
            knn_candidates=getattr(model, "partial_ot_knn_candidates", 10),
            phantom_cost=getattr(model, "partial_ot_phantom_cost", None),
            partial=getattr(model, "partial_ot_partial", "mass"),
            min_mass=getattr(model, "partial_ot_min_mass", 1e-3),
            frontier_cap=getattr(model, "partial_ot_frontier_cap", 200),
            max_candidates=getattr(model, "partial_ot_max_candidates", 400),
            wave_max=getattr(model, "partial_ot_wave_max", None),
            frontier_mode=getattr(model, "partial_ot_frontier_mode", "front"),
            edge_weight=getattr(model, "partial_ot_edge_weight", "mechanistic"),
            alpha_switch=getattr(model, "partial_ot_alpha_switch", 0.5),
            dtau=getattr(model, "partial_ot_dtau", None),
            geom_space=getattr(model, "partial_ot_geom_space", "expression"),
            feature=getattr(model, "partial_ot_feature", "posterior"),
            feature_weight=getattr(model, "partial_ot_feature_weight", "gene_weights"),
            n_pcs=getattr(model, "partial_ot_n_pcs", 0),
            num_iter_max=getattr(model, "partial_ot_num_iter_max", 200),
            seed=getattr(model, "seed", 0) or 0,
        )

    # ------------------------------------------------------------------------------------------
    def run(self, model, roots=None, verb: bool = False) -> PartialOTResult:
        """
        Pseudotime of every cell of ``model`` (needs ``model.modes``, and ``model.counts`` /
        ``model.cell_prot`` for the geometric / mechanistic parts; see
        ``PseudotimeNetworkModel.initialise_cell_states``).

        ``roots``: boolean mask (or indices) of the root cells.
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
        prot = None
        if self.edge_weight == "mechanistic":
            if getattr(model, "cell_prot", None) is None:
                raise ValueError(
                    "mechanistic step costs need the protein state of the cells: call "
                    "PseudotimeNetworkModel.initialise_cell_states(data) first")
            prot = np.asarray(model.cell_prot, dtype=float)[:, ns:]
        stim = 1.0 - root_mask.astype(float)          # the stimulus is on after the root population

        F = feature_matrix(model, self.feature, self.feature_weight, self.n_pcs, self.seed)
        k_fit = int(min(max(self.knn_candidates + 1, 16), max(N - 1, 1)))
        knn = NearestNeighbors(n_neighbors=k_fit).fit(F)
        rng = np.random.default_rng(self.seed)

        tau = np.full(N, np.inf)
        tau[root_mask] = 0.0
        reached = root_mask.copy()
        frontier = np.flatnonzero(root_mask)
        dtau = ctx.dtau
        n_max = 200 if self.max_steps is None else int(self.max_steps)

        history, wide_pool, n_fallback = [], False, 0
        for step in range(n_max):
            unknown = ~reached
            if not unknown.any():
                break
            src = self._frontier(frontier, tau, rng)
            tgt = np.flatnonzero(unknown) if wide_pool else self._candidates(knn, F, src, unknown)
            if tgt.size == 0:                          # frontier isolated: widen the pool
                wide_pool, n_fallback = True, n_fallback + 1
                tgt = np.flatnonzero(unknown)
            M = self._cost_matrix(src, tgt, modes, counts, prot, ctx, stim)
            gamma, mass = self._solve(M)
            col = gamma.sum(axis=0)
            keep = col > self.min_mass
            if self.wave_max and int(keep.sum()) > int(self.wave_max):
                idx = np.argsort(-col, kind="stable")[: int(self.wave_max)]
                keep = np.zeros_like(keep)
                keep[idx] = True
            if not keep.any():
                if wide_pool:
                    break
                wide_pool, n_fallback = True, n_fallback + 1
                continue
            nj = np.flatnonzero(keep)
            tau[tgt[nj]] = ((gamma[:, nj] * (tau[src][:, None] + dtau)).sum(axis=0)
                            / np.maximum(col[nj], EPS))
            reached[tgt[nj]] = True
            frontier = np.concatenate([frontier, tgt[nj]])
            history.append(dict(step=int(step), n_src=int(src.size), n_candidates=int(tgt.size),
                                n_new=int(nj.size), mass=float(mass),
                                cost_mean=float(np.mean(M))))

        n_unreached = int((~np.isfinite(tau)).sum())
        if n_unreached:
            tau = self._fill_unreached(knn, F, tau, ctx.dtau)

        tau = np.nan_to_num(tau, nan=0.0, posinf=0.0)
        tau = np.maximum(tau - np.min(tau[root_mask]), 0.0)
        tau[root_mask] = 0.0
        spread = float(np.max(tau))
        tau = tau / spread if spread > 0 else tau

        diagnostics = dict(
            method="partial_ot", edge_weight=self.edge_weight, partial=self.partial,
            restrict=self.restrict, feature=self.feature, feature_weight=self.feature_weight,
            n_roots=n_roots, n_steps=len(history), n_unreached=n_unreached,
            n_fallback=n_fallback, knn_candidates=self.knn_candidates,
            frontier_cap=self.frontier_cap, frontier_mode=self.frontier_mode,
            mass_per_step=self.mass_per_step, reg=self.reg, dtau=ctx.dtau,
            alpha_geom=self.alpha_geom, wave_sizes=[h["n_new"] for h in history],
            history=history, tau_max=spread,
        )
        if verb:
            print(f"  [partial_ot/{self.edge_weight}] {N} cells, {n_roots} roots, "
                  f"{len(history)} waves, {n_unreached} filled at the end, "
                  f"{n_fallback} pool widenings")
        return PartialOTResult(tau=tau, diagnostics=diagnostics)

    # ------------------------------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------------------------------
    def _dtau(self, model):
        """Step of the wave: the bin width when the axis has a duration, else one relaxation time."""
        return step_of(model, self.dtau)

    @staticmethod
    def _roots(model, roots, N: int) -> np.ndarray:
        return root_mask(model, roots, N)

    def _frontier(self, frontier: np.ndarray, tau: np.ndarray, rng) -> np.ndarray:
        """
        Cells of the frontier that transport mass this wave.  ``'front'`` (default) keeps the
        leading edge (largest :math:`\\tau`, so the wave always moves forward), ``'random'`` samples
        uniformly (a stochastic approximation over the waves), ``'all'`` keeps everything.
        ``frontier_cap=None`` disables the cap — the partial-OT matrix is then
        ``|frontier| x |candidates|`` and the cost grows quadratically.
        """
        cap = None if self.frontier_cap is None else int(self.frontier_cap)
        if self.frontier_mode == "all" or cap is None or frontier.size <= cap:
            return frontier
        if self.frontier_mode == "random":
            return np.sort(frontier[rng.choice(frontier.size, cap, replace=False)])
        order = np.argsort(-tau[frontier], kind="stable")[:cap]
        return frontier[order]

    def _candidates(self, knn, F, src: np.ndarray, unknown: np.ndarray) -> np.ndarray:
        """Unexplained cells that are close (in the feature space) to at least one frontier cell."""
        k = int(min(self.knn_candidates, F.shape[0] - 1))
        dist, idx = knn.kneighbors(F[src], n_neighbors=max(k, 1))
        dist, idx = dist.ravel(), idx.ravel()
        mask = unknown[idx]
        dist, idx = dist[mask], idx[mask]
        if idx.size == 0:
            return np.empty(0, dtype=int)
        order = np.argsort(dist, kind="stable")
        uniq, first = np.unique(idx[order], return_index=True)
        uniq = uniq[np.argsort(first, kind="stable")]
        if self.max_candidates:
            uniq = uniq[: int(self.max_candidates)]
        return uniq

    def _cost_matrix(self, src, tgt, modes, counts, prot, ctx, stim) -> np.ndarray:
        """(n_src, n_candidates) matrix of the transition costs of one wave."""
        n_s, n_t = int(src.size), int(tgt.size)
        ii = np.repeat(src, n_t)
        jj = np.tile(tgt, n_s)
        kw = dict(rna_i=None if counts is None else counts[ii],
                  rna_j=None if counts is None else counts[jj],
                  stim=stim[jj])
        if self.edge_weight == "mechanistic":
            kw["P0"] = prot[ii]
        elif self.edge_weight == "plausibility":
            kw["sig_i"] = source_sigmoid(modes[ii], ctx, stim=stim[ii])
        cost = edge_cost(self.edge_weight, modes[ii], modes[jj], ctx, **kw)
        return np.asarray(cost, dtype=float).reshape(n_s, n_t)

    def _solve(self, M: np.ndarray):
        """Partial OT plan of one wave, and its total mass."""
        n_s, n_t = M.shape
        a = np.full(n_s, 1.0 / n_s)
        b = np.full(n_t, 1.0 / n_t)
        scale = max(float(np.median(M)), EPS)
        if self.partial == "lagrange":
            gamma = ot.partial.partial_wasserstein_lagrange(
                a, b, M, reg_m=self.phantom_cost)
        else:
            gamma = ot.partial.entropic_partial_wasserstein(
                a, b, M, reg=max(float(self.reg) * scale, EPS),
                m=float(min(self.mass_per_step, 1.0)), numItermax=self.num_iter_max)
        gamma = np.asarray(gamma, dtype=float).reshape(n_s, n_t)
        return gamma, float(gamma.sum())

    @staticmethod
    def _fill_unreached(knn, F, tau: np.ndarray, dtau: float, n_iter: int = 20,
                        n_neighbors: int = 15) -> np.ndarray:
        """
        Cells that never received mass (a component disconnected from the roots): the mean
        pseudo-time of their reached neighbours plus one step, relaxed a few times.
        """
        tau = tau.copy()
        for _ in range(int(n_iter)):
            bad = ~np.isfinite(tau)
            if not bad.any():
                break
            n_ok = int((~bad).sum())
            if n_ok == 0:
                break
            k = int(min(n_neighbors, n_ok, F.shape[0] - 1))
            _, idx = knn.kneighbors(F[bad], n_neighbors=max(k, 1))
            nb = tau[idx]
            ok = np.isfinite(nb)
            cnt = ok.sum(axis=1)
            s = np.where(ok, nb, 0.0).sum(axis=1)
            good = cnt > 0
            tau[np.flatnonzero(bad)[good]] = s[good] / cnt[good] + dtau
        if not np.all(np.isfinite(tau)):
            finite = tau[np.isfinite(tau)]
            tau[~np.isfinite(tau)] = (float(np.max(finite)) if finite.size else 0.0) + dtau
        return tau
