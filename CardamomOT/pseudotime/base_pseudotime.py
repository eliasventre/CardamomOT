"""
CardamomOT with the time labels replaced by an inferred pseudotime.

``PseudotimeNetworkModel`` subclasses :class:`CardamomOT.model.base.NetworkModel` and leaves the base
implementation untouched: it only adds the pseudotime state (``tau``, the bins) and the pieces of
the modified cycle that do not depend on the ordering method.  The three ordering options
(partial OT, path-dependent TSP, mechanistic Palantir) plug into ``update_pseudotime`` later; see
``PLAN.md``.

This module implements the **mixture on bins** increment of ``PLAN.md`` (§4.1, §10.6): the NB
mixture is fitted with the bins as time labels, so that

* each bin has its own mode masses :math:`\\nu_g(k)` (imposed as the target marginal of the basin
  EMD, ``force_basins``) — the burst-frequency distribution becomes a function of the pseudotime,
  which is the signal the network has to explain;
* the mean-preservation constraint is enforced *per bin* instead of globally;
* the extreme mode amplitudes are calibrated on the extreme bins (``_apply_temporal_constraints``
  in ``inference/mixture.py``), so they span the dynamic range observed along the axis.

The fit is reused unchanged (``NetworkModel.fit_mixture``) by writing the bin time in an obs column
and passing it as ``time_key``.  Three guards limit the circularity (bins -> mixture -> bins):

1. **fit from a Theta-free pseudotime** (``pseudotime_refit_mixture`` is False during the first
   iterations, or the caller passes the tau of the previous iteration): the mixture is fitted on
   bins that do not come from the network;
2. **re-fit only every** ``pseudotime_refit_every`` iterations (``refit_mixture_if_due``);
3. **ramp** ``force_basins`` from ``pseudotime_force_basins_start`` to ``self.force_basins`` over
   ``min_n_loops`` iterations (``force_basins_at``), instead of imposing the bins' masses at full
   strength from the first iteration.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import ot
from joblib import Parallel, delayed

from ..inference.network import (PrevProt, active_regulators, gene_weights, inference_network_multi,
                                 sigmoid_amplitudes)
from ..inference.trajectory import inference_alpha, minimal_repetition_choice, s1_rows
from ..model.base import NetworkModel
from .binning import BinScheme, bin_from_tau, tau_from_time
from .evaluate import spearman


class PseudotimeNetworkModel(NetworkModel):
    """CardamomOT whose time labels are replaced by an inferred pseudotime."""

    # --- pseudotime parameters (settable through Data/CardamomOT_inputs.xlsx) -------------------
    pseudotime_bins = 10              # K, root bin included
    pseudotime_bin_rule = "equal_width"   # 'equal_width' (keeps the density -> growth) | 'equal_count'
    pseudotime_bin_min_cells = 50     # bins with fewer cells are merged
    pseudotime_total_duration = None  # duration of the axis in model time units; None = normalised to 1
    pseudotime_time_key = "pseudotime_bin"        # obs column holding the bin *time* (read by fit_mixture)
    pseudotime_bin_index_key = "pseudotime_bin_index"  # obs column holding the bin index
    pseudotime_root_key = "is_root"               # obs bool column of the root cells
    pseudotime_root_group_key = "root_group"      # optional obs column: several root populations
    # guards against the bins -> mixture -> bins circularity
    pseudotime_refit_mixture = True   # fit the mixture on the bins (False: pooled fit only)
    pseudotime_refit_every = 5        # re-fit every n EM iterations (the first iteration always fits)
    pseudotime_force_basins_start = 0.3   # force_basins at iteration 1, ramping to self.force_basins
    # ordering step (PLAN.md §5)
    pseudotime_method = "palantir"        # 'palantir' | 'partial_ot' | 'tsp'
    palantir_n_neighbors = 60             # denser graph -> smoother tau -> better bins (§0.7)
    palantir_n_waypoints = 50
    palantir_n_waypoints_local = 5        # nearest waypoints used by the consensus
    palantir_feature = "posterior"        # 'posterior' (expected modes) | 'hard' | 'expression'
    palantir_edge_weight = "geometric"    # 'geometric' | 'plausibility' | 'mechanistic'
                                          # | 'auto' (plausibility while Theta=0, then mechanistic)
                                          # geometric is the best inside the cycle so far (§0.5)
    palantir_alpha_geom = 0.5
    palantir_alpha_switch = 0.5
    palantir_dtau = None                  # step of the graph; None = one protein relaxation time
    palantir_n_consensus = 3
    palantir_consensus = "min"           # 'min' (tightest perspective) | 'mean' (Palantir weighted mean)
    palantir_symmetrise = True
    palantir_feature_weight = "gene_weights"   # 'gene_weights' | 'std' | 'none'
    palantir_n_pcs = 0
    palantir_geom_space = "expression"    # geometric term of the edge costs: 'expression' | 'modes'
    # Option 1: waves of partial OT from the root population (PLAN.md §5.2)
    partial_ot_mass_per_step = 0.25      # beta: fraction of the wave mass transported per step
    partial_ot_max_steps = None          # K; None = until every cell is reached (bounded at 200)
    partial_ot_reg = 0.05                # entropic regularisation, relative to the median cost
    partial_ot_alpha_geom = 1.0          # weight of the geometric term (0.2 lets the fitted network
                                         # override the expression continuity -> tau collapses, §0.8)
    partial_ot_restrict = "knn"          # candidate pool: 'knn' | 'all'
    partial_ot_knn_candidates = 10       # nearest unexplained cells of each frontier cell
    partial_ot_partial = "mass"          # 'mass' (fixed beta) | 'lagrange' (phantom column)
    partial_ot_phantom_cost = None       # lambda of the phantom column, 'lagrange' only
    partial_ot_min_mass = 1e-3           # column mass above which a cell joins the frontier
    partial_ot_frontier_cap = 200        # leading-edge cells solved per wave (None: no cap)
    partial_ot_max_candidates = 400      # candidates kept per wave (None: no cap)
    partial_ot_wave_max = None           # cap of the cells added by one wave
    partial_ot_frontier_mode = "front"   # 'front' (largest tau) | 'all' | 'random'
    partial_ot_edge_weight = "mechanistic"   # 'mechanistic' | 'plausibility' | 'geometric'
    partial_ot_alpha_switch = 0.5
    partial_ot_dtau = None               # step of the wave; None = one relaxation time / bin width
    partial_ot_geom_space = "expression"  # 'expression' | 'modes'
    partial_ot_feature = "posterior"     # space of the candidate search
    partial_ot_feature_weight = "gene_weights"
    partial_ot_n_pcs = 0
    partial_ot_num_iter_max = 200        # Sinkhorn iterations of one partial-OT solve
    # Option 2: path-dependent TSP refined by a windowed 2-opt (PLAN.md §5.3)
    tsp_init = "palantir"                # option that gives the first tour: 'palantir'
                                         # | 'partial_ot' | 'previous'
    tsp_n_moves = 1000                   # random windows tried per EM iteration
    tsp_window = 40                      # longest segment reversed by one 2-opt move
    tsp_use_path_length = True           # tau = cumulative path length (False: the tour rank)
    tsp_edge_weight = "mechanistic"      # 'mechanistic' | 'geometric'
    tsp_alpha_geom = 1.0                 # weight of the geometric term in the step cost
    tsp_alpha_switch = 0.5
    tsp_dtau = None                      # step of the propagation; None = one relaxation time
    tsp_geom_space = "expression"         # 'expression' | 'modes'
    # the cycle (PLAN.md §4.4)
    pseudotime_max_iter = None        # None = self.max_iter
    pseudotime_count_max = 2          # consecutive stable iterations to stop
    pseudotime_stable = 0.999         # Spearman(tau_n, tau_{n-1}) of a stable iteration
    pseudotime_weight_prev = 0.0      # flow-matching term of the M-step (0: disabled in v1)
    pseudotime_update_alpha = True    # recompute the switching fractions each iteration
    pseudotime_growth = True          # growth pass (net proliferation) after the loop
    pseudotime_fixed_tau = False      # DEBUG: keep the pseudotime fixed (validate the cycle on true bins)
    # duration of the pseudotime axis (PLAN.md §4.3): only d1 * dtau is identified
    pseudotime_duration_mode = "prior"   # 'prior' (bins span ~relaxation/d1) | 'fixed' | 'grid'
    pseudotime_relaxation = 1.0          # bins span this many protein relaxation times (1/d1)
    pseudotime_dt_grid = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)   # candidates, in multiples of 1/median(d1)

    def __init__(self, n_genes: Optional[int] = None, n_stimuli: int = 1, times=None):
        super().__init__(n_genes=n_genes, n_stimuli=n_stimuli, times=times)
        self.tau = None            # (N,) pseudotime of each real cell (distance / step count, >= 0)
        self.bins = None           # BinScheme
        self.roots = None          # (N,) bool mask of the root cells
        self.counts = None         # (N, G) raw counts of the cells (geometric term, diagnostics)
        self.cell_prot = None      # (N, G_tot) protein state of each real cell
        self.mixture_fit_iter = None   # EM iteration of the last mixture fit
        self.mixture_fits = []         # one summary dict per mixture fit
        self.pseudotime_iter = None    # EM iteration of the last ordering
        self.pseudotime_diagnostics = None
        self.pseudotime_history = []   # one dict per iteration of the cycle

    # ------------------------------------------------------------------------------------------
    # Pseudotime / bins
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def tau_from_time(times: Sequence[float]) -> np.ndarray:
        """Pseudotime from known times (oracle validation only)."""
        return tau_from_time(times)

    def root_mask(self, data) -> np.ndarray:
        """
        Boolean mask of the root cells: ``obs[pseudotime_root_key]`` (boolean or 0/1), else the cells
        with the smallest ``tau``.  Raises if neither is available.
        """
        if self.tau is not None:
            if self.pseudotime_root_key and self.pseudotime_root_key in data.obs:
                flag = np.asarray(data.obs[self.pseudotime_root_key])
                if flag.dtype == bool or set(np.unique(flag)).issubset({0, 1, 0.0, 1.0, True, False}):
                    return flag.astype(bool)
            return np.asarray(self.tau) <= np.min(self.tau) + 1e-12
        if self.pseudotime_root_key and self.pseudotime_root_key in data.obs:
            return np.asarray(data.obs[self.pseudotime_root_key]).astype(bool)
        raise ValueError(
            f"no root cells: set the pseudotime (set_tau) or the obs column "
            f"'{self.pseudotime_root_key}'"
        )

    def set_bins(self, scheme: BinScheme) -> BinScheme:
        """Set the bin structure (see binning.bin_from_tau)."""
        self.bins = scheme
        self.roots = scheme.bin_of_cell == 0
        return scheme

    def set_tau(self, tau, roots=None, n_bins: Optional[int] = None,
                rule: Optional[str] = None, min_cells: Optional[int] = None,
                total_duration: Optional[float] = None) -> BinScheme:
        """
        Set the per-cell pseudotime and bin it (binning.bin_from_tau).

        ``tau`` must be an *absolute* coordinate (a distance or a step count, 0 on the roots), not a
        rank: the growth pass reads the density of cells along the axis (PLAN.md §6.2).
        """
        tau = np.asarray(tau, dtype=float).ravel()
        if self.n_cells_known() is not None and tau.size != self.n_cells_known():
            raise ValueError(f"{tau.size} pseudotimes for {self.n_cells_known()} cells")
        self.tau = tau
        return self.set_bins(bin_from_tau(
            tau,
            n_bins=self.pseudotime_bins if n_bins is None else n_bins,
            roots=roots,
            rule=self.pseudotime_bin_rule if rule is None else rule,
            min_cells=self.pseudotime_bin_min_cells if min_cells is None else min_cells,
            total_duration=self.pseudotime_total_duration if total_duration is None else total_duration,
        ))

    def set_total_duration(self, total_duration: float) -> BinScheme:
        """Re-bin the current tau with another duration of the axis (used by the duration gauge)."""
        if self.tau is None:
            raise ValueError("no pseudotime: call set_tau first")
        self.pseudotime_total_duration = float(total_duration)
        return self.set_bins(bin_from_tau(
            self.tau, n_bins=self.bins.n_bins if self.bins is not None else self.pseudotime_bins,
            roots=self.roots,            # keep the root population (else only the earliest cell)
            rule=self.pseudotime_bin_rule, min_cells=self.pseudotime_bin_min_cells,
            total_duration=float(total_duration),
        ))

    def n_cells_known(self) -> Optional[int]:
        """Number of cells of the current mixture fit (None before any fit)."""
        if self.basins is not None:
            return int(self.basins.shape[0])
        return None

    def obs_columns(self) -> dict:
        """obs columns giving the bins (``pseudotime_bin`` = bin time, read by fit_mixture)."""
        if self.bins is None:
            raise ValueError("no bins: call set_tau / set_bins first")
        return self.bins.as_obs(self.pseudotime_time_key, self.pseudotime_bin_index_key)

    def write_obs(self, data) -> None:
        """Write ``obs['pseudotime_bin']`` (bin time) and ``obs['pseudotime_bin_index']``."""
        for key, values in self.obs_columns().items():
            data.obs[key] = values

    @property
    def bin_times(self) -> Optional[np.ndarray]:
        return None if self.bins is None else self.bins.bin_times

    @property
    def dtau(self) -> Optional[float]:
        return None if self.bins is None else self.bins.dtau

    # ------------------------------------------------------------------------------------------
    # Ordering step (PLAN.md §5) and per-cell states
    # ------------------------------------------------------------------------------------------
    def pseudotime_option(self):
        """Ordering option of ``pseudotime_method``, built from the ``palantir_*`` parameters."""
        from .options import get_option
        return get_option(self.pseudotime_method, self)

    def initialise_cell_states(self, data):
        """
        Raw counts and protein state of each real cell, at the equilibrium of the assigned modes
        (the same initialisation as ``estimate_trajectories_given_model``: ``adaptive_shrinkage_init``
        of the observed mRNA, i.e. what the burst rate of the cell's basin implies).

        Reads the roots from ``obs[pseudotime_root_key]`` when no pseudotime is set yet.  The loop
        replaces ``cell_prot`` by the trajectories of the E-step after the first iteration.
        """
        if self.modes is None:
            raise ValueError("fit the mixture first (fit_mixture_on_bins / fit_mixture_pooled)")
        ns = self.n_stimuli
        data_rna = self._parse_input(data)
        self.counts = np.asarray(data_rna[:, ns:], dtype=float)
        if self.roots is None and self.pseudotime_root_key in data.obs:
            self.roots = np.asarray(data.obs[self.pseudotime_root_key]).astype(bool)

        s1 = self._mixture_terms()[1]
        s1 = s1[0] if np.ndim(s1) == 2 else s1
        prot = np.zeros_like(np.asarray(self.modes, dtype=float))
        prot[:, ns:] = self.adaptive_shrinkage_init(self.counts * s1, self.modes[:, ns:])
        stim = np.ones(prot.shape[0])
        if self.roots is not None:
            stim[np.asarray(self.roots)] = 0.0     # the stimulus switches on after the roots
        prot[:, :ns] = stim[:, None]
        self.cell_prot = prot
        return prot

    def update_pseudotime(self, n_iter: int = 1, roots=None, verb: bool = True,
                          total_duration: Optional[float] = None):
        """
        Ordering step of the cycle: recompute the pseudotime from the current network
        (:math:`\\Theta` = zeros at the first iteration) and proteins, and re-bin the cells.

        Returns ``tau``; ``self.bins`` / ``self.tau`` are updated (see :meth:`set_tau`).
        """
        if self.modes is None:
            raise ValueError("fit the mixture first (fit_mixture_on_bins / fit_mixture_pooled)")
        if self.cell_prot is None:
            raise ValueError("call initialise_cell_states(data) before update_pseudotime")
        roots = self.roots if roots is None else roots
        result = self.pseudotime_option().run(self, roots=roots, verb=verb)
        self.set_tau(result.tau, roots=roots, total_duration=total_duration)
        self.pseudotime_iter = int(n_iter)
        self.pseudotime_diagnostics = dict(result.diagnostics)
        self.pseudotime_diagnostics["n_iter"] = int(n_iter)
        return result.tau

    # ------------------------------------------------------------------------------------------
    # Guard 3: ramped force_basins
    # ------------------------------------------------------------------------------------------
    def force_basins_at(self, n_iter: int) -> float:
        """
        Strength of the per-bin mode masses in the basin update (``force_basins``) at iteration
        ``n_iter``: ``pseudotime_force_basins_start`` at iteration 1, ramping linearly to
        ``self.force_basins`` at ``min_n_loops``.  Imposing the bins' masses at full strength from
        the start would partly re-assert the ordering the mixture was fitted on (PLAN.md §10.6).
        """
        start, end = float(self.pseudotime_force_basins_start), float(self.force_basins)
        if start == end:
            return end
        span = max(self.min_n_loops - 1, 1)
        lam = min(1.0, max(0.0, (n_iter - 1) / span))
        return end + (start - end) * (1.0 - lam)

    # ------------------------------------------------------------------------------------------
    # Mixture on bins
    # ------------------------------------------------------------------------------------------
    def fit_mixture_on_bins(self, data, tau=None, roots=None, n_bins: Optional[int] = None,
                            rule: Optional[str] = None, min_cells: Optional[int] = None,
                            total_duration: Optional[float] = None, gene_names=None,
                            min_components: int = 2, max_components: int = 2,
                            max_iter_kinetics: int = 0, refilter: float = 0,
                            stimulus_schedule=None, verb: bool = True, n_iter: int = 1):
        """
        Fit the NB mixture with the pseudotime bins as time labels (PLAN.md §4.1, §10.6).

        ``tau`` is set (and binned) first if given.  The bin time is written in
        ``obs[pseudotime_time_key]`` and passed as ``time_key`` to the unchanged
        ``NetworkModel.fit_mixture``, so that the mixture runs its per-group basin EMD (per-bin
        target masses with the mean constraint) and calibrates the extreme modes on the extreme
        bins.  ``self.modes``, ``self.basins``, ``self.proba_init``, ``self.pi_init`` and ``self.a``
        are replaced: call :meth:`mixture_terms` again afterwards (the caller of a re-fit must
        refresh ks / s1 / k1).

        Returns the per-bin masses, ``(K, G_tot, M)`` (see :meth:`bin_masses`).
        """
        if tau is not None:
            self.set_tau(tau, roots=roots, n_bins=n_bins, rule=rule, min_cells=min_cells,
                         total_duration=total_duration)
        elif self.bins is None:
            raise ValueError("no bins: pass tau=... (or call set_tau / set_bins) first")

        self.write_obs(data)
        if gene_names is None:
            gene_names = [str(g) for g in data.var_names]

        self.fit_mixture(
            data, refilter=refilter, gene_names=gene_names,
            min_components=min_components, max_components=max_components,
            max_iter_kinetics=max_iter_kinetics, verb=verb,
            stimulus_schedule=stimulus_schedule, time_key=self.pseudotime_time_key,
        )

        self.mixture_fit_iter = int(n_iter)
        self.mixture_fits.append(dict(
            n_iter=int(n_iter), n_bins=int(self.bins.n_bins), dtau=float(self.bins.dtau),
            counts=[int(c) for c in self.bins.counts],
            mode_amplitudes=self.mode_amplitudes().tolist(),
        ))
        return self.bin_masses()

    def fit_mixture_pooled(self, data, gene_names=None, stimulus_schedule=None, verb: bool = True,
                           time_key: str = "pseudotime_bin", **kwargs):
        """
        Pooled mixture (one group for every cell), the comparator of the binned fit: same code path,
        a single group, so the mean-preservation constraint is global and the mode masses are
        constant along the pseudotime.
        """
        data.obs[time_key] = np.zeros(data.n_obs)
        if gene_names is None:
            gene_names = [str(g) for g in data.var_names]
        self.fit_mixture(data, gene_names=gene_names, stimulus_schedule=stimulus_schedule,
                         verb=verb, time_key=time_key, **kwargs)
        return self.bin_masses()

    def refit_mixture_if_due(self, data, n_iter: int, verb: bool = True, **kwargs) -> bool:
        """
        Guard 2: re-fit the mixture on the current bins only every ``pseudotime_refit_every``
        iterations (the first iteration always fits).  Returns True when a fit happened.
        """
        if not self.pseudotime_refit_mixture:
            return False
        every = max(1, int(self.pseudotime_refit_every))
        if n_iter != 1 and (n_iter - 1) % every != 0:
            return False
        self.fit_mixture_on_bins(data, n_iter=n_iter, verb=verb, **kwargs)
        return True

    def mixture_terms(self):
        """``ks, s1, k1`` of the current mixture (call again after a re-fit)."""
        return self._mixture_terms()

    # ------------------------------------------------------------------------------------------
    # Diagnostics: per-bin mode masses and implied means
    # ------------------------------------------------------------------------------------------
    def bin_masses(self, n_modes: Optional[int] = None):
        """
        ``(K, G_tot, M)`` per-bin mode masses ``nu[k, g, m]`` of the current mixture, or None.

        ``self.pi_init`` is a list over genes of ``{bin time: masses}``; this stacks it in the bin
        order of ``self.bins``, padded to the largest number of modes (0 elsewhere).  The stimulus
        columns are left at 0.  With a pooled mixture (one group) its single mass vector is used for
        every bin.
        """
        if not self.pi_init or self.a is None or self.bins is None:
            return None
        ns = self.n_stimuli
        G_tot = int(self.a.shape[-1])
        times = self.bins.bin_times
        if n_modes is None:
            n_modes = max(int(np.asarray(v).size) for d in self.pi_init for v in d.values())
        out = np.zeros((len(times), G_tot, int(n_modes)))
        for g, masses in enumerate(self.pi_init):
            # a pooled mixture has a single group: the same masses for every bin
            single = next(iter(masses.values())) if len(masses) == 1 else None
            for k, t in enumerate(times):
                nu = masses.get(float(t), single)
                if nu is None:
                    continue
                nu = np.asarray(nu, dtype=float).ravel()
                out[k, g + ns, :min(out.shape[2], nu.size)] = nu[:out.shape[2]]
        return out

    def mode_amplitudes(self) -> np.ndarray:
        """``(G_tot, M)`` mode amplitudes ks of each gene (from self.a), 0 for the unused modes."""
        if self.a is None:
            raise ValueError("no mixture: fit it first")
        return np.asarray(self.a[:-1], dtype=float)   # a[-1] holds the dispersion c

    def mixture_masses(self, n_modes: Optional[int] = None):
        """
        ``(G_tot, M)`` mode masses averaged over the groups: the per-bin masses weighted by the bin
        masses when bins are set, else the single (pooled) mass vector.
        """
        nu = self.bin_masses(n_modes=n_modes)
        if nu is not None:
            w = self.bins.masses()[:, None, None]
            return (nu * w).sum(axis=0)
        if not self.pi_init or self.a is None:
            return None
        ns = self.n_stimuli
        G_tot = int(self.a.shape[-1])
        if n_modes is None:
            n_modes = max(int(np.asarray(v).size) for d in self.pi_init for v in d.values())
        out = np.zeros((G_tot, int(n_modes)))
        for g, masses in enumerate(self.pi_init):
            for nu_g in masses.values():     # a single entry when the mixture had one group
                nu_g = np.asarray(nu_g, dtype=float).ravel()
                out[g + ns, :min(out.shape[1], nu_g.size)] = nu_g[:out.shape[1]]
                break
        return out

    def bin_expected_means(self):
        """
        ``(K, G_tot)`` mean expression implied by the per-bin masses,
        ``sum_m nu[k, g, m] * ks[m, g] / c[g]``: it should match the observed mean of each bin (the
        mean-preservation constraint of the mixture), a direct correctness check of the binned fit.
        """
        nu = self.bin_masses()
        if nu is None:
            return None
        ks = np.asarray(self.a[:-1], dtype=float)          # (M, G_tot)
        c = np.asarray(self.a[-1], dtype=float)            # (G_tot,)
        M = min(nu.shape[2], ks.shape[0])
        means = np.zeros(nu.shape[:2])
        for m in range(M):
            means += nu[:, :, m] * ks[m][None, :]
        return means / np.maximum(c[None, :], 1e-16)

    # ------------------------------------------------------------------------------------------
    # The cycle: ordering -> E-step -> M-step -> alpha -> basins (PLAN.md §4.4)
    # ------------------------------------------------------------------------------------------
    def prepare_pseudotime(self, data, gene_names=None, roots=None, verb: bool = True):
        """
        Mixture (pooled, or on the bins if a pseudotime is already set), protein state of the cells and
        first pseudotime.  Entry point of :meth:`fit_pseudotime`.

        The degradation rates are read from ``data.var['d1']`` / ``data.var['d0']`` when present (the
        protocol of the benchmark figures), so that the prior time unit ``1 / d1`` is the real one.
        """
        ns = int(self.n_stimuli)
        var = getattr(data, "var", None)
        for key, row in (("d1", 1), ("d0", 0)):
            if var is None or key not in getattr(var, "columns", []):
                continue
            values = np.asarray(var[key].values, dtype=float)
            if values.size == np.shape(self.d)[1] - ns:
                self.d[row, ns:] = values
            elif verb:
                print(f"[pseudotime] var['{key}'] has {values.size} values for "
                      f"{np.shape(self.d)[1] - ns} genes: ignored")
        if roots is not None:
            roots = np.asarray(roots)
            if roots.dtype != bool:
                mask = np.zeros(self.n_cells_or_modes(), dtype=bool)
                mask[roots.astype(int)] = True
                roots = mask
            self.roots = roots
        roots_eff = self.roots if roots is None else roots
        if self.modes is None:
            kw = dict(gene_names=gene_names or [str(g) for g in data.var_names],
                      min_components=2, max_components=2, max_iter_kinetics=0, verb=verb)
            if self.tau is not None and self.bins is not None:
                self.fit_mixture_on_bins(data, tau=self.tau, roots=roots_eff, **kw)
            else:
                self.fit_mixture_pooled(data, **kw)
        if self.roots is None:
            self.roots = self.root_mask(data)
        self.initialise_cell_states(data)
        if self.tau is None:
            self.update_pseudotime(n_iter=0, roots=self.roots, verb=verb)
        return self

    def n_cells_or_modes(self) -> int:
        """Number of cells of the current mixture (or of the counts, before any fit)."""
        if self.modes is not None:
            return int(np.shape(self.modes)[0])
        if self.counts is not None:
            return int(np.shape(self.counts)[0])
        raise ValueError("no cells known yet: fit the mixture first")

    # --- duration of the pseudotime axis (PLAN.md §4.3) -----------------------------------------
    def prior_dtau(self) -> float:
        """Bin width of the time unit anchored on d1: ``pseudotime_relaxation / median(d1)``."""
        d1 = np.asarray(self.d[1, self.n_stimuli:], dtype=float)
        d1 = d1[d1 > 0]
        if d1.size == 0:
            return 1.0
        return float(self.pseudotime_relaxation / max(float(np.median(d1)), 1e-12))

    def resolve_duration(self, n_bins: int) -> float:
        """Total duration of the axis (model time) for ``n_bins`` bins, from the duration mode."""
        if self.pseudotime_total_duration is not None:
            return float(self.pseudotime_total_duration)
        return max(int(n_bins) - 1, 1) * self.prior_dtau()

    def apply_duration(self, n_bins: Optional[int] = None, verb: bool = True) -> float:
        """Set (and re-bin with) the duration of the axis resolved from the current mode."""
        if self.tau is None or self.bins is None:
            raise ValueError("no pseudotime to re-bin: call set_tau first")
        k = int(self.bins.n_bins if n_bins is None else n_bins)
        total = self.resolve_duration(k)
        if not np.isclose(total, self.bins.total_duration):
            self.pseudotime_total_duration = total
            self.set_total_duration(total)
            if verb:
                print(f"[pseudotime] duration of the axis: {total:.1f} model time units "
                      f"(dtau = {self.bins.dtau:.3f} per bin, {k} bins)")
        return total

    def select_duration_grid(self, data, verb: bool = True, n_iter: int = 1) -> float:
        """
        ``pseudotime_duration_mode='grid'``: run ``n_iter`` iterations for each candidate bin width
        (``pseudotime_dt_grid`` x one relaxation time) and keep the one with the lowest model error
        (the OT/trajectory fit is what identifies ``d1 * dtau``, PLAN.md §4.3).
        """
        import copy

        k = int(self.bins.n_bins)
        base = self.prior_dtau()
        best_error, best_factor = np.inf, 1.0
        for factor in self.pseudotime_dt_grid:
            trial = copy.deepcopy(self)
            trial.pseudotime_history = []
            trial.pseudotime_duration_mode = "fixed"
            trial.pseudotime_total_duration = (k - 1) * float(factor) * base
            trial.set_total_duration(trial.pseudotime_total_duration)
            trial.loop_pseudotime(data, verb=False, max_iter=n_iter)
            error = float(trial.pseudotime_history[-1]["error_after"])
            if verb:
                print(f"  [duration] dtau = {factor:g} / d1 = {factor * base:.2f} "
                      f"-> error {error:.4f}")
            if error < best_error:
                best_error, best_factor = error, float(factor)
        self.pseudotime_duration_mode = "grid"
        self.pseudotime_total_duration = (k - 1) * best_factor * base
        self.set_total_duration(self.pseudotime_total_duration)
        if verb:
            print(f"[pseudotime] duration selected: {best_factor:g} / d1 per bin "
                  f"({self.pseudotime_total_duration:.1f} in total)")
        return self.pseudotime_total_duration

    def _scaffolding(self, data_rna, n_full: Optional[int] = None) -> dict:
        """
        Everything the base E-step expects, from the current bins: the trajectory slots per bin
        (``N_full``, ``N_samples``), the real target cells of each interval and the initial cells (the
        roots, repeated).  Rebuilt at each iteration since the cells change bin.
        """
        times = np.asarray(self.bins.bin_times, dtype=float)
        labels = np.asarray(self.bins.bin_of_cell)
        K = int(times.size)
        counts = np.bincount(labels, minlength=K)
        if n_full is None:
            n_full = int(np.median(counts[1:])) if K > 1 else int(counts.max())
        n_full = int(max(10, min(n_full, 5000)))
        n_samples = int(min(max(1, int(self.batch_size_traj)), n_full))
        n_batches = int(np.ceil(n_full / n_samples))

        root_idx = np.flatnonzero(labels == 0)
        init_cells = minimal_repetition_choice(root_idx.size, n_full)

        batches = []
        for k in range(K - 1):
            idx = np.flatnonzero(labels == k + 1)
            if idx.size == 0:
                batches.append([idx for _ in range(n_batches)])
                continue
            k_nom = max(int(self.batch_size_traj_exp), n_samples)
            size = int(min(idx.size, max(k_nom, np.ceil(idx.size / n_batches))))
            total = n_batches * size
            reps = int(np.ceil(total / idx.size))
            tiled = np.concatenate([np.random.permutation(idx) for _ in range(reps)])[:total]
            batches.append([tiled[b * size:(b + 1) * size] for b in range(n_batches)])

        return dict(times=times, labels=labels, vect_t=times[labels], K=K, n_full=n_full,
                    n_samples=n_samples, n_batches=n_batches, init_cells=init_cells,
                    real_cell_batches=[batches], observed=np.ones((1, K), dtype=bool),
                    valid=np.ones(K * n_full, dtype=bool), n_cells=int(labels.size))

    def _estep(self, data_rna, S: dict, basal, inter, ks, s1, n_iter: int,
               verb: bool = False) -> dict:
        """E-step: the base trajectory inference, for every batch of the current bins."""
        ns = int(self.n_stimuli)
        N_tot, K = S["n_full"], S["K"]
        G_tot = int(data_rna.shape[1])
        y_prot = np.zeros((K * N_tot, G_tot))
        y_kon = np.zeros_like(y_prot)
        y_rna = np.zeros_like(y_prot)
        y_alpha = np.random.uniform(0.1, 0.9, size=(max(K - 1, 1), N_tot, G_tot - ns))
        y_samples = np.zeros(K * N_tot, dtype=int)
        R_opt = np.zeros(K * N_tot)
        y_real = np.full(K * N_tot, -1, dtype=int)
        to_keep = np.zeros(S["n_cells"], dtype=bool)
        formodes = np.zeros_like(np.asarray(self.modes, dtype=float))
        samples_id = np.zeros(S["n_cells"], dtype=int)

        offset = 0
        while offset < N_tot:
            n_tmp = int(min(S["n_samples"], N_tot - offset))
            init_cells = [S["init_cells"][offset:offset + n_tmp]]
            batch_idx = [int(min(offset // S["n_samples"], S["n_batches"] - 1))]
            (y_prot, formodes, y_rna, y_kon, y_alpha, y_samples, R_opt, to_keep) = \
                self.estimate_trajectories_given_model(
                    S["vect_t"], S["times"], samples_id, [0], data_rna,
                    y_prot, formodes, y_kon, y_rna, y_alpha, y_samples,
                    basal, inter, s1, ks, init_cells, R_opt, to_keep,
                    offset_init=[offset], N_full=[N_tot], N_samples=[n_tmp],
                    n_iter=n_iter, intensity_prior=self.intensity_prior,
                    real_cell_batches=S["real_cell_batches"], batch_idx=batch_idx,
                    sim_real_idx=y_real, observed=S["observed"],
                    reg_factor=self._reg_factor(n_iter, None))
            offset += n_tmp

        return dict(y_prot=y_prot, y_kon=y_kon, y_rna=y_rna, y_alpha=y_alpha, y_samples=y_samples,
                    R_opt=R_opt, y_real=y_real, to_keep=to_keep, formodes=formodes,
                    vect_t_sim=np.repeat(S["times"], N_tot))

    def _mstep(self, S: dict, E: dict, ks, n_iter: int, verb: bool = False):
        """M-step: the base network inference (per-gene L-BFGS on the burst-rate target)."""
        G_tot = int(self.inter.shape[-1])
        _, prev_cols = active_regulators(self.ref_network, np.zeros((G_tot, G_tot)), False)
        weight_prev = float(self.pseudotime_weight_prev)
        if weight_prev > 0:
            # the flow-matching states of _fill_prev_prot are not built in the pseudotime cycle yet
            raise ValueError("pseudotime_weight_prev must be 0 in this version (no flow-matching "
                             "states along the pseudotime bins)")
        # the base fit indexes the flow-matching states even with a zero weight: placeholder
        y_prot_mod = PrevProt.zeros(prev_cols, E["vect_t_sim"].size)

        def fit_on(sels):
            return inference_network_multi(
                sels, E["y_samples"], E["y_kon"], E["y_prot"], y_prot_mod, ks,
                n_stimuli=int(self.n_stimuli), samples_id=[0], ref_network=self.ref_network,
                basal_init=self.basal, inter_init=self.inter,
                basal_ref=np.zeros((1, G_tot)), inter_ref=np.zeros((G_tot, G_tot)),
                scale=self.scale_pen, weight_prev=weight_prev, final=0,
                constrain_basal_uniform=self.constrain_basal_uniform,
                hard_forcing_ref=False, ref_constraint_pct=self.ref_constraint_pct,
                seuil_zero_min_ref=self.seuil_zero_min_ref)

        return self._fit_theta_averaged(fit_on, E["vect_t_sim"], E["y_samples"], None,
                                        self.n_network_fits,
                                        self._n_params_per_target(prev_cols, 1),
                                        valid=S["valid"])

    def _update_alpha(self, S: dict, E: dict, ks, s1, basal, inter) -> dict:
        """Switching fractions of each interval (base inference_alpha), from the current trajectories."""
        ns = int(self.n_stimuli)
        K, N_tot = S["K"], S["n_full"]
        if K < 2:
            return E
        s1r = s1_rows(s1, E["y_samples"])
        modes = self.adaptive_shrinkage(E["y_rna"][:, ns:] * s1r, E["y_kon"][:, ns:]) / np.maximum(s1r, 1e-16)
        kon_vec = E["y_kon"].copy()
        kon_vec[:, ns:] = self._kon_ref_per_sample(E["y_prot"], ks, inter, basal, samples_id=[0],
                                                   samples_data=E["y_samples"])[:, ns:]
        alpha = E["y_alpha"]
        for a in range(K - 1):
            b = a + 1
            r0 = np.arange(a * N_tot, (a + 1) * N_tot)
            r1 = np.arange(b * N_tot, (b + 1) * N_tot)
            alpha[a] = inference_alpha(
                self.d[1, ns:], s1, alpha[a],
                E["y_kon"][r0], kon_vec[r0], E["y_prot"][r0], E["y_rna"][r0],
                E["y_kon"][r1], kon_vec[r1], E["y_prot"][r1], E["y_rna"][r1],
                modes[r0], modes[r1], basal, inter, ks, S["times"][b] - S["times"][a],
                tol=self.alpha_threshold, n_pas=self.n_pas,
                samples_data=E["y_samples"][r0],
                stim_vals=self._stim_schedule.per_cell(S["times"][b], E["y_samples"][r0]),
                scale_proteins=self.scale_proteins)
        return E

    def _update_basins(self, S: dict, E: dict, ks, basal, inter, n_iter: int) -> None:
        """
        Basin / mode update of each gene (the per-gene EMD of ``loop_trajectories``), with the
        per-bin target masses of the binned mixture and the ramped ``force_basins`` (§10.6).
        """
        ns = int(self.n_stimuli)
        G_tot = int(np.shape(self.modes)[1])
        labels, times, K = S["labels"], S["times"], S["K"]
        force = self.force_basins_at(n_iter)
        weight_prob = max(0.96 ** (n_iter - 1), 0.1)
        masses = self.bin_masses()
        if masses is None:
            return
        # mode amplitudes of each gene (as in loop_trajectories: ks is (G_tot, n_modes), or per sample)
        ks_gene = np.asarray(ks).max(axis=0) if np.ndim(ks) == 3 else np.asarray(ks)
        n_cells = int(self.proba_init.shape[0])
        samples_data = np.zeros(n_cells, dtype=int)
        kon_formodes = np.array(E["formodes"], dtype=float, copy=True)
        kon_formodes[:, ns:] = self._kon_ref_per_sample(E["formodes"], ks, inter, basal,
                                                        samples_id=[0],
                                                        samples_data=samples_data)[:, ns:]
        to_keep = np.asarray(E["to_keep"], dtype=float)

        def run_gene(g: int):
            l_max = 1 + int(np.argmax(ks_gene[g]))
            obj = ks_gene[g, :l_max][None, :]
            basins = np.zeros(n_cells, dtype=np.int8)
            modes_g = np.zeros(n_cells)
            for k in range(K):
                idx = labels == k
                n_i = int(idx.sum())
                if n_i == 0:
                    continue
                proba = np.array(self.proba_init[idx, g, :l_max], dtype=float)
                post = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-16)
                proba = proba / np.maximum(proba.max(axis=1, keepdims=True), 1e-16)
                nu = (np.asarray(masses[k, g, :l_max], dtype=float) * force
                      + post.sum(axis=0) * (1.0 - force))
                nu = np.maximum(nu, 0.0)
                if nu.sum() <= 0:
                    nu = post.sum(axis=0) / max(post.sum(), 1e-16)
                nu = nu / nu.sum()

                mu = np.ones(n_i) / n_i
                diff_k = np.maximum(1.0 - np.abs(kon_formodes[idx, g][:, None] - obj), 1e-3)
                diff_k /= np.maximum(diff_k.max(axis=1, keepdims=True), 1e-16)
                dist = -(np.log(np.maximum(proba, 1e-16))
                         + (1.0 - weight_prob) * to_keep[idx][:, None] * np.log(diff_k))
                dist = np.clip(dist, 0, 100)
                idx_bin = np.argmax(ot.emd(mu, nu, dist, numItermax=int(1e7)), axis=1)
                basins[idx] = idx_bin
                modes_g[idx] = obj[0, idx_bin]
            return basins, modes_g

        results = Parallel(n_jobs=-1)(delayed(run_gene)(g) for g in range(ns, G_tot))
        for i, g in enumerate(range(ns, G_tot)):
            self.basins[:, g], self.modes[:, g] = results[i]

    def _pseudo_error(self, E: dict, S: dict, ks, basal, inter) -> float:
        """Model error of the trajectories (the stopping criterion of the base loop)."""
        valid = S["valid"]
        return self._count_errors_per_sample(E["y_prot"][valid], E["y_kon"][valid], ks, inter, basal,
                                             samples_id=[0], samples_data=E["y_samples"][valid])

    def loop_pseudotime(self, data, verb: bool = True, roots=None,
                        max_iter: Optional[int] = None) -> "PseudotimeNetworkModel":
        """
        The cycle: (mixture on the bins) -> ordering -> E-step -> M-step -> alpha -> basins, until the
        pseudotime and the error are stable (PLAN.md §4.4).

        At the first iteration the network is zero (hence the plausibility edge weights, and a random
        network term in the OT cost); from the second one the ordering uses the fitted network and the
        protein trajectories of the previous E-step (§0.3).
        """
        data_rna = self._parse_input(data)
        G_tot = int(data_rna.shape[1])
        ns = int(self.n_stimuli)
        max_iter = int(self.max_iter if max_iter is None else max_iter)
        count_max = int(max(self.pseudotime_count_max, 1))
        gene_names = [str(g) for g in data.var_names]

        if self.basal is None or np.shape(self.basal)[-1] != G_tot:
            self.basal = np.zeros((1, G_tot))
        if self.inter is None or np.shape(self.inter)[-1] != G_tot:
            self.inter = np.zeros((G_tot, G_tot))
        self.basal_tmp, self.inter_tmp = self.basal.copy(), self.inter.copy()

        self.pseudotime_history = []
        prev_error, count_end = np.inf, 0
        E = S = None
        # the time unit of the axis (prior on d1, or the given / grid-selected duration)
        if self.pseudotime_duration_mode in ("prior", "fixed"):
            self.apply_duration(verb=verb)
        for n_iter in range(1, max_iter + 1):
            # (a) mixture on the bins of the previous pseudotime (guard 2; always at the first iteration)
            if self.refit_mixture_if_due(data, n_iter=n_iter, verb=verb, gene_names=gene_names):
                self.initialise_cell_states(data)
            ks, s1, _ = self.mixture_terms()

            # (b) ordering with the current network / proteins -> new bins
            prev_tau = None if self.tau is None else np.array(self.tau, dtype=float, copy=True)
            if not self.pseudotime_fixed_tau:
                self.update_pseudotime(n_iter=n_iter, roots=self.roots, verb=verb)
            elif self.bins is None:
                raise ValueError("pseudotime_fixed_tau needs bins: call set_tau before the loop")

            # (c) E-step on the new bins
            S = self._scaffolding(data_rna)
            self._stim_schedule = self._build_stimulus_schedule(S["times"])
            E = self._estep(data_rna, S, self.basal, self.inter, ks, s1, n_iter, verb=verb)
            error_before = self._pseudo_error(E, S, ks, self.basal, self.inter)

            # (d) M-step
            basal, inter, _, _ = self._mstep(S, E, ks, n_iter, verb=verb)
            self.basal, self.inter = basal, inter
            error_after = self._pseudo_error(E, S, ks, basal, inter)

            # (e) switching fractions, (f) basins and modes
            if self.pseudotime_update_alpha:
                E = self._update_alpha(S, E, ks, s1, basal, inter)
            self._update_basins(S, E, ks, basal, inter, n_iter)

            # (g) the protein state of each cell feeds the ordering of the next iteration
            self.cell_prot = np.array(E["formodes"], dtype=float, copy=True)

            rho = None if prev_tau is None else spearman(self.tau, prev_tau)
            weight_kind = (self.pseudotime_diagnostics or {}).get("edge_weight")
            self.pseudotime_history.append(dict(
                n_iter=n_iter, n_bins=int(S["K"]), n_full=int(S["n_full"]),
                error_before=float(error_before), error_after=float(error_after),
                spearman_prev=None if rho is None else float(rho),
                dtau=float(self.bins.dtau), force_basins=float(self.force_basins_at(n_iter)),
                total_duration=float(self.bins.total_duration),
                edge_weight=weight_kind))

            improvement = prev_error - error_after
            stable = rho is not None and rho >= float(self.pseudotime_stable)
            converged = n_iter > 1 and abs(improvement) < 1e-4 * max(abs(prev_error), 1e-12)
            count_end = count_end + 1 if (stable and converged) else 0
            if verb:
                print(f"[pseudotime] iter {n_iter}: K={S['K']} bins, N={S['n_full']}/bin, "
                      f"error {error_before:.4f} -> {error_after:.4f}, "
                      f"tau vs previous {('%.3f' % rho) if rho is not None else '-'}, "
                      f"weight={weight_kind}, "
                      f"force_basins={self.force_basins_at(n_iter):.2f}")
            prev_error = error_after
            if count_end >= count_max:
                if verb:
                    print(f"[pseudotime] converged after {n_iter} iterations")
                break

        # --- the state the downstream stages (growth, degradations, report) expect ---
        self.times_data = E["vect_t_sim"]
        self.prot, self.kon_beta, self.rna = E["y_prot"], E["y_kon"], E["y_rna"]
        self.alpha, self.samples_data = E["y_alpha"], E["y_samples"]
        self.traj_real_idx, self.traj_valid = E["y_real"], S["valid"]
        self.traj_cell_types = None
        self.R_opt, self.couplings = E["R_opt"], None
        self._last_scaffolding = S

        if self.pseudotime_growth:
            self.growth_pass_pseudotime(data_rna, S, ks, s1, verb=verb)
        return self

    def growth_pass_pseudotime(self, data_rna, S: dict, ks, s1, verb: bool = True) -> None:
        """
        Growth pass of the base loop (``_growth_log_mass``), on the final bins: the net proliferation
        rate of each trajectory state from the density of cells along the pseudotime (PLAN.md §6).
        """
        ns = int(self.n_stimuli)
        N_tot = S["n_full"]
        R_opt = np.full(S["K"] * N_tot, np.nan)
        self.couplings = []
        offset = 0
        while offset < N_tot:
            n_tmp = int(min(S["n_samples"], N_tot - offset))
            init_cells = [S["init_cells"][offset:offset + n_tmp]]
            batch_idx = [int(min(offset // S["n_samples"], S["n_batches"] - 1))]
            out = self.estimate_trajectories_given_model(
                S["vect_t"], S["times"], np.zeros(S["n_cells"], dtype=int), [0], data_rna,
                self.prot, np.array(self.modes, dtype=float, copy=True), self.kon_beta, self.rna,
                self.alpha, self.samples_data, self.basal, self.inter, s1, ks, init_cells,
                R_opt, np.ones(S["n_cells"], dtype=bool),
                offset_init=[offset], N_full=[N_tot], N_samples=[n_tmp],
                n_iter=1, intensity_prior=0.0, real_cell_batches=S["real_cell_batches"],
                batch_idx=batch_idx, sim_real_idx=np.array(self.traj_real_idx, copy=True),
                observed=S["observed"], growth_only=True)
            R_opt = out[6]
            offset += n_tmp
        self.R_opt = R_opt
        if verb and np.any(np.isfinite(R_opt)):
            print(f"[pseudotime] growth pass: net rate per unit pseudotime "
                  f"mean {np.nanmean(R_opt):+.4f} (min {np.nanmin(R_opt):+.4f}, "
                  f"max {np.nanmax(R_opt):+.4f})")

    def fit_pseudotime(self, data, gene_names=None, roots=None, verb: bool = True,
                       refine: bool = False, max_iter: Optional[int] = None):
        """
        Full run: mixture + first pseudotime, the cycle, then (``refine``) the degradation rates /
        proliferation model of the base (``refine_network_degradations``) in the pseudotime unit.
        """
        self.prepare_pseudotime(data, gene_names=gene_names, roots=roots, verb=verb)
        if self.pseudotime_duration_mode == "grid":
            self.select_duration_grid(data, verb=verb, n_iter=1)
        self.loop_pseudotime(data, verb=verb, roots=self.roots, max_iter=max_iter)
        if refine:
            self.refine_network_degradations(verb=verb)
        return self
