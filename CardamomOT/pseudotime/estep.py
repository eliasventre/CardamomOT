r"""
E-step of the prototype, with the time gap of each pair chosen inside the step
(PROTOTYPE_PLAN.md §4).

A reduced copy of ``NetworkModel.estimate_trajectories_given_model`` (``model/base.py``), which
stays untouched:

* **kept**: one sample, the cost ``my_otdistance``, ``_solve_ot``, the marginals
  ``_traj_marginals``, the flows ``find_next_prot`` / ``find_next_prot_dilution`` (dilution with
  birth rates), ``to_keep_for_update``, ``sim_real_idx``, the root initial state, the stimulus;
* **dropped**: several samples, unobserved timepoints, ``alpha_pool`` (test set), ``growth_only``
  (the growth pass keeps calling the base function); lineage and cell-type transition constraints
  are left for later (§9);
* **not supported**: ``mrna_bridge = True``.

Two changes of structure:

1. **pairs outside, batches inside**: the gap of pair k is chosen once (``gaps.py``) from the
   source states of *all* batches at bin k, then every batch is transported with it.  With one
   batch the order of the random draws is the base one, so the output is identical to the base
   E-step when the search is off.
2. **switching fractions carried across iterations** (§4.7): a newly propagated state takes the
   switching fraction of the closest state of the previous iteration whose time lies near the
   time of its bin (``AlphaPool``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial.distance import cdist

from ..inference.network import gene_weights, sigmoid_amplitudes, sigmoid_target
from ..inference.trajectory import (birth_alpha, find_next_prot, find_next_prot_dilution, inter_of,
                                    ks_of, my_otdistance, s1_of, s1_rows)
from ..model.base import _rows
from .gaps import GapSearch, PairInputs, choose_gap, damp, gap_curves

PROPAGATE = ("chosen", "given")


# ----------------------------------------------------------------------------------------------
# Switching fractions of the previous iteration (§4.7)
# ----------------------------------------------------------------------------------------------
@dataclass
class AlphaPool:
    """States of the previous iteration with the switching fraction of the step leaving them."""

    prot: np.ndarray     # (M, G) proteins (genes)
    kon: np.ndarray      # (M, G) modes (genes)
    alpha: np.ndarray    # (M, G) switching fraction of the step leaving the state
    time: np.ndarray     # (M,) time of the bin of the state

    @classmethod
    def from_arrays(cls, prot, kon, alpha, times, n_full: int, ns: int) -> "AlphaPool":
        """
        Pool of the trajectories ``prot`` / ``kon`` ((K * n_full, G_tot)), switching fractions
        ``alpha`` ((K - 1, n_full, G)) and bin times ``times`` (K,): every state of bins 0..K-2.
        """
        times = np.asarray(times, dtype=float)
        K = int(times.size)
        alpha = np.asarray(alpha, dtype=float)
        if K < 2:
            raise ValueError("a pool needs at least two bins")
        rows = np.arange((K - 1) * int(n_full))
        return cls(prot=np.asarray(prot, dtype=float)[rows, ns:],
                   kon=np.asarray(kon, dtype=float)[rows, ns:],
                   alpha=alpha[:K - 1].reshape(-1, alpha.shape[-1]),
                   time=np.repeat(times[:K - 1], int(n_full)))

    @classmethod
    def from_estep(cls, E: dict, n_full: int, ns: int) -> "AlphaPool":
        """Pool of an E-step result (call it after the switching fractions were re-estimated)."""
        return cls.from_arrays(E["y_prot"], E["y_kon"], E["y_alpha"], E["times"], n_full, ns)

    def match(self, prot, kon, t_lo: float, t_hi: float, G_tot: int) -> Optional[np.ndarray]:
        """
        Switching fractions (n, G) of the closest old state (weighted L1 on proteins and modes, the
        weights of the base: 1/G_tot and (G_tot - 1)/G_tot) among the states of time in
        ``[t_lo, t_hi]``; None when no state is in the window.
        """
        cand = np.flatnonzero((self.time >= t_lo) & (self.time <= t_hi))
        if cand.size == 0:
            return None
        w_p, w_k = 1.0 / G_tot, (G_tot - 1.0) / G_tot
        d = cdist(np.hstack([prot * w_p, kon * w_k]),
                  np.hstack([self.prot[cand] * w_p, self.kon[cand] * w_k]), "cityblock")
        return self.alpha[cand[np.argmin(d, axis=1)]]


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def _batches(S: dict) -> List[tuple]:
    """(offset, size, batch index, initial cells) of each batch of trajectory slots (as ``_estep``)."""
    out, offset = [], 0
    N_tot, n_samples, n_batches = S["n_full"], S["n_samples"], S["n_batches"]
    while offset < N_tot:
        n_tmp = int(min(n_samples, N_tot - offset))
        out.append((offset, n_tmp, int(min(offset // n_samples, n_batches - 1)),
                    np.asarray(S["init_cells"][offset:offset + n_tmp])))
        offset += n_tmp
    return out


def _subsample(rng, idx: np.ndarray, groups: np.ndarray, n_max: int) -> np.ndarray:
    """At most ``n_max`` entries of ``idx``, stratified by ``groups`` (proportional allocation)."""
    if idx.size <= n_max:
        return idx
    out = []
    labels, counts = np.unique(groups, return_counts=True)
    quota = np.maximum(1, np.floor(counts * n_max / idx.size)).astype(int)
    for g, q in zip(labels, quota):
        members = idx[groups == g]
        out.append(rng.choice(members, size=min(q, members.size), replace=False))
    return np.sort(np.concatenate(out))


def stimulus_per_bin(model, times: Sequence[float]):
    """(K, ns) stimulus value of each bin and the schedule, from the default schedule on ``times``."""
    sched = model._build_stimulus_schedule(np.asarray(times, dtype=float))
    return np.array([np.asarray(sched[t], dtype=float) for t in sched._times]), sched


# ----------------------------------------------------------------------------------------------
# The E-step
# ----------------------------------------------------------------------------------------------
def estep_with_gaps(model, data_rna, S: dict, basal, inter, ks, s1, n_iter: int,
                    gaps: Sequence[float], y_alpha=None, search: Optional[GapSearch] = None,
                    pool: Optional[AlphaPool] = None, propagate: str = "chosen",
                    reg_factor: Optional[float] = None, verb: bool = False) -> dict:
    """
    Trajectories between the consecutive bins of the scaffolding ``S`` (``_scaffolding``), with the
    gap of each pair chosen by ``search`` (None: the given ``gaps`` are used, as the base E-step).

    Parameters
    ----------
    gaps : (K - 1,) current gaps (starting values of the search, reference of its scales and
        prior of the neighbour smoothing).
    y_alpha : (K - 1, n_full, G) switching fractions; None draws them uniformly in [0.1, 0.9]
        (as ``_estep``, same random stream).
    pool : switching fractions of the previous iteration (§4.7); None skips the re-assignment
        (as ``_estep``, which starts from zero proteins).
    propagate : ``'chosen'`` (the trajectories use the chosen gaps) or ``'given'`` (the search is
        run and recorded, but the trajectories use ``gaps``: the sources stay fixed, gate G0).

    Returns the arrays of ``_estep`` plus ``times``, ``gaps`` (the gaps used), ``gap_diag`` (one
    dict per pair: curves, chosen gap, flags) and ``vect_t_sim``.
    """
    if getattr(model, "mrna_bridge", False):
        raise NotImplementedError("mrna_bridge = True is not supported by the prototype E-step")
    if propagate not in PROPAGATE:
        raise ValueError(f"propagate must be one of {PROPAGATE}")
    ns = int(model.n_stimuli)
    N_tot, K = int(S["n_full"]), int(S["K"])
    labels = np.asarray(S["labels"])
    G_tot = int(data_rna.shape[1])
    gaps_in = np.asarray(gaps, dtype=float).ravel()
    if gaps_in.size != K - 1:
        raise ValueError(f"{gaps_in.size} gaps for {K} bins")
    if np.any(gaps_in <= 0):
        raise ValueError("the gaps must be positive")
    gaps_used = gaps_in.copy()

    # arrays of _estep (same order of the random draws)
    y_prot = np.zeros((K * N_tot, G_tot))
    y_kon = np.zeros_like(y_prot)
    y_rna = np.zeros_like(y_prot)
    if y_alpha is None:
        y_alpha = np.random.uniform(0.1, 0.9, size=(max(K - 1, 1), N_tot, G_tot - ns))
    y_alpha = np.array(y_alpha, dtype=float, copy=True)
    y_samples = np.zeros(K * N_tot, dtype=int)
    R_opt = np.zeros(K * N_tot)
    y_real = np.full(K * N_tot, -1, dtype=int)
    to_keep = np.zeros(int(S["n_cells"]), dtype=bool)
    formodes = np.zeros_like(np.asarray(model.modes, dtype=float))
    samples_id = np.zeros(int(S["n_cells"]), dtype=int)
    modes = np.asarray(model.modes)
    d1 = model.d[1, ns:]
    batches = _batches(S)
    if reg_factor is None:
        reg_factor = model._reg_factor(n_iter, None)

    # stimulus of each bin: the default schedule (bin 0 off, the others on), indexed by bin
    stim_bin, sched = stimulus_per_bin(model, np.concatenate([[0.0], np.cumsum(gaps_in)]))
    for k in range(K):
        sl = slice(k * N_tot, (k + 1) * N_tot)
        val = stim_bin[k]
        formodes[labels == k, :ns] = val * model.scale_proteins
        y_rna[sl, :ns] = val * model.scale_mrnas
        y_prot[sl, :ns] = val * model.scale_proteins
        y_kon[sl, :ns] = (val >= 0.5)

    # initial states (the roots), batch by batch as the base (adaptive_shrinkage_init is per batch)
    start_cells = labels == 0
    global_idx = np.flatnonzero(start_cells)
    for offset, n_tmp, _, sel in batches:
        rows = np.arange(offset, offset + n_tmp)
        y_kon[rows, ns:] = modes[start_cells][sel, ns:]
        y_rna[rows, ns:] = data_rna[start_cells][sel, ns:]
        y_real[rows] = global_idx[sel]
        s1_traj = s1_rows(s1, y_samples)
        y_prot[rows, ns:] = model.adaptive_shrinkage_init(
            y_rna[rows, ns:] * _rows(s1_traj, rows), y_kon[rows, ns:]) * model._dilution(y_real[rows])[1]
        if n_iter == 1:
            s1_cells = s1_rows(s1, samples_id)
            formodes[start_cells, ns:] = model.adaptive_shrinkage_init(
                data_rna[start_cells, ns:] * _rows(s1_cells, start_cells), modes[start_cells, ns:]) \
                * model._dilution(global_idx)[1]

    s1_s, ks_s = s1_of(s1, 0), ks_of(ks, 0)
    kb_s, w_s = sigmoid_amplitudes(ks_s), gene_weights(ks_s)[ns:]
    basal_s = basal[0] if np.ndim(basal) == 2 else basal
    inter_s = inter_of(inter, 0)
    rna_to_x = model.scale_proteins / model.fact_simple
    run_search = search is not None and n_iter >= int(search.start)
    grid = search.grid(d1) if search is not None else None
    log_prior = float(np.log(model.pseudotime_relaxation / max(float(np.median(d1[d1 > 0])), 1e-12))) \
        if search is not None else 0.0
    gap_diag: List[Dict] = []

    for k in range(K - 1):
        b = k + 1
        batch_cells = S["real_cell_batches"][0][k]
        times = np.concatenate([[0.0], np.cumsum(gaps_used)])

        # (i) gap of the pair, from the sources of every batch and the cells of bin k+1
        if run_search:
            diag = _search_pair(model, data_rna, y_prot, y_kon, y_rna, y_alpha, y_real, k, N_tot,
                                batches, batch_cells, s1_s, kb_s, w_s, basal_s, inter_s, d1,
                                stim_bin[b], search, grid, log_prior, gaps_in, gaps_used, n_iter)
            gap_diag.append(diag)
            if propagate == "chosen":
                gaps_used[k] = diag["gap_used"]
                times = np.concatenate([[0.0], np.cumsum(gaps_used)])
        delta_t = float(gaps_used[k])

        # (ii) every batch with this gap (the body of the base E-step)
        for offset, n_tmp, batch_idx, _ in batches:
            cell_idx = batch_cells[batch_idx]
            N_cells = len(cell_idx)
            if not (n_tmp and N_cells):
                continue
            current = np.arange(k * N_tot + offset, k * N_tot + offset + n_tmp)
            nxt = np.arange(b * N_tot + offset, b * N_tot + offset + n_tmp)
            a_idx = np.arange(offset, offset + n_tmp)
            prot_init = y_prot[current, ns:]
            alpha_init = y_alpha[k, a_idx]
            mode_init = model.adaptive_shrinkage(y_rna[current, ns:] * s1_s, y_kon[current, ns:]) / s1_s
            mode_end = model.adaptive_shrinkage(data_rna[cell_idx, ns:] * s1_s, modes[cell_idx, ns:]) / s1_s
            rate_src, c_src, b_src = model._dilution(y_real[current])
            rate_tgt, c_tgt, b_tgt = model._dilution(cell_idx)
            b_src = np.zeros(0) if b_src is None else b_src
            b_tgt = np.zeros(0) if b_tgt is None else b_tgt
            cost = my_otdistance(
                y_kon[current, ns:], modes[cell_idx, ns:], prot_init,
                y_rna[current, ns:], data_rna[cell_idx, ns:], mode_init, mode_end, alpha_init,
                s1_s, d1, delta_t, basal_s, inter_s,
                sigmoid_target(modes[cell_idx, ns:], kb_s[ns:, 0], kb_s[ns:, 1]), w_s,
                n_iter=n_iter, intensity_prior=model.intensity_prior, n_stimuli=ns,
                stim_vals=np.asarray(stim_bin[b], dtype=np.float64),
                scale_proteins=model.scale_proteins, weight_fixed=-1.0,
                b_init=b_src, b_end=b_tgt, alpha_tol=model.alpha_threshold,
                d0=np.zeros(0), rna_to_x=rna_to_x, beta=np.zeros(0), z=np.zeros((0, 0)))
            src_real = y_real[current]
            cost = cost - cost.min(axis=0, keepdims=True)
            tmp = float(np.median(cost))
            if tmp <= 1e-12:
                tmp = float(np.mean(cost)) if np.mean(cost) > 1e-12 else 1.0
            it_reg = n_iter
            reg = max(model.init_entropic_noise * tmp * (1 / it_reg) ** (1 - 1 / it_reg), 1e-6 * tmp) * reg_factor
            ramp_uot = model._uot_ramp(it_reg, 0)
            m_ref = max(model.min_n_loops, 1)
            reg_ref = max(model.init_entropic_noise * tmp * (1 / m_ref) ** (1 - 1 / m_ref), 1e-6 * tmp)
            mu, nu, reg_m = model._traj_marginals(src_real, cell_idx, delta_t, reg, tmp, ramp_uot, reg_ref)
            coupling = model._solve_ot(mu, nu, cost, reg, reg_m)

            cdf = np.cumsum(coupling, axis=1)
            u = np.random.random(n_tmp) * cdf[:, -1]
            m_idx = np.minimum((cdf < u[:, None]).sum(axis=1), N_cells - 1)
            tgt = cell_idx[m_idx]
            if b_tgt.shape[0] == 0:
                next_prot = find_next_prot(
                    rate_src, prot_init, y_rna[current, ns:], data_rna[tgt, ns:],
                    mode_init * c_src, mode_end[m_idx] * c_src, alpha_init, s1_s, delta_t)
            else:
                rate_end, c_end = rate_tgt[m_idx], c_tgt[m_idx]
                alpha_b = birth_alpha(alpha_init, y_kon[current, ns:], modes[tgt, ns:],
                                      model.alpha_threshold)
                next_prot = find_next_prot_dilution(
                    rate_src, rate_end, prot_init, mode_init * c_src, mode_end[m_idx] * c_end,
                    alpha_init, alpha_b, s1_s, delta_t)

            y_kon[nxt] = modes[tgt]
            y_rna[nxt, ns:] = data_rna[tgt, ns:]
            y_prot[nxt, ns:] = next_prot
            formodes[tgt, ns:] = next_prot
            to_keep[tgt] = 1
            y_real[nxt] = tgt

            # switching fraction of the next step: closest previous-iteration state near this time
            if pool is not None and b < K - 1:
                after = gaps_used[b]
                alpha_new = pool.match(next_prot, y_kon[nxt, ns:], times[b] - delta_t / 2,
                                       times[b] + after / 2, G_tot)
                if alpha_new is None:
                    alpha_new = np.random.uniform(0.1, 0.9, size=(n_tmp, G_tot - ns))
                y_alpha[b, a_idx] = alpha_new

    times = np.concatenate([[0.0], np.cumsum(gaps_used)])
    model._stim_schedule = model._build_stimulus_schedule(times)
    if verb and run_search:
        print("[prototype] gaps: " + ", ".join(f"{g:.2f}" for g in gaps_used))
    return dict(y_prot=y_prot, y_kon=y_kon, y_rna=y_rna, y_alpha=y_alpha, y_samples=y_samples,
                R_opt=R_opt, y_real=y_real, to_keep=to_keep, formodes=formodes,
                vect_t_sim=np.repeat(times, N_tot), times=times, gaps=gaps_used,
                gap_diag=gap_diag)


def _search_pair(model, data_rna, y_prot, y_kon, y_rna, y_alpha, y_real, k, N_tot, batches,
                 batch_cells, s1_s, kb_s, w_s, basal_s, inter_s, d1, stim, search: GapSearch, grid,
                 log_prior, gaps_in, gaps_used, n_iter) -> dict:
    """Gap of pair k (§4.2–§4.3): subsample, curves on the grid, choice with the penalties."""
    ns = int(model.n_stimuli)
    modes = np.asarray(model.modes)
    rng = np.random.default_rng(int(search.seed) + 1000 * int(n_iter) + int(k))
    # sources: slots of bin k of every batch, stratified by batch
    slot = np.arange(N_tot)
    slot_batch = np.zeros(N_tot, dtype=int)
    for i, (offset, n_tmp, _, _) in enumerate(batches):
        slot_batch[offset:offset + n_tmp] = i
    src_slots = _subsample(rng, slot, slot_batch, int(search.max_cells))
    rows = k * N_tot + src_slots
    # targets: the cells of bin k+1 (union of the target batches)
    tgt_all = np.unique(np.concatenate([np.asarray(c) for c in batch_cells]))
    tgt = _subsample(rng, tgt_all, np.zeros(tgt_all.size, dtype=int), int(search.max_cells))

    _, _, b_src = model._dilution(y_real[rows])
    _, _, b_tgt = model._dilution(tgt)
    inp = PairInputs(
        P0=y_prot[rows, ns:],
        mode_init=model.adaptive_shrinkage(y_rna[rows, ns:] * s1_s, y_kon[rows, ns:]) / s1_s,
        alpha=y_alpha[k, src_slots], rna_src=y_rna[rows, ns:], kon_src=y_kon[rows, ns:],
        mode_end=model.adaptive_shrinkage(data_rna[tgt, ns:] * s1_s, modes[tgt, ns:]) / s1_s,
        tau_end=sigmoid_target(modes[tgt, ns:], kb_s[ns:, 0], kb_s[ns:, 1]),
        rna_tgt=data_rna[tgt, ns:], kon_tgt=modes[tgt, ns:], s1=np.asarray(s1_s, dtype=float),
        d1=np.asarray(d1, dtype=float), basal=np.asarray(basal_s, dtype=float),
        inter=np.asarray(inter_s, dtype=float), w=np.asarray(w_s, dtype=float),
        stim=np.asarray(stim, dtype=float), ns=ns, scale_proteins=float(model.scale_proteins),
        b_src=b_src, b_tgt=b_tgt, alpha_tol=float(model.alpha_threshold),
        k_low=np.asarray(kb_s[ns:, 0], dtype=float), k_high=np.asarray(kb_s[ns:, 1], dtype=float))

    spec = search.formulation
    if not isinstance(spec, str) and (search.lambda_prot or search.lambda_rna):
        spec = (float(search.lambda_prot), float(search.lambda_rna))
    formulations = {"search": spec}
    formulations.update(search.extra)
    null_inp = None
    if any(isinstance(f, str) and f == "self_null" for f in formulations.values()):
        null_inp = _self_null_inputs(model, data_rna, tgt_all, rng, int(search.max_cells), inp)
    ref = float(gaps_in[k]) if search.scale_ref == "current" else float(np.exp(log_prior))
    curves = gap_curves(inp, grid, ref_delta=ref, formulations=formulations, null_inp=null_inp)

    neighbours = []
    if k > 0:
        neighbours.append(float(np.log(gaps_used[k - 1])))
    if k + 1 < gaps_in.size:
        neighbours.append(float(np.log(gaps_in[k + 1])))
    choice = choose_gap(grid, curves["search"], log_prior, search.prior_weight, neighbours,
                        search.smooth_weight, search.flat_tol)
    if choice["flat"]:
        gap_used = float(gaps_in[k])                     # not identified: keep the current gap
    else:
        gap_used = float(np.exp(damp(np.log(gaps_in[k]), choice["log_gap"], search.damping)))
    return dict(pair=int(k), grid=np.asarray(grid), curves=curves, gap_chosen=choice["gap"],
                gap_used=gap_used, gap_in=float(gaps_in[k]), flat=choice["flat"],
                boundary=choice["boundary"], relative_depth=choice["relative_depth"],
                n_sources=int(src_slots.size), n_targets=int(tgt.size))


def _self_null_inputs(model, data_rna, cells, rng, n_max: int, inp: PairInputs) -> PairInputs:
    """
    Inputs of the null of the ``self_null`` formulation: two disjoint subsamples of the cells of
    the target bin, the first as sources (protein state of the cells from the previous E-step,
    ``model.cell_prot``; switching fraction 1/2), the second as targets.  No time separates them.
    """
    ns = int(model.n_stimuli)
    modes = np.asarray(model.modes)
    cells = rng.permutation(np.asarray(cells))
    half = int(min(n_max, cells.size // 2))
    if half < 2:
        raise ValueError("the target bin is too small for the self_null formulation")
    src, tgt = np.sort(cells[:half]), np.sort(cells[half:2 * half])
    s1 = inp.s1
    prot = getattr(model, "cell_prot", None)
    if prot is None:
        P0 = model.adaptive_shrinkage_init(data_rna[src, ns:] * s1, modes[src, ns:])
    else:
        P0 = np.asarray(prot, dtype=float)[src, ns:]
    kb_low, kb_high = inp.k_low, inp.k_high
    return PairInputs(
        P0=P0, mode_init=model.adaptive_shrinkage(data_rna[src, ns:] * s1, modes[src, ns:]) / s1,
        alpha=np.full((half, P0.shape[1]), 0.5), rna_src=data_rna[src, ns:], kon_src=modes[src, ns:],
        mode_end=model.adaptive_shrinkage(data_rna[tgt, ns:] * s1, modes[tgt, ns:]) / s1,
        tau_end=sigmoid_target(modes[tgt, ns:], kb_low, kb_high), rna_tgt=data_rna[tgt, ns:],
        kon_tgt=modes[tgt, ns:], s1=s1, d1=inp.d1, basal=inp.basal, inter=inp.inter, w=inp.w,
        stim=inp.stim, ns=ns, scale_proteins=inp.scale_proteins, alpha_tol=inp.alpha_tol,
        k_low=kb_low, k_high=kb_high)
