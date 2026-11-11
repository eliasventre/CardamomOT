r"""
Time gaps between consecutive bins (PROTOTYPE_PLAN.md §4.2–§4.4, §5).

For a pair (bin k -> bin k+1), the gap :math:`\Delta t_k` is the gap that minimises the optimal
transport cost between the source states of bin k and the real cells of bin k+1:

.. math::

    W_k(\Delta) = \min_{\gamma\in\Pi(\mu,\nu)} \langle\gamma, C(\Delta)\rangle,\qquad
    C_{ij}(\Delta) = \mathrm{net}_{ij}(\Delta)
      + \lambda_P \tfrac{m}{s_P}\,\mathrm{prot}_{ij}(\Delta)
      + \lambda_x \tfrac{m}{s_x}\,\mathrm{rna}_{ij}

* ``net``: the network term of ``my_otdistance`` — gene-weighted relative cross-entropy between the
  sigmoid of the network at the end proteins :math:`P_{ij}(\Delta)` (the flow ``find_next_prot``,
  or ``find_next_prot_dilution`` with birth rates) and the sigmoid target of the target modes;
* ``prot``: :math:`\|(P_{ij}(\Delta) - P_i)/\text{scale\_proteins}\|^2` (always favours Δ -> 0);
* ``rna``: :math:`\|s_1(x_j - x_i)\|^2` (does not depend on Δ, anchors the pairing).

Unlike the E-step cost, the scales :math:`m, s_P, s_x` (medians of the three terms) are computed
**once**, at a reference gap, so that the values at different Δ are comparable, and no column
minimum is subtracted.  The OT is the exact EMD (no entropic term, whose size would depend on the
cost scale).  The formulations of §5 are the choices of :math:`(\lambda_P, \lambda_x)`:
F1 = (0, 0), F2 = (λ_P, 0), F3 = (λ_P or 0, λ_x).  Two corrections of F1's bias towards large
gaps (G0, PROTOTYPE_PLAN.md §11.1: as Δ grows the end protein relaxes towards the target's *own*
modes, so each target is "explained by itself"):

* ``"forecast"`` (F6): the source protein is propagated by the **network's own dynamics**
  :math:`\dot P = d_1(\text{scale}\cdot k_{on}(P;\Theta) - P)`, with no target state in the
  predictor, and :math:`k_{on}` is compared with the target modes;
* ``"self_null"`` (F1 − null): the F1 curve minus the F1 curve from the cells of bin k+1 to other
  cells of bin k+1 (no time between them), both divided by their mean: what the null gains at
  large Δ is the self-explanation alone.

Everything here is a pure function of arrays: the E-step (``estep.py``) builds the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import ot
from scipy.special import expit

from ..inference.network import eps_CE
from ..inference.trajectory import birth_alpha

EPS = 1e-12

FORMULATIONS = {"F1": (0.0, 0.0), "F2": (1.0, 0.0), "F3": (1.0, 1.0)}


@dataclass
class GapSearch:
    """Settings of the gap search (the ``gap_*`` parameters of the prototype)."""

    grid_range: Tuple[float, float] = (0.01, 5.0)   # candidates, in multiples of 1 / median(d1)
    n_grid: int = 21                                # log-spaced candidates
    max_cells: int = 300                            # sources and targets subsampled per pair
    lambda_prot: float = 0.0                        # λ_P (0: formulation F1)
    lambda_rna: float = 0.0                         # λ_x
    prior_weight: float = 0.01                      # ρ_1, pull towards the prior gap
    smooth_weight: float = 0.01                     # ρ_2, pull towards the neighbouring gaps
    damping: float = 1.0                            # η, 1 = no damping across iterations
    flat_tol: float = 0.02                          # relative range below which a curve is flat
    start: int = 2                                  # first EM iteration with a search (Θ = 0 before)
    seed: int = 0                                   # subsampling of the search
    scale_ref: str = "current"                      # gap where the cost scales are fixed:
                                                    # 'current' (the gap of the pair) | 'prior'
    formulation: object = (0.0, 0.0)               # curve used for the choice: (λ_P, λ_x) | 'forecast' | 'self_null'
    extra: Dict[str, object] = field(default_factory=dict)  # extra curves recorded (G0), same specs

    def grid(self, d1) -> np.ndarray:
        """Candidate gaps (model time units, hours with the d1 prior)."""
        d1 = np.asarray(d1, dtype=float)
        d1 = d1[d1 > 0]
        unit = 1.0 / max(float(np.median(d1)), EPS) if d1.size else 1.0
        lo, hi = self.grid_range
        return np.geomspace(lo * unit, hi * unit, int(self.n_grid))


# ----------------------------------------------------------------------------------------------
# Flow, network term and the three cost terms
# ----------------------------------------------------------------------------------------------
def relative_ce(y_pred, y_true, eps: float = eps_CE):
    """Element-wise relative cross-entropy (0 when y_pred = y_true): ``network.cross_entropy``."""
    p = np.clip(y_pred, eps, 1 - eps)
    t = np.clip(y_true, eps, 1 - eps)
    return -(y_true * np.log(p / t) + (1 - y_true) * np.log((1 - p) / (1 - t)))


def flow(P0, mode_init, mode_end, alpha, s1, d1, delta):
    """``find_next_prot``, broadcast over any leading axes (d1, s1: (G,))."""
    p_int = mode_init * s1 + (P0 - mode_init * s1) * np.exp(-d1 * delta * alpha)
    return mode_end * s1 + (p_int - mode_end * s1) * np.exp(-d1 * delta * (1.0 - alpha))


def flow_dilution(d1, d1_end, P0, mode_init, mode_end, alpha, alpha_b, s1, delta):
    """``find_next_prot_dilution``, broadcast over any leading axes."""
    a1 = np.minimum(alpha, alpha_b)
    a2 = np.maximum(alpha, alpha_b)
    f = 1.0 * (alpha < alpha_b)
    mode_mid = f * (mode_end * d1_end / d1) + (1.0 - f) * (mode_init * d1 / d1_end)
    rate_mid = f * d1 + (1.0 - f) * d1_end
    P1 = mode_init * s1 + (P0 - mode_init * s1) * np.exp(-d1 * delta * a1)
    P2 = mode_mid * s1 + (P1 - mode_mid * s1) * np.exp(-rate_mid * delta * (a2 - a1))
    return mode_end * s1 + (P2 - mode_end * s1) * np.exp(-d1_end * delta * (1.0 - a2))


@dataclass
class PairInputs:
    """
    Inputs of the gap search of one pair, genes only (stimuli excluded) unless stated.

    Sources (n_s rows): ``P0`` proteins, ``mode_init`` shrunk modes of the E-step, ``alpha``
    switching fractions, ``rna_src`` counts, ``kon_src`` modes (for the birth fraction).
    Targets (n_t rows): ``mode_end`` shrunk modes, ``tau_end`` sigmoid targets of their modes,
    ``rna_tgt`` counts, ``kon_tgt`` modes.  Network: ``basal`` (G_tot,), ``inter`` (G_tot, G_tot),
    ``w`` gene weights (G,), ``stim`` (ns,) stimulus values of the target bin.
    Dilution: ``b_src`` (n_s,), ``b_tgt`` (n_t,) birth rates, or None (no dilution).
    """

    P0: np.ndarray
    mode_init: np.ndarray
    alpha: np.ndarray
    rna_src: np.ndarray
    kon_src: np.ndarray
    mode_end: np.ndarray
    tau_end: np.ndarray
    rna_tgt: np.ndarray
    kon_tgt: np.ndarray
    s1: np.ndarray
    d1: np.ndarray
    basal: np.ndarray
    inter: np.ndarray
    w: np.ndarray
    stim: np.ndarray
    ns: int = 1
    scale_proteins: float = 1.0
    b_src: Optional[np.ndarray] = None
    b_tgt: Optional[np.ndarray] = None
    alpha_tol: float = 0.6
    k_low: Optional[np.ndarray] = None    # (G,) sigmoid bounds of the modes (forecast only)
    k_high: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.P0.shape[0]), int(self.mode_end.shape[0])


def end_proteins(inp: PairInputs, delta: float, rows: slice) -> np.ndarray:
    """End proteins (n_rows, n_t, G) of the sources ``rows`` towards every target, after ``delta``."""
    P0 = inp.P0[rows, None, :]
    alpha = inp.alpha[rows, None, :]
    if inp.b_src is None or inp.b_tgt is None:
        return flow(P0, inp.mode_init[rows, None, :], inp.mode_end[None, :, :], alpha,
                    inp.s1, inp.d1, delta)
    d1_i = inp.d1[None, :] + inp.b_src[rows, None]                   # (n_rows, G)
    c_i = inp.d1[None, :] / d1_i
    d1_j = inp.d1[None, :] + inp.b_tgt[:, None]                      # (n_t, G)
    alpha_b = birth_alpha(np.broadcast_to(alpha, (alpha.shape[0], d1_j.shape[0], alpha.shape[2])),
                          inp.kon_src[rows, None, :], inp.kon_tgt[None, :, :], inp.alpha_tol)
    return flow_dilution(d1_i[:, None, :], d1_j[None, :, :], P0,
                         (inp.mode_init[rows] * c_i)[:, None, :],
                         inp.mode_end[None, :, :] * (inp.d1[None, :] / d1_j)[None, :, :],
                         alpha, alpha_b, inp.s1, delta)


def pair_terms(inp: PairInputs, delta: float, chunk_elems: int = 2_000_000
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    ``(net, prot)`` matrices (n_s, n_t) at the gap ``delta``.  The sources are processed in chunks
    of about ``chunk_elems`` (pair × gene) values, so the memory stays bounded for many genes.
    """
    n_s, n_t = inp.shape
    G = int(inp.P0.shape[1])
    net = np.empty((n_s, n_t))
    prot = np.empty((n_s, n_t))
    step = int(max(1, chunk_elems // max(n_t * G, 1)))
    stim = np.asarray(inp.stim, dtype=float).ravel() * inp.scale_proteins
    z_stim = inp.basal + stim @ inp.inter[:inp.ns]                    # (G_tot,)
    inter_g = inp.inter[inp.ns:, inp.ns:]                             # gene -> gene
    for lo in range(0, n_s, step):
        rows = slice(lo, min(lo + step, n_s))
        P = end_proteins(inp, delta, rows)                           # (c, n_t, G)
        sig = expit(z_stim[inp.ns:] + P @ inter_g)
        net[rows] = (relative_ce(sig, inp.tau_end[None, :, :]) * inp.w).sum(axis=-1)
        diff = (P - inp.P0[rows, None, :]) / inp.scale_proteins
        prot[rows] = (diff * diff).sum(axis=-1)
    return net, prot


def rna_term(inp: PairInputs) -> np.ndarray:
    """mRNA term (n_s, n_t), independent of the gap."""
    a = inp.rna_src * inp.s1
    b = inp.rna_tgt * inp.s1
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)


def forecast_sigmoids(inp: PairInputs, grid: Sequence[float], steps_per_relaxation: int = 20
                      ) -> np.ndarray:
    r"""
    Network sigmoids (n_grid, n_s, G) of the sources propagated by the network's own mean-field
    dynamics :math:`\dot P = d_1(\text{scale}\cdot k_{on}(P) - P)`,
    :math:`k_{on} = k_{low} + (k_{high} - k_{low})\,\sigma(\theta_0 + \Theta[\text{stim}; P])`
    (RK4, about ``steps_per_relaxation`` steps per protein relaxation time 1/d1), recorded at the
    increasing gaps of ``grid``.  No dilution.
    """
    if inp.k_low is None or inp.k_high is None:
        raise ValueError("the forecast needs the sigmoid bounds k_low / k_high")
    grid = np.asarray(grid, dtype=float)
    if np.any(np.diff(grid) <= 0):
        raise ValueError("the grid must be increasing")
    stim = np.asarray(inp.stim, dtype=float).ravel() * inp.scale_proteins
    z0 = (inp.basal + stim @ inp.inter[:inp.ns])[inp.ns:]
    inter_g = inp.inter[inp.ns:, inp.ns:]
    span = inp.k_high - inp.k_low

    def sig(P):
        return expit(z0 + P @ inter_g)

    def rhs(P):
        return inp.d1 * (inp.scale_proteins * (inp.k_low + span * sig(P)) - P)

    h_max = 1.0 / (steps_per_relaxation * max(float(np.max(inp.d1)), EPS))
    P = np.array(inp.P0, dtype=float, copy=True)
    out = np.empty((grid.size,) + P.shape)
    t = 0.0
    for m, t_next in enumerate(grid):
        n_sub = int(max(1, np.ceil((t_next - t) / h_max)))
        h = (t_next - t) / n_sub
        for _ in range(n_sub):
            k1 = rhs(P)
            k2 = rhs(P + 0.5 * h * k1)
            k3 = rhs(P + 0.5 * h * k2)
            k4 = rhs(P + h * k3)
            P = P + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = float(t_next)
        out[m] = sig(P)
    return out


def forecast_cost(sig_src: np.ndarray, inp: PairInputs) -> np.ndarray:
    """(n_s, n_t) gene-weighted relative cross-entropy between forecast sigmoids and target modes."""
    return (relative_ce(sig_src[:, None, :], inp.tau_end[None, :, :]) * inp.w).sum(axis=-1)


def median_scale(x: np.ndarray) -> float:
    """Median of a term (its mean if the median is 0, 1 if the term is 0): ``_median_scale``."""
    m = float(np.median(x))
    if m <= EPS:
        m = float(np.mean(x))
    return m if m > EPS else 1.0


# ----------------------------------------------------------------------------------------------
# Curves W_k(Δ) on the grid
# ----------------------------------------------------------------------------------------------
def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _emd(C: np.ndarray, mu=None, nu=None) -> float:
    mu = _uniform(C.shape[0]) if mu is None else mu
    nu = _uniform(C.shape[1]) if nu is None else nu
    return float(ot.emd2(mu, nu, C, numItermax=int(1e7)))


def gap_curves(inp: PairInputs, grid: Sequence[float], ref_delta: float,
               formulations: Dict[str, object], mu=None, nu=None,
               null_inp: Optional[PairInputs] = None) -> Dict[str, np.ndarray]:
    """
    Curves ``{name: (n_grid,)}`` of the exact OT cost for each formulation: ``(λ_P, λ_x)`` (scales
    fixed at ``ref_delta``), ``"forecast"`` (F6) or ``"self_null"`` (F1 − null, which needs
    ``null_inp``: sources and targets taken among the cells of the target bin).
    ``mu`` / ``nu``: marginals of ``inp`` (uniform by default).
    """
    grid = np.asarray(grid, dtype=float)
    weighted = {n: f for n, f in formulations.items() if not isinstance(f, str)}
    kinds = {n: f for n, f in formulations.items() if isinstance(f, str)}
    for n, f in kinds.items():
        if f not in ("forecast", "self_null"):
            raise ValueError(f"unknown formulation {f!r} for {n!r}")
    if "self_null" in kinds.values() and null_inp is None:
        raise ValueError("the 'self_null' formulation needs null_inp")
    out = {name: np.empty(grid.size) for name in formulations}

    need_net = bool(weighted) or "self_null" in kinds.values()
    if need_net:
        net_ref, prot_ref = pair_terms(inp, float(ref_delta))
        rna = rna_term(inp)
        m = median_scale(net_ref)
        k_prot = m / median_scale(prot_ref)
        k_rna = m / median_scale(rna)
        f1 = np.empty(grid.size)
        for i, delta in enumerate(grid):
            net, prot = pair_terms(inp, float(delta))
            f1[i] = _emd(net, mu, nu)
            for name, (lp, lx) in weighted.items():
                if lp == 0 and lx == 0:
                    out[name][i] = f1[i]
                else:
                    out[name][i] = _emd(net + lp * k_prot * prot + lx * k_rna * rna, mu, nu)
    if "self_null" in kinds.values():
        f1_null = np.array([_emd(pair_terms(null_inp, float(d))[0]) for d in grid])
        diff = f1 / median_scale(np.abs(f1)) - f1_null / median_scale(np.abs(f1_null))
        for name, f in kinds.items():
            if f == "self_null":
                out[name] = diff
    if "forecast" in kinds.values():
        sig = forecast_sigmoids(inp, grid)
        fc = np.array([_emd(forecast_cost(sig[i], inp), mu, nu) for i in range(grid.size)])
        for name, f in kinds.items():
            if f == "forecast":
                out[name] = fc
    return out


# ----------------------------------------------------------------------------------------------
# Choosing the gap
# ----------------------------------------------------------------------------------------------
def _normalised(curve: np.ndarray) -> np.ndarray:
    """Curve divided by the mean of its absolute value (the weights ρ are then dimensionless; a
    difference curve, which can be negative, keeps its sign)."""
    curve = np.asarray(curve, dtype=float)
    m = float(np.mean(np.abs(curve)))
    return curve / m if m > EPS else curve


def _parabola_vertex(x: np.ndarray, y: np.ndarray, i: int) -> float:
    """Vertex of the parabola through the points i-1, i, i+1, clipped to [x[i-1], x[i+1]]."""
    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)
    if abs(den) < EPS:
        return float(x1)
    num = (x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0)
    return float(np.clip(x1 - 0.5 * num / den, x0, x2))


def curve_flags(curve: np.ndarray, flat_tol: float) -> dict:
    """``flat`` (relative range below ``flat_tol``) and ``boundary`` (raw minimum at an end)."""
    curve = np.asarray(curve, dtype=float)
    mean = float(np.mean(np.abs(curve)))
    rng = float(curve.max() - curve.min())
    i = int(np.argmin(curve))
    return dict(flat=bool(rng <= flat_tol * max(mean, EPS)),
                boundary=bool(i in (0, curve.size - 1)),
                raw_argmin=i, relative_depth=rng / max(mean, EPS))


def choose_gap(grid: Sequence[float], curve: Sequence[float], log_prior: float,
               prior_weight: float = 0.0, neighbours: Sequence[float] = (),
               smooth_weight: float = 0.0, flat_tol: float = 0.02) -> dict:
    r"""
    Gap of one pair: argmin over the grid of
    :math:`W/\bar W + \rho_1(\log\Delta - \log\Delta_{prior})^2 + \rho_2\sum_n(\log\Delta - \log n)^2`
    (``neighbours``: log of the neighbouring gaps), refined by a parabola in log Δ.

    Returns ``gap``, ``log_gap``, ``index``, the objective ``J`` and the flags of the raw curve.
    """
    x = np.log(np.asarray(grid, dtype=float))
    c = np.asarray(curve, dtype=float)
    J = _normalised(c) + prior_weight * (x - log_prior) ** 2
    for n in neighbours:
        J = J + smooth_weight * (x - float(n)) ** 2
    i = int(np.argmin(J))
    log_gap = _parabola_vertex(x, J, i) if 0 < i < x.size - 1 else float(x[i])
    out = dict(gap=float(np.exp(log_gap)), log_gap=log_gap, index=i, J=J)
    out.update(curve_flags(c, flat_tol))
    return out


def joint_gaps_dp(grid: Sequence[float], curves: np.ndarray, log_prior: float,
                  prior_weight: float = 0.0, smooth_weight: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Exact joint choice of the gaps of all pairs on the grid (dynamic programming, as Viterbi):
    minimise :math:`\sum_k [W_k/\bar W_k + \rho_1(\log\Delta_k - \log\Delta_{prior})^2]
    + \rho_2\sum_k(\log\Delta_{k+1} - \log\Delta_k)^2`.  Valid when the curves do not depend on each
    other's gaps (fixed sources, gate G0).  Returns ``(indices (P,), gaps (P,))``.
    """
    x = np.log(np.asarray(grid, dtype=float))
    curves = np.atleast_2d(np.asarray(curves, dtype=float))
    P, n = curves.shape
    unary = np.stack([_normalised(c) for c in curves]) + prior_weight * (x - log_prior)[None, :] ** 2
    pair = smooth_weight * (x[:, None] - x[None, :]) ** 2            # (previous, current)
    cost = unary[0].copy()
    back = np.zeros((P, n), dtype=int)
    for k in range(1, P):
        tot = cost[:, None] + pair
        back[k] = np.argmin(tot, axis=0)
        cost = tot[back[k], np.arange(n)] + unary[k]
    idx = np.zeros(P, dtype=int)
    idx[-1] = int(np.argmin(cost))
    for k in range(P - 1, 0, -1):
        idx[k - 1] = back[k, idx[k]]
    return idx, np.exp(x[idx])


def damp(log_old: float, log_new: float, damping: float) -> float:
    """Damped update of a log-gap across iterations: (1 - η) log_old + η log_new."""
    return (1.0 - damping) * float(log_old) + damping * float(log_new)
