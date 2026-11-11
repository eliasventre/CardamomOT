r"""
Mechanistic edge costs shared by the pseudotime ordering options (``PLAN.md`` §5).

The three options all need the same quantity: how plausible is a transition from cell :math:`c`
(with its burst modes :math:`k^{\mathrm{mix}}_c` and protein state :math:`P_c`) to cell :math:`c'`
(with modes :math:`k^{\mathrm{mix}}_{c'}`), given the current network :math:`\Theta`?  Two costs
are provided, following ``plan.tex``:

* **mechanistic** :math:`w_\Theta(c,c') = \mathrm{loss}(k_{on}(P_{c\to c'};\Theta), k^{\mathrm{mix}}_{c'})
  + \alpha\,\|k^{\mathrm{mix}}_{c'} - k^{\mathrm{mix}}_c\|_2`, where :math:`P_{c\to c'}` is the
  deterministic flow of the bursty model over a step :math:`\Delta\tau` (the *same* formula as
  ``find_next_prot`` of the E-step, ``inference/trajectory.py``) and ``loss`` is the gene-weighted
  cross-entropy used by ``my_otdistance``;
* **plausibility** (no :math:`P` needed, so it can also *build* the graph):
  :math:`\alpha\|\log\frac{1+M_{c'}}{1+M_c}\|_2 + \sum_g [0,\ 1-\sigma(z_{c,g}),\ \sigma(z_{c,g})]`
  according to whether the mode of gene :math:`g` is unchanged, increases or decreases;
* **geometric** (the non-mechanistic baseline, i.e. Palantir on the expression space):
  :math:`\|\log(1+M_{c'}) - \log(1+M_c)\|_2`.

All the functions work on *pairs* given as arrays of shape ``(n_pairs, G)`` so that the memory stays
:math:`O(n_{\text{pairs}} \times G)` — a kNN graph of :math:`N \times k` edges is handled at once.

Note on the time unit: the flow only depends on :math:`d_1\,\Delta\tau` (``PLAN.md`` §4.3), and
:math:`\Delta\tau` here is the step of the *graph* (a scale parameter), not the pseudotime unit of
the bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import expit

from ..inference.network import gene_weights, sigmoid_amplitudes, sigmoid_target

EPS = 1e-16

EDGE_COSTS = ("mechanistic", "plausibility", "geometric")
GEOM_SPACES = ("expression", "modes")


def step_of(model, dtau: Optional[float] = None) -> Optional[float]:
    """
    Duration of one step of the ordering graph: ``dtau`` when given, else the bin width when the
    pseudotime axis has a known duration (``PLAN.md`` §4.3), else ``None`` — one protein relaxation
    time ``1/d1``, the time unit not being identified yet (resolved by ``CostContext.from_model``).
    """
    if dtau is not None:
        return float(dtau)
    if getattr(model, "pseudotime_total_duration", None) and getattr(model, "dtau", None):
        return float(model.dtau)
    return None


# ----------------------------------------------------------------------------------------------
# Cost context: the arrays shared by every edge of a graph
# ----------------------------------------------------------------------------------------------
@dataclass
class CostContext:
    """Arrays shared by all the edges of the ordering graph (genes only for the per-gene terms)."""

    s1: np.ndarray                 # (G,) mRNA -> protein factor of the mixture
    d1: np.ndarray                 # (G,) protein degradation rates
    dtau: float                    # step of the graph (only d1 * dtau matters)
    basal: np.ndarray              # (G_tot,) basal activities of the network
    inter: np.ndarray              # (G_tot, G_tot) interactions, node 0 = stimulus
    k_low: np.ndarray              # (G,) lowest mode amplitude of each gene
    k_high: np.ndarray             # (G,) highest mode amplitude of each gene
    w: np.ndarray                  # (G,) gene weights (separation of the two modes)
    ns: int = 1                    # number of stimulus nodes
    alpha: float = 0.5             # switching fraction inside a step (the E-step's alpha)
    alpha_geom: float = 0.5        # weight of the geometric term
    geom_space: str = "expression"  # 'expression' (log1p counts) | 'modes' (burst-mode distance)

    def __post_init__(self):
        if self.geom_space not in GEOM_SPACES:
            raise ValueError(f"geom_space must be one of {GEOM_SPACES}, got {self.geom_space!r}")

    @classmethod
    def from_model(cls, model, dtau: Optional[float] = None, alpha: float = 0.5,
                   alpha_geom: float = 0.5, inter=None, basal=None,
                   geom_space: str = "expression") -> "CostContext":
        """
        Context of the current state of ``model`` (a PseudotimeNetworkModel).

        ``inter`` / ``basal`` override the network (e.g. zeros before the first M-step); ``dtau``
        defaults to one protein relaxation time ``1 / d1`` (relaxation 1), a scale-free choice while
        the duration gauge (``PLAN.md`` §4.3) is not implemented.
        """
        ns = int(model.n_stimuli)
        ks, s1, _ = model._mixture_terms()
        ks = ks[0] if np.ndim(ks) == 3 else ks            # single sample
        s1 = s1[0] if np.ndim(s1) == 2 else s1
        kb = sigmoid_amplitudes(ks)                        # (G_tot, 2)
        G_tot = int(ks.shape[0])
        d1 = np.asarray(model.d[1, ns:], dtype=float)

        if inter is None:
            inter = model.inter
        inter = np.zeros((G_tot, G_tot)) if inter is None else np.asarray(inter, dtype=float)
        if basal is None:
            basal = model.basal
        if basal is None:
            basal = np.zeros(G_tot)
        basal = np.asarray(basal, dtype=float).reshape(-1)
        if basal.size != G_tot:                            # (n_samples, G_tot)
            basal = basal[0]

        if dtau is None:
            dtau = float(1.0 / max(np.median(d1[d1 > 0]), EPS))
        return cls(s1=np.asarray(s1, dtype=float), d1=d1, dtau=float(dtau), basal=basal,
                   inter=inter, k_low=kb[ns:, 0].copy(), k_high=kb[ns:, 1].copy(),
                   w=gene_weights(ks)[ns:].copy(), ns=ns, alpha=float(alpha),
                   alpha_geom=float(alpha_geom), geom_space=geom_space)


# ----------------------------------------------------------------------------------------------
# Elementary terms
# ----------------------------------------------------------------------------------------------
def bridge_end(d1, P0, mode_init, mode_end, s1, dtau, alpha: float = 0.5):
    """
    Protein state after a step ``dtau`` from ``P0``, the modes switching at ``alpha`` (the flow of
    ``find_next_prot`` in ``inference/trajectory.py``, vectorised over the pairs).

    ``d1`` (G,), ``s1`` (G,); ``P0``, ``mode_init``, ``mode_end`` (n_pairs, G).
    """
    m_init = mode_init * s1
    m_end = mode_end * s1
    p_mid = m_init + (P0 - m_init) * np.exp(-d1 * dtau * alpha)
    return m_end + (p_mid - m_end) * np.exp(-d1 * dtau * (1.0 - alpha))


def network_cost(P_end, kon_end, ctx: CostContext, stim=1.0) -> np.ndarray:
    """
    Gene-weighted cross-entropy between the sigmoid of the network at the end states ``P_end``
    (n_pairs, G) and the sigmoid target of the target modes ``kon_end`` (n_pairs, G) — the network
    term of ``my_otdistance``.  ``stim``: stimulus level of the target cells (scalar or (n_pairs,)).
    """
    n = P_end.shape[0]
    if np.ndim(stim) == 0:
        stim = np.full(n, float(stim))
    stim = np.asarray(stim, dtype=float).ravel()
    p_full = np.hstack([np.broadcast_to(stim[:, None], (n, ctx.ns)), P_end])

    z = ctx.basal[None, :] + p_full @ ctx.inter
    sig = np.clip(expit(z)[:, ctx.ns:], 1e-12, 1.0 - 1e-12)
    target = np.clip(sigmoid_target(kon_end, ctx.k_low, ctx.k_high), 1e-6, 1.0 - 1e-6)
    ce = -(target * np.log(sig) + (1.0 - target) * np.log(1.0 - sig))
    return (ce * ctx.w).sum(axis=1) / max(float(ctx.w.sum()), EPS)


def mechanistic_cost(modes_i, modes_j, P0, ctx: CostContext, stim=1.0,
                     rna_i=None, rna_j=None) -> np.ndarray:
    """Mechanistic transition cost (network term + geometric term), shape (n_pairs,)."""
    p_end = bridge_end(ctx.d1, P0, modes_i, modes_j, ctx.s1, ctx.dtau, ctx.alpha)
    ce = network_cost(p_end, modes_j, ctx, stim=stim)
    return ce + ctx.alpha_geom * geometric_cost(rna_i, rna_j, modes_i, modes_j, ctx.geom_space)


def plausibility_cost(modes_i, modes_j, sig_i, ctx: CostContext,
                      rna_i=None, rna_j=None) -> np.ndarray:
    """
    Transition-plausibility cost (``plan.tex`` §mech-palantir): 0 if the mode of a gene does not
    change, ``1 - sigmoid(z)`` if it increases, ``sigmoid(z)`` if it decreases, where ``sig_i``
    (n_pairs, G) is the network sigmoid at the *source* modes.  Plus the geometric term if the
    counts ``rna_i`` / ``rna_j`` (n_pairs, G) are given.
    """
    up, down = modes_j > modes_i, modes_j < modes_i
    cost_g = np.where(up, 1.0 - sig_i, np.where(down, sig_i, 0.0))
    cost = (cost_g * ctx.w).sum(axis=1) / max(float(ctx.w.sum()), EPS)
    if rna_i is not None or ctx.geom_space == "modes":
        cost = cost + ctx.alpha_geom * geometric_cost(rna_i, rna_j, modes_i, modes_j, ctx.geom_space)
    return cost


def geometric_cost(rna_i, rna_j, modes_i=None, modes_j=None, space: str = "expression") -> np.ndarray:
    """
    Geometric term, shape (n_pairs,): the same term in the three costs (``ctx.geom_space``).

    * ``'expression'`` (default): :math:`\\|\\log(1+M_{c'}) - \\log(1+M_c)\\|_2` — continuous, averaged
      over the genes, but it needs the counts (modes are used if they are missing);
    * ``'modes'``: :math:`\\|k^{\\mathrm{mix}}_{c'} - k^{\\mathrm{mix}}_c\\|_2` — the burst-mode distance
      (quantised, no counts needed).
    """
    if space == "modes" or rna_i is None or rna_j is None:
        if modes_i is None or modes_j is None:
            raise ValueError("geometric_cost needs the counts or the modes")
        return np.linalg.norm(modes_j - modes_i, axis=1)
    return np.linalg.norm(np.log1p(rna_j) - np.log1p(rna_i), axis=1)


def source_sigmoid(modes, ctx: CostContext, stim=1.0) -> np.ndarray:
    """Network sigmoid of each cell at its own modes, (N, G) genes only (for plausibility_cost)."""
    n = modes.shape[0]
    if np.ndim(stim) == 0:
        stim = np.full(n, float(stim))
    stim = np.asarray(stim, dtype=float).ravel()
    p_full = np.hstack([np.broadcast_to(stim[:, None], (n, ctx.ns)), modes])
    return expit(ctx.basal[None, :] + p_full @ ctx.inter)[:, ctx.ns:]


def edge_cost(kind: str, modes_i, modes_j, ctx: CostContext, P0=None, sig_i=None,
              rna_i=None, rna_j=None, stim=1.0) -> np.ndarray:
    """Dispatch to one of ``EDGE_COSTS`` (all the arguments except the used ones are ignored)."""
    if kind == "mechanistic":
        return mechanistic_cost(modes_i, modes_j, P0, ctx, stim=stim, rna_i=rna_i, rna_j=rna_j)
    if kind == "plausibility":
        return plausibility_cost(modes_i, modes_j, sig_i, ctx, rna_i=rna_i, rna_j=rna_j)
    if kind == "geometric":
        return geometric_cost(rna_i, rna_j, modes_i, modes_j, ctx.geom_space)
    raise ValueError(f"kind must be one of {EDGE_COSTS}, got {kind!r}")
