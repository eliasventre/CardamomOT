"""
Tests for Option 1 of the ordering step (partial OT from the root population):
``CardamomOT.pseudotime.options.partial_ot``.

Run with:
    python -m pytest tests/test_pseudotime_partial_ot.py -v
"""

import numpy as np
import pytest

from CardamomOT.pseudotime.evaluate import spearman
from CardamomOT.pseudotime.options import OPTIONS, get_option
from CardamomOT.pseudotime.options.partial_ot import PartialOT


# ----------------------------------------------------------------------------------------------
# duck-typed model (same contract as tests/test_pseudotime_order.py)
# ----------------------------------------------------------------------------------------------
class FakeModel:
    """Minimal stand-in for PseudotimeNetworkModel used by the ordering options."""

    n_stimuli = 1
    pseudotime_total_duration = None

    def __init__(self, counts, modes=None, d1=0.05):
        counts = np.asarray(counts, dtype=float)
        if modes is None:
            modes = (counts - counts.min(axis=0)) / (np.ptp(counts, axis=0) + 1e-12)
        modes = np.asarray(modes, dtype=float)
        G = counts.shape[1]
        self.counts = counts
        self.modes = np.hstack([np.ones((counts.shape[0], 1)), modes])
        self.cell_prot = np.hstack([np.ones((counts.shape[0], 1)), modes])
        self.d = np.stack([np.full(G + 1, 0.1), np.full(G + 1, d1)])
        self.inter = np.zeros((G + 1, G + 1))
        self.basal = np.zeros((1, G + 1))
        self.tau = None
        self._ks = np.stack([np.full(G + 1, 0.1), np.ones(G + 1)], axis=1)

    def _mixture_terms(self):
        return self._ks, np.ones(self.counts.shape[1]), self._ks.max(axis=1)


def _manifold(n_cells=80, n_genes=3, roots=3):
    """Cells on a 1-D manifold parameterised by a hidden time t in [0, 1]."""
    t = np.linspace(0.0, 1.0, n_cells)
    coef = np.array([1.0, 1.5, 2.0, 2.5])[:n_genes]
    counts = 10.0 * np.exp(t[:, None] * coef[None, :])
    is_root = np.zeros(n_cells, dtype=bool)
    is_root[:roots] = True
    return FakeModel(counts), t, is_root


def _run(model, is_root, **kw):
    kw.setdefault("mass_per_step", 0.25)
    return PartialOT(**kw).run(model, roots=is_root)


# ----------------------------------------------------------------------------------------------
# the wave reaches every cell
# ----------------------------------------------------------------------------------------------
def test_reaches_every_cell_and_zero_on_the_roots():
    model, _, is_root = _manifold()
    res = _run(model, is_root)

    tau = res.tau
    assert tau.shape == (model.counts.shape[0],)
    assert np.all(np.isfinite(tau))
    assert np.allclose(tau[is_root], 0.0)
    assert np.isclose(tau.max(), 1.0)
    assert res.diagnostics["n_unreached"] == 0


def test_tau_increases_along_the_manifold():
    """On a clean 1-D manifold the ordering must recover the hidden time exactly."""
    model, t, is_root = _manifold()
    res = _run(model, is_root)
    assert spearman(res.tau, t) > 0.95


def test_tau_is_deterministic():
    model, _, is_root = _manifold()
    a = _run(model, is_root).tau
    b = _run(model, is_root).tau
    assert np.allclose(a, b)


# ----------------------------------------------------------------------------------------------
# the partial-OT solve
# ----------------------------------------------------------------------------------------------
def test_partial_plan_respects_the_mass_budget_and_the_marginals():
    rng = np.random.default_rng(0)
    M = rng.random((7, 11))
    opt = PartialOT(mass_per_step=0.3)
    gamma, mass = opt._solve(M)

    assert gamma.shape == M.shape
    assert np.all(gamma >= 0)
    assert np.isclose(mass, 0.3, atol=1e-8)                       # 1^T gamma 1 = beta
    assert np.all(gamma.sum(axis=1) <= 1.0 / 7 + 1e-8)            # rows <= mu
    assert np.all(gamma.sum(axis=0) <= 1.0 / 11 + 1e-8)           # cols <= nu


def test_the_plan_uses_the_cheap_edges_first():
    """With a mass budget smaller than 1, only the cheapest columns receive mass."""
    M = np.array([[10.0, 1.0, 20.0, 30.0]])
    opt = PartialOT(mass_per_step=0.25)
    gamma, _ = opt._solve(M)
    assert np.argmax(gamma[0]) == 1


def test_a_larger_mass_budget_fills_more_candidates():
    """beta is what sets the width of the wave: more mass -> more candidates above min_mass."""
    rng = np.random.default_rng(0)
    M = rng.random((5, 200))
    narrow, _ = PartialOT(mass_per_step=0.1)._solve(M)
    wide, _ = PartialOT(mass_per_step=0.5)._solve(M)
    assert (wide.sum(axis=0) > 1e-3).sum() > (narrow.sum(axis=0) > 1e-3).sum()


def test_lagrange_variant_produces_a_valid_ordering():
    model, t, is_root = _manifold()
    res = _run(model, is_root, partial="lagrange")
    assert np.all(np.isfinite(res.tau))
    assert np.allclose(res.tau[is_root], 0.0)
    assert spearman(res.tau, t) > 0.5


# ----------------------------------------------------------------------------------------------
# candidate pool, caps and fallbacks
# ----------------------------------------------------------------------------------------------
def test_restrict_all_and_knn_both_reach_every_cell():
    model, t, is_root = _manifold()
    a = _run(model, is_root, restrict="knn")
    b = _run(model, is_root, restrict="all")
    assert a.diagnostics["n_unreached"] == b.diagnostics["n_unreached"] == 0
    assert spearman(a.tau, t) > 0.9 and spearman(b.tau, t) > 0.9


def test_frontier_and_candidate_caps_still_reach_every_cell():
    model, t, is_root = _manifold(n_cells=120)
    res = _run(model, is_root, frontier_cap=10, max_candidates=20)
    assert res.diagnostics["n_unreached"] == 0
    assert spearman(res.tau, t) > 0.9


def test_a_disconnected_component_is_filled_instead_of_left_without_a_time():
    """The far cluster is never a kNN candidate of the near one: the pool must widen."""
    near = 10.0 * np.exp(np.linspace(0, 1, 60)[:, None] * np.array([1.0, 1.5, 2.0]))
    far = 1e4 * np.exp(np.linspace(0, 1, 40)[:, None] * np.array([1.0, 1.5, 2.0]))
    model = FakeModel(np.vstack([near, far]))
    is_root = np.zeros(model.counts.shape[0], dtype=bool)
    is_root[:3] = True

    res = _run(model, is_root, knn_candidates=5)
    assert np.all(np.isfinite(res.tau))
    assert res.diagnostics["n_unreached"] == 0
    assert res.diagnostics["n_fallback"] >= 1
    assert np.all(res.tau[60:] > 0.0)            # the far cells get a forward time, not the roots'


def test_wave_max_caps_the_cells_added_per_wave():
    model, _, is_root = _manifold(n_cells=120)
    res = _run(model, is_root, wave_max=5)
    sizes = res.diagnostics["wave_sizes"]
    assert max(sizes) <= 5
    assert res.diagnostics["n_unreached"] == 0


# ----------------------------------------------------------------------------------------------
# parameters, registry
# ----------------------------------------------------------------------------------------------
def test_from_model_reads_the_partial_ot_parameters():
    class WithParams:
        partial_ot_mass_per_step = 0.4
        partial_ot_restrict = "all"
        partial_ot_edge_weight = "geometric"
        partial_ot_frontier_cap = 33
        seed = 7

    opt = PartialOT.from_model(WithParams())
    assert opt.mass_per_step == 0.4
    assert opt.restrict == "all"
    assert opt.edge_weight == "geometric"
    assert opt.frontier_cap == 33
    assert opt.seed == 7
    assert opt.knn_candidates == 10               # untouched default


def test_partial_ot_is_registered_and_built_by_get_option():
    assert OPTIONS["partial_ot"] is PartialOT

    class WithParams:
        partial_ot_mass_per_step = 0.2

    assert isinstance(get_option("partial_ot", WithParams()), PartialOT)


def test_the_geometric_step_cost_does_not_need_the_proteins():
    model, t, is_root = _manifold()
    model.cell_prot = None
    res = _run(model, is_root, edge_weight="geometric")
    assert np.all(np.isfinite(res.tau))
    assert spearman(res.tau, t) > 0.9


def test_mechanistic_step_cost_requires_the_cell_proteins():
    model, _, is_root = _manifold()
    model.cell_prot = None
    with pytest.raises(ValueError, match="initialise_cell_states"):
        _run(model, is_root, edge_weight="mechanistic")


@pytest.mark.parametrize("kwargs, match", [
    (dict(mass_per_step=0.0), "mass_per_step"),
    (dict(mass_per_step=1.5), "mass_per_step"),
    (dict(restrict="ball"), "restrict"),
    (dict(partial="oracle"), "partial"),
    (dict(frontier_mode="edges"), "frontier_mode"),
    (dict(edge_weight="diffusion"), "edge_weight"),
])
def test_invalid_parameters_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PartialOT(**kwargs)


def test_no_root_cell_raises():
    model, _, _ = _manifold()
    with pytest.raises(ValueError, match="no root cell"):
        PartialOT().run(model, roots=np.zeros(model.counts.shape[0], dtype=bool))
