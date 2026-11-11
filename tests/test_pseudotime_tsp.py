"""
Tests for Option 2 of the ordering step (path-dependent TSP):
``CardamomOT.pseudotime.options.tsp``.

Run with:
    python -m pytest tests/test_pseudotime_partial_ot.py tests/test_pseudotime_tsp.py -v
"""

import numpy as np
import pytest

from CardamomOT.pseudotime.costs import CostContext, bridge_end, mechanistic_cost
from CardamomOT.pseudotime.evaluate import spearman
from CardamomOT.pseudotime.options import OPTIONS, get_option
from CardamomOT.pseudotime.options.tsp import PathTSP


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


def _manifold(n_cells=60, n_genes=3, roots=3):
    """Cells on a 1-D manifold parameterised by a hidden time t in [0, 1]."""
    t = np.linspace(0.0, 1.0, n_cells)
    coef = np.array([1.0, 1.5, 2.0, 2.5])[:n_genes]
    counts = 10.0 * np.exp(t[:, None] * coef[None, :])
    is_root = np.zeros(n_cells, dtype=bool)
    is_root[:roots] = True
    return FakeModel(counts), t, is_root


def _run(model, is_root, **kw):
    kw.setdefault("n_moves", 120)
    kw.setdefault("window", 12)
    return PathTSP(**kw).run(model, roots=is_root)


# ----------------------------------------------------------------------------------------------
# the tour recovers the axis
# ----------------------------------------------------------------------------------------------
def test_of_the_manifold_is_recovered_with_the_palantir_init():
    model, t, is_root = _manifold()
    res = _run(model, is_root)

    assert res.tau.shape == (model.counts.shape[0],)
    assert np.all(np.isfinite(res.tau))
    assert np.allclose(res.tau[is_root], 0.0)
    assert np.isclose(res.tau.max(), 1.0)
    assert spearman(res.tau, t) > 0.95
    assert res.diagnostics["init"] == "palantir"


def test_the_root_block_stays_at_the_front_of_the_tour():
    model, _, is_root = _manifold()
    res = _run(model, is_root, n_moves=200, window=10)
    n_roots = int(is_root.sum())
    first = np.argsort(res.tau, kind="stable")[:n_roots]
    assert set(first.tolist()) == set(np.flatnonzero(is_root).tolist())


def test_tau_is_deterministic():
    model, _, is_root = _manifold()
    a = _run(model, is_root).tau
    b = _run(model, is_root).tau
    assert np.allclose(a, b)


def test_partial_ot_can_be_the_init_option():
    model, t, is_root = _manifold()
    res = _run(model, is_root, init="partial_ot", n_moves=50)
    assert res.diagnostics["init"] == "partial_ot"
    assert spearman(res.tau, t) > 0.85


# ----------------------------------------------------------------------------------------------
# the 2-opt
# ----------------------------------------------------------------------------------------------
def test_the_tour_never_gets_worse():
    model, _, is_root = _manifold(n_cells=70)
    res = _run(model, is_root, n_moves=150, window=15)
    d = res.diagnostics
    assert d["tour_length_end"] <= d["tour_length_start"] + 1e-9
    assert d["tour_improvement"] >= 0.0
    assert d["n_moves_accepted"] <= d["n_moves_tried"]


def test_zero_moves_keeps_the_initial_ordering():
    """With n_moves=0 the tour is the warm start, so tau is a monotone map of the init tau."""
    model, t, is_root = _manifold()
    res = _run(model, is_root, n_moves=0)
    assert res.diagnostics["n_moves_tried"] == 0
    assert spearman(res.tau, t) > 0.9


def test_path_length_and_tour_rank_are_the_two_coordinate_choices():
    model, _, is_root = _manifold(n_cells=50)
    n_roots = int(is_root.sum())
    length = _run(model, is_root, n_moves=0, use_path_length=True)
    rank = _run(model, is_root, n_moves=0, use_path_length=False)

    # the rank coordinate has exactly one distinct value per tour position
    assert len(np.unique(np.round(rank.tau, 12))) == 50 - n_roots + 1
    # the path-length coordinate is continuous (uneven steps)
    assert len(np.unique(np.round(length.tau, 12))) > 50 - n_roots
    assert spearman(length.tau, rank.tau) > 0.99


# ----------------------------------------------------------------------------------------------
# the step cost is the shared mechanistic cost, with the propagated protein
# ----------------------------------------------------------------------------------------------
def test_the_step_cost_is_the_mechanistic_cost_with_the_propagated_protein():
    model, _, _ = _manifold()
    opt = PathTSP(n_moves=0, alpha_geom=1.0)
    ctx = CostContext.from_model(model, dtau=20.0, alpha=0.5, alpha_geom=1.0)
    modes = model.modes[:, 1:]
    counts = model.counts
    stim = np.zeros(model.counts.shape[0])

    P = model.cell_prot[0, 1:]
    cost, p_next = opt._step(0, 1, P, modes, counts, ctx, stim)

    ref = mechanistic_cost(modes[0:1], modes[1:2], P[None, :], ctx, stim=stim[1],
                           rna_i=counts[0:1], rna_j=counts[1:2])[0]
    p_ref = bridge_end(ctx.d1, P[None, :], modes[0:1], modes[1:2], ctx.s1, ctx.dtau, ctx.alpha)[0]
    assert cost == pytest.approx(float(ref), rel=1e-9)
    assert np.allclose(p_next, p_ref)


def test_the_protein_is_propagated_along_the_tour_not_read_from_the_cells():
    """A tour step must use the state reached at the previous step, not the cell's own state."""
    model, _, _ = _manifold()
    opt = PathTSP(n_moves=0)
    ctx = CostContext.from_model(model, dtau=20.0, alpha=0.5)
    modes, counts = model.modes[:, 1:], model.counts
    stim = np.zeros(model.counts.shape[0])

    P_own = model.cell_prot[5, 1:]
    _, p_next = opt._step(5, 40, P_own, modes, counts, ctx, stim)
    assert not np.allclose(p_next, P_own)           # the state moved
    assert not np.allclose(p_next, model.cell_prot[40, 1:])   # and it is not the target's own


# ----------------------------------------------------------------------------------------------
# parameters, registry
# ----------------------------------------------------------------------------------------------
def test_previous_init_requires_an_existing_pseudotime():
    model, _, is_root = _manifold()
    with pytest.raises(ValueError, match="init='previous'"):
        _run(model, is_root, init="previous")


def test_previous_init_warm_starts_from_the_current_tau():
    model, t, is_root = _manifold()
    model.tau = t.copy()                            # a perfect warm start
    res = _run(model, is_root, init="previous", n_moves=0)
    assert res.diagnostics["init"] == "previous"
    assert spearman(res.tau, t) > 0.99


def test_from_model_reads_the_tsp_parameters():
    class WithParams:
        tsp_init = "partial_ot"
        tsp_n_moves = 12
        tsp_window = 7
        tsp_use_path_length = False
        tsp_edge_weight = "geometric"
        seed = 3

    opt = PathTSP.from_model(WithParams())
    assert opt.init == "partial_ot"
    assert opt.n_moves == 12
    assert opt.window == 7
    assert opt.use_path_length is False
    assert opt.edge_weight == "geometric"
    assert opt.seed == 3


def test_tsp_is_registered_and_built_by_get_option():
    assert OPTIONS["tsp"] is PathTSP

    class WithParams:
        tsp_window = 9

    assert isinstance(get_option("tsp", WithParams()), PathTSP)


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
    (dict(init="random"), "init"),
    (dict(edge_weight="plausibility"), "edge_weight"),
    (dict(window=1), "window"),
    (dict(n_moves=-5), "n_moves"),
])
def test_invalid_parameters_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PathTSP(**kwargs)


def test_no_root_cell_raises():
    model, _, _ = _manifold()
    with pytest.raises(ValueError, match="no root cell"):
        PathTSP().run(model, roots=np.zeros(model.counts.shape[0], dtype=bool))
