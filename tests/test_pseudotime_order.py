"""
Tests for the ordering step of the pseudotime cycle (Option 3, mechanistic Palantir):
``CardamomOT.pseudotime.costs``, ``options.palantir_like`` and ``evaluate``.

Run with:
    python -m pytest tests/test_pseudotime_order.py -v
"""

import numpy as np
import pytest

from CardamomOT.pseudotime.costs import (CostContext, bridge_end, edge_cost, geometric_cost,
                                         mechanistic_cost, plausibility_cost)
from CardamomOT.pseudotime.evaluate import pseudotime_metrics, spearman
from CardamomOT.pseudotime.options.palantir_like import PalantirLike


# ----------------------------------------------------------------------------------------------
# A duck-typed model: the ordering options only need the mixture terms, the modes, the counts and
# the cell proteins (no mixture fit, so the tests are deterministic and fast)
# ----------------------------------------------------------------------------------------------
class FakeModel:
    """Minimal stand-in for PseudotimeNetworkModel used by the ordering options."""

    n_stimuli = 1

    def __init__(self, counts, modes, prot=None, inter=None, basal=None, d1=0.05):
        counts = np.asarray(counts, dtype=float)
        modes = np.asarray(modes, dtype=float)
        G = counts.shape[1]
        self.counts = counts
        self.modes = np.hstack([np.ones((counts.shape[0], 1)), modes])
        prot = modes if prot is None else np.asarray(prot, dtype=float)
        self.cell_prot = np.hstack([np.ones((counts.shape[0], 1)), prot])
        self.d = np.stack([np.full(G + 1, 0.1), np.full(G + 1, d1)])
        self.inter = np.zeros((G + 1, G + 1)) if inter is None else np.asarray(inter, float)
        self.basal = np.zeros((1, G + 1)) if basal is None else np.asarray(basal, float)
        self.tau = None
        # modes are in kon units normalised by the gene maximum (as _mixture_terms), i.e. in [0, 1]
        ks = np.stack([np.full(G + 1, 0.1), np.ones(G + 1)], axis=1)   # (G_tot, n_modes)
        self._ks = ks

    def _mixture_terms(self):
        return self._ks, np.ones(self.counts.shape[1]), self._ks.max(axis=1)


def _manifold(n_cells=200, n_genes=3, gap=None, seed=0):
    """Cells on a 1-D manifold parameterised by a hidden time t in [0, 1]."""
    t = np.linspace(0.0, 1.0, n_cells)
    coef = np.array([1.0, 1.5, 2.0, 2.5])[:n_genes]
    counts = 10.0 * np.exp(t[:, None] * coef[None, :])
    if gap is not None:                       # two well separated clusters
        counts[n_cells // 2:] *= float(gap)
    modes = (counts - counts.min(axis=0)) / (np.ptp(counts, axis=0) + 1e-12)
    return FakeModel(counts, modes), t


# ----------------------------------------------------------------------------------------------
# costs
# ----------------------------------------------------------------------------------------------
def test_bridge_end_matches_find_next_prot():
    """The vectorised flow must be exactly the E-step's find_next_prot."""
    from CardamomOT.inference.trajectory import find_next_prot

    d1 = np.array([0.05, 0.1, 0.02])
    p0 = np.array([1.0, 2.0, 0.5])
    mode_i = np.array([1.5, 0.5, 2.0])
    mode_end = np.array([3.0, 1.0, 0.2])
    s1 = np.array([2.0, 1.0, 0.7])
    ref = find_next_prot(d1, p0, np.zeros(3), np.zeros(3), mode_i, mode_end, 0.5, s1, 3.0)
    got = bridge_end(d1, p0, mode_i, mode_end, s1, 3.0, 0.5)
    assert np.allclose(got, ref)


def test_bridge_end_goes_to_the_mode_target():
    """After a long step, the protein reaches the equilibrium of the target mode."""
    d1 = np.array([0.5])
    got = bridge_end(d1, np.array([0.0]), np.array([1.0]), np.array([4.0]), np.array([2.0]), 50.0)
    assert np.isclose(got[0], 8.0, rtol=1e-4)          # mode_end * s1


def test_edge_costs_are_finite_and_geometric_is_symmetric():
    model, _ = _manifold()
    ctx = CostContext.from_model(model)
    i, j = np.array([0, 5, 10]), np.array([1, 6, 11])
    mi, mj = model.modes[i, 1:], model.modes[j, 1:]

    cij = edge_cost("geometric", mi, mj, ctx, rna_i=model.counts[i], rna_j=model.counts[j])
    cji = edge_cost("geometric", mi, mj, ctx, rna_i=model.counts[j], rna_j=model.counts[i])
    assert np.allclose(cij, cji) and np.all(cij >= 0)

    for kind in ("mechanistic", "plausibility"):
        c = edge_cost(kind, mi, mj, ctx, P0=model.cell_prot[i, 1:],
                      sig_i=np.full((3, model.counts.shape[1]), 0.5),
                      rna_i=model.counts[i], rna_j=model.counts[j])
        assert np.all(np.isfinite(c)) and np.all(c >= 0)


def test_plausibility_prefers_the_direction_the_network_pushes():
    """With a stimulus->gene activation, increasing the mode must be cheaper than decreasing it."""
    G = 2
    counts = np.ones((4, G))
    modes = np.tile([0.5, 0.5], (4, 1))
    inter = np.zeros((G + 1, G + 1))
    inter[0, 1:] = 5.0                     # stimulus activates both genes
    model = FakeModel(counts, modes, inter=inter)
    ctx = CostContext.from_model(model)

    up = plausibility_cost(np.array([[0.1, 0.1]]), np.array([[0.9, 0.9]]),
                           np.array([[0.9, 0.9]]), ctx)
    down = plausibility_cost(np.array([[0.9, 0.9]]), np.array([[0.1, 0.1]]),
                             np.array([[0.9, 0.9]]), ctx)
    assert up[0] < down[0]


def test_mechanistic_cost_is_the_network_term_plus_the_geometric_term():
    """Decomposition of the mechanistic cost: cross-entropy(Theta) + alpha_geom * geometric term."""
    from CardamomOT.pseudotime.costs import network_cost

    model, _ = _manifold()
    i, j = np.array([0, 40]), np.array([-1, -2])
    ctx = CostContext.from_model(model, alpha_geom=0.7, geom_space="modes")
    mi, mj = model.modes[i, 1:], model.modes[j, 1:]
    p0 = model.cell_prot[i, 1:]
    got = mechanistic_cost(mi, mj, p0, ctx)
    p_end = bridge_end(ctx.d1, p0, mi, mj, ctx.s1, ctx.dtau, ctx.alpha)
    expected = network_cost(p_end, mj, ctx) + 0.7 * np.linalg.norm(mj - mi, axis=1)
    assert np.allclose(got, expected)
    # and the same with the expression-space geometric term
    ctx_exp = CostContext.from_model(model, alpha_geom=0.7, geom_space="expression")
    got_exp = mechanistic_cost(mi, mj, p0, ctx_exp, rna_i=model.counts[i], rna_j=model.counts[j])
    expected_exp = network_cost(p_end, mj, ctx_exp) + 0.7 * np.linalg.norm(
        np.log1p(model.counts[j]) - np.log1p(model.counts[i]), axis=1)
    assert np.allclose(got_exp, expected_exp)


def test_network_cost_rewards_an_aligned_network():
    """A basal pushing the genes to their target mode must cost less than no network at all."""
    from CardamomOT.pseudotime.costs import network_cost

    model, _ = _manifold()
    i, j = np.array([0]), np.array([-1])
    mi, mj = model.modes[i, 1:], model.modes[j, 1:]
    ctx0 = CostContext.from_model(model)
    p_end = bridge_end(ctx0.d1, model.cell_prot[i, 1:], mi, mj, ctx0.s1, ctx0.dtau, ctx0.alpha)

    target = model.modes[j, 1:]                      # target modes (all high, ~1)
    aligned = CostContext.from_model(model, basal=np.full(model.inter.shape[0], 8.0))
    assert network_cost(p_end, mj, aligned)[0] < network_cost(p_end, mj, ctx0)[0]



# ----------------------------------------------------------------------------------------------
# Palantir
# ----------------------------------------------------------------------------------------------
def test_palantir_orders_a_monotone_manifold():
    model, t = _manifold(n_cells=200)
    roots = np.arange(5)                       # the earliest cells
    result = PalantirLike(n_neighbors=10, n_waypoints=20, edge_weight="geometric").run(
        model, roots=roots)
    rho = spearman(result.tau, t)
    assert rho > 0.95, f"Spearman(tau, t) = {rho:.3f}"
    assert result.tau[roots].max() == pytest.approx(0.0, abs=1e-9)
    assert result.tau.min() >= 0 and result.tau.max() == pytest.approx(1.0)


def test_palantir_mechanistic_weights_also_order():
    model, t = _manifold(n_cells=150)
    roots = np.arange(5)
    result = PalantirLike(n_neighbors=10, n_waypoints=15, edge_weight="mechanistic").run(
        model, roots=roots)
    assert spearman(result.tau, t) > 0.8
    assert np.all(np.isfinite(result.tau))


def test_palantir_is_deterministic():
    model, _ = _manifold(n_cells=120)
    roots = np.arange(4)
    a = PalantirLike(n_neighbors=8, n_waypoints=10).run(model, roots=roots).tau
    b = PalantirLike(n_neighbors=8, n_waypoints=10).run(model, roots=roots).tau
    assert np.array_equal(a, b)


def test_palantir_reports_unreachable_cells():
    """Two disconnected components: the cells without a path from the roots are counted and finite."""
    model, _ = _manifold(n_cells=120, gap=1e6)
    result = PalantirLike(n_neighbors=5, n_waypoints=10, edge_weight="geometric").run(
        model, roots=np.arange(4))
    assert result.diagnostics["n_unreached"] > 0
    assert np.all(np.isfinite(result.tau))
    assert np.all((result.tau >= 0) & (result.tau <= 1))


def test_palantir_requires_roots():
    model, _ = _manifold(n_cells=50)
    with pytest.raises(ValueError):
        PalantirLike().run(model)


# ----------------------------------------------------------------------------------------------
# integration with the model, and metrics
# ----------------------------------------------------------------------------------------------
def test_update_pseudotime_sets_the_bins():
    from CardamomOT.pseudotime import PseudotimeNetworkModel

    _, t = _manifold(n_cells=120)
    import anndata as ad
    import pandas as pd

    data = ad.AnnData(X=np.round(10.0 * np.exp(t[:, None] * np.array([[1.0, 1.5, 2.0]]))).astype(int),
                      obs=pd.DataFrame({"is_root": t <= np.quantile(t, 0.05)},
                                       index=[f"c{i}" for i in range(t.size)]),
                      var=pd.DataFrame(index=["G1", "G2", "G3"]))
    model = PseudotimeNetworkModel(n_genes=3)
    model.seed = 0
    model.fit_mixture_pooled(data, verb=False, min_components=2, max_components=2,
                             max_iter_kinetics=0)
    model.initialise_cell_states(data)
    tau = model.update_pseudotime(roots=None, verb=False)

    assert tau.shape == (data.n_obs,)
    assert model.bins is not None and model.bins.n_bins >= 2
    assert model.roots.sum() == int((t <= np.quantile(t, 0.05)).sum())
    assert model.pseudotime_diagnostics["method"] == "palantir"
    assert np.allclose(model.bins.cell_time, model.bins.bin_times[model.bins.bin_of_cell])


def test_pseudotime_metrics_sanity():
    t = np.linspace(0, 1, 50)
    assert spearman(t, t) == pytest.approx(1.0)
    assert abs(spearman(t, -t)) == pytest.approx(1.0)
    m = pseudotime_metrics(t, t)
    assert m["spearman"] == pytest.approx(1.0) and m["kendall"] == pytest.approx(1.0)
    nan = np.zeros(10)
    assert pseudotime_metrics(nan, nan)["spearman"] == 0.0    # undefined -> 0


def test_pseudotime_metrics_per_branch():
    """Two branches: each must be well ordered by a tau that is not monotone globally."""
    rng = np.random.default_rng(0)
    n = 120
    t = rng.uniform(0, 1, n)
    branch = (rng.random(n) < 0.5).astype(int)
    counts = np.zeros((n, 2))
    counts[:, 0] = (branch == 0) * 50 + t * 10
    counts[:, 1] = (branch == 1) * 50 + t * 10
    tau = t + 3 * branch                     # globally non-monotone in t
    m = pseudotime_metrics(tau, t, counts=counts, n_branches=2, late_quantile=0.5)
    assert m["spearman"] < 0.9               # the branches are confounded
    assert m["spearman_branch_mean"] > 0.9   # but ordered inside each branch


# ----------------------------------------------------------------------------------------------
# the cycle
# ----------------------------------------------------------------------------------------------
def test_loop_pseudotime_one_iteration():
    """Smoke test of the full cycle: ordering -> E-step -> M-step -> basins (no growth pass)."""
    import anndata as ad
    import pandas as pd

    from CardamomOT.pseudotime import PseudotimeNetworkModel

    _, t = _manifold(n_cells=120)
    X = np.round(10.0 * np.exp(t[:, None] * np.array([[1.0, 1.5, 2.0]]))).astype(int)
    data = ad.AnnData(X=X,
                      obs=pd.DataFrame({"is_root": t <= np.quantile(t, 0.05)},
                                       index=[f"c{i}" for i in range(t.size)]),
                      var=pd.DataFrame(index=["G1", "G2", "G3"]))
    model = PseudotimeNetworkModel(n_genes=3)
    model.seed = 0
    model.pseudotime_bins = 4
    model.pseudotime_growth = False
    model.fit_pseudotime(data, verb=False, max_iter=1)

    assert model.inter.shape == (4, 4)                 # 3 genes + stimulus
    assert model.tau.shape == (data.n_obs,)
    assert model.bins is not None and model.bins.n_bins >= 2
    assert len(model.pseudotime_history) == 1
    assert np.isfinite(model.pseudotime_history[0]["error_after"])
    assert (model.times_data.shape[0] == model.prot.shape[0] == model.kon_beta.shape[0])
    # the protein state of the cells is kept for the ordering of the next iteration
    assert model.cell_prot.shape == (data.n_obs, 4)


def test_loop_pseudotime_frozen_tau_keeps_the_bins():
    """pseudotime_fixed_tau: the bins of the given pseudotime are not re-estimated."""
    import anndata as ad
    import pandas as pd

    from CardamomOT.pseudotime import PseudotimeNetworkModel

    _, t = _manifold(n_cells=100)
    X = np.round(10.0 * np.exp(t[:, None] * np.array([[1.0, 1.5, 2.0]]))).astype(int)
    data = ad.AnnData(X=X,
                      obs=pd.DataFrame({"is_root": t <= np.quantile(t, 0.05)},
                                       index=[f"c{i}" for i in range(t.size)]),
                      var=pd.DataFrame(index=["G1", "G2", "G3"]))
    model = PseudotimeNetworkModel(n_genes=3)
    model.seed = 0
    model.pseudotime_bins = 5
    model.pseudotime_growth = False
    model.pseudotime_fixed_tau = True
    model.set_tau(t, roots=data.obs["is_root"].values)     # the oracle bins
    bins_before = model.bins.bin_of_cell.copy()
    model.fit_pseudotime(data, verb=False, max_iter=2)
    assert np.array_equal(model.bins.bin_of_cell, bins_before)
    assert len(model.pseudotime_history) == 2
    assert np.any(model.inter != 0)          # the network was fitted on the oracle bins


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
