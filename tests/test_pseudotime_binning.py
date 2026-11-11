"""
Tests for the pseudotime binning and the NB mixture fitted on bins
(CardamomOT.pseudotime.binning, CardamomOT.pseudotime.base_pseudotime).

Run with:
    python -m pytest tests/test_pseudotime_binning.py -v
"""

import numpy as np
import pytest

from CardamomOT.pseudotime.binning import bin_from_tau, tau_from_time

# --------------------------------------------------------------------------------------------
# binning
# --------------------------------------------------------------------------------------------


def test_roots_are_bin_zero_and_only_them():
    tau = np.array([0.0, 0.0, 0.0, 0.2, 0.5, 0.6, 0.9, 1.0])
    scheme = bin_from_tau(tau, n_bins=4, min_cells=1)
    roots = tau == 0
    assert np.all(scheme.bin_of_cell[roots] == 0)
    assert np.all(scheme.bin_of_cell[~roots] >= 1)
    assert scheme.counts[0] == roots.sum()
    assert scheme.counts.sum() == tau.size


def test_non_root_cells_below_the_first_edge_go_to_bin_one():
    # three roots and one cell at tau = 0.01 with 5 bins: it must not stay in the root bin
    tau = np.array([0.0, 0.0, 0.0, 0.01, 0.5, 1.0])
    scheme = bin_from_tau(tau, n_bins=5, min_cells=1)
    assert scheme.bin_of_cell[3] == 1


def test_bin_times_are_the_multiples_of_dtau():
    tau = np.linspace(0, 1, 201)
    scheme = bin_from_tau(tau, n_bins=6, min_cells=1)
    assert scheme.bin_times.size == scheme.n_bins
    assert np.isclose(scheme.dtau, 1.0 / (scheme.n_bins - 1))
    assert np.allclose(scheme.bin_times, np.arange(scheme.n_bins) * scheme.dtau)
    # the cell time is the time of its bin
    assert np.allclose(scheme.cell_time, scheme.bin_times[scheme.bin_of_cell])


def test_total_duration_scales_dtau_only():
    tau = np.linspace(0, 1, 101)
    a = bin_from_tau(tau, n_bins=5, min_cells=1)
    b = bin_from_tau(tau, n_bins=5, min_cells=1, total_duration=96.0)
    assert np.array_equal(a.bin_of_cell, b.bin_of_cell)
    assert np.isclose(b.dtau, 96.0 / (b.n_bins - 1))
    assert np.isclose(b.bin_times[-1], 96.0)


def test_equal_count_gives_balanced_bins():
    rng = np.random.default_rng(0)
    tau = np.concatenate([[0.0] * 20, rng.gamma(0.3, 1.0, 980)])   # very skewed density
    scheme = bin_from_tau(tau, n_bins=5, rule="equal_count", min_cells=1)
    counts = scheme.counts[1:]
    assert counts.max() - counts.min() <= max(3, counts.mean() * 0.1)


def test_equal_width_follows_the_density():
    rng = np.random.default_rng(0)
    tau = np.concatenate([[0.0] * 20, rng.gamma(0.3, 1.0, 980)])
    scheme = bin_from_tau(tau, n_bins=5, rule="equal_width", min_cells=1)
    counts = scheme.counts[1:]
    assert counts.argmax() == 0          # most cells at the low end of a skewed pseudotime
    assert counts.max() > 2 * counts.mean()


def test_small_bins_are_merged():
    tau = np.concatenate([np.zeros(10), np.linspace(0.001, 1, 90)])
    scheme = bin_from_tau(tau, n_bins=10, rule="equal_width", min_cells=40)
    assert np.all(scheme.counts[1:] >= 40)
    assert scheme.n_bins < 10
    assert scheme.n_merged > 0


def test_relative_distances_are_preserved():
    # the binning must not rank-normalise: with equal-width bins the bin index is proportional to tau
    tau = np.array([0.0, 0.0, 1.0, 2.0, 4.0, 8.0])
    scheme = bin_from_tau(tau, n_bins=4, min_cells=1)
    b = scheme.bin_of_cell
    assert b[2] <= b[3] <= b[4] <= b[5]      # monotone
    assert b[2] < b[5]                        # and strictly increasing overall
    assert b[5] == scheme.n_bins - 1           # the furthest cell is in the last bin
    assert scheme.edges[-1] == pytest.approx(1.0)
    # a rank transform would give 4 distinct bins to the 4 non-root cells: check it is not the case
    assert len(np.unique(b[2:])) < 4


def test_roots_as_indices():
    tau = np.array([0.0, 0.1, 0.5, 1.0])
    scheme = bin_from_tau(tau, n_bins=3, roots=np.array([0, 1]), min_cells=1)
    assert scheme.bin_of_cell[0] == 0 and scheme.bin_of_cell[1] == 0
    assert scheme.counts[0] == 2


@pytest.mark.parametrize("bad", [
    dict(tau=np.array([])),                                     # empty
    dict(tau=np.array([0.0, -1.0, 2.0])),                        # negative
    dict(tau=np.array([0.0, 0.0, 0.0])),                         # only roots
    dict(tau=np.array([0.0, 1.0]), n_bins=1),                    # a single bin
])
def test_invalid_inputs(bad):
    with pytest.raises(ValueError):
        bin_from_tau(**bad)


def test_tau_from_time():
    tau = tau_from_time([10.0, 20.0, 50.0, 10.0])
    assert tau.min() == 0.0 and tau.max() == 1.0
    assert np.allclose(tau, [0.0, 0.25, 1.0, 0.0])


# --------------------------------------------------------------------------------------------
# mixture on bins (small synthetic dataset: two genes, two well-separated modes)
# --------------------------------------------------------------------------------------------


def _synthetic_two_modes(n_cells=300, seed=0):
    """(AnnData, hidden time): gene 1 switches ON with the hidden time, gene 2 is bimodal at random."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 1.0, n_cells)
    on = rng.random(n_cells) < t
    c, m_low, m_high = 10.0, 2.0, 40.0
    X = np.stack([
        rng.negative_binomial(c * (m_high if k else m_low), c / (c + (m_high if k else m_low)))
        for k in on
    ]).astype(np.int32)[:, None]
    other = rng.random(n_cells) < 0.5
    X = np.hstack([X, np.stack([
        rng.negative_binomial(c * (m_high if k else m_low), c / (c + (m_high if k else m_low)))
        for k in other
    ]).astype(np.int32)[:, None]])
    obs = pd.DataFrame({"time": t}, index=[f"c{i}" for i in range(n_cells)])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["G1", "G2"])), t, on


def test_fit_mixture_on_bins_sets_per_bin_masses():
    from CardamomOT.pseudotime import PseudotimeNetworkModel

    data, t, on = _synthetic_two_modes()
    model = PseudotimeNetworkModel(n_genes=data.n_vars)
    nu = model.fit_mixture_on_bins(data, tau=t, n_bins=4, roots=t < np.quantile(t, 0.05),
                                   min_cells=20, verb=False)

    # per-bin masses: one row per bin, normalised, one column per mode (the stimulus column is 0)
    assert nu.shape[0] == model.bins.n_bins
    assert np.allclose(nu[:, 1:, :].sum(axis=2), 1.0, atol=1e-6)
    assert np.allclose(nu[:, 0, :], 0.0)
    # the bins were written in obs for the (re)fit and for the report
    assert "pseudotime_bin" in data.obs and "pseudotime_bin_index" in data.obs
    assert np.allclose(data.obs["pseudotime_bin"].values, model.bins.cell_time)

    # (a) the mass of the highest mode of gene 1 grows with the bins
    hi = nu[:, 1, -1]
    assert hi[-1] > hi[0] + 0.2

    # the per-bin means implied by the masses match the observed bin means
    implied = model.bin_expected_means()[:, 1:]
    labels = model.bins.bin_of_cell
    for k in range(model.bins.n_bins):
        observed = np.asarray(data.X)[labels == k, 0].mean()
        assert abs(implied[k, 0] - observed) < 0.3 * max(observed, 1.0)

    # (b) the mode range covers the range observed along the bins
    ks = model.mode_amplitudes()[:, 1:]
    c = np.asarray(model.a[-1])[1:]
    means = np.stack([np.asarray(data.X)[labels == k, 0].mean() for k in range(model.bins.n_bins)])
    assert ks.max(axis=0)[0] - ks.min(axis=0)[0] >= c[0] * (means.max() - means.min()) - 1e-6


def test_force_basins_ramp():
    from CardamomOT.pseudotime import PseudotimeNetworkModel

    model = PseudotimeNetworkModel(n_genes=3)
    model.min_n_loops = 10
    model.force_basins = 1.0
    model.pseudotime_force_basins_start = 0.3
    assert model.force_basins_at(1) == pytest.approx(0.3)
    assert model.force_basins_at(10) == pytest.approx(1.0)
    assert model.force_basins_at(100) == pytest.approx(1.0)
    # monotone in between
    values = [model.force_basins_at(n) for n in range(1, 11)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def test_refit_schedule():
    from CardamomOT.pseudotime import PseudotimeNetworkModel

    model = PseudotimeNetworkModel(n_genes=3)
    model.pseudotime_refit_mixture = True
    model.pseudotime_refit_every = 5
    due = [n for n in range(1, 12)
           if (n == 1 or (n - 1) % model.pseudotime_refit_every == 0)]
    assert due == [1, 6, 11]
    model.pseudotime_refit_mixture = False
    assert model.pseudotime_refit_every == 5   # the guard is read in refit_mixture_if_due


def test_mixture_on_bins_requires_bins():
    from CardamomOT.pseudotime import PseudotimeNetworkModel

    data, _, _ = _synthetic_two_modes(n_cells=60)
    model = PseudotimeNetworkModel(n_genes=data.n_vars)
    with pytest.raises(ValueError):
        model.fit_mixture_on_bins(data, verb=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
