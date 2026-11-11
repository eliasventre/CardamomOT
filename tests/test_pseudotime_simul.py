"""
Tests for the M5 metrics (``PLAN.md`` §0.10): simulated mRNA marginals and degradation recovery.

Run with:
    python -m pytest tests/test_pseudotime_simul.py -v
"""

import numpy as np
import pytest

from CardamomOT.pseudotime.evaluate import degradation_metrics, marginal_agreement


class _Bins:
    def __init__(self, bin_of_cell, bin_times):
        self.bin_of_cell = np.asarray(bin_of_cell)
        self.bin_times = np.asarray(bin_times, dtype=float)


class SimModel:
    """
    Minimal stand-in for a fitted + simulated PseudotimeNetworkModel: only what the M5 metrics read.

    The burst size is ``kz = max(sigmoid_amplitudes(ks)) * k1`` and the expected count of a state is
    ``kon * kz / c`` (mean of ``negative_binomial(kz*kon, c/(c+1))``).
    """

    n_stimuli = 1

    def __init__(self, n_bins=3, n_per=10, genes=2, kon=1.0, kz=2.0, c=0.5):
        G_tot = genes + self.n_stimuli
        self.ks = np.tile(np.array([0.0, 1.0]), (G_tot, 1))
        self.k1 = np.full(G_tot, float(kz))
        self.a = np.vstack([np.zeros((1, G_tot)), np.ones((1, G_tot)),
                            np.full((1, G_tot), float(c))])
        n_slots = n_bins * n_per
        self.kon_theta = np.full((n_slots, G_tot), float(kon))
        self.times_data = np.repeat(np.arange(n_bins, dtype=float), n_per)
        self.bins = _Bins(np.repeat(np.arange(n_bins), n_per), np.arange(n_bins, dtype=float))

    def mixture_terms(self):
        return self.ks, np.ones(self.ks.shape[0]), self.k1

    @property
    def expected(self):
        """Expected count per gene of a slot: kon * kz / c."""
        return self.kon_theta[:, self.n_stimuli:] * (self.k1[-1] / self.a[-1, -1])


class _Data:
    def __init__(self, X):
        self.X = np.asarray(X, dtype=float)


class _SparseData:
    def __init__(self, X):
        import scipy.sparse as sp
        self.X = sp.csr_matrix(np.asarray(X, dtype=float))


def _data_like(model, scale=1.0):
    """Observed counts equal to the model's expectation (x ``scale``), per slot."""
    return _Data(model.expected * scale)


# ----------------------------------------------------------------------------------------------
# marginals
# ----------------------------------------------------------------------------------------------
def test_marginals_are_exactly_one_when_the_simulation_matches_the_data():
    model = SimModel()
    out = marginal_agreement(model, _data_like(model))

    assert out["frac_within_2x"] == 1.0
    assert np.allclose([b["ratio"] for b in out["per_bin"]], 1.0)
    assert np.allclose(out["per_gene"]["ratio"], 1.0)
    assert out["ratio_min"] == pytest.approx(1.0) and out["ratio_max"] == pytest.approx(1.0)


def test_expected_count_uses_kz_over_c():
    """kon * kz / c, not kon * kz: the negative-binomial mean with p = c/(c+1)."""
    model = SimModel(kon=1.0, kz=2.0, c=0.5)
    assert np.allclose(model.expected, 4.0)


def test_a_global_scale_error_is_reported_per_bin_and_per_gene():
    model = SimModel()
    out = marginal_agreement(model, _data_like(model, scale=3.0))   # simulated 3x too low

    assert np.allclose([b["ratio"] for b in out["per_bin"]], 1.0 / 3.0)
    assert np.allclose(out["per_gene"]["ratio"], 1.0 / 3.0)
    assert out["frac_within_2x"] == 0.0
    assert out["ratio_min"] == pytest.approx(1 / 3, rel=1e-6)


def test_marginals_are_within_2x_for_a_moderate_error():
    model = SimModel()
    out = marginal_agreement(model, _data_like(model, scale=1.5))
    assert out["frac_within_2x"] == 1.0


def test_marginals_accept_a_sparse_matrix():
    model = SimModel()
    dense = marginal_agreement(model, _data_like(model))
    sparse = marginal_agreement(model, _SparseData(model.expected))
    assert sparse["frac_within_2x"] == dense["frac_within_2x"]


def test_marginals_report_the_number_of_real_cells_per_bin():
    model = SimModel(n_bins=3, n_per=4)
    out = marginal_agreement(model, _data_like(model))
    assert [b["n_cells"] for b in out["per_bin"]] == [4, 4, 4]


def test_gene_count_mismatch_raises():
    model = SimModel(genes=2)
    with pytest.raises(ValueError, match="genes"):
        marginal_agreement(model, _Data(np.zeros((30, 5))))


# ----------------------------------------------------------------------------------------------
# degradations
# ----------------------------------------------------------------------------------------------
def test_degradation_metrics_recover_d1_and_the_d0_d1_ratio():
    model = SimModel()
    G_tot = model.k1.size
    true_d = np.vstack([np.full(G_tot, 0.5), np.full(G_tot, 0.1)])
    model.d_t = np.tile(np.array([[0.5] * G_tot, [0.08] * G_tot]), (2, 1, 1))
    model.ratios = np.tile(np.full(G_tot, 6.25), (2, 1))

    out = degradation_metrics(model, true_d)
    assert out["d1_inferred"] == pytest.approx([0.08, 0.08])
    assert out["d1_true"] == pytest.approx([0.1, 0.1])
    assert out["d1_ratio"] == pytest.approx([0.8, 0.8])
    assert out["d1_within_2x"] == 1.0
    assert out["d0_d1_inferred"] == pytest.approx([6.25, 6.25])
    assert out["d0_d1_true"] == pytest.approx([5.0, 5.0])


def test_degradation_metrics_flags_a_factor_three_error():
    model = SimModel()
    G_tot = model.k1.size
    true_d = np.vstack([np.full(G_tot, 0.5), np.full(G_tot, 0.1)])
    model.d_t = np.tile(np.array([[0.5] * G_tot, [0.3] * G_tot]), (1, 1, 1))
    out = degradation_metrics(model, true_d)
    assert out["d1_within_2x"] == 0.0
    assert out["d1_ratio"] == pytest.approx([3.0, 3.0])
