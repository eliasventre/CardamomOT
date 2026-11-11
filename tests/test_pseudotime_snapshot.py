"""
Tests for the snapshot (irregular grid) design and its metrics (``PLAN.md`` §0.11):

* ``simulate_pseudotime_data.snapshot_times`` / ``sample_times``;
* ``evaluate.snapshot_metrics`` — order accuracy, separation and recovered gap ratios.

Run with:
    python -m pytest tests/test_pseudotime_snapshot.py -v
"""

import numpy as np
import pytest

from CardamomOT.pseudotime.evaluate import snapshot_metrics
from CardamomOT.pseudotime.simulate_pseudotime_data import sample_times, snapshot_times


# ----------------------------------------------------------------------------------------------
# the sampling design
# ----------------------------------------------------------------------------------------------
def test_snapshot_times_span_the_interval_with_geometric_gaps():
    snaps = snapshot_times(0.0, 96.0, n_snapshots=10, ratio=1.3)

    assert snaps.size == 10
    assert snaps[0] == pytest.approx(0.0)
    assert snaps[-1] == pytest.approx(96.0)
    gaps = np.diff(snaps)
    assert np.all(gaps > 0)                       # strictly increasing
    assert np.allclose(np.diff(gaps) / gaps[:-1], 0.3)   # geometric, ratio 1.3
    # the last gap is ratio**(n-2) times the first: strongly irregular, as a real design
    assert gaps[-1] / gaps[0] == pytest.approx(1.3 ** 8)


def test_snapshot_times_are_shifted_with_t_min():
    snaps = snapshot_times(5.0, 10.0, n_snapshots=4, ratio=2.0)
    assert snaps[0] == pytest.approx(5.0)
    assert snaps[-1] == pytest.approx(10.0)
    assert np.allclose(np.diff(snaps), np.diff(snaps)[0] * 2 ** np.arange(3))


def test_grid_design_puts_equal_cells_on_each_snapshot():
    times = sample_times(2000, 0.0, 96.0, np.random.default_rng(0), design="grid",
                         n_snapshots=10)
    values, counts = np.unique(times, return_counts=True)
    assert values.size == 10
    assert np.all(counts == 200)
    assert values[0] == pytest.approx(0.0) and values[-1] == pytest.approx(96.0)


def test_grid_design_spreads_a_remainder_evenly():
    times = sample_times(23, 0.0, 10.0, np.random.default_rng(0), design="grid", n_snapshots=4)
    _, counts = np.unique(times, return_counts=True)
    assert counts.sum() == 23
    assert counts.max() - counts.min() <= 1


def test_continuous_design_gives_one_time_per_cell():
    times = sample_times(500, 0.0, 96.0, np.random.default_rng(0), design="continuous")
    assert times.size == 500
    assert np.unique(times).size == 500
    assert times.min() >= 0.0 and times.max() <= 96.0


def test_grid_and_continuous_differ_in_their_tie_structure():
    rng = np.random.default_rng(0)
    cont = sample_times(300, 0.0, 96.0, rng, design="continuous")
    grid = sample_times(300, 0.0, 96.0, rng, design="grid", n_snapshots=10)
    assert np.unique(cont).size == 300
    assert np.unique(grid).size == 10


@pytest.mark.parametrize("kwargs, match", [
    (dict(design="ladder"), "design must be"),
    (dict(design="grid", n_snapshots=1), "n_snapshots"),
])
def test_invalid_design_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sample_times(10, 0.0, 1.0, np.random.default_rng(0), **kwargs)


def test_snapshot_times_rejects_a_bad_interval():
    with pytest.raises(ValueError, match="t_max"):
        snapshot_times(10.0, 1.0, n_snapshots=5)


# ----------------------------------------------------------------------------------------------
# the metrics
# ----------------------------------------------------------------------------------------------
def _snapshot_labels(times, n_per=10):
    return np.repeat(np.asarray(times, dtype=float), n_per)


def test_a_perfect_pseudotime_scores_one_everywhere():
    true_t = _snapshot_labels([0.0, 3.0, 12.0, 40.0, 96.0])
    out = snapshot_metrics(true_t.copy(), true_t)

    assert out["n_snapshots"] == 5
    assert out["order_accuracy"] == 1.0
    assert out["separation"] == 1.0
    assert out["gap_spearman"] == pytest.approx(1.0)
    assert out["gap_ratio_within_2x"] == 1.0
    assert np.allclose(out["gap_ratio"], 1.0, atol=1e-8)


def test_the_order_can_be_right_while_the_spacing_is_wrong():
    """
    The rank coordinate of a snapshot design: tau = 0, 1/3, 2/3, 1 whatever the true gaps.
    The ordering is perfect, the separation is perfect, but the spacing is not recovered — this is
    exactly what the gap-ratio metric is there to expose.
    """
    times = [0.0, 1.0, 5.0, 20.0]
    true_t = _snapshot_labels(times)
    tau_rank = np.repeat(np.arange(len(times), dtype=float), 10)
    out = snapshot_metrics(tau_rank, true_t)

    assert out["order_accuracy"] == 1.0
    assert out["separation"] == 1.0
    assert out["gap_ratio_within_2x"] < 0.5
    assert out["gap_ratio_max"] > 2.0            # the small gap is over-stated
    assert out["gap_ratio_min"] < 0.5            # the large gap is under-stated


def test_a_random_pseudotime_orders_the_snapshots_at_chance():
    rng = np.random.default_rng(0)
    true_t = _snapshot_labels([0.0, 5.0, 20.0, 60.0, 96.0], n_per=40)
    out = snapshot_metrics(rng.random(true_t.size), true_t)
    assert 0.2 < out["order_accuracy"] < 0.8
    assert out["separation"] < 0.3


def test_a_shuffled_order_is_detected():
    true_t = _snapshot_labels([0.0, 5.0, 20.0, 60.0], n_per=20)
    out = snapshot_metrics(-true_t, true_t)       # reversed axis
    assert out["order_accuracy"] == 0.0
    assert out["separation"] == 0.0
    assert out["gap_ratio_within_2x"] == 0.0


def test_snapshot_metrics_rejects_a_continuous_design():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="not a snapshot"):
        snapshot_metrics(rng.random(500), rng.random(500))


def test_snapshot_metrics_rejects_inconsistent_sizes():
    with pytest.raises(ValueError, match="pseudotimes"):
        snapshot_metrics(np.zeros(5), np.zeros(7))


def test_snapshot_metrics_ignores_non_finite_values():
    true_t = _snapshot_labels([0.0, 1.0, 2.0, 3.0])
    tau = true_t.copy()
    tau[:5] = np.nan
    out = snapshot_metrics(tau, true_t)
    assert out["order_accuracy"] == 1.0
