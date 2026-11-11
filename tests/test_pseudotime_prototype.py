"""
Tests of the prototype: roots, bins sized by the roots and the growth, the gap search and the
E-step with the gap chosen inside it (PROTOTYPE_PLAN.md §8.1).

Run with:
    python -m pytest tests/test_pseudotime_prototype.py -v
"""

import numpy as np
import pytest

from CardamomOT.pseudotime.binning import (bin_by_sizes, cell_times, gaps_from_cell_times,
                                           growth_sizes)
from CardamomOT.pseudotime.gaps import (GapSearch, PairInputs, choose_gap, curve_flags, end_proteins,
                                        flow, flow_dilution, gap_curves, joint_gaps_dp, pair_terms,
                                        relative_ce)


# ----------------------------------------------------------------------------------------------
# Roots
# ----------------------------------------------------------------------------------------------
def _line_data(n=500, seed=0):
    """Cells along a line: expression increases with a hidden time t in [0, 1]."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(n))
    mean = 2.0 + 40.0 * t[:, None] * np.array([[1.0, 0.7, 0.4]])
    X = rng.poisson(mean).astype(float)
    data = ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                      var=pd.DataFrame(index=["G1", "G2", "G3"]))
    return data, t


def test_roots_from_the_user_column():
    from CardamomOT.pseudotime.roots import select_roots

    data, t = _line_data(100)
    data.obs["is_root"] = t < 0.1
    mask, info = select_roots(data)
    assert info["source"] == "user"
    assert np.array_equal(mask, t < 0.1)


def test_fallback_roots_are_two_percent_at_the_low_end():
    from CardamomOT.pseudotime.roots import select_roots

    data, t = _line_data(500)
    mask, info = select_roots(data, fraction=0.02, seed=0)
    assert info["source"] == "fallback"
    assert mask.sum() == int(np.ceil(0.02 * 500))
    assert t[mask].mean() < 0.15                     # the least expressed end = the start
    again, _ = select_roots(data, fraction=0.02, seed=0)
    assert np.array_equal(mask, again)               # deterministic


def test_fallback_roots_side_can_be_forced():
    from CardamomOT.pseudotime.roots import select_roots

    data, t = _line_data(500)
    low, info_low = select_roots(data, side="low", seed=0)
    high, info_high = select_roots(data, side="high", seed=0)
    assert not np.any(low & high)
    assert info_low["side"] == "low" and info_high["side"] == "high"
    assert abs(t[low].mean() - t[high].mean()) > 0.7


# ----------------------------------------------------------------------------------------------
# Bin sizes, bins by rank, time per cell
# ----------------------------------------------------------------------------------------------
def test_sizes_without_growth_are_the_number_of_roots():
    sizes = growth_sizes(200, 2000)
    assert np.array_equal(sizes, np.full(10, 200))


def test_sizes_follow_a_constant_growth():
    sizes = growth_sizes(100, 3000, gaps=np.ones(30), rates=np.full(30, 0.1), k_max=30)
    assert sizes.sum() == 3000 and sizes[0] == 100
    ratios = sizes[2:] / sizes[1:-1]
    assert np.allclose(ratios, np.exp(0.1), atol=0.03)


def test_sizes_respect_k_max_and_keep_equal_ratios():
    sizes = growth_sizes(10, 2000, k_max=30)
    assert sizes.size == 30 and sizes.sum() == 2000 and sizes[0] == 10
    assert sizes[1:].max() - sizes[1:].min() <= 1


def test_sizes_respect_k_min_and_n_min():
    sizes = growth_sizes(500, 1000, k_min=4)
    assert sizes.size == 4 and sizes.sum() == 1000
    sizes = growth_sizes(10, 200, n_min=50, k_min=2)
    assert np.all(sizes[1:] >= 50) and sizes.sum() == 200


def test_bins_by_rank_keep_the_roots_and_the_order():
    rng = np.random.default_rng(0)
    tau = np.concatenate([np.zeros(20), rng.random(180)])
    roots = tau == 0
    sizes = np.array([20, 50, 60, 70])
    times = np.array([0.0, 1.0, 3.0, 7.0])
    scheme = bin_by_sizes(tau, sizes, roots, bin_times=times)
    assert np.array_equal(scheme.counts, sizes)
    assert np.all(scheme.bin_of_cell[roots] == 0)
    for k in range(1, 3):                            # every cell of bin k precedes those of bin k+1
        assert tau[scheme.bin_of_cell == k].max() <= tau[scheme.bin_of_cell == k + 1].min()
    assert np.allclose(scheme.bin_times, times)
    assert np.allclose(cell_times(scheme), times[scheme.bin_of_cell])
    with pytest.raises(ValueError):
        bin_by_sizes(tau, np.array([21, 49, 60, 70]), roots)
    with pytest.raises(ValueError):
        bin_by_sizes(tau, sizes, roots, bin_times=[0.0, 2.0, 1.0, 3.0])


def test_starting_gaps_come_from_the_time_of_the_cells():
    labels = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3])
    t_old = np.array([0, 0, 2.0, 4.0, 3.0, 5.0, 7.0, 10.0, 14.0])
    gaps = gaps_from_cell_times(labels, t_old, min_gap=0.1)
    assert np.allclose(gaps, [3.0, 3.0, 6.0])
    labels2 = np.array([0, 1, 2])                    # decreasing medians -> floored
    assert np.allclose(gaps_from_cell_times(labels2, np.array([0.0, 5.0, 4.0]), 0.5), [5.0, 0.5])


# ----------------------------------------------------------------------------------------------
# Gap search: elementary terms
# ----------------------------------------------------------------------------------------------
def test_flows_match_the_e_step():
    from CardamomOT.inference.trajectory import find_next_prot, find_next_prot_dilution

    rng = np.random.default_rng(0)
    G = 4
    d1, s1 = rng.uniform(0.02, 0.2, G), rng.uniform(0.5, 2, G)
    P0, mi, me, al = rng.random(G), rng.random(G), rng.random(G), rng.random(G)
    assert np.allclose(flow(P0, mi, me, al, s1, d1, 7.0),
                       find_next_prot(d1, P0, P0, P0, mi, me, al, s1, 7.0))
    d1e, ab = d1 + rng.uniform(0, 0.05, G), rng.random(G)
    assert np.allclose(flow_dilution(d1, d1e, P0, mi, me, al, ab, s1, 7.0),
                       find_next_prot_dilution(d1, d1e, P0, mi, me, al, ab, s1, 7.0))


def test_relative_ce_matches_the_network_module():
    from CardamomOT.inference.network import cross_entropy

    p, t = np.array([0.1, 0.5, 0.9, 1e-9]), np.array([0.2, 0.5, 1.0, 0.0])
    assert np.allclose(relative_ce(p, t), cross_entropy(p, t))
    assert np.all(relative_ce(p, t) >= -1e-12)


def _pair(n_s=6, n_t=5, G=3, ns=1, seed=0, dilution=False):
    rng = np.random.default_rng(seed)
    G_tot = G + ns
    inter = rng.normal(0, 1.5, (G_tot, G_tot))
    return PairInputs(
        P0=rng.random((n_s, G)), mode_init=rng.random((n_s, G)), alpha=rng.uniform(0.1, 0.9, (n_s, G)),
        rna_src=rng.poisson(5, (n_s, G)).astype(float), kon_src=rng.random((n_s, G)),
        mode_end=rng.random((n_t, G)), tau_end=rng.random((n_t, G)),
        rna_tgt=rng.poisson(5, (n_t, G)).astype(float), kon_tgt=rng.random((n_t, G)),
        s1=rng.uniform(0.5, 2, G), d1=rng.uniform(0.03, 0.2, G), basal=rng.normal(0, 1, G_tot),
        inter=inter, w=rng.random(G), stim=np.ones(ns), ns=ns, scale_proteins=1.0,
        b_src=rng.uniform(0, 0.05, n_s) if dilution else None,
        b_tgt=rng.uniform(0, 0.05, n_t) if dilution else None)


def test_network_term_matches_a_pair_by_pair_computation():
    from CardamomOT.inference.network import cross_entropy
    from CardamomOT.inference.trajectory import base_kon_vector, find_next_prot

    inp = _pair()
    delta = 5.0
    net, prot = pair_terms(inp, delta)
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            P = find_next_prot(inp.d1, inp.P0[i], inp.P0[i], inp.P0[i], inp.mode_init[i],
                               inp.mode_end[j], inp.alpha[i], inp.s1, delta)
            full = np.concatenate([inp.stim * inp.scale_proteins, P])[None, :]
            sig = base_kon_vector(inp.basal, inp.inter, full)[0, inp.ns:, 1]
            assert np.isclose(net[i, j], np.sum(inp.w * cross_entropy(sig, inp.tau_end[j])))
            assert np.isclose(prot[i, j], np.sum((P - inp.P0[i]) ** 2))


def test_dilution_end_proteins_match_my_otdistance_flow():
    from CardamomOT.inference.trajectory import _birth_alpha_row, find_next_prot_dilution

    inp = _pair(dilution=True)
    delta = 4.0
    P = end_proteins(inp, delta, slice(None))
    for i in range(inp.shape[0]):
        d1_i = inp.d1 + inp.b_src[i]
        c_i = inp.d1 / d1_i
        for j in range(inp.shape[1]):
            d1_j = inp.d1 + inp.b_tgt[j]
            ab = _birth_alpha_row(inp.alpha[i], inp.kon_src[i], inp.kon_tgt[j], inp.alpha_tol)
            ref = find_next_prot_dilution(d1_i, d1_j, inp.P0[i], inp.mode_init[i] * c_i,
                                          inp.mode_end[j] * (inp.d1 / d1_j), inp.alpha[i], ab,
                                          inp.s1, delta)
            assert np.allclose(P[i, j], ref)


def test_terms_do_not_depend_on_the_chunk_size():
    inp = _pair(n_s=9, n_t=7)
    a = pair_terms(inp, 3.0, chunk_elems=10)
    b = pair_terms(inp, 3.0, chunk_elems=10 ** 7)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_scales_are_fixed_so_curve_values_do_not_depend_on_the_grid():
    inp = _pair()
    grid = np.array([1.0, 3.0, 9.0, 27.0])
    f = {"F2": (1.0, 0.0), "F3": (1.0, 1.0)}
    a = gap_curves(inp, grid, ref_delta=5.0, formulations=f)
    b = gap_curves(inp, grid[::-1], ref_delta=5.0, formulations=f)
    c = gap_curves(inp, grid[2:3], ref_delta=5.0, formulations=f)
    for name in f:
        assert np.allclose(a[name], b[name][::-1])
        assert np.isclose(a[name][2], c[name][0])


# ----------------------------------------------------------------------------------------------
# Gap search: recovery, null, choice
# ----------------------------------------------------------------------------------------------
def _known_gap_pair(delta_true=12.0, n=40, seed=0):
    """Gene 1 driven by the protein of gene 0, rising after a switch: the targets are exactly the
    network output at delta_true, so the network term is 0 there and positive elsewhere."""
    rng = np.random.default_rng(seed)
    G, ns = 2, 1
    inter = np.zeros((G + ns, G + ns))
    inter[1, 2] = 6.0                                # gene 0 -> gene 1
    basal = np.array([0.0, 0.0, -3.0])
    P0 = np.column_stack([rng.uniform(0.0, 0.1, n), rng.uniform(0.0, 0.1, n)])
    mode_init = np.column_stack([np.ones(n), np.full(n, 0.1)])
    mode_end = mode_init.copy()
    alpha = np.full((n, G), 0.5)
    s1, d1 = np.ones(G), np.full(G, 0.1)
    P_true = flow(P0, mode_init, mode_end, alpha, s1, d1, delta_true)
    from scipy.special import expit
    tau_end = expit(basal[ns:] + P_true @ inter[ns:, ns:] + inter[0, ns:])
    return PairInputs(P0=P0, mode_init=mode_init, alpha=alpha, rna_src=np.ones((n, G)),
                      kon_src=mode_init, mode_end=mode_end, tau_end=tau_end, rna_tgt=np.ones((n, G)),
                      kon_tgt=mode_end, s1=s1, d1=d1, basal=basal, inter=inter, w=np.ones(G),
                      stim=np.ones(ns), ns=ns)


def test_a_known_gap_is_recovered():
    delta_true = 12.0
    inp = _known_gap_pair(delta_true)
    grid = GapSearch(n_grid=21).grid(inp.d1)
    curve = gap_curves(inp, grid, ref_delta=10.0, formulations={"F1": (0.0, 0.0)})["F1"]
    choice = choose_gap(grid, curve, log_prior=np.log(10.0))
    assert not choice["boundary"] and not choice["flat"]
    assert abs(choice["gap"] / delta_true - 1) < 0.2


def test_no_time_signal_gives_a_flat_curve():
    inp = _pair()
    inp.inter = np.zeros_like(inp.inter)             # the network does not see the proteins
    grid = np.geomspace(1, 100, 9)
    curve = gap_curves(inp, grid, ref_delta=10.0, formulations={"F1": (0.0, 0.0)})["F1"]
    assert curve_flags(curve, flat_tol=0.02)["flat"]


def test_choice_without_penalty_is_the_minimum_and_the_prior_pulls():
    grid = np.geomspace(1, 100, 11)
    curve = (np.log(grid) - np.log(10.0)) ** 2 + 1.0
    free = choose_gap(grid, curve, log_prior=np.log(50.0))
    assert np.isclose(free["gap"], 10.0, rtol=0.05)
    pulled = choose_gap(grid, curve, log_prior=np.log(50.0), prior_weight=100.0)
    assert pulled["gap"] > 30.0
    assert choose_gap(grid, np.linspace(2, 1, 11), log_prior=0.0)["boundary"]


def test_dp_equals_brute_force_and_its_limits():
    import itertools

    rng = np.random.default_rng(1)
    grid = np.geomspace(1, 16, 5)
    curves = rng.random((4, 5)) + 0.5
    lp, r1, r2 = np.log(4.0), 0.1, 0.3
    idx, _ = joint_gaps_dp(grid, curves, lp, r1, r2)
    x = np.log(grid)
    norm = curves / curves.mean(axis=1, keepdims=True)

    def total(path):
        u = sum(norm[k, i] + r1 * (x[i] - lp) ** 2 for k, i in enumerate(path))
        return u + sum(r2 * (x[path[k + 1]] - x[path[k]]) ** 2 for k in range(len(path) - 1))

    best = min(itertools.product(range(5), repeat=4), key=total)
    assert np.isclose(total(tuple(idx)), total(best))
    idx0, _ = joint_gaps_dp(grid, curves, lp, 0.0, 0.0)
    assert np.array_equal(idx0, np.argmin(curves, axis=1))
    idx_big, _ = joint_gaps_dp(grid, curves, lp, 0.0, 1e6)
    assert np.all(idx_big == idx_big[0])


# ----------------------------------------------------------------------------------------------
# Corrections of F1: forecast (F6) and self null
# ----------------------------------------------------------------------------------------------
def _with_bounds(inp):
    G = inp.P0.shape[1]
    inp.k_low, inp.k_high = np.full(G, 0.1), np.full(G, 1.0)
    return inp


def test_forecast_without_network_is_flat():
    from CardamomOT.pseudotime.gaps import forecast_sigmoids

    inp = _with_bounds(_pair())
    inp.inter = np.zeros_like(inp.inter)
    sig = forecast_sigmoids(inp, np.geomspace(1, 100, 6))
    assert np.allclose(sig, sig[0])                  # the sigmoid ignores the proteins
    curve = gap_curves(inp, np.geomspace(1, 100, 6), 10.0, {"F6": "forecast"})["F6"]
    assert curve_flags(curve, 0.02)["flat"]


def test_forecast_reaches_the_fixed_point_of_the_network():
    from scipy.special import expit
    from CardamomOT.pseudotime.gaps import forecast_sigmoids

    inp = _with_bounds(_pair())
    inp.inter = np.zeros_like(inp.inter)
    inp.basal = np.full(inp.basal.shape, 4.0)
    long = forecast_sigmoids(inp, np.array([1.0, 2000.0]))[-1]
    assert np.allclose(long, expit(4.0))
    # the proteins themselves reach scale * kon: check through a gene-to-gene network
    inp2 = _with_bounds(_pair(seed=3))
    a = forecast_sigmoids(inp2, np.array([5.0]), steps_per_relaxation=20)
    b = forecast_sigmoids(inp2, np.array([5.0]), steps_per_relaxation=400)
    assert np.allclose(a, b, atol=1e-6)              # RK4 converged


def test_forecast_recovers_a_gap_generated_by_the_network():
    """Targets = the network's own forecast after a known gap: the forecast curve is minimal there."""
    from CardamomOT.pseudotime.gaps import forecast_sigmoids

    rng = np.random.default_rng(0)
    G, ns, n = 2, 1, 40
    inter = np.zeros((G + ns, G + ns))
    inter[0, 1] = 4.0                                # stimulus -> gene 0
    inter[1, 2] = 6.0                                # gene 0 -> gene 1
    inp = PairInputs(P0=rng.uniform(0, 0.1, (n, G)), mode_init=np.zeros((n, G)),
                     alpha=np.full((n, G), 0.5), rna_src=np.ones((n, G)), kon_src=np.zeros((n, G)),
                     mode_end=np.zeros((n, G)), tau_end=np.zeros((n, G)), rna_tgt=np.ones((n, G)),
                     kon_tgt=np.zeros((n, G)), s1=np.ones(G), d1=np.full(G, 0.1),
                     basal=np.array([0.0, -2.0, -3.0]), inter=inter, w=np.ones(G), stim=np.ones(1),
                     ns=ns, k_low=np.full(G, 0.0), k_high=np.full(G, 1.0))
    delta_true = 12.0
    inp.tau_end = forecast_sigmoids(inp, np.array([delta_true]))[0]
    grid = np.geomspace(1, 100, 25)
    curve = gap_curves(inp, grid, 10.0, {"F6": "forecast"})["F6"]
    choice = choose_gap(grid, curve, log_prior=np.log(10.0))
    assert not choice["boundary"] and abs(choice["gap"] / delta_true - 1) < 0.2


def test_self_null_is_the_difference_of_two_normalised_f1_curves():
    from CardamomOT.pseudotime.gaps import median_scale

    inp, null = _pair(seed=0), _pair(seed=1)
    grid = np.geomspace(1, 50, 5)
    out = gap_curves(inp, grid, 5.0, {"F1": (0.0, 0.0), "N": "self_null"}, null_inp=null)
    f1_null = gap_curves(null, grid, 5.0, {"F1": (0.0, 0.0)})["F1"]
    expected = out["F1"] / median_scale(np.abs(out["F1"])) - f1_null / median_scale(np.abs(f1_null))
    assert np.allclose(out["N"], expected)
    with pytest.raises(ValueError):
        gap_curves(inp, grid, 5.0, {"N": "self_null"})


def test_normalisation_keeps_the_sign_of_a_difference_curve():
    curve = np.array([0.5, -1.0, 0.2])
    assert choose_gap(np.array([1.0, 2.0, 4.0]), curve, log_prior=0.0)["index"] == 1


# ----------------------------------------------------------------------------------------------
# The E-step (a small fitted model)
# ----------------------------------------------------------------------------------------------
SNAP_TIMES = np.array([0.0, 2.0, 5.0, 10.0, 20.0])


@pytest.fixture(scope="module")
def fitted():
    """5 snapshots of 60 cells, 3 genes switching on at different speeds; oracle cycle, 1 iteration."""
    import anndata as ad
    import pandas as pd

    from CardamomOT.pseudotime import PseudotimeNetworkModel

    rng = np.random.default_rng(0)
    n_per = 60
    snap = np.repeat(np.arange(SNAP_TIMES.size), n_per)
    t = SNAP_TIMES[snap]
    p_on = 1 - np.exp(-t[:, None] / np.array([[3.0, 8.0, 15.0]]))
    on = rng.random(p_on.shape) < p_on
    X = rng.negative_binomial(2, 2 / (2 + np.where(on, 30.0, 2.0))).astype(float)
    data = ad.AnnData(X=X, obs=pd.DataFrame({"is_root": snap == 0, "time": t},
                                            index=[f"c{i}" for i in range(X.shape[0])]),
                      var=pd.DataFrame({"d1": [0.1] * 3, "d0": [0.5] * 3}, index=["G1", "G2", "G3"]))
    np.random.seed(0)
    model = PseudotimeNetworkModel(n_genes=3)
    model.seed = 0
    model.pseudotime_growth = False
    model.prepare_pseudotime(data, verb=False)
    model.tau = t.astype(float)
    model.set_bins(bin_by_sizes(t, np.bincount(snap), snap == 0, bin_times=SNAP_TIMES))
    model.pseudotime_fixed_tau = True
    model.pseudotime_duration_mode = "grid"
    model.loop_pseudotime(data, verb=False, roots=model.roots, max_iter=1)
    data_rna = model._parse_input(data)
    ks, s1, _ = model.mixture_terms()
    return model, data, data_rna, ks, s1


def _scaffold(model, data_rna):
    S = model._scaffolding(data_rna)
    model._stim_schedule = model._build_stimulus_schedule(S["times"])
    return S


def test_estep_without_search_is_the_base_estep(fitted):
    from CardamomOT.pseudotime.estep import estep_with_gaps

    model, _, data_rna, ks, s1 = fitted
    S = _scaffold(model, data_rna)
    assert S["n_batches"] == 1
    np.random.seed(3)
    Eb = model._estep(data_rna, S, model.basal, model.inter, ks, s1, 2)
    np.random.seed(3)
    En = estep_with_gaps(model, data_rna, S, model.basal, model.inter, ks, s1, 2, np.diff(S["times"]))
    for key in ("y_prot", "y_kon", "y_rna", "y_real", "to_keep", "formodes", "y_alpha"):
        assert np.array_equal(Eb[key], En[key]), key
    assert np.allclose(En["times"], S["times"]) and np.allclose(En["gaps"], np.diff(S["times"]))


def test_estep_search_chooses_one_gap_per_pair(fitted):
    from CardamomOT.pseudotime.estep import estep_with_gaps

    model, _, data_rna, ks, s1 = fitted
    S = _scaffold(model, data_rna)
    gaps = np.diff(S["times"])
    search = GapSearch(start=1, n_grid=7, seed=1)
    runs = []
    for _ in range(2):
        np.random.seed(4)
        runs.append(estep_with_gaps(model, data_rna, S, model.basal, model.inter, ks, s1, 2, gaps,
                                    search=search))
    E = runs[0]
    assert len(E["gap_diag"]) == S["K"] - 1
    assert np.allclose(E["times"], np.concatenate([[0], np.cumsum(E["gaps"])]))
    assert np.allclose(E["vect_t_sim"], np.repeat(E["times"], S["n_full"]))
    assert np.allclose(runs[0]["gaps"], runs[1]["gaps"])          # deterministic search
    for d in E["gap_diag"]:
        assert np.isclose(E["gaps"][d["pair"]], d["gap_used"])


def test_estep_given_mode_records_but_keeps_the_gaps(fitted):
    from CardamomOT.pseudotime.estep import estep_with_gaps

    model, _, data_rna, ks, s1 = fitted
    S = _scaffold(model, data_rna)
    gaps = np.diff(S["times"])
    np.random.seed(5)
    E = estep_with_gaps(model, data_rna, S, model.basal, model.inter, ks, s1, 2, gaps,
                        search=GapSearch(start=1, n_grid=5), propagate="given")
    assert np.allclose(E["gaps"], gaps) and len(E["gap_diag"]) == S["K"] - 1


def test_estep_flat_curves_keep_the_current_gaps(fitted):
    from CardamomOT.pseudotime.estep import estep_with_gaps

    model, _, data_rna, ks, s1 = fitted
    S = _scaffold(model, data_rna)
    gaps = np.diff(S["times"])
    zero_b, zero_i = np.zeros_like(model.basal), np.zeros_like(model.inter)
    np.random.seed(6)
    E = estep_with_gaps(model, data_rna, S, zero_b, zero_i, ks, s1, 2, gaps,
                        search=GapSearch(start=1, n_grid=5))
    assert all(d["flat"] for d in E["gap_diag"])
    assert np.allclose(E["gaps"], gaps)


def test_estep_search_waits_for_its_first_iteration(fitted):
    from CardamomOT.pseudotime.estep import estep_with_gaps

    model, _, data_rna, ks, s1 = fitted
    S = _scaffold(model, data_rna)
    np.random.seed(7)
    E = estep_with_gaps(model, data_rna, S, model.basal, model.inter, ks, s1, 1, np.diff(S["times"]),
                        search=GapSearch(start=2))
    assert E["gap_diag"] == []


def test_estep_with_several_batches(fitted):
    from CardamomOT.pseudotime.estep import _subsample, estep_with_gaps

    model, _, data_rna, ks, s1 = fitted
    old = model.batch_size_traj
    try:
        model.batch_size_traj = 20
        S = _scaffold(model, data_rna)
        assert S["n_batches"] == 3
        gaps = np.diff(S["times"])
        np.random.seed(8)
        E = estep_with_gaps(model, data_rna, S, model.basal, model.inter, ks, s1, 2, gaps,
                            search=GapSearch(start=1, n_grid=5, max_cells=30))
        assert np.allclose(E["times"], np.concatenate([[0], np.cumsum(E["gaps"])]))
        assert all(d["n_sources"] == 30 for d in E["gap_diag"])
        # without the search: same distribution of the trajectories as the base E-step
        # (the search left the schedule of its chosen times: back to the times of S)
        model._stim_schedule = model._build_stimulus_schedule(S["times"])
        np.random.seed(9)
        Eb = model._estep(data_rna, S, model.basal, model.inter, ks, s1, 2)
        np.random.seed(10)
        En = estep_with_gaps(model, data_rna, S, model.basal, model.inter, ks, s1, 2, gaps)
        N = S["n_full"]
        for k in range(1, S["K"]):
            a, b = Eb["y_prot"][k * N:(k + 1) * N, 1:], En["y_prot"][k * N:(k + 1) * N, 1:]
            se = np.sqrt(a.var(axis=0) / N + b.var(axis=0) / N) + 1e-9
            assert np.all(np.abs(a.mean(axis=0) - b.mean(axis=0)) < 4 * se), k
    finally:
        model.batch_size_traj = old
    # the search subsample takes cells of every batch
    groups = np.repeat([0, 1, 2], [20, 20, 20])
    sub = _subsample(np.random.default_rng(0), np.arange(60), groups, 30)
    assert sub.size == 30 and set(groups[sub]) == {0, 1, 2}


def test_mrna_bridge_is_not_supported(fitted):
    from CardamomOT.pseudotime.estep import estep_with_gaps

    model, _, data_rna, ks, s1 = fitted
    S = _scaffold(model, data_rna)
    model.mrna_bridge = True
    try:
        with pytest.raises(NotImplementedError):
            estep_with_gaps(model, data_rna, S, model.basal, model.inter, ks, s1, 2, np.diff(S["times"]))
    finally:
        model.mrna_bridge = False


# ----------------------------------------------------------------------------------------------
# Switching fractions carried across iterations (§4.7)
# ----------------------------------------------------------------------------------------------
def test_pool_takes_the_closest_state_inside_the_time_window():
    from CardamomOT.pseudotime.estep import AlphaPool

    pool = AlphaPool(prot=np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
                     kon=np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
                     alpha=np.array([[0.1, 0.1], [0.2, 0.2], [0.9, 0.9]]),
                     time=np.array([0.0, 0.0, 5.0]))
    # identical state at time 0 -> its alpha exactly
    assert np.allclose(pool.match(np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]]), -1, 1, 3), 0.2)
    # the state at time 5 is identical to the query but outside the window: never chosen
    assert np.allclose(pool.match(np.zeros((1, 2)), np.zeros((1, 2)), -1, 1, 3), 0.1)
    assert pool.match(np.zeros((1, 2)), np.zeros((1, 2)), 10, 12, 3) is None


def test_pool_reproduces_the_base_reassignment_when_bins_do_not_change(fitted):
    from CardamomOT.pseudotime.estep import AlphaPool, estep_with_gaps

    model, _, data_rna, ks, s1 = fitted
    S = _scaffold(model, data_rna)
    gaps = np.diff(S["times"])
    np.random.seed(11)
    E1 = estep_with_gaps(model, data_rna, S, model.basal, model.inter, ks, s1, 2, gaps)
    N = S["n_full"]
    model._stim_schedule = model._build_stimulus_schedule(S["times"])
    old = {k: np.array(E1[k], copy=True) for k in ("y_prot", "y_kon", "y_rna", "y_alpha", "y_samples",
                                                    "y_real", "formodes")}
    np.random.seed(12)
    base = model.estimate_trajectories_given_model(
        S["vect_t"], S["times"], np.zeros(S["n_cells"], dtype=int), [0], data_rna,
        old["y_prot"], old["formodes"], old["y_kon"], old["y_rna"], old["y_alpha"], old["y_samples"],
        model.basal, model.inter, s1, ks, [S["init_cells"][:N]], np.zeros(S["K"] * N),
        np.zeros(S["n_cells"], dtype=bool), offset_init=[0], N_full=[N], N_samples=[N], n_iter=2,
        intensity_prior=model.intensity_prior, real_cell_batches=S["real_cell_batches"],
        batch_idx=[0], sim_real_idx=old["y_real"], observed=S["observed"],
        reg_factor=model._reg_factor(2, None))
    np.random.seed(12)
    pool = AlphaPool.from_estep(E1, N, int(model.n_stimuli))
    En = estep_with_gaps(model, data_rna, S, model.basal, model.inter, ks, s1, 2, gaps,
                         y_alpha=E1["y_alpha"], pool=pool)
    assert np.allclose(base[0], En["y_prot"])
    assert np.allclose(base[4], En["y_alpha"])


def test_pool_survives_a_change_of_the_bins(fitted):
    from CardamomOT.pseudotime.estep import AlphaPool, estep_with_gaps

    model, _, data_rna, ks, s1 = fitted
    S = _scaffold(model, data_rna)
    np.random.seed(13)
    E1 = estep_with_gaps(model, data_rna, S, model.basal, model.inter, ks, s1, 2, np.diff(S["times"]))
    pool = AlphaPool.from_estep(E1, S["n_full"], int(model.n_stimuli))
    # new bins: 3 bins instead of 5 (snapshots 2-4 merged), hence another slot count
    saved = model.bins
    try:
        labels = np.minimum(saved.bin_of_cell, 2)
        model.set_bins(bin_by_sizes(model.tau, np.bincount(labels), labels == 0,
                                    bin_times=np.array([0.0, 2.0, 7.0])))
        S2 = _scaffold(model, data_rna)
        assert S2["K"] == 3 and S2["n_full"] != S["n_full"]
        np.random.seed(14)
        E2 = estep_with_gaps(model, data_rna, S2, model.basal, model.inter, ks, s1, 2,
                             np.diff(S2["times"]), pool=pool)
        # bin 1 kept its time: the alphas of the step leaving it come from the pool
        old_alphas = {tuple(np.round(a, 12)) for a in pool.alpha}
        assert all(tuple(np.round(r, 12)) in old_alphas for r in E2["y_alpha"][1])
    finally:
        model.set_bins(saved)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
