"""Network conditions and mixed model of the network fit (inference/network.py)."""
import numpy as np
import pytest
from scipy.optimize import check_grad

from CardamomOT.inference import network as net


def _targets(C, K=6, n=200, seed=0):
    rng = np.random.default_rng(seed)
    ys = np.repeat(np.arange(C), n)
    X = np.clip(.5 + .2 * rng.normal(size=(len(ys), K)), 0, 1)
    th = np.zeros(K)
    th[:2] = 4.0, -4.0
    y = rng.binomial(1, 1 / (1 + np.exp(-(X - .5) @ th))).astype(float)
    return ys, X, y


def _grad_error(mixed):
    rng = np.random.default_rng(0)
    C, G, ns = 3, 5, 4
    ys = np.repeat(np.arange(ns), 40)
    X = np.clip(.5 + .2 * rng.normal(size=(len(ys), G)), 0, 1)
    cond = np.array([0, 0, 1, 2])
    if mixed:
        levels = net.ConditionLevels(1.0, 2.0, np.zeros(C), np.zeros(C), np.ones(C) / C, np.zeros(C))
        extra = dict(mixed=net.MixedPrior(rng.uniform(1, 5, G), 3.0, rng.uniform(0, 1, G), cond))
    else:
        levels = net.ConditionLevels(1.0, 2.0, np.array([.5, 1., 1.5]), np.array([1., 2., 3.]), np.ones(C) / C, np.ones(C) / C)
        extra = dict(ridge=np.full(G, .1))
    kw = dict(weights_samples=[1., 1.5, 1., 2.], ys=ys, yp=X, ypm=None, yt=rng.uniform(size=len(ys)), G=G, n_samples=ns,
              cond=cond, levels=levels, theta_ref=np.zeros((C, G)), ref_shared=np.zeros(G), delta_ref=np.zeros((C, G)),
              w=np.ones(G), weight_flow_matching=0., final=0, **extra)
    return check_grad(lambda x: net._loss_grad_conditions(x, **kw)[0], lambda x: net._loss_grad_conditions(x, **kw)[1],
                      rng.normal(size=(C + 1) * G + ns))


@pytest.mark.parametrize('mixed', [False, True])
def test_gradient(mixed):
    assert _grad_error(mixed) < 1e-3


def test_levels_limits():
    """pen = 0: each condition at its own level; pen >= 1: shared network and deviations; pooled level as qut_penalty."""
    C, K = 3, 6
    ys, X, y = _targets(C)
    args = (ys, X, None, y, np.ones(K), K, K, C)
    pool = net.qut_penalty(*args, 0., .5, 20)
    cond = np.arange(C)
    low = net.qut_conditions(*args, 0., cond, 0.0)
    assert low.l_shared == 0 and np.all(low.l_fuse == 0) and np.all(low.l_cond > 0)
    one = net.qut_conditions(*args, 0., cond, 1.0)
    assert np.all(one.l_cond == 0) and one.l_shared > 0 and np.all(one.l_fuse > 0)
    assert one.l_pen == pytest.approx(pool)
    high = net.qut_conditions(*args, 0., cond, 1e4)
    assert high.l_shared == pytest.approx(pool)  # only the shared route is left: kappa = 1
    assert np.allclose(high.ridge_w, 1 / C) and np.allclose(low.ridge_w, 1.0)


def test_large_pen_gives_common_network():
    C, K = 3, 6
    ys, X, y = _targets(C)
    inter = net.main_loop_inference(K, ys, X, None, y, np.zeros(C * K + C), np.zeros(C * K + C), np.array([0., 1.]), K, C,
                                    np.ones(K), weight_flow_matching=0., cond=np.arange(C), condition_pen=1e4)[1]
    assert np.max(np.abs(inter - inter[0])) < 1e-6


def test_mixed_weights_never_inflate():
    ys = np.repeat(np.arange(3), [150, 500, 3000])
    weights, m = net._mixed_weights(ys)
    ws = np.mean(weights) / np.array(weights)
    assert m == pytest.approx(np.mean(weights))
    assert np.allclose(ws / m, [1.0, 1.0, 500 / 3000])


def test_patient_mean_term():
    """Only the patient means are penalised: contrasts between the samples of a patient are free."""
    patient = np.array([0, 0, 1, 2])
    free = np.ones(4, bool)
    base = np.array([1., 1., 0., 2.])
    v0, _ = net._patient_mean_term(base, free, patient, 2.0)
    v1, g1 = net._patient_mean_term(base + np.array([.5, -.5, 0., 0.]), free, patient, 2.0)
    assert v0 == pytest.approx(v1) and g1[0] == pytest.approx(g1[1])
    eps = 1e-6
    numeric = [(net._patient_mean_term(base + eps * np.eye(4)[i], free, patient, 2.0)[0] - v0) / eps for i in range(4)]
    assert np.allclose(numeric, net._patient_mean_term(base, free, patient, 2.0)[1], atol=1e-4)


def test_mixed_replicates_and_outputs():
    """Replicates: the deviations stay small; statistics returned; the fit is deterministic."""
    C, K = 3, 6
    ys, X, y = _targets(C, n=400)
    kw = dict(weight_flow_matching=0., cond=np.arange(C), mixed=(.05, .05))
    run = lambda: net.main_loop_inference(K, ys, X, None, y, np.zeros(C * K + C), np.zeros(C * K + C), np.array([0., 1.]),
                                          K, C, np.ones(K), **kw)
    res = run()
    assert len(res) == 6
    s_theta, n_theta, s_basal, n_basal, z = res[5]
    assert n_theta > 0 and s_theta >= 0 and n_basal == C - 1 and z.shape == (C, K)
    inter = res[1]
    assert np.all(np.abs(inter[:, :2]) > 0)  # the true edges are found in every patient
    assert np.array_equal(inter, run()[1])


def test_mixed_multi_edge_cases():
    """Several samples per patient, a target without regulator, a target flat in one sample, sub-samples."""
    rng = np.random.default_rng(1)
    S, n, K = 4, 150, 5
    G = K + 1
    ys = np.repeat(np.arange(S), n)
    P = np.c_[np.ones(len(ys)), np.clip(.5 + .2 * rng.normal(size=(len(ys), K)), 0, 1)]
    Y = np.zeros((len(ys), G))
    Y[:, 1:] = rng.binomial(1, 1 / (1 + np.exp(-4 * (P[:, [2]] - .5))), size=(len(ys), K))
    Y[ys == 1, 3] = 0.0  # target 3 flat in sample 1
    ref = np.ones((G, G))
    ref[:, 4] = 0.0  # target 4 without regulator
    ks = np.tile(np.array([0., 1.]), (G, 1))
    fits = net.inference_network_multi([np.arange(0, len(ys), 2), np.arange(1, len(ys), 2)], ys, Y, P, None, ks, n_stimuli=1,
                                       ref_network=ref, samples_id=np.arange(S), weight_flow_matching=0.,
                                       sample_conditions=np.array([0, 0, 1, 2]), condition_pen=0.0, mixed=(.25, .25))
    assert len(fits) == 2
    basal, inter, _, _, _, (stats, z) = fits[0]
    assert inter.shape == (S, G, G) and basal.shape == (S, G) and z.shape == (3, G, G)
    assert np.array_equal(inter[0], inter[1])  # two samples of one patient share its network
    assert stats.shape == (G, 4) and stats[4, 1] == 0  # no statistics for the target without regulator
    assert np.all(inter[:, :, 4] == 0) and np.all(np.isfinite(stats)) and stats[:, 3].sum() > 0
    assert np.all(np.isfinite(inter)) and np.all(np.isfinite(basal))


def test_mixed_tau_per_target():
    """One tau2 per target: each target is fitted as with its own value alone."""
    rng = np.random.default_rng(2)
    S, n, K = 3, 300, 4
    G = K + 1
    ys = np.repeat(np.arange(S), n)
    P = np.c_[np.ones(len(ys)), np.clip(.5 + .2 * rng.normal(size=(len(ys), K)), 0, 1)]
    Y = np.zeros((len(ys), G))
    Y[:, 1:] = rng.binomial(1, 1 / (1 + np.exp(-6 * (P[:, [2]] - .5))), size=(len(ys), K))
    ks = np.tile(np.array([0., 1.]), (G, 1))
    run = lambda tau2: net.inference_network_multi([None], ys, Y, P, None, ks, n_stimuli=1, samples_id=np.arange(S),
                                                   weight_flow_matching=0., sample_conditions=np.arange(S),
                                                   mixed=(tau2, .25))[0][1]
    tau2 = np.array([1., .01, 1., .25, 4.])
    both = run(tau2)
    for t in (.01, .25, 4.):
        g = np.flatnonzero(tau2 == t)[0]
        assert np.array_equal(both[:, :, g], run(t)[:, :, g])


def test_learn_mixed_variances():
    """Roots of the score equations: pooled, per target with shrinkage, a target without edge at the pooled value, no
    variance at the bottom of the bracket."""
    G, d0 = 4, 8.0
    n = np.array([0.0, 8.0, 40.0, 40.0])
    a = np.array([0.0, 2.0, 4.0, 0.25])  # s = n (a + tau2 / 2): fixed point 2 a, in closed form below

    def stats(tau2, tau2_b):
        s = np.zeros((G, 4))
        s[:, 0], s[:, 1] = n * (a + np.broadcast_to(tau2, (G,)) / 2), n
        s[:, 2], s[:, 3] = 3 * tau2_b / 2, 3.0  # tau2_b -> 0
        return s

    tau2, tau2_b, tau2_0 = net.learn_mixed_variances(stats, G, d0, n_steps=30)
    root0 = 2 * np.sum(n * a) / np.sum(n)
    assert tau2_0 == pytest.approx(root0, rel=1e-3) and tau2_b == pytest.approx(1e-4, rel=1e-3)
    assert tau2[0] == pytest.approx(root0, rel=1e-3)
    expected = (n * a + d0 * root0) / (n / 2 + d0)
    assert np.allclose(tau2, expected, rtol=1e-3)
