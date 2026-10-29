"""
Oracle of the threshold s_ij of the exact Herbach form, on the true proteins and the true sigma of the cells (true
support and true m of every target, no mixture, no loop): does a threshold learned per edge (Gaussian prior on
log s_ij around the threshold s_j of the regulator) let the exact form reach the effects that the bound
-log(1 + (Pbar / s_j)^m) of a fixed threshold forbids, and is it identified?

Data (generated here, ml_sim.py, C cells, cached in bench/<network>/oracle_threshold/):
- approx: approximate form, m drawn in HERBACH_M, s = Pbar / 2 (the benchmark);
- exact_s05: exact form, m drawn, s = Pbar / 2;
- exact_svar: exact form, m drawn, threshold of each regulator drawn in S_GRID x Pbar.
Fits of each target (cross-entropy against the true sigma, 5-fold held-out), thresholds of the inference
s_j = Pbar / 2: 'approx', 'exact' (s_j fixed), 'exact_s' (log s_ij free, prior sd TAU). Outputs: excess held-out
cross-entropy (KL to the true sigma) per target, effects theta Pbar of the edges against the true ones, learned
thresholds against the true ones.

Usage: python oracle_threshold.py [network ...] [--runs 1-3] [--cells 1500]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from networks import Network, NETWORKS  # noqa: E402
from ml_sim import cell_times, simulate_snapshot  # noqa: E402
from CardamomOT.inference.sigma import (kon_matrix, set_family, reset_edge_thresholds, PHI_HERBACH, SPEC_THRESH,  # noqa: E402
                                        SPEC_PBAR, SPEC_M, HERBACH_U_MIN)

S_GRID = (0.1, 0.2, 0.35, 0.5)  # thresholds of the regulators (x Pbar) of exact_svar
TAU = 1.0                       # prior sd of log s_ij around log s_j
RIDGE = 1e-6                    # ridge on theta Pbar (per cell), numerical only
LOG_S_BOUNDS = (np.log(0.01), np.log(2.0))
EPS = 1e-9


def dataset(name, run, kind, C):
    """(P (C, G+1), sigma (C, G), net) of a run, simulated once and cached."""
    out = os.path.join(HERE, 'bench', name, 'oracle_threshold')
    os.makedirs(out, exist_ok=True)
    f = os.path.join(out, f'{kind}_{run}.npz')
    net = Network(name, run)
    rng = np.random.default_rng([run, 7])
    if kind != 'approx':
        for g in range(1, net.G + 1):
            set_family(net.spec, g, (PHI_HERBACH, 0, net.spec[g, SPEC_M]))
        if kind == 'exact_svar':
            net.spec[1:, SPEC_THRESH] = rng.choice(S_GRID, net.G) * net.spec[1:, SPEC_PBAR]
            reset_edge_thresholds(net.spec)  # edges at the thresholds of their regulators
    if not os.path.exists(f):
        np.random.seed(run)
        _, P = simulate_snapshot(net, cell_times(C))
        np.savez(f, P=P)
    P = np.load(f)['P']
    sig = kon_matrix(P, net.ks, net.inter, net.basal, net.spec)[:, 1:] / net.ks[1:, 1]
    return P, sig, net


def psi_exact(th, log_s, X, pbar, m):
    """Contributions (n, k) of the regulators X (n, k): theta pbar = effect at pbar, thresholds exp(log_s)."""
    L = m * np.log(np.maximum(X, 1e-12) / np.exp(log_s))
    lbar = m * (np.log(pbar) - log_s)
    u = np.maximum(th * pbar + np.logaddexp(0.0, lbar), HERBACH_U_MIN)
    t = u + np.log(-np.expm1(-u)) - lbar
    return np.logaddexp(0.0, t + L) - np.logaddexp(0.0, L)


def phi_approx(X, log_s, pbar, m):
    h = lambda x: x ** m / (1 + x ** m)
    return pbar * h(np.maximum(X, 0) / np.exp(log_s)) / h(pbar / np.exp(log_s))


def ce(y, s):
    s = np.clip(s, EPS, 1 - EPS)
    return -np.mean(y * np.log(s) + (1 - y) * np.log(1 - s))


class Fit:
    """One target: model 'approx', 'exact' (thresholds fixed) or 'exact_s' (log thresholds learned)."""

    def __init__(self, kind, s0, pbar, m):
        self.kind, self.ls0, self.pbar, self.m = kind, np.log(s0), pbar, m
        self.k = len(s0)

    def unpack(self, x):
        beta, th = x[0], x[1:1 + self.k]
        ls = x[1 + self.k:] if self.kind == 'exact_s' else self.ls0
        return beta, th, ls

    def sigma(self, x, X):
        beta, th, ls = self.unpack(x)
        if self.kind == 'approx':
            z = phi_approx(X, ls, self.pbar, self.m) @ th
        else:
            z = psi_exact(th, ls, X, self.pbar, self.m).sum(axis=1)
        return expit(beta + z)

    def loss(self, x, X, y):
        _, th, ls = self.unpack(x)
        out = ce(y, self.sigma(x, X)) + RIDGE * np.sum((th * self.pbar) ** 2)
        if self.kind == 'exact_s':
            out += np.sum((ls - self.ls0) ** 2) / (2 * TAU ** 2) / len(y)
        return out

    def fit(self, X, y, starts):
        bounds = [(-30, 30)] + [(-30, 30)] * self.k + ([LOG_S_BOUNDS] * self.k if self.kind == 'exact_s' else [])
        best = None
        for x0 in starts:
            x0 = np.r_[x0[:1 + self.k], self.ls0] if self.kind == 'exact_s' else x0[:1 + self.k]
            r = minimize(self.loss, x0, args=(X, y), method='L-BFGS-B', bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        return best.x


def run_target(net, P, sig, i, rng, n_folds=5):
    """One target i (gene index >= 1): held-out excess CE per model, fitted effects and thresholds."""
    J = np.flatnonzero(net.inter[:, i])
    X, y = P[:, J], sig[:, i - 1]
    pbar, m = net.spec[J, SPEC_PBAR], net.spec[i, SPEC_M]
    s0_fit = 0.5 * pbar  # thresholds of the inference, Pbar / 2 (k_low = 0)
    folds = rng.permutation(len(y)) % n_folds
    starts = [np.zeros(1 + len(J)), np.r_[0.0, np.sign(net.inter[J, i]) * 3.0]]
    res = {}
    for kind in ('approx', 'exact', 'exact_s'):
        f = Fit(kind, s0_fit, pbar, m)
        exc = []
        for k in range(n_folds):
            tr, te = folds != k, folds == k
            x = f.fit(X[tr], y[tr], starts)
            exc.append(ce(y[te], f.sigma(x, X[te])) - ce(y[te], y[te]))
        x = f.fit(X, y, starts)
        _, th, ls = f.unpack(x)
        res[kind] = (np.mean(exc), th * pbar, np.exp(ls))
    return J, res, net.spec[J, SPEC_THRESH]


def main(names, runs, C):
    rows, edges = [], []
    for name in names:
        for run in runs:
            for kind in ('approx', 'exact_s05', 'exact_svar'):
                P, sig, net = dataset(name, run, kind, C)
                rng = np.random.default_rng(run)
                for i in range(1, net.G + 1):
                    J, res, s_true = run_target(net, P, sig, i, rng)
                    m = net.spec[i, SPEC_M]
                    rows.append(dict(net=name, run=run, data=kind, target=i, m=m,
                                     **{f'kl_{k}': v[0] for k, v in res.items()}))
                    for e, j in enumerate(J):
                        edges.append(dict(net=name, run=run, data=kind, target=i, reg=j, m=m, true=net.inter[j, i],
                                          s_true=s_true[e], bound_fixed=-np.logaddexp(0, m * np.log(2.0)),
                                          **{f'eff_{k}': v[1][e] for k, v in res.items()},
                                          s_learned=res['exact_s'][2][e]))
                print(f'{name} run {run} {kind}', flush=True)
    out = os.path.join(HERE, 'bench')
    pd.DataFrame(rows).to_csv(os.path.join(out, 'oracle_threshold_targets.csv'), index=False)
    pd.DataFrame(edges).to_csv(os.path.join(out, 'oracle_threshold_edges.csv'), index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('networks', nargs='*', default=list(NETWORKS))
    ap.add_argument('--runs', default='1-3')
    ap.add_argument('--cells', type=int, default=1500)
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.runs.split('-')) if '-' in a.runs else (int(a.runs),) * 2
    main(a.networks, range(lo, hi + 1), a.cells)
