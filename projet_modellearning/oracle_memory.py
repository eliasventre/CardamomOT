"""
Oracle of the family selection, part 2: which of the mRNA memory and the count noise makes the targets of the mixture
lose the family (oracle_families.py), and does a regulation function averaged along the mRNA memory recover it?

Fresh cells of the run (same network and families, full model with explicit mRNA), each with its true protein path
on a grid over the last H = 5 / d0 hours before its observation. Targets:
- 'true':  sigmoid of the true burst frequency at the observation (noise-free, instantaneous);
- 'nb':    posteriors of the mixture for counts drawn from the NB of the true burst frequency (count noise, no memory);
- 'mix':   posteriors of the mixture for the counts of the explicit-mRNA simulation, Poisson(M) (memory and noise),
           as the method sees them.
Predictions: 'inst', sigma at the proteins of the observation; 'mem', sigma averaged along the mRNA memory,
sum_k w_k sigma(P(s_k)), w_k ~ d0 exp(-d0 (t - s_k)) (mass before the window on its first node); 'lagmed' / 'lagmean',
sigma at the proteins a single lag before the observation, the median ln2 / d0 / the mean 1 / d0 of the memory.
Each (targets, prediction) pair: both families fitted per target by a logistic regression with a weak ridge (same
fitter for all), scored by 5-fold held-out cross-entropy; the family of lowest score is selected. Loss 'nbll': the
counts themselves, NB likelihood of the burst frequency k0 + (k1 - k0) sigma of the continuous model (no discrete
modes, mixture parameters k0, k1, c).

Usage: python oracle_memory.py <network> [runs]          (mixture of the run: <network>/runs/run<r>_{base,ml})
Summary appended to <network>/oracle_memory_summary.csv.
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np
import torch
from scipy.stats import nbinom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from networks import Network  # noqa: E402
from ml_sim import cell_times, BURNIN  # noqa: E402
from CardamomOT.inference.simulations import simulate_next_mrna_pdmp  # noqa: E402
from CardamomOT.inference.sigma import sigma_pairs, PHI_LOG  # noqa: E402

H_MEMORY = 5.0     # window of the mRNA memory, in units of 1 / d0
DT = 0.5           # step of the protein grid (h)
KAPPA = 0.1        # scale of the regulators of the Hill family (Pbar = 1)


def simulate_paths(net, time, rng_seed):
    """Per cell: protein path (K, G+1) on the grid s_k = t - H + k DT (state of the burn-in end, stimulus 0, before
    t = 0), mRNA M (G+1,) and proteins at the observation; grid offsets (K,) (t - s_k)."""
    np.random.seed(rng_seed)
    H = H_MEMORY / net.d0[1]
    offs = np.arange(0.0, H + 1e-9, DT)[::-1]           # t - s_k, decreasing: s_k increasing
    G1 = net.G + 1
    paths, Ms = [], []
    for t in time:
        sim = lambda ts, P0, M0, stim: simulate_next_mrna_pdmp(
            net.d0, net.d1, net.ks, net.c_nb, net.basal, net.inter, np.atleast_1d(ts).astype(float), 1.0,
            P0=P0, M0=M0, ns=1, stim_vals=np.full(1, stim), spec=net.spec)
        b = sim(BURNIN, np.zeros(G1), np.zeros(G1), 0.0)
        p_burn = np.concatenate([[0.0], b.p[-1]])
        s = t - offs
        pos = s >= 0
        out = sim(s[pos], np.concatenate([[1.0], b.p[-1]]), np.concatenate([[0.0], b.m[-1]]), 1.0)
        path = np.tile(p_burn, (len(s), 1))
        path[pos, 1:] = out.p
        path[pos, 0] = 1.0
        paths.append(path)
        Ms.append(np.concatenate([[0.0], out.m[-1]]))
    return np.array(paths), np.array(Ms), offs


def memory_weights(offs, d0):
    """Weights (K,) of the nodes in the mRNA memory d0 exp(-d0 (t - s)), mass before the window on the first node."""
    w = d0 * np.exp(-d0 * offs) * DT
    w[0] += np.exp(-d0 * offs[0])
    return w / w.sum()


def lag_weights(offs, lag):
    """Weights (K,) of a single node, the one of the grid nearest to the lag before the observation."""
    w = np.zeros(len(offs))
    w[int(np.argmin(np.abs(offs - lag)))] = 1.0
    return w


def posteriors(x, time, a, pi_init):
    """Expected position tau (C, G) of the modes under the posteriors of the mixture (a: (M+1, G+1), last row c;
    pi_init: per gene, {time: weights})."""
    C, G = x.shape
    tau = np.zeros((C, G))
    for g in range(G):
        ks, c = a[:-1, g + 1], a[-1, g + 1]
        ll = np.stack([nbinom.logpmf(x[:, g], k, c / (c + 1.0)) for k in ks], axis=1)
        prior = np.array([pi_init[g][float(t)][:len(ks)] for t in time])
        lp = ll + np.log(np.maximum(prior, 1e-300))
        p = np.exp(lp - lp.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        pos = (ks - ks.min()) / max(ks.max() - ks.min(), 1e-12)
        tau[:, g] = p @ pos
    return tau


def phi(X, family):
    return torch.log1p(torch.relu(X) / KAPPA) / np.log1p(1.0 / KAPPA) if family == PHI_LOG else X


def fit_score(paths, w, tau, j, family, mem, folds, ridge=1e-4, counts=None, nb=None, no_self=False, self_path=None):
    """Held-out cross-entropy of target j for one family, prediction instantaneous or averaged on the memory; with
    counts (C,) and nb = (k0, k1, c) of the mixture: held-out NB negative log-likelihood of the counts, burst
    frequency k0 + (k1 - k0) sigma (the continuous model, without the discrete modes). self_path (C, K): protein of
    the target used as its own regulator instead of its true path (own protein lagged or propagated)."""
    if self_path is not None:
        paths = paths.copy()
        paths[:, :, j + 1] = self_path
    X = torch.tensor(paths if mem else paths[:, -1:], dtype=torch.float64)   # (C, K or 1, G+1)
    W = torch.tensor(w if mem else np.ones(1), dtype=torch.float64)
    F = phi(X, family)
    if no_self:  # protein of the target left out of its regulators (translated from its own mRNA)
        F = F.clone()
        F[..., j + 1] = 0.0
    y = torch.tensor(np.clip(tau[:, j], 1e-6, 1 - 1e-6), dtype=torch.float64)
    ce = lambda s, yy: -(yy * torch.log(s.clamp(1e-12)) + (1 - yy) * torch.log((1 - s).clamp(1e-12)))
    if counts is not None:
        k0, k1, c = nb
        xc = torch.tensor(counts, dtype=torch.float64)
        lp = np.log(c / (c + 1.0)), np.log(1.0 / (c + 1.0))
        def ce(s, idx):  # NB negative log-likelihood of the counts of the cells idx
            r = k0 + (k1 - k0) * s
            xx = xc[idx]
            return -(torch.lgamma(xx + r) - torch.lgamma(r) - torch.lgamma(xx + 1) + r * lp[0] + xx * lp[1])
        y = torch.arange(len(counts))
        ce_ref = lambda yy: 0.0
    else:
        ce_ref = lambda yy: ce(yy, yy)
    out = []
    for k, test in enumerate(folds):
        train = np.concatenate([f for m, f in enumerate(folds) if m != k])
        th = torch.zeros(F.shape[-1] + 1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th], max_iter=300, line_search_fn='strong_wolfe')
        pred = lambda idx: (torch.sigmoid(th[0] + F[idx] @ th[1:]) * W).sum(-1)

        def closure():
            opt.zero_grad()
            loss = ce(pred(train), y[train]).mean() + ridge * (th[1:] ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            out.append((ce(pred(test), y[test]) - ce_ref(y[test])).mean().item())
    return float(np.mean(out))


def run(name, r, seed=0):
    net = Network(name, r)
    time = cell_times(1000)
    paths, M, offs = simulate_paths(net, time, rng_seed=10_000 + r)
    G1 = net.G + 1
    cdir = next(d for d in (os.path.join(HERE, name, 'runs', f'run{r}_{m}', 'cardamomOT') for m in ('base', 'ml'))
                if os.path.exists(os.path.join(d, 'mixture_parameters.npy')))
    a = np.load(os.path.join(cdir, 'mixture_parameters.npy'))
    pi_init = pickle.load(open(os.path.join(cdir, 'pi_init.pkl'), 'rb'))
    rng = np.random.default_rng([seed, r])
    sig_true = sigma_pairs(net.basal, net.inter, paths[:, -1], net.spec)[:, :, 1]          # (C, G+1)
    kon = net.kz[:, 0] + (net.kz[:, 1] - net.kz[:, 0]) * sig_true
    p_nb = net.c_nb / (net.c_nb + 1.0)
    x_nb = rng.negative_binomial(np.maximum(kon[:, 1:], 1e-8), p_nb[None, 1:])
    x_mix = rng.poisson(M[:, 1:])
    taus = {'true': sig_true[:, 1:], 'nb': posteriors(x_nb, time, a, pi_init), 'mix': posteriors(x_mix, time, a, pi_init)}
    # Continuous targets of the pipeline (continuous_targets): E[sigma | counts] under the continuous model fitted from
    # the mixture of the run, for the explicit-mRNA counts
    from CardamomOT.inference.mixture import fit_grid_mixture
    tau_ct = np.zeros_like(taus['mix'])
    for g in range(net.G):
        kz = a[:-1, g + 1]
        grid = fit_grid_mixture(x_mix[:, g], time, kz.min(), kz.max(), a[-1, g + 1])
        tau_ct[:, g] = taus['mix'][:, g] if grid is None else grid[3] @ np.linspace(0, 1, grid[3].shape[1])
    taus['ct'] = tau_ct
    w = memory_weights(offs, net.d0[1])
    folds = np.array_split(rng.permutation(len(time)), 5)
    hill = net.hill[1:]
    rows = []
    counts = {'nb': x_nb, 'mix': x_mix}
    # Own protein of each target: lagged (frozen at the start of the memory window) or propagated from there with the
    # mode in force at the start of the window (no switch): it cannot contain the translation of the counted mRNA
    P0 = paths[:, 0, :]                                                        # (C, G+1) start of the window
    m_start = sigma_pairs(net.basal, net.inter, P0, net.spec)[:, :, 1]         # level of the start mode (k_low = 0)
    since = offs[0] - offs                                                     # time since the window start (K,)
    own = {'selflag': np.repeat(P0[:, None, :], len(offs), axis=1),
           'selfprop': m_start[:, None, :] + (P0 - m_start)[:, None, :] * np.exp(-net.d1[None, None, :] * since[None, :, None])}
    combos = [('true', 'inst', 'ce'), ('nb', 'inst', 'ce'), ('mix', 'inst', 'ce'), ('mix', 'mem', 'ce'),
              ('true', 'mem', 'ce'), ('nb', 'inst', 'nbll'), ('mix', 'inst', 'nbll'), ('mix', 'mem', 'nbll'),
              ('mix', 'inst', 'nbll-noself'), ('mix', 'mem', 'nbll-noself'), ('mix', 'mem', 'ce-noself'),
              ('mix', 'mem', 'ce-selflag'), ('mix', 'mem', 'ce-selfprop'), ('true', 'mem', 'ce-selflag'),
              ('true', 'mem', 'ce-selfprop')]
    if os.environ.get('ORACLE_COMBOS'):  # subset, e.g. 'mix:mem:nbll-noself,mix:inst:nbll-noself'
        combos = [tuple(c.split(':')) for c in os.environ['ORACLE_COMBOS'].split(',')]
    for target, pred, loss in combos:
        gains = []
        for j in range(net.G):
            kw = {} if loss.startswith('ce') else dict(counts=counts[target][:, j],
                                                       nb=(a[:-1, j + 1].min(), a[:-1, j + 1].max(), a[-1, j + 1]))
            kw['no_self'] = loss.endswith('noself')
            for key in own:
                if loss.endswith(key):
                    kw['self_path'] = own[key][:, :, j + 1]
            w_p = {'mem': w, 'lagmed': lag_weights(offs, np.log(2) / net.d0[1]),
                   'lagmean': lag_weights(offs, 1.0 / net.d0[1])}.get(pred, w)
            s_lin = fit_score(paths, w_p, taus[target], j, 0, pred != 'inst', folds, **kw)
            s_log = fit_score(paths, w_p, taus[target], j, PHI_LOG, pred != 'inst', folds, **kw)
            gains.append((s_lin - s_log) / max(s_lin, 1e-12))
        sel = np.array(gains) > 0
        res = dict(network=name, run=r, targets=target, prediction=pred, loss=loss, family_acc=float(np.mean(sel == hill)),
                   true_hill=''.join('H' if h else 'S' for h in hill), selected=''.join('H' if h else 'S' for h in sel),
                   gain_log=' '.join(f'{g:+.3f}' for g in gains))
        print(', '.join(f'{k} {v:.2f}' if isinstance(v, float) else f'{k} {v}' for k, v in res.items()), flush=True)
        rows.append(res)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('network')
    ap.add_argument('runs', nargs='?', default='1-10')
    a = ap.parse_args()
    runs = list(range(int(a.runs.split('-')[0]), int(a.runs.split('-')[-1]) + 1)) if '-' in a.runs \
        else [int(x) for x in a.runs.split(',')]
    path = os.path.join(HERE, a.network, 'oracle_memory_summary.csv')
    for r in runs:
        rows = run(a.network, r)
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                wr.writeheader()
            wr.writerows(rows)
