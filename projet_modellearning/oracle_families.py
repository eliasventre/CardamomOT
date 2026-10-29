"""
Level 0 of the benchmark: the family selection of fit_network (same network fit, same candidates, same held-out
score) on the TRUE proteins of the observed cells, without the loop (no trajectories, no basin updates).

Targets of the fits and of the score:
- 'mixture': expected mode of each gene under the posteriors of the mixture (targets of the selection of fit_network);
- 'true': true burst frequency between the extreme modes (noise-free upper bound).

EM mode (--em): level 1 of the benchmark without the scale and the proliferation, on the true proteins. From the
sigmoid for every target, each iteration fits every family on the expected levels under the current posteriors
(E-step: q(l) proportional to NB(x | k_l) p(l | sigma), network prior of precision kappa), scores them with --scores
('ce' against the current posteriors, 'marginal' likelihood of the counts, as fit_network) and switches
a target with the rule of fit_network (margin, two consecutive selections), for --iters iterations.

Usage: python oracle_families.py <network> [runs] [--targets mixture,true] [--fits K] [--weak-penalty]
       python oracle_families.py <network> [runs] --em [--scores ce,em,marginal] [--iters 10]
Needs the mixture of each run, bench/<network>/runs/run<r>_obs_*/cardamomOT (run_inference.py), and the true proteins,
bench/<network>/True/prot_<r>.npy (generate.py). Summary appended to bench/<network>/oracle_summary.csv.
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, 'bench')
sys.path.insert(0, os.path.dirname(HERE))

from CardamomOT.model.base import NetworkModel  # noqa: E402
from CardamomOT.inference import inference_network_multi  # noqa: E402
from CardamomOT.inference.network import sigmoid_amplitudes, sigmoid_target, cross_entropy  # noqa: E402
from CardamomOT.inference.sigma import (sigma_pairs, default_spec, set_family, family_name, is_herbach, family_index,  # noqa: E402
                                        SPEC_PHI, SPEC_M, N_SPEC)


def subsamples(time, K, rng):
    """K disjoint sub-samples of the cells, each with the same share of every time point."""
    sels = [[] for _ in range(K)]
    for t in np.unique(time):
        idx = rng.permutation(np.flatnonzero(time == t))
        for k, part in enumerate(np.array_split(idx, K)):
            sels[k].append(part)
    return [np.concatenate(s) for s in sels]


def fit_spec(spec, edges):
    """spec with the edge thresholds learned by one fit (exact Herbach targets)."""
    if edges is None:
        return spec
    out = spec.copy()
    out[:, N_SPEC:] = edges
    return out


def held_out_ce(fits, sels, spec, P, tau):
    """Cross-entropy (G+1,) of each target, of the fit of each sub-sample on the cells of the other ones."""
    acc = np.zeros(P.shape[1])
    for k, (basal, inter, _, _, edges) in enumerate(fits):
        held = np.concatenate([s for m, s in enumerate(sels) if m != k])
        sig = sigma_pairs(np.asarray(basal[0], float), np.asarray(inter, float), P[held], fit_spec(spec, edges))[:, :, 1]
        ce = cross_entropy(np.ascontiguousarray(sig).ravel(), np.ascontiguousarray(tau[held]).ravel())
        acc += ce.reshape(sig.shape).mean(axis=0)
    return acc / len(fits)


def token(phi, m):
    """S sigmoid, H Hill type, B<m> Herbach type."""
    return 'S' if phi == 0 else ('H' if not is_herbach(phi) else f'B{m:g}')


def true_tokens(spec, cands):
    """Tokens of the true families; a Hill target is B1 when the Hill type is not a candidate (data generated with
    the Hill type: same shape as the Herbach type of m = 1 on [0, Pbar])."""
    from CardamomOT.inference.sigma import PHI_LOG
    hill = any(int(f[0]) == PHI_LOG for f in cands)
    tok = [token(int(f), m) for f, m in zip(spec[1:, SPEC_PHI], spec[1:, SPEC_M])]
    return [t if hill or t != 'H' else 'B1' for t in tok]


def oracle_run(name, run, targets, K=5, seed=0, weak_penalty=False, relaxed=False):
    import anndata as ad
    truth = np.load(os.path.join(BENCH, name, 'True', f'truth_{run}.npz'))
    P = np.load(os.path.join(BENCH, name, 'True', f'prot_{run}.npy'))       # (C, G+1), column 0 = stimulus
    time = ad.read_h5ad(os.path.join(BENCH, name, 'Data', f'data_{run}.h5ad')).obs['time'].values
    # Mixture of the run (independent of the model learning)
    cdir = next(d for d in (os.path.join(BENCH, name, 'runs', f'run{run}_{m}', 'cardamomOT')
                            for m in ('obs_base', 'obs_ml', 'hid_base', 'hid_ml'))
                if os.path.exists(os.path.join(d, 'proba_init.npy')))
    G1 = P.shape[1]
    model = NetworkModel(G1 - 1, n_stimuli=1)
    model.a = np.load(os.path.join(cdir, 'mixture_parameters.npy'))
    proba = np.load(os.path.join(cdir, 'proba_init.npy'))
    ks = model._mixture_terms()[0]
    kb = sigmoid_amplitudes(ks)
    samples = np.zeros(len(time), dtype=int)
    tau_t = {'mixture': sigmoid_target(model._posterior_modes(proba, ks, samples, [0]), kb[:, 0], kb[:, 1]),
             'true': sigma_pairs(truth['basal'], truth['inter'], P, truth['spec'])[:, :, 1]}
    ref = np.ones((G1, G1))
    sels = subsamples(time, K, np.random.default_rng([seed, run]))
    cands = model._model_learning_candidates()
    fam_true = true_tokens(truth['spec'], cands)
    # Penalties of the method, or nearly none (upper bound: weak ridge, low sparsity threshold)
    pen = dict(model._penalty_kw(), **(dict(network_ridge=1e6, qut_level=0.01) if weak_penalty else {}))
    pen['network_relaxed'] = relaxed
    true_idx = family_index(truth['spec'], cands)
    out = []
    for target in targets:
        tau = tau_t[target]
        y_kon = kb[:, 0] + (kb[:, 1] - kb[:, 0]) * tau        # sigmoid target tau in the network fit
        scores, inters = [], []
        for fam in cands:
            spec = default_spec(G1, kappa=model.model_learning_kappa * model.scale_proteins, pbar=model.scale_proteins,
                                thresh=model._herbach_thresholds(G1))
            set_family(spec, slice(1, None), fam)
            fits = inference_network_multi(
                sels, samples, y_kon, P, None, ks, n_stimuli=1, samples_id=[0], ref_network=ref,
                weight_flow_matching=0.0, final=0, seuil_min_network_intensity=model.seuil_min_network_intensity,
                sigma_spec_grn=spec, **pen)
            scores.append(held_out_ce(fits, sels, spec, P, tau)[1:])
            inters.append(np.mean([f[1] for f in fits], axis=0))
        best = np.argmin(np.array(scores), axis=0)
        sel = [token(cands[b][0], cands[b][2]) for b in best]
        kind = lambda s: s[0]
        # Network of the true family of each target (estimation alone): error on theta, wrong edges (off diagonal)
        est = np.stack([inters[true_idx[g]][:, g] if g else inters[0][:, 0] for g in range(G1)], axis=1)
        off = ~np.eye(G1, dtype=bool)
        off[:, 0] = False
        on_true, on_est = truth['inter'][off] != 0, np.abs(est[off]) > model.seuil_min_network_intensity
        res = dict(network=name, run=run, targets=target,
                   penalty=('weak' if weak_penalty else 'default') + ('+relaxed' if relaxed else ''),
                   family_acc=float(np.mean([kind(a) == kind(b) for a, b in zip(sel, fam_true)])),
                   herbach_m_acc=float(np.mean([a == b for a, b in zip(sel, fam_true) if kind(a) == kind(b) == 'B']))
                   if any(kind(a) == kind(b) == 'B' for a, b in zip(sel, fam_true)) else np.nan,
                   theta_mae=float(np.mean(np.abs(est[off] - truth['inter'][off]))),
                   false_pos=int(np.sum(on_est & ~on_true)), false_neg=int(np.sum(~on_est & on_true)),
                   true_family=' '.join(fam_true), selected=' '.join(sel),
                   candidates=' '.join(family_name(f) for f in cands))
        print(', '.join(f'{k} {v:.2f}' if isinstance(v, float) else f'{k} {v}' for k, v in res.items()), flush=True)
        out.append(res)
    return out


def em_run(name, run, scores, iters=10, K=5, seed=0, start='sigmoid', levels='grid', n_quad=51, prior='kl', hard=False):
    """Family selection along an EM on the true proteins, for each score (see the module docstring). start: family of
    every target at the first iteration, 'identity' (fitted and scored as the current family, never selected unless a
    candidate), 'hill' (needs 'logarithm' among the candidates) or 'b<m>' (Herbach of coefficient m, as fit_network
    with 'b2'). levels: 'grid', the levels of the mixture, or 'quad', n_quad points of sigma in [0, 1] with the exact
    NB likelihood of the counts (the network alone places the mass). prior: network prior on the levels, 'kl'
    (exp(-kappa KL(tau_l || sigma)), mean pulled towards 1/2 with more than two levels) or 'beta' (Beta(kappa sigma,
    kappa (1 - sigma)) at the levels, mean sigma). hard: the network is fitted on basins (one_hot_probas = True), the
    cells of each time ranked by their expected level under q and given the levels by the masses of the mixture (EMD
    of ordered levels), and 'ce' scores against them; kappa from the basins."""
    import anndata as ad
    truth = np.load(os.path.join(BENCH, name, 'True', f'truth_{run}.npz'))
    P = np.load(os.path.join(BENCH, name, 'True', f'prot_{run}.npy'))
    time = ad.read_h5ad(os.path.join(BENCH, name, 'Data', f'data_{run}.h5ad')).obs['time'].values
    cdir = next(d for d in (os.path.join(BENCH, name, 'runs', f'run{run}_{m}', 'cardamomOT')
                            for m in ('obs_base', 'obs_ml', 'hid_base', 'hid_ml'))
                if os.path.exists(os.path.join(d, 'proba_init.npy')))
    G1 = P.shape[1]
    model = NetworkModel(G1 - 1, n_stimuli=1)
    model.a = np.load(os.path.join(cdir, 'mixture_parameters.npy'))
    model.proba_init = np.load(os.path.join(cdir, 'proba_init.npy'))
    ks = model._mixture_terms()[0]
    kb = sigmoid_amplitudes(ks)
    samples = np.zeros(len(time), dtype=int)
    loglik, tau_l = model._level_likelihoods(ks, time, samples, [0])
    pi = model.proba_init[..., :loglik.shape[2]]
    pi = pi / np.maximum(pi.sum(axis=2, keepdims=True), 1e-12)
    tau0 = (pi * tau_l).sum(axis=2)                               # mixture alone
    if levels == 'quad':
        # NB likelihood of the counts at n_quad levels of sigma (stimulus column: flat)
        X = np.asarray(ad.read_h5ad(os.path.join(BENCH, name, 'Data', f'data_{run}.h5ad')).X, dtype=float)
        x = np.concatenate([np.zeros((len(X), 1)), X], axis=1)
        grid = (np.arange(n_quad) + 0.5) / n_quad
        loglik = np.stack([-nb_nll(model, np.full(x.shape, q), x) for q in grid], axis=2)
        loglik[:, 0] = 0.0
        tau_l = np.broadcast_to(grid, loglik.shape).copy()
    sig_true = sigma_pairs(truth['basal'], truth['inter'], P, truth['spec'])[:, :, 1]
    multi = np.isfinite(loglik).sum(axis=2) > 2                   # genes with intermediate levels
    ref = np.ones((G1, G1))
    cands = model._model_learning_candidates()
    fam_true = true_tokens(truth['spec'], cands)
    n_c = len(cands)
    if start in ('identity', 'sigmoid') and not any(int(f[0]) == 0 for f in cands):
        cands = cands + [(0, 0, 0.0)]                             # starting family only
    pen = model._penalty_kw()
    rng = np.random.default_rng([seed, run])
    kind = lambda s: s[0]
    out = []
    for score in scores:
        cur = np.full(G1, start_index(cands, start))
        cur[0] = 0  # stimulus: unused column
        pending, q_tau, kappa = {}, (hard_levels(tau0, pi, tau_l, time) if hard else tau0.copy()), np.ones(pi.shape[:2])
        for it in range(iters):
            sels = subsamples(time, K, rng)
            y_kon = kb[:, 0] + (kb[:, 1] - kb[:, 0]) * q_tau
            sc, sig_f = np.full((len(cands), G1), np.inf), {}
            for f, fam in enumerate(cands):
                spec = default_spec(G1, kappa=model.model_learning_kappa * model.scale_proteins,
                                    pbar=model.scale_proteins, thresh=model._herbach_thresholds(G1))
                set_family(spec, slice(1, None), fam)
                fits = inference_network_multi(
                    sels, samples, y_kon, P, None, ks, n_stimuli=1, samples_id=[0], ref_network=ref,
                    weight_flow_matching=0.0, final=0, seuil_min_network_intensity=model.seuil_min_network_intensity,
                    sigma_spec_grn=spec, **pen)
                acc = np.zeros(G1)
                for k, (basal, inter, _, _, edges) in enumerate(fits):
                    held = np.concatenate([x for m, x in enumerate(sels) if m != k])
                    sig = sigma_pairs(np.asarray(basal[0], float), np.asarray(inter, float), P[held],
                                      fit_spec(spec, edges))[:, :, 1]
                    if score == 'marginal':
                        lp_h = loglik[held] + _log_prior(sig, tau_l[held], kappa[held], loglik[held], prior)
                        acc += -logsumexp(lp_h, axis=2).mean(axis=0)
                    else:
                        tau = q_tau[held]
                        acc += cross_entropy(np.ascontiguousarray(sig).ravel(),
                                             np.ascontiguousarray(tau).ravel()).reshape(sig.shape).mean(axis=0)
                sc[f] = acc / len(fits)
                basal = np.mean([fi[0] for fi in fits], axis=0)
                inter = np.mean([fi[1] for fi in fits], axis=0)
                sig_f[f] = sigma_pairs(np.asarray(basal[0], float), np.asarray(inter, float), P, spec)[:, :, 1]
            # Switches: margin model_learning_margin, two consecutive selections
            best = np.argmin(sc[:n_c], axis=0)
            for g in range(1, G1):
                gain = (sc[cur[g], g] - sc[best[g], g]) / max(abs(sc[cur[g], g]), 1e-12)
                if best[g] != cur[g] and gain > model.model_learning_margin:
                    prev = pending.get(g)
                    pending[g] = (best[g], prev[1] + 1 if prev is not None and prev[0] == best[g] else 1)
                    if pending[g][1] >= 2:
                        cur[g] = best[g]
                        del pending[g]
                else:
                    pending.pop(g, None)
            # E-step with the network of the current families: q(l) proportional to NB(x | k_l) p(l | sigma)
            sig_cur = np.stack([sig_f[cur[g]][:, g] for g in range(G1)], axis=1)
            lp = loglik + _log_prior(sig_cur, tau_l, kappa, loglik, prior)
            q_post = np.exp(lp - logsumexp(lp, axis=2, keepdims=True))
            q_tau = (q_post * tau_l).sum(axis=2) / np.maximum(q_post.sum(axis=2), 1e-12)
            dev2 = ((q_post * (tau_l - sig_cur[..., None]) ** 2).sum(axis=2)).mean(axis=0)
            if hard:
                q_tau = hard_levels(q_tau, pi, tau_l, time)
                dev2 = ((q_tau - sig_cur) ** 2).mean(axis=0)
            k_g = np.clip(np.mean(sig_cur * (1 - sig_cur), axis=0) / np.maximum(dev2, 1e-12) - (prior == 'beta'),
                          1.0, model.grid_network_precision_max)
            kappa = np.where(multi, k_g[None, :], 1.0)
        sel = [token(cands[b][0], cands[b][2]) for b in cur[1:]]
        herb = [a == b for a, b in zip(sel, fam_true) if kind(a) == kind(b) == 'B']
        # Calibration of the final targets: mean target per decile of the true sigma (targets)
        b = np.clip((sig_true[:, 1:] * 10).astype(int), 0, 9)
        calib = ' '.join(f'{q_tau[:, 1:][b == k].mean():.2f}' if (b == k).sum() > 20 else 'nan' for k in range(10))
        res = dict(network=name, run=run, targets='hard' if hard else 'soft', score=score, start=start, levels=levels,
                   prior=prior, iters=iters,
                   family_acc=float(np.mean([kind(a) == kind(b) for a, b in zip(sel, fam_true)])),
                   exact_acc=float(np.mean([a == b for a, b in zip(sel, fam_true)])),
                   herbach_m_acc=float(np.mean(herb)) if herb else np.nan,
                   true_family=' '.join(fam_true), selected=' '.join(sel), calibration=calib)
        print(', '.join(f'{k} {v:.2f}' if isinstance(v, float) else f'{k} {v}' for k, v in res.items()), flush=True)
        out.append(res)
    return out


def nb_nll(model, sig, x):
    """Negative log-likelihood (n, G1) of the counts x under the NB of the burst frequency k_low + (k_high - k_low) sigma
    (lowest / highest mode and rate of the mixture, single sample, no depth factor)."""
    from scipy.special import gammaln
    from CardamomOT.inference.depth_model import nb_shape_rate, set_depth_model
    set_depth_model(model.depth_model)
    a = np.asarray(model.a, dtype=float)
    a = a[0] if a.ndim == 3 else a
    k_lo, k_hi, c = a[:-1].min(axis=0), a[:-1].max(axis=0), a[-1]
    r, rate = nb_shape_rate(k_lo + (k_hi - k_lo) * sig, c, None)
    r, rate = np.maximum(np.asarray(r, dtype=float), 1e-10), np.asarray(rate, dtype=float)
    return -(gammaln(x + r) - gammaln(r) - gammaln(x + 1) + r * np.log(rate / (1 + rate)) - x * np.log1p(rate))


def hard_levels(order, pi, tau_l, time):
    """Basins (N, G1) as levels tau: at each time, the cells ranked by order (expected level) take the levels in
    increasing order with the masses of the mixture (mean posteriors pi of the time)."""
    out = np.zeros(order.shape)
    for t in np.unique(time):
        rows = np.flatnonzero(time == t)
        for g in range(order.shape[1]):
            w = pi[rows, g].mean(axis=0)
            cum = np.cumsum(w) / max(w.sum(), 1e-12)
            rank = (np.argsort(np.argsort(order[rows, g], kind='stable'), kind='stable') + 0.5) / len(rows)
            lev = np.minimum(np.searchsorted(cum, rank), len(w) - 1)
            out[rows, g] = tau_l[rows[0], g, lev]
    return out


def start_index(cands, start):
    """Starting candidate of every target, 'identity' ('sigmoid'), 'hill' or 'b<m>'; the stimulus stays a sigmoid."""
    from CardamomOT.inference.sigma import PHI_LOG
    if start in ('identity', 'sigmoid'):
        return next(f for f, fam in enumerate(cands) if int(fam[0]) == 0)
    if start == 'hill':
        return next(f for f, fam in enumerate(cands) if int(fam[0]) == PHI_LOG)
    return next(f for f, fam in enumerate(cands) if is_herbach(int(fam[0])) and fam[2] == float(start[1:]))


def _log_prior(sig, tau_l, kappa, loglik, prior='kl'):
    """log p(l | sigma) of the network prior on the levels: 'kl' (NetworkModel._marginal_nll) or 'beta'."""
    if prior == 'beta':
        from scipy.special import xlogy
        t = np.clip(tau_l, 1e-3, 1 - 1e-3)
        a = np.clip(kappa[..., None] * sig[..., None], 1e-3, None)
        b = np.clip(kappa[..., None] * (1 - sig[..., None]), 1e-3, None)
        logp = np.where(np.isfinite(loglik), xlogy(a - 1, t) + xlogy(b - 1, 1 - t), -np.inf)
        return logp - logsumexp(logp, axis=2, keepdims=True)
    sig_b = np.ascontiguousarray(np.broadcast_to(sig[..., None], tau_l.shape)).ravel()
    kl = cross_entropy(sig_b, np.ascontiguousarray(tau_l).ravel()).reshape(tau_l.shape)
    logp = np.where(np.isfinite(loglik), -kappa[..., None] * kl, -np.inf)
    return logp - logsumexp(logp, axis=2, keepdims=True)


def append(name, rows, fname='oracle_summary.csv'):
    path = os.path.join(BENCH, name, fname)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('network')
    ap.add_argument('runs', nargs='?', default='1-10')
    ap.add_argument('--targets', default='mixture,true')
    ap.add_argument('--fits', type=int, default=5)
    ap.add_argument('--weak-penalty', action='store_true')
    ap.add_argument('--relaxed', action='store_true')
    ap.add_argument('--em', action='store_true')
    ap.add_argument('--scores', default='ce,marginal')
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--start', default='sigmoid')
    ap.add_argument('--levels', default='grid')
    ap.add_argument('--prior', default='kl')
    ap.add_argument('--hard', action='store_true')
    a = ap.parse_args()
    runs = list(range(int(a.runs.split('-')[0]), int(a.runs.split('-')[-1]) + 1)) if '-' in a.runs \
        else [int(x) for x in a.runs.split(',')]
    for r in runs:
        if a.em:
            append(a.network, em_run(a.network, r, a.scores.split(','), iters=a.iters, K=a.fits, start=a.start,
                                         levels=a.levels, prior=a.prior, hard=a.hard), 'oracle_em6_summary.csv')
            continue
        append(a.network, oracle_run(a.network, r, a.targets.split(','), K=a.fits, weak_penalty=a.weak_penalty,
                                     relaxed=a.relaxed), 'oracle_theta6_summary.csv')
