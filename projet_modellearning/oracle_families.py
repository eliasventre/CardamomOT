"""
Oracle of the family selection: the selection of fit_network (same network fit, same held-out score, inference/sigma.py
families) on the TRUE proteins of the observed cells instead of the reconstructed trajectories.

Targets of the fits and of the score:
- 'mixture': expected mode of each gene under the posteriors of the mixture (as the selection of fit_network);
- 'true': position of the true burst frequency between the extreme modes (noise-free upper bound).

Usage: python oracle_families.py <network> [runs] [--targets mixture,true] [--fits K] [--weak-penalty]
Needs the mixture of each run, <network>/runs/run<r>_{base,ml}/cardamomOT (run_inference.py), and the true proteins,
<network>/True/prot_<r>.npy (generate.py). Summary appended to <network>/oracle_summary.csv.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from CardamomOT.model.base import NetworkModel  # noqa: E402
from CardamomOT.inference import inference_network_multi  # noqa: E402
from CardamomOT.inference.network import sigmoid_amplitudes, sigmoid_target, cross_entropy  # noqa: E402
from CardamomOT.inference.sigma import (sigma_pairs, default_spec, SPEC_PHI, PHI_IDENTITY, PHI_LOG,  # noqa: E402
                                        )


def subsamples(time, K, rng):
    """K disjoint sub-samples of the cells, each with the same share of every time point."""
    sels = [[] for _ in range(K)]
    for t in np.unique(time):
        idx = rng.permutation(np.flatnonzero(time == t))
        for k, part in enumerate(np.array_split(idx, K)):
            sels[k].append(part)
    return [np.concatenate(s) for s in sels]


def held_out_ce(fits, sels, spec, P, tau):
    """Cross-entropy (G+1,) of each target, of the fit of each sub-sample on the cells of the other ones."""
    acc = np.zeros(P.shape[1])
    for k, (basal, inter, _, _) in enumerate(fits):
        held = np.concatenate([s for m, s in enumerate(sels) if m != k])
        sig = sigma_pairs(np.asarray(basal[0], float), np.asarray(inter, float), P[held], spec)[:, :, 1]
        ce = cross_entropy(np.ascontiguousarray(sig).ravel(), np.ascontiguousarray(tau[held]).ravel())
        acc += ce.reshape(sig.shape).mean(axis=0)
    return acc / len(fits)


def oracle_run(name, run, targets, K=5, seed=0, weak_penalty=False):
    import anndata as ad
    truth = np.load(os.path.join(HERE, name, 'True', f'truth_{run}.npz'))
    P = np.load(os.path.join(HERE, name, 'True', f'prot_{run}.npy'))       # (C, G+1), stimulus 1 after the burn-in
    time = ad.read_h5ad(os.path.join(HERE, name, 'Data', f'data_{run}.h5ad')).obs['time'].values
    # Mixture of the run (independent of the model learning): base project, else the model-learning one
    cdir = next(d for d in (os.path.join(HERE, name, 'runs', f'run{run}_{m}', 'cardamomOT') for m in ('base', 'ml'))
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
    hill = truth['hill'].astype(bool)
    # Penalties of the method, or nearly none (upper bound: weak ridge, low sparsity threshold)
    pen = dict(model._penalty_kw(), **(dict(network_ridge=1e6, qut_level=0.01) if weak_penalty else {}))
    out = []
    for target in targets:
        tau = tau_t[target]
        y_kon = kb[:, 0] + (kb[:, 1] - kb[:, 0]) * tau        # sigmoid target tau in the network fit
        scores = {}
        for phi in (PHI_IDENTITY, PHI_LOG):
            spec = default_spec(G1, kappa=model.model_learning_kappa * model.scale_proteins, pbar=model.scale_proteins)
            spec[1:, SPEC_PHI] = phi
            fits = inference_network_multi(
                sels, samples, y_kon, P, None, ks, n_stimuli=1, samples_id=[0], ref_network=ref,
                weight_flow_matching=0.0, final=0, seuil_min_network_intensity=model.seuil_min_network_intensity,
                sigma_spec_grn=spec, **pen)
            scores[phi] = held_out_ce(fits, sels, spec, P, tau)
        sel = scores[PHI_LOG][1:] < scores[PHI_IDENTITY][1:]
        gain = (scores[PHI_IDENTITY][1:] - scores[PHI_LOG][1:]) / np.maximum(scores[PHI_IDENTITY][1:], 1e-12)
        res = dict(network=name, run=run, targets=target, penalty='weak' if weak_penalty else 'default', family_acc=float(np.mean(sel == hill[1:])),
                   true_hill=''.join('H' if h else 'S' for h in hill[1:]),
                   selected=''.join('H' if h else 'S' for h in sel),
                   gain_log=' '.join(f'{g:+.3f}' for g in gain))
        print(', '.join(f'{k} {v:.2f}' if isinstance(v, float) else f'{k} {v}' for k, v in res.items()), flush=True)
        out.append(res)
    return out


def append(name, rows):
    path = os.path.join(HERE, name, 'oracle_summary.csv')
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
    a = ap.parse_args()
    runs = list(range(int(a.runs.split('-')[0]), int(a.runs.split('-')[-1]) + 1)) if '-' in a.runs \
        else [int(x) for x in a.runs.split(',')]
    for r in runs:
        append(a.network, oracle_run(a.network, r, a.targets.split(','), K=a.fits, weak_penalty=a.weak_penalty))
