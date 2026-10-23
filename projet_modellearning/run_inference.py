"""
CardamomOT on the runs of the benchmark networks, with and without model learning of the GRN sigma, then
evaluation: family of each target (model learning) and network (AUPR, sign of the true edges).

Usage: python run_inference.py <network> [runs] [--modes ml,base[,soft,ct,ml_ct,late,ml_late]] [--compare-only]
       (e.g. python run_inference.py FN4 1-3; runs '1-3' or '1,4,7', default 1)
Projects: <network>/runs/run<r>_<mode>/; summary appended to <network>/summary.csv.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STEPS = ['split_dataset', 'select_genes', 'get_degradation_rates', 'infer_mixture', 'infer_network_structure']
PARAMS = {'split': 'full', 'select_genes': False, 'build_prior_network': False, 'estimate_proliferation_rates': False,
          'simulate_with_proliferation': False, 'simulate_perturbations': False, 'run_test': False,
          'run_classical_OT': False, 'allow_depth_correction': False, 'compute_depth_factor': False, 'seed': 0}
# ml: model learning (soft EM forced); base: default parameters; soft: soft EM without model learning (isolates
# the effect of the model learning when the default is the hard EM); ct, ml_ct: continuous targets E[sigma | counts]; late, ml_late: sigma averaged on the mRNA memory
MODES = {'ml': {'model_learning_grn': True}, 'base': {'model_learning_grn': False},
         'soft': {'model_learning_grn': False, 'one_hot_probas': False},
         'ct': {'model_learning_grn': False, 'max_grid_levels': 11},
         'ml_ct': {'model_learning_grn': True, 'max_grid_levels': 11},
         'late': {'model_learning_grn': False, 'targets_accounting_late_mrna': True},
         'ml_late': {'model_learning_grn': True, 'targets_accounting_late_mrna': True}}


def setup(name, run, mode):
    proj = os.path.join(HERE, name, 'runs', f'run{run}_{mode}')
    os.makedirs(os.path.join(proj, 'Data'), exist_ok=True)
    shutil.copy(os.path.join(HERE, name, 'Data', f'data_{run}.h5ad'), os.path.join(proj, 'Data', 'data.h5ad'))
    from openpyxl import load_workbook
    wb = load_workbook(os.path.join(REPO, 'CardamomOT_inputs.xlsx'))
    ws = wb['model_parameters']
    hdr = [c.value for c in ws[1]]
    ip, iv = hdr.index('parameter'), hdr.index('value')
    rows = {row[ip].value: row for row in ws.iter_rows(min_row=2)}
    for k, v in {**PARAMS, **MODES[mode]}.items():
        if k in rows:
            rows[k][iv].value = v
        else:
            ws.append([k, v] + [None] * (len(hdr) - 2))
    wb.save(os.path.join(proj, 'Data', 'CardamomOT_inputs.xlsx'))
    return proj


def run_steps(proj):
    for step in STEPS:
        print(f'=== {step}', flush=True)
        subprocess.run([sys.executable, '-m', 'CardamomOT.cli', 'step', step, '-i', proj], cwd=REPO, check=True)


def aupr(score, y):
    """Average precision of the scores against the binary labels y."""
    order = np.argsort(-score)
    tp = np.cumsum(y[order])
    prec = tp / np.arange(1, len(order) + 1)
    return float(np.sum(prec * y[order]) / max(y.sum(), 1))


def evaluate(name, run, mode, proj):
    from CardamomOT.inference.sigma import SPEC_PHI, PHI_LOG
    truth = np.load(os.path.join(HERE, name, 'True', f'truth_{run}.npz'))
    inter = np.load(os.path.join(proj, 'cardamomOT', 'inter.npy'))
    inter = inter[..., 0] if inter.ndim == 3 and inter.shape[-1] == 1 else inter
    T = np.sign(truth['inter'])
    G1 = T.shape[0]
    off = ~np.eye(G1, dtype=bool)
    off[:, 0] = False                        # no edge towards the stimulus
    y = (np.abs(T) > 0)[off]
    res = dict(network=name, run=run, mode=mode, aupr=aupr(np.abs(inter)[off], y), aupr_random=float(y.mean()))
    edges = np.argwhere((np.abs(T) > 0) & off)
    res['sign_ok'] = float(np.mean([np.sign(inter[i, j]) == T[i, j] for i, j in edges]))
    hill = truth['hill'].astype(bool)
    spec_path = os.path.join(proj, 'cardamomOT', 'sigma_spec_grn.npy')
    if os.path.exists(spec_path):
        sel = np.load(spec_path)[:, SPEC_PHI] == PHI_LOG
        res.update(family_acc=float(np.mean(sel[1:] == hill[1:])),
                   hill_recall=float(np.mean(sel[1:][hill[1:]])) if hill[1:].any() else np.nan,
                   sigmoid_recall=float(np.mean(~sel[1:][~hill[1:]])) if (~hill[1:]).any() else np.nan,
                   true_hill=''.join('H' if h else 'S' for h in hill[1:]),
                   selected=''.join('H' if h else 'S' for h in sel[1:]))
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    print(f'\n{name} run {run} [{mode}]: inferred inter (rows = regulators, 0 = stimulus)\n{inter}')
    print(', '.join(f'{k} {v:.2f}' if isinstance(v, float) else f'{k} {v}' for k, v in res.items()))
    return res


def append_summary(name, res):
    path = os.path.join(HERE, name, 'summary.csv')
    keys = ['network', 'run', 'mode', 'aupr', 'aupr_random', 'sign_ok', 'family_acc', 'hill_recall', 'sigmoid_recall',
            'true_hill', 'selected']
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if new:
            w.writeheader()
        w.writerow({k: res.get(k, '') for k in keys})


def parse_runs(s):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('network')
    ap.add_argument('runs', nargs='?', default='1')
    ap.add_argument('--modes', default='ml,base')
    ap.add_argument('--compare-only', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, REPO)
    for run in parse_runs(a.runs):
        for mode in a.modes.split(','):
            proj = setup(a.network, run, mode) if not a.compare_only else \
                os.path.join(HERE, a.network, 'runs', f'run{run}_{mode}')
            if not a.compare_only:
                run_steps(proj)
            append_summary(a.network, evaluate(a.network, run, mode, proj))
