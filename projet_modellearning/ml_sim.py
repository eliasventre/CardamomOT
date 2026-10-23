"""
Simulation of the benchmark networks with the full model of CardamomOT (bursty PDMP with explicit mRNA,
inference/simulations.py) and the families of sigma of each gene, as the Harissa benchmarks: independent cells,
each with a burn-in of 5 h without stimulus, then the stimulus from t = 0, observed once (counts Poisson(M)).
"""
import os
import sys

import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from CardamomOT.inference.simulations import simulate_next_mrna_pdmp  # noqa: E402

TIMES = np.array([0, 6, 12, 24, 36, 48, 60, 72, 84, 96])
BURNIN = 5.0


def cell_times(C=1000):
    """Time of each cell: C / 10 cells per time point."""
    k = np.linspace(0, C, len(TIMES) + 1, dtype='int')
    time = np.zeros(C, dtype='int')
    for i in range(len(TIMES)):
        time[k[i]:k[i + 1]] = TIMES[i]
    return time


def simulate_cell(net, t):
    """mRNA and protein levels M, P (G+1,) of one cell at time t (column 0 = stimulus)."""
    sim = lambda times, P0, M0, stim: simulate_next_mrna_pdmp(
        net.d0, net.d1, net.ks, net.c_nb, net.basal, net.inter, np.atleast_1d(times).astype(float), 1.0,
        P0=P0, M0=M0, ns=1, stim_vals=np.full(1, stim), spec=net.spec)
    G1 = net.G + 1
    b = sim(BURNIN, np.zeros(G1), np.zeros(G1), 0.0)
    out = sim(t, np.concatenate([[1.0], b.p[-1]]), np.concatenate([[0.0], b.m[-1]]), 1.0)
    return np.concatenate([[0.0], out.m[-1]]), np.concatenate([[1.0], out.p[-1]])


def simulate_snapshot(net, time):
    """Counts (C, G) of independent cells observed at time[n], and their true proteins (C, G+1), column 0 = stimulus
    (1 after the burn-in)."""
    MP = [simulate_cell(net, t) for t in time]
    M, P = np.array([m for m, _ in MP]), np.array([p for _, p in MP])
    return np.random.poisson(M[:, 1:]), P


def write_txt(fname, time, rna):
    """Text format of simulate_data_FN4.py: row 0 = gene index, column 0 = time, column 1 = stimulus."""
    C, G = rna.shape
    data = np.zeros((C + 1, G + 2), dtype='int')
    data[0][1:] = np.arange(G + 1)
    data[1:, 0] = time
    data[1:, 1] = 100 * (time > 0)
    data[1:, 2:] = rna
    np.savetxt(fname, data.T, fmt='%d', delimiter='\t')


def write_h5ad(fname, time, rna, net, P=None):
    """Input of CardamomOT: counts, obs['time'], var d0 / d1, ground truth in uns (and obsm['true_P'], proteins of the
    genes, if given)."""
    genes = [f'G{g}' for g in range(1, rna.shape[1] + 1)]
    obs = pd.DataFrame({'time': np.asarray(time, dtype=float)}, index=pd.Index([f'cell_{i}' for i in range(len(time))]))
    var = pd.DataFrame({'d0': net.d0[1:], 'd1': net.d1[1:]}, index=pd.Index(genes))
    adata = ad.AnnData(X=np.asarray(rna, dtype=np.int64), obs=obs, var=var)
    adata.uns['truth'] = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in net.truth().items()}
    if P is not None:
        adata.obsm['true_P'] = np.asarray(P)[:, 1:]
    adata.write_h5ad(fname)
