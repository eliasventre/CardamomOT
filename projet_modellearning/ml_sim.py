"""
Simulation of the benchmark networks with the model of the inference of CardamomOT: bursty PDMP of the proteins only
(inference/simulations.py, parametrised as simulate_network: protein bursts c = k1 d0 / d1, scale 1, so that the high
mode of every gene is at P = 1) with the families of sigma of each gene, and mRNA counts drawn at the observation from
the negative binomial of the burst frequency, NB(kon(P) / d0, c_nb) (no memory of the mRNA). Independent cells, each
with a burn-in of 5 h without stimulus, then the stimulus from t = 0, observed once.
"""
import os
import sys

import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from CardamomOT.inference.simulations import simulate_next_prot_pdmp  # noqa: E402
from CardamomOT.inference.sigma import kon_matrix  # noqa: E402

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
    """Proteins P (G+1,) of one cell at time t (column 0 = stimulus, 0 at t = 0: not yet stimulated)."""
    c_prot = net.kz.max(axis=1) * net.d0 / net.d1
    sim = lambda times, P0, stim: simulate_next_prot_pdmp(
        net.d1, net.ks, c_prot, net.basal, net.inter, np.atleast_1d(times).astype(float), 1.0,
        P0=P0, ns=1, stim_vals=np.full(1, stim), spec=net.spec)
    b = sim(BURNIN, np.zeros(net.G + 1), 0.0)
    out = sim(t, np.concatenate([[1.0], b.p[-1]]), 1.0)
    return np.concatenate([[float(t > 0)], out.p[-1]])


def simulate_snapshot(net, time):
    """Counts (C, G) of independent cells observed at time[n], NB of their burst frequency at that time, and their
    proteins (C, G+1), column 0 = stimulus (1 after t = 0)."""
    P = np.array([simulate_cell(net, t) for t in time])
    kon = kon_matrix(P, net.ks, net.inter, net.basal, net.spec)[:, 1:]
    c = net.c_nb[1:]
    rna = np.random.negative_binomial(np.maximum(kon / net.d0[1:], 1e-12), c / (c + 1))
    return rna, P


def _adata(X, time, net):
    genes = [f'G{g}' for g in range(1, X.shape[1] + 1)]
    obs = pd.DataFrame({'time': np.asarray(time, dtype=float)}, index=pd.Index([f'cell_{i}' for i in range(len(time))]))
    var = pd.DataFrame({'d0': net.d0[1:], 'd1': net.d1[1:]}, index=pd.Index(genes))
    return ad.AnnData(X=X, obs=obs, var=var)


def write_h5ad(fname, time, rna, net):
    """Input of CardamomOT: counts, obs['time'], var d0 / d1, ground truth in uns."""
    adata = _adata(np.asarray(rna, dtype=np.int64), time, net)
    adata.uns['truth'] = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in net.truth().items()}
    adata.write_h5ad(fname)


def write_prot_h5ad(fname, time, P, net):
    """Data/data_prot.h5ad: proteins of the genes (units of cardamomOT/data_prot.npy), same cells and genes."""
    _adata(np.asarray(P, dtype=float)[:, 1:], time, net).write_h5ad(fname)
