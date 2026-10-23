"""
Generation of the runs of the benchmark networks with mixed families of sigma.

Usage: python generate.py [network ...] [--runs N] [--cells C] [--proteins-only]
       (default: FN4 CN5 BN8 FN8, 10 runs, 1000 cells)
Per network and run r: <name>/True/truth_r.npz, inter_r.npy (binary topology), hill_r.npy (Hill targets), prot_r.npy
(true proteins of the cells, column 0 = stimulus) and <name>/Data/data_r.txt (format of simulate_data_FN4.py),
data_r.h5ad (input of CardamomOT, true proteins in obsm['true_P']). --proteins-only: the same simulation (same seed)
writes only True/prot_r.npy, after checking that the counts are those of the existing data_r.h5ad.
"""
import argparse
import os

import numpy as np

from networks import NETWORKS, Network
from ml_sim import cell_times, simulate_snapshot, write_txt, write_h5ad

HERE = os.path.dirname(os.path.abspath(__file__))


def generate(name, N=10, C=1000, seed=0, proteins_only=False):
    np.random.seed(seed)
    root = os.path.join(HERE, name)
    os.makedirs(os.path.join(root, 'True'), exist_ok=True)
    os.makedirs(os.path.join(root, 'Data'), exist_ok=True)
    time = cell_times(C)
    for r in range(1, N + 1):
        net = Network(name, r, seed=seed)
        print(f'{name} run {r}: Hill targets {list(np.flatnonzero(net.hill))}', flush=True)
        rna, P = simulate_snapshot(net, time)
        np.save(os.path.join(root, 'True', f'prot_{r}'), P)
        if proteins_only:
            import anndata as ad
            old = ad.read_h5ad(os.path.join(root, 'Data', f'data_{r}.h5ad')).X
            if not np.array_equal(np.asarray(old), rna):
                raise RuntimeError(f'{name} run {r}: the counts differ from the existing data')
            continue
        np.savez(os.path.join(root, 'True', f'truth_{r}.npz'), **net.truth())
        np.save(os.path.join(root, 'True', f'inter_{r}'), 1 * (np.abs(net.inter) > 0))
        np.save(os.path.join(root, 'True', f'hill_{r}'), net.hill)
        write_txt(os.path.join(root, 'Data', f'data_{r}.txt'), time, rna)
        write_h5ad(os.path.join(root, 'Data', f'data_{r}.h5ad'), time, rna, net, P=P)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('networks', nargs='*', default=list(NETWORKS))
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--cells', type=int, default=1000)
    ap.add_argument('--proteins-only', action='store_true')
    a = ap.parse_args()
    for name in a.networks:
        generate(name, N=a.runs, C=a.cells, proteins_only=a.proteins_only)
