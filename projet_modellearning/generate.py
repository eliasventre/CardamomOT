"""
Generation of the runs of the benchmark networks with mixed families of sigma (model of the inference: proteins only,
mRNA counts drawn from the NB of the burst frequency).

Usage: python generate.py [network ...] [--runs N] [--cells C]
       (default: FN4 CN5 BN8 FN8, 10 runs, 1000 cells)
Per network and run r, in bench/<name>/: True/truth_r.npz, inter_r.npy (binary topology), prot_r.npy (true proteins
of the cells, column 0 = stimulus) and Data/data_r.h5ad (counts, input of CardamomOT), data_prot_r.h5ad (proteins of
the genes, Data/data_prot.h5ad of the runs with observed proteins).
"""
import argparse
import os

import numpy as np

from networks import NETWORKS, Network
from ml_sim import cell_times, simulate_snapshot, write_h5ad, write_prot_h5ad

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, 'bench')


def generate(name, N=10, C=1000, seed=0):
    np.random.seed(seed)
    root = os.path.join(BENCH, name)
    os.makedirs(os.path.join(root, 'True'), exist_ok=True)
    os.makedirs(os.path.join(root, 'Data'), exist_ok=True)
    time = cell_times(C)
    for r in range(1, N + 1):
        net = Network(name, r, seed=seed)
        print(f'{name} run {r}: families {list(net.family[1:])}', flush=True)
        rna, P = simulate_snapshot(net, time)
        np.save(os.path.join(root, 'True', f'prot_{r}'), P)
        np.savez(os.path.join(root, 'True', f'truth_{r}.npz'), **net.truth())
        np.save(os.path.join(root, 'True', f'inter_{r}'), 1 * (np.abs(net.inter) > 0))
        write_h5ad(os.path.join(root, 'Data', f'data_{r}.h5ad'), time, rna, net)
        write_prot_h5ad(os.path.join(root, 'Data', f'data_prot_{r}.h5ad'), time, P, net)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('networks', nargs='*', default=list(NETWORKS))
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--cells', type=int, default=1000)
    a = ap.parse_args()
    for name in a.networks:
        generate(name, N=a.runs, C=a.cells)
