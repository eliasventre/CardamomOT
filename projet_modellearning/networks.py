"""
Benchmark networks of CardamomOT (results_article/figures/simulate_data_{FN4,CN5,BN8,FN8}.py, Harissa defaults:
k_low = 0, k_high = 2, mRNA burst size 50) in which each gene is a Herbach-type target of coefficient m = 1, 2 or 4
(HERBACH_M, probability 1/3 each), in the approximate form (saturating Hill steps of threshold 1/2) or the exact one
(thermodynamic model of Herbach et al., same thresholds), probability 1/2 each: the candidate families of
inference/sigma.py, drawn per run (the stimulus stays linear).

Both forms are parametrised by the effect of each edge at the high mode (phi(0) = 0, theta at P = 1), so that every
family keeps the interactions of the network and switches between the same burst frequencies, with different shapes;
in the exact form, a repression beyond -log(1 + 2^m) is at its bound (full repressor of threshold 1/2).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

KAPPA = 0.1           # scale of the regulators of the Hill type, relative to the protein level of the high mode
HERBACH_M = (1.0, 2.0, 4.0)  # Hill coefficients of the Herbach targets

# name: (d0, d1, basal of the genes, {(regulator, target): interaction}), gene 0 = stimulus
NETWORKS = {
    'FN4': (0.2, 0.04, [-5] * 4,
            {(0, 1): 10, (1, 2): 10, (1, 3): 10, (3, 4): 10, (4, 1): -10, (2, 2): 10, (3, 3): 10}),
    'CN5': (0.5, 0.1, [-5, 4, 4, -5, -5],
            {(0, 1): 10, (1, 2): -10, (2, 3): -10, (3, 4): 10, (4, 5): 10, (5, 1): -10}),
    'BN8': (0.25, 0.05, [-4] * 8,
            {(0, 1): 10, (1, 2): 10, (1, 3): 10, (3, 2): -10, (2, 3): -10, (2, 2): 5, (3, 3): 5, (2, 4): 10,
             (3, 5): 10, (2, 5): -10, (3, 4): -10, (4, 7): -10, (5, 6): -10, (4, 6): 10, (5, 7): 10, (7, 8): 10,
             (6, 8): -10}),
    'FN8': (0.4, 0.08, [-5] * 8,
            {(0, 1): 10, (1, 2): 10, (2, 3): 10, (3, 4): 10, (3, 5): 10, (3, 6): 10, (4, 1): -10, (5, 1): -10,
             (6, 1): -10, (4, 4): 10, (5, 5): 10, (6, 6): 10, (4, 8): -10, (4, 7): -10, (6, 7): 10, (7, 6): 10,
             (8, 8): 10}),
}


class Network:
    """Parameters of one run: G genes (index 0 = stimulus), kz (G+1, 2) k_low / k_high (NB shape units, burst
    rate kz d0), c_nb (G+1,), d0, d1 (G+1,), basal, inter, spec (families of sigma, inference/sigma.py)."""

    def __init__(self, name, run, seed=0):
        from CardamomOT.inference.sigma import default_spec, set_family, SPEC_PHI, PHI_HERBACH_APPROX, PHI_HERBACH
        d0, d1, basal, edges = NETWORKS[name]
        self.name, self.run = name, run
        G = len(basal)
        self.G = G
        self.d0, self.d1 = np.full(G + 1, float(d0)), np.full(G + 1, float(d1))
        self.kz = np.tile([0.0, 2.0], (G + 1, 1))
        self.c_nb = np.full(G + 1, 0.02)
        self.basal = np.concatenate([[0.0], np.asarray(basal, float)])
        self.inter = np.zeros((G + 1, G + 1))
        for (i, j), v in edges.items():
            self.inter[i, j] = v
        # Family of each gene (k > 0 Herbach of coefficient HERBACH_M[k - 1], 0 = stimulus) and its form (exact or
        # approximate), drawn per run; threshold of the Herbach regulators 1/2
        rng = np.random.default_rng([seed, list(NETWORKS).index(name), run])
        fam = np.concatenate([[0], rng.integers(1, 1 + len(HERBACH_M), G)])
        exact = np.concatenate([[False], rng.random(G) < 0.5])
        self.spec = default_spec(G + 1, kappa=KAPPA, pbar=1.0)
        for g in np.flatnonzero(fam > 0):
            set_family(self.spec, g, (PHI_HERBACH if exact[g] else PHI_HERBACH_APPROX, 0, HERBACH_M[fam[g] - 1]))
        self.family = self.spec[:, SPEC_PHI].astype(int)
        self.ks = self.kz * self.d0[:, None]

    def truth(self):
        return dict(inter=self.inter, basal=self.basal, spec=self.spec, family=self.family, kz=self.kz,
                    c_nb=self.c_nb, d0=self.d0, d1=self.d1, kappa=np.array(KAPPA))
