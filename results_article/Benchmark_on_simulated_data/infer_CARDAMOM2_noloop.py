# Script pour lancer tous les benchmarks de CARDAMOM
import sys; sys.path += ['./../../']
import time as timer
import numpy as np
from CardamomOT import NetworkModel

# Number of runs
N = 10
# Print information
verb = 1
# number of initial couplings
n_repet=2

# Inference for Network4
for r in range(0, N):
    fname = 'FN4/Data/data_{}.txt'.format(r + 1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]
    x = data.T
    x[:, 0] = time
    G = np.size(x, 1)
    model = NetworkModel(G - 1)
    model.recompute_degradations=0
    model.d = np.loadtxt('FN4/Data/Rates/degradation_rates.txt', dtype=float, delimiter='\t').T 
    model.max_iter=1
    model.fit(x, verb=verb)
    score = model.inter_t[-1]
    for n in range(n_repet):
        model = NetworkModel(G - 1)
        model.recompute_degradations=0
        model.d = np.loadtxt('FN4/Data/Rates/degradation_rates.txt', dtype=float, delimiter='\t').T 
        model.max_iter=1
        model.fit(x, verb=verb)
        score += model.inter_t[-1]
    np.save('FN4/noloop/score_{}'.format(r+1), score)

# Inference for Cycle
for r in range(0, N):
    fname = 'CN5/Data/data_{}.txt'.format(r + 1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:, 1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0, 1:]
    x = data.T
    x[:, 0] = time
    G = np.size(x, 1)
    model = NetworkModel(G - 1)
    model.recompute_degradations=0
    model.d = np.loadtxt('CN5/Data/Rates/degradation_rates.txt', dtype=float, delimiter='\t').T 
    model.max_iter=1
    model.fit(x, verb=verb)
    score = model.inter_t[-1]
    for n in range(n_repet):
        model = NetworkModel(G - 1)
        model.recompute_degradations=0
        model.d = np.loadtxt('CN5/Data/Rates/degradation_rates.txt', dtype=float, delimiter='\t').T 
        model.max_iter=1
        model.fit(x, verb=verb)
        score += model.inter_t[-1]
    np.save('CN5/noloop/score_{}'.format(r+1), score)

# Inference for Bifurcation
for r in range(0, N):
    fname = 'BN8/Data/data_{}.txt'.format(r + 1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:, 1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0, 1:]
    x = data.T
    x[:, 0] = time
    G = np.size(x, 1)
    model = NetworkModel(G - 1)
    model.recompute_degradations=0
    model.d = np.loadtxt('BN8/Data/Rates/degradation_rates.txt', dtype=float, delimiter='\t').T 
    model.max_iter=1
    model.fit(x, verb=verb)
    score = model.inter_t[-1]
    for n in range(n_repet):
        model = NetworkModel(G - 1)
        model.recompute_degradations=0
        model.d = np.loadtxt('BN8/Data/Rates/degradation_rates.txt', dtype=float, delimiter='\t').T
        model.max_iter=1 
        model.fit(x, verb=verb)
        score += model.inter_t[-1]
    np.save('BN8/noloop/score_{}'.format(r+1), score)


# Inference for Trifurcation
for r in range(0, N):
    fname = 'FN8/Data/data_{}.txt'.format(r + 1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:, 1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0, 1:]
    x = data.T
    x[:, 0] = time
    G = np.size(x, 1)
    model = NetworkModel(G - 1)
    model.recompute_degradations=0
    model.d = np.loadtxt('FN8/Data/Rates/degradation_rates.txt', dtype=float, delimiter='\t').T 
    model.max_iter=1
    model.fit(x, verb=verb)
    score = model.inter_t[-1]
    for n in range(n_repet):
        model = NetworkModel(G - 1)
        model.recompute_degradations=0
        model.d = np.loadtxt('FN8/Data/Rates/degradation_rates.txt', dtype=float, delimiter='\t').T 
        model.max_iter=1
        model.fit(x, verb=verb)
        score += model.inter_t[-1]
    np.save('FN8/noloop/score_{}'.format(r+1), score)

print("We are at trees")
# Inference for tree-like networks
for n in [5, 10, 20, 50, 100]:
    runtime = np.zeros(N)
    for r in range(N):
        fname = 'Trees{}/Data/data_{}.txt'.format(n,r+1)
        data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
        time = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]
        x = data.T
        x[:,0] = time
        G = np.size(x, 1)
        model = NetworkModel(G - 1)
        model.recompute_degradations=0
        model.d = np.loadtxt('Trees{}/Data/Rates/degradation_rates.txt'.format(n), dtype=float, delimiter='\t').T 
        model.max_iter=1
        t0 = timer.time()
        model.fit(x, verb=verb)
        t1 = timer.time()
        runtime[r] = t1 - t0
        score = model.inter_t[-1]
        for _ in range(n_repet):
            model = NetworkModel(G - 1)
            model.recompute_degradations=0
            model.d = np.loadtxt('Trees{}/Data/Rates/degradation_rates.txt'.format(n), dtype=float, delimiter='\t').T 
            model.max_iter=1
            model.fit(x, verb=verb)
            score += model.inter_t[-1]
        np.save('Trees{}/noloop/score_{}'.format(n,r+1), score)
    # Save running times
    np.savetxt('Trees{}/noloop/runtime.txt'.format(n), runtime.T)
