# Script pour lancer tous les benchmarks de CARDAMOM
import sys; sys.path += ['./../../']
import time as timer
import numpy as np
from CardamomOT import NetworkModel

# Number of runs
N = 10
# Print information
verb = 1

scaledeg = 5

# Inference for Network4
for r in range(0, N):
    fname = 'FN4/Data/data_{}.txt'.format(r + 1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:,1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0,1:]
    x = data.T
    x[:, 0] = time
    G = np.size(x, 1)
    model = NetworkModel(G - 1)
    model.d = np.loadtxt('FN4/Data/Rates/degradation_rates.txt', dtype=float, delimiter='\t').T 
    print(model.d)
    model.d += np.random.normal(loc=0, scale=np.mean(model.d[1])/scaledeg, size=model.d.shape)
    model.d = np.maximum(model.d, .002)
    print(model.d)
    model.fit(x, verb=verb)
    score = model.inter_t[-1]
    np.save('FN4/CARDAMOM2_degmult2/score_{}'.format(r+1), score)

# Inference for Cycle
for r in range(0, N):
    fname = 'CN5/Data/data_{}.txt'.format(r + 1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:, 1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0, 1:]
    x = data.T
    x[:, 0] = time
    G = np.size(x, 1)
    model = NetworkModel(G - 1)
    model.d = np.loadtxt('CN5/Data/Rates/degradation_rates.txt', dtype=float, delimiter='\t').T 
    model.d += np.random.normal(loc=0, scale=np.mean(model.d[1])/scaledeg, size=model.d.shape)
    model.d = np.maximum(model.d, .002)
    model.fit(x, verb=verb)
    score = model.inter_t[-1]
    np.save('CN5/CARDAMOM2_degmult2/score_{}'.format(r+1), score)

# Inference for Bifurcation
for r in range(0, N):
    fname = 'BN8/Data/data_{}.txt'.format(r + 1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:, 1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0, 1:]
    x = data.T
    x[:, 0] = time
    G = np.size(x, 1)
    model = NetworkModel(G - 1)
    model.d = np.loadtxt('BN8/Data/Rates/degradation_rates.txt', dtype=float, delimiter='\t').T 
    model.d += np.random.normal(loc=0, scale=np.mean(model.d[1])/scaledeg, size=model.d.shape)
    model.d = np.maximum(model.d, .002)
    model.fit(x, verb=verb)
    score = model.inter_t[-1]
    np.save('BN8/CARDAMOM2_degmult2/score_{}'.format(r+1), score)


# Inference for Trifurcation
for r in range(0, N):
    fname = 'FN8/Data/data_{}.txt'.format(r + 1)
    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:, 1:]
    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0, 1:]
    x = data.T
    x[:, 0] = time
    G = np.size(x, 1)
    model = NetworkModel(G - 1)
    model.d = np.loadtxt('FN8/Data/Rates/degradation_rates.txt', dtype=float, delimiter='\t').T 
    model.d += np.random.normal(loc=0, scale=np.mean(model.d[1])/scaledeg, size=model.d.shape)
    model.d = np.maximum(model.d, .002)
    model.fit(x, verb=verb)
    score = model.inter_t[-1]
    np.save('FN8/CARDAMOM2_degmult2/score_{}'.format(r+1), score)

