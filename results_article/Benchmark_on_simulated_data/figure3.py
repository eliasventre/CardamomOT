# Benchmark (directed|undirected) for all test networks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.metrics import precision_recall_curve, auc

path = './'

algoD = ['CardamomOT', 'CARDAMOM', 'Reference Fitting', 'GENIE3', 'SINCERITIES']
algoU = ['CardamomOT', 'CARDAMOM', 'Reference Fitting', 'GENIE3', 'SINCERITIES', 'PIDC', 'PEARSON']

# Map display names to folder names on disk
algo_file = {'CardamomOT': 'CARDAMOM2', 'CARDAMOM': 'CARDAMOM1', 'Reference Fitting': 'REFERENCE_FITTING'}
def file_name(algo): return algo_file.get(algo, algo)

benchmarks = ['FN4', 'CN5', 'BN8', 'FN8',
    'Trees5', 'Trees10', 'Trees20', 'Trees50', 'Trees100']
N = 10

cmap = plt.get_cmap('tab20')
c = {
    'CardamomOT':        (cmap(6),  cmap(7)),
    'CARDAMOM':          (cmap(8),  cmap(9)),
    'Reference Fitting': (cmap(18), cmap(19)),
    'GENIE3':            (cmap(0),  cmap(1)),
    'SINCERITIES':       (cmap(2),  cmap(3)),
    'PIDC':              (cmap(4),  cmap(5)),
    'PEARSON':           (cmap(14), cmap(15)),
    'Random':            2*('lightgray',)
}

# Figure: 6.85 in wide (=174 mm, double-column A4), 7.5 in tall (~190 mm)
# hspace increased slightly to accommodate rotated labels
fig = plt.figure(figsize=(6.85, 7.5))
grid = gs.GridSpec(5, 3, hspace=0.55, wspace=0.18,
    height_ratios=[1,1,1,1,1], width_ratios=[6,1,7])

def configure(ax):
    w = 0.7
    ax.tick_params(direction='out', length=3, width=w)
    ax.tick_params(axis='x', pad=2, labelsize=5.5)
    ax.tick_params(axis='y', pad=0.5, labelsize=5.5)
    for side in ['top','bottom','left','right']: ax.spines[side].set_linewidth(w)
    ax.set_ylim(0,1)

opt_box = {'patch_artist': True, 'widths': [.25]}
def configure_box(box, col):
    w = 0.8
    for item in ['boxes','whiskers','fliers','medians','caps']:
        plt.setp(box[item], color=col[0], lw=w)
    plt.setp(box['boxes'], facecolor=col[1])
    plt.setp(box['fliers'], markeredgecolor=col[0], ms=3,
             markerfacecolor=col[1], markeredgewidth=w)

x, y = -11, 10
xn, yn = -0.142, 0.875
opt = {'xy': (0,1), 'xycoords': 'axes fraction', 'fontsize': 10,
       'textcoords': 'offset points', 'annotation_clip': False}

auprTreesD = {algo: [] for algo in algoD+['Random']}
auprTreesU = {algo: [] for algo in algoU+['Random']}

for n, benchmark in enumerate(benchmarks):
    auprD = {algo: [] for algo in algoD+['Random']}
    auprU = {algo: [] for algo in algoU+['Random']}

    for r in range(1, N+1):
        inter = abs(np.load(path+f'{benchmark}/True/inter_{r}.npy'))
        G = inter.shape[0]

        edges = [(i,j) for i in range(G) for j in set(range(1,G))-{i}]
        y0 = np.array([inter[i,j] for (i,j) in edges])
        auprD['Random'].append(np.mean(y0))
        for algo in algoD:
            score = abs(np.load(path+f'{benchmark}/{file_name(algo)}/score_{r}.npy'))
            if algo == 'GENIE3': score = score.T
            y1 = np.array([score[i,j] for (i,j) in edges])
            precision, recall, _ = precision_recall_curve(y0, y1)
            auprD[algo].append(auc(recall, precision))

        edges = [(i,j) for i in range(G) for j in range(i+1,G)]
        y0 = np.array([max(inter[i,j],inter[j,i]) for (i,j) in edges])
        auprU['Random'].append(np.mean(y0))
        for algo in algoU:
            score = abs(np.load(path+f'{benchmark}/{file_name(algo)}/score_{r}.npy'))
            y1 = np.array([max(score[i,j],score[j,i]) for (i,j) in edges])
            precision, recall, _ = precision_recall_curve(y0, y1)
            auprU[algo].append(auc(recall, precision))

    if benchmark[:5] == 'Trees':
        for algo in algoD+['Random']: auprTreesD[algo].append(np.mean(auprD[algo]))
        for algo in algoU+['Random']: auprTreesU[algo].append(np.mean(auprU[algo]))

    name = ['FN4', 'CN5', 'BN8', 'FN8']
    if n < 4:
        # Panel A — Directed
        ax = plt.subplot(grid[n,0])
        configure(ax)
        if n == 0:
            ax.annotate('A', xytext=(x+0.2,y), fontweight='bold', **opt)
            ax.annotate('Snapshot-based', xytext=(x+14,y), **opt)
        b = np.mean(auprD['Random'])
        ax.plot([0,5],[b,b], color='lightgray', ls='--', lw=0.8, zorder=0)
        ax.set_xlim(0.8, 5.5)
        for i, algo in enumerate(algoD):
            box = ax.boxplot([auprD[algo]], positions=[i+1], **opt_box)
            configure_box(box, c[algo])
        ax.set_xticks(range(1, len(algoD)+1))
        ax.set_xticklabels(algoD, fontsize=4.5, rotation=30, ha='right')
        ax.set_ylabel('AUPR', fontsize=6)
        optn = {'fontsize': 9, 'transform': ax.transAxes, 'ha': 'right'}
        ax.text(xn, yn, name[n], **optn)
        ax.text(xn, yn+0.01, name[n], color='none', zorder=0,
                bbox=dict(boxstyle='round,pad=0.2',fc='none',ec='lightgray',lw=0.8), **optn)

        # Panel B — Undirected
        ax = plt.subplot(grid[n,2])
        configure(ax)
        if n == 0:
            ax.annotate('B', xytext=(x,y), fontweight='bold', **opt)
            ax.annotate('Snapshot-based (undirected edges)', xytext=(x+14,y), **opt)
        b = np.mean(auprU['Random'])
        ax.plot([0,7],[b,b], color='lightgray', ls='--', lw=0.8, zorder=0)
        ax.set_xlim(0.8, 7.4)
        for i, algo in enumerate(algoU):
            box = ax.boxplot([auprU[algo]], positions=[i+1], **opt_box)
            configure_box(box, c[algo])
        ax.set_xticks(range(1, len(algoU)+1))
        ax.set_xticklabels(algoU, fontsize=4.5, rotation=30, ha='right')

# Trees panels
s = {'ls': '--', 'lw': 0.85, 'marker': '.', 'ms': 4}
p = {'borderaxespad': 0, 'frameon': False, 'fontsize': 4.5,
     'handlelength': 1.2, 'handletextpad': 0.5}
size = [5, 10, 20, 50, 100]

ax = plt.subplot(grid[4,0])
configure(ax)
for algo in algoD+['Random']:
    ax.plot(size, auprTreesD[algo], color=c[algo][0], label=algo, **s)
ax.legend(loc='upper right', **p)
ax.set_xticks(size)
ax.set_xlabel('No. of genes', fontsize=6, labelpad=1.8)
ax.set_ylabel('AUPR', fontsize=6)
optn = {'fontsize': 9, 'transform': ax.transAxes, 'ha': 'right'}
ax.text(xn, yn, 'Trees', **optn)
ax.text(xn, yn+0.01, 'Trees', color='none', zorder=0,
        bbox=dict(boxstyle='round,pad=0.2',fc='none',ec='lightgray',lw=0.8), **optn)

ax = plt.subplot(grid[4,2])
configure(ax)
for algo in algoU+['Random']:
    ax.plot(size, auprTreesU[algo], color=c[algo][0], label=algo, **s)
ax.legend(loc='upper right', ncol=2, **p)
ax.set_xticks(size)
ax.set_xlabel('No. of genes', fontsize=6, labelpad=1.8)

fig.savefig('figure3.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()