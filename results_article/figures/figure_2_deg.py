# ============================================================
# Figure: Directed benchmarks + Trees + Global boxplot
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.metrics import precision_recall_curve, auc

path = './'
N = 10

algoD = ['CardamomOT', 'random init', 'noloop', 'random + noloop']

algo_file = {'CardamomOT': 'CARDAMOM2'}
def file_name(algo): return algo_file.get(algo, algo)

algoG = [
    'CARDAMOM2_deg',
    'CARDAMOM2_degmult2', 'CARDAMOM2_degmult4',
    'CARDAMOM2_degmult8', 'CARDAMOM2_degmult16',
    'CARDAMOM2_degover2', 'CARDAMOM2_degover4', 'CARDAMOM2_degover8'
]
labelG = ['real', '+-20%', '+-50%', '+-100%', '+-150%', '+-200%', '+-300%', '+-400%']

benchmarks = ['FN4', 'FN8', 'CN5', 'BN8']
trees = ['Trees5', 'Trees10', 'Trees20', 'Trees50', 'Trees100']
size = [5, 10, 20, 50, 100]

cmap = plt.get_cmap('tab20')
c = {
    'CardamomOT':      (cmap(6),  cmap(7)),
    'random init':     (cmap(8),  cmap(9)),
    'noloop':          (cmap(0),  cmap(1)),
    'random + noloop': (cmap(2),  cmap(3)),
    'Random':          2*('lightgray',)
}
for a in algoG: c[a] = ('black', 'grey')

# Figure: same width as figure2.py (6.85 in = 174 mm), 6.0 in tall (~152 mm)
fig = plt.figure(figsize=(6.85, 6.0))
grid = gs.GridSpec(3, 2, hspace=0.45, wspace=0.30, height_ratios=[1,1,1])

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
    plt.setp(box['fliers'], markeredgecolor=col[0],
             markerfacecolor=col[1], ms=3, markeredgewidth=w)

x, y = -11, 10
xn, yn = -0.14, 0.87
opt_panel = dict(xy=(0,1), xycoords='axes fraction',
    textcoords='offset points', fontsize=10, annotation_clip=False)

# Panels A-D
panel_letters = ['A','B','C','D']
positions = [(0,0),(0,1),(1,0),(1,1)]

for i, bench in enumerate(benchmarks):
    aupr = {a: [] for a in algoD+['Random']}
    for r in range(1, N+1):
        inter = abs(np.load(f'{path}{bench}/True/inter_{r}.npy'))
        G = inter.shape[0]
        edges = [(ii,jj) for ii in range(G) for jj in set(range(1,G))-{ii}]
        y0 = np.array([inter[ii,jj] for ii,jj in edges])
        aupr['Random'].append(np.mean(y0))
        for a in algoD:
            score = abs(np.load(f'{path}{bench}/{file_name(a)}/score_{r}.npy'))
            y1 = np.array([score[ii,jj] for ii,jj in edges])
            p, rcl, _ = precision_recall_curve(y0, y1)
            aupr[a].append(auc(rcl, p))

    ax = plt.subplot(grid[positions[i]])
    configure(ax)
    kw = dict(fontweight='bold', **opt_panel) if i == 0 else opt_panel
    ax.annotate(panel_letters[i], xytext=(x,y), **kw)
    b = np.mean(aupr['Random'])
    ax.plot([0,len(algoD)+1],[b,b],'--',color='lightgray',lw=0.8)
    for j, a in enumerate(algoD):
        box = ax.boxplot([aupr[a]], positions=[j+1], **opt_box)
        configure_box(box, c[a])
    ax.set_xlim(0.8, len(algoD)+0.5)
    ax.set_xticklabels(algoD, fontsize=4.5)
    ax.set_ylabel('AUPR', fontsize=6)
    optn = dict(fontsize=9, transform=ax.transAxes, ha='right')
    ax.text(xn, yn, bench, **optn)
    ax.text(xn, yn+0.01, bench, color='none',
            bbox=dict(boxstyle='round,pad=0.2',fc='none',ec='lightgray',lw=0.8), **optn)

# Panel E: Trees
auprTrees = {a: [] for a in algoD+['Random']}
for bench in trees:
    tmp = {a: [] for a in algoD+['Random']}
    for r in range(1, N+1):
        inter = abs(np.load(f'{path}{bench}/True/inter_{r}.npy'))
        G = inter.shape[0]
        edges = [(ii,jj) for ii in range(G) for jj in set(range(1,G))-{ii}]
        y0 = np.array([inter[ii,jj] for ii,jj in edges])
        tmp['Random'].append(np.mean(y0))
        for a in algoD:
            score = abs(np.load(f'{path}{bench}/{file_name(a)}/score_{r}.npy'))
            y1 = np.array([score[ii,jj] for ii,jj in edges])
            p, rcl, _ = precision_recall_curve(y0, y1)
            tmp[a].append(auc(rcl, p))
    for a in tmp: auprTrees[a].append(np.mean(tmp[a]))

ax = plt.subplot(grid[2,0])
configure(ax)
ax.annotate('E', xytext=(x,y), fontweight='bold', **opt_panel)
for a in algoD+['Random']:
    ax.plot(size, auprTrees[a], '--', marker='.', ms=4, lw=0.85, color=c[a][0], label=a)
ax.legend(frameon=False, fontsize=4.5)
ax.set_xticks(size)
ax.set_xlabel('No. of genes', fontsize=6)
ax.set_ylabel('AUPR', fontsize=6)
optn = dict(fontsize=9, transform=ax.transAxes, ha='right')
ax.text(xn, yn, 'Trees', **optn)
ax.text(xn, yn+0.01, 'Trees', color='none',
        bbox=dict(boxstyle='round,pad=0.2',fc='none',ec='lightgray',lw=0.8), **optn)

# Panel F: Global boxplot
auprG = {a: [] for a in algoG}
auprR = []
for r in range(1, N+1):
    rand = []
    for b in benchmarks:
        inter = abs(np.load(f'{path}{b}/True/inter_{r}.npy'))
        G = inter.shape[0]
        edges = [(ii,jj) for ii in range(G) for jj in set(range(1,G))-{ii}]
        rand.append(np.mean([inter[ii,jj] for ii,jj in edges]))
    auprR.append(np.mean(rand))
    for a in algoG:
        vals = []
        for b in benchmarks:
            inter = abs(np.load(f'{path}{b}/True/inter_{r}.npy'))
            score = abs(np.load(f'{path}{b}/{a}/score_{r}.npy'))
            G = inter.shape[0]
            edges = [(ii,jj) for ii in range(G) for jj in set(range(1,G))-{ii}]
            y0 = np.array([inter[ii,jj] for ii,jj in edges])
            y1 = np.array([score[ii,jj] for ii,jj in edges])
            p, rcl, _ = precision_recall_curve(y0, y1)
            vals.append(auc(rcl, p))
        auprG[a].append(np.mean(vals))

ax = plt.subplot(grid[2,1])
configure(ax)
ax.annotate('F', xytext=(x,y), fontweight='bold', **opt_panel)
for i, a in enumerate(algoG):
    box = ax.boxplot([auprG[a]], positions=[i+1], widths=0.5, patch_artist=True)
    configure_box(box, c[a])
ax.plot([0,len(algoG)+1],[np.mean(auprR)]*2,'--',color='lightgray',lw=1)
ax.set_xticks(range(1, len(labelG)+1))
ax.set_xticklabels(labelG, rotation=45, ha='right', fontsize=5)
ax.set_ylabel('AUPR', fontsize=6)

fig.savefig('figure4.pdf', dpi=300, bbox_inches='tight', pad_inches=0.04)
plt.show()