import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.metrics import precision_recall_curve, auc


path = './'

algo = 'CardamomOT'
algo_file = {'CardamomOT': 'CARDAMOM2', 'CARDAMOM': 'CARDAMOM1', 'Reference Fitting': 'REFERENCE_FITTING'}
def file_name(a): return algo_file.get(a, a)

benchmarks_traj = ['FN4', 'CN5', 'BN8', 'FN8']
N = 10

# ── Same palette as figure 3 ──────────────────────────────────────────────────
cmap = plt.get_cmap('tab20')
color_bench = {
    'FN4': (cmap(0),  cmap(1)),
    'CN5': (cmap(2),  cmap(3)),
    'BN8': (cmap(4),  cmap(5)),
    'FN8': (cmap(6),  cmap(7)),
}

# ── Padding helpers ───────────────────────────────────────────────────────────

def compute_aupr_traj(score_traj, inter):
    """
    score_traj : (T, G, G, 1)  — matrices de scores à chaque step
    inter      : (G, G)         — matrice de référence (valeurs absolues)
    Retourne un array (T,) d'AUPR (undirected, comme panel B de fig3).
    """
    score_traj = score_traj[..., 0]          # (T, G, G)
    G = inter.shape[0]
    edges = [(i, j) for i in range(G) for j in range(i + 1, G)]
    y0 = np.array([max(inter[i, j], inter[j, i]) for (i, j) in edges])
    aupr_traj = []
    for t in range(score_traj.shape[0]):
        s = np.abs(score_traj[t])
        y1 = np.array([max(s[i, j], s[j, i]) for (i, j) in edges])
        precision, recall, _ = precision_recall_curve(y0, y1)
        aupr_traj.append(auc(recall, precision))
    return np.array(aupr_traj)

def pad_1d(arr, target_len):
    """Pad 1D array by repeating last value."""
    if len(arr) == target_len:
        return arr
    return np.pad(arr, (0, target_len - len(arr)), mode='edge')

# ── Load trajectories ─────────────────────────────────────────────────────────
traj_data = {}
for b in benchmarks_traj:
    loss_list, aupr_list = [], []
    for r in range(1, N + 1):
        inter      = np.abs(np.load(f"{path}{b}/True/inter_{r}.npy"))
        loss       = np.load(f"{path}{b}/{file_name(algo)}/loss_traj_{r}.npy")
        score_traj = np.load(f"{path}{b}/{file_name(algo)}/score_traj_{r}.npy")
        loss_list.append(loss)
        aupr_list.append(compute_aupr_traj(score_traj, inter))

    max_len      = max(len(x) for x in loss_list)
    padded_loss  = np.array([pad_1d(x, max_len) for x in loss_list])
    padded_aupr  = np.array([pad_1d(x, max_len) for x in aupr_list])

    t = np.arange(max_len)
    traj_data[b] = dict(
        t          = t,
        mean_loss  = padded_loss.mean(axis=0),
        std_loss   = padded_loss.std(axis=0)  / np.sqrt(N),
        mean_score = padded_aupr.mean(axis=0),
        std_score  = padded_aupr.std(axis=0)  / np.sqrt(N),
    )

# ── Figure layout — same width as fig3, 2-row height ─────────────────────────
fig = plt.figure(figsize=(6.85, 3.2))
grid = gs.GridSpec(1, 2, hspace=0.55, wspace=0.28,
                   width_ratios=[1, 1])

# ── Shared axis styler (mirrors configure() in fig3) ─────────────────────────
def configure(ax):
    w = 0.7
    ax.tick_params(direction='out', length=3, width=w)
    ax.tick_params(axis='x', pad=2, labelsize=5.5)
    ax.tick_params(axis='y', pad=0.5, labelsize=5.5)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['bottom', 'left']:
        ax.spines[side].set_linewidth(w)

# ── Panel-label annotation options (same as fig3) ────────────────────────────
x, y = -11, 10
opt = {'xy': (0, 1), 'xycoords': 'axes fraction', 'fontsize': 10,
       'textcoords': 'offset points', 'annotation_clip': False}

# ── Panel A : Training loss ───────────────────────────────────────────────────
ax_loss = plt.subplot(grid[0, 0])
configure(ax_loss)
ax_loss.annotate('A', xytext=(x + 0.2, y), fontweight='bold', **opt)
ax_loss.annotate('Training loss', xytext=(x + 14, y), **opt)

for b in benchmarks_traj:
    d   = traj_data[b]
    col = color_bench[b]
    ax_loss.plot(d['t'], d['mean_loss'], color=col[0], lw=0.9, label=b)
    ax_loss.fill_between(d['t'],
                         d['mean_loss'] - d['std_loss'],
                         d['mean_loss'] + d['std_loss'],
                         color=col[1], alpha=0.45)

ax_loss.set_xlabel('Optimization step', fontsize=6, labelpad=1.8)
ax_loss.set_ylabel('Loss',              fontsize=6)
ax_loss.legend(loc='upper right', borderaxespad=0, frameon=False,
               fontsize=4.5, handlelength=1.2, handletextpad=0.5)

# ── Panel B : AUPR over training ──────────────────────────────────────────────
ax_score = plt.subplot(grid[0, 1])
configure(ax_score)
ax_score.annotate('B', xytext=(x, y), fontweight='bold', **opt)
ax_score.annotate('AUPR over training', xytext=(x + 14, y), **opt)
ax_score.set_ylim(0, 1.1)

for b in benchmarks_traj:
    d   = traj_data[b]
    col = color_bench[b]
    ax_score.plot(d['t'], d['mean_score'], color=col[0], lw=0.9, label=b)
    ax_score.fill_between(d['t'],
                          d['mean_score'] - d['std_score'],
                          d['mean_score'] + d['std_score'],
                          color=col[1], alpha=0.45)

ax_score.set_xlabel('Optimization step', fontsize=6, labelpad=1.8)
ax_score.set_ylabel('AUPR',              fontsize=6)
ax_score.legend(loc='lower right', borderaxespad=0, frameon=False,
                fontsize=4.5, handlelength=1.2, handletextpad=0.5)

fig.savefig('figureS7.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()