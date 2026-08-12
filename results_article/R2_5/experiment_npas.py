# ============================================================
# Experiment: Sensitivity of inference to n_pas (alpha timesteps)
# ============================================================
"""
Varies NetworkModel.n_pas across {2, 4, 6, 10, 25} on the 4 directed
benchmarks (FN4, FN8, CN5, BN8) and plots AUPR boxplots.

n_pas controls the number of discretization steps between consecutive
timepoints used by inference_alpha in loop_trajectories. The default
value in CardamomOT is 25.
"""
import os, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.metrics import precision_recall_curve, auc

# ---- Paths and constants ----
BASE_DIR = Path(__file__).resolve().parent  # results_article/R2_5/
FIGURES_DIR = BASE_DIR.parent / 'figures'   # results_article/figures/
REPO_ROOT = BASE_DIR.parents[1]             # repository root

NPAS_VALUES = [2, 4, 6, 10, 25]
NPAS_LABELS = ['n_pas=2', 'n_pas=4', 'n_pas=6', 'n_pas=10', 'n_pas=25']
BENCHMARKS = ['FN4', 'FN8', 'CN5', 'BN8']
N = 10          # number of replicates
N_REPET = 2     # number of initial couplings (averaged)

# ============================================================
# Inference
# ============================================================

def run_inference():
    """Run CARDAMOM2 inference for each (benchmark, n_pas, replicate).
    Resumes per score file if it already exists."""
    sys.path.insert(0, str(REPO_ROOT))
    from CardamomOT import NetworkModel

    verb = 1
    any_work_done = False

    for bench in BENCHMARKS:
        data_dir = FIGURES_DIR / bench / 'Data'
        true_dir = FIGURES_DIR / bench / 'True'
        rates_file = data_dir / 'Rates' / 'degradation_rates.txt'

        for npas in NPAS_VALUES:
            out_subdir = BASE_DIR / bench / f'npas_{npas}' / 'CARDAMOM2'
            out_subdir.mkdir(parents=True, exist_ok=True)

            for r in range(1, N + 1):
                score_path = out_subdir / f'score_{r}.npy'
                if score_path.exists():
                    continue
                any_work_done = True
                print(f'  n_pas experiment: {bench} n_pas={npas} run {r}/{N}')

                fname = data_dir / f'data_{r}.txt'
                data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:, 1:]
                time = np.loadtxt(fname, dtype=int, delimiter='\t')[0, 1:]
                x = data.T
                x[:, 0] = time
                G = x.shape[1]
                d = np.loadtxt(rates_file, dtype=float, delimiter='\t').T

                model = NetworkModel(G - 1)
                model.d = d
                model.n_pas = npas
                model.fit(x, verb=verb)
                score = model.inter

                for _ in range(N_REPET):
                    model = NetworkModel(G - 1)
                    model.d = d
                    model.n_pas = npas
                    model.fit(x, verb=verb)
                    score += model.inter

                np.save(score_path, score)

    if not any_work_done:
        print('n_pas inference already complete, skipping.')


# ============================================================
# Compute AUPR scores
# ============================================================

def compute_aupr():
    """Compute AUPR for each (benchmark, n_pas, replicate)."""
    aupr_per_bench = {b: {npas: [] for npas in NPAS_VALUES} for b in BENCHMARKS}
    aupr_global   = {npas: [] for npas in NPAS_VALUES}   # averaged across benchmarks per rep
    aupr_random   = []                                     # random baseline

    for bench in BENCHMARKS:
        true_dir = FIGURES_DIR / bench / 'True'
        for r in range(1, N + 1):
            inter = abs(np.load(true_dir / f'inter_{r}.npy'))
            G = inter.shape[0]
            edges = [(i, j) for i in range(G) for j in set(range(1, G)) - {i}]
            y0 = np.array([inter[i, j] for i, j in edges])

            # Random baseline (same for all n_pas)
            aupr_random.append(np.mean(y0))

            for npas in NPAS_VALUES:
                score_path = BASE_DIR / bench / f'npas_{npas}' / 'CARDAMOM2' / f'score_{r}.npy'
                score = abs(np.load(score_path))
                y1 = np.array([score[i, j] for i, j in edges])
                p, rcl, _ = precision_recall_curve(y0, y1)
                aupr_per_bench[bench][npas].append(auc(rcl, p))

    # Average across benchmarks per replicate
    for npas in NPAS_VALUES:
        for r in range(N):
            vals = [aupr_per_bench[b][npas][r] for b in BENCHMARKS]
            aupr_global[npas].append(np.mean(vals))

    return aupr_per_bench, aupr_global, aupr_random


# ============================================================
# Plotting
# ============================================================

def plot_results(aupr_per_bench, aupr_global, aupr_random):
    """Create a figure with per-benchmark + global AUPR boxplots vs n_pas."""

    cmap = plt.get_cmap('tab20')
    col_bench = {'FN4': (cmap(6), cmap(7)),
                 'FN8': (cmap(8), cmap(9)),
                 'CN5': (cmap(0), cmap(1)),
                 'BN8': (cmap(2), cmap(3))}

    fig = plt.figure(figsize=(6.85, 3.5))
    grid = gs.GridSpec(1, 2, hspace=0.55, wspace=0.50)

    def configure(ax):
        w = 0.7
        ax.tick_params(direction='out', length=3, width=w)
        ax.tick_params(axis='x', pad=2, labelsize=5.5)
        ax.tick_params(axis='y', pad=0.5, labelsize=5.5)
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_linewidth(w)
        ax.set_ylim(0, 1.1)

    opt_box = {'patch_artist': True, 'widths': [0.25]}

    def configure_box(box, col):
        w = 0.8
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box[item], color=col[0], lw=w)
        plt.setp(box['boxes'], facecolor=col[1])
        plt.setp(box['fliers'], markeredgecolor=col[0],
                 markerfacecolor=col[1], ms=3, markeredgewidth=w)

    n_npas = len(NPAS_VALUES)
    x_offset = -11
    y_offset = 10
    xn, yn = -0.14, 0.87
    opt_panel = dict(xy=(0, 1), xycoords='axes fraction',
                     textcoords='offset points', fontsize=10, annotation_clip=False)

    # ---- Panel A: Per-benchmark line plot of mean AUPR vs n_pas ----
    ax = plt.subplot(grid[0, 0])
    configure(ax)
    ax.annotate('A', xytext=(x_offset, y_offset), fontweight='bold', **opt_panel)

    # Random baseline
    b_rand = np.mean(aupr_random)
    ax.plot([0, n_npas + 1], [b_rand, b_rand], '--', color='lightgray', lw=0.8)

    for bench in BENCHMARKS:
        means = np.array([np.mean(aupr_per_bench[bench][n]) for n in NPAS_VALUES])
        stds  = np.array([np.std(aupr_per_bench[bench][n]) for n in NPAS_VALUES])
        ax.fill_between(range(1, n_npas + 1),
                        means - stds, means + stds,
                        color=col_bench[bench][0], alpha=0.12, lw=0)
        ax.plot(range(1, n_npas + 1), means, '.-', ms=5, lw=0.85,
                color=col_bench[bench][0], label=bench)

    ax.legend(frameon=False, fontsize=5)
    ax.set_xlim(0.5, n_npas + 0.5)
    ax.set_xticks(range(1, n_npas + 1))
    ax.set_xticklabels(NPAS_LABELS, rotation=30, ha='right', fontsize=5)
    ax.set_ylabel('AUPR', fontsize=6)
    optn = dict(fontsize=8, transform=ax.transAxes, ha='right')
    ax.text(xn, yn, 'Per-benchmark', **optn)

    # ---- Panel B: Global boxplot (averaged across benchmarks) ----
    ax = plt.subplot(grid[0, 1])
    configure(ax)
    ax.annotate('B', xytext=(x_offset, y_offset), fontweight='bold', **opt_panel)

    ax.plot([0, n_npas + 1], [b_rand, b_rand], '--', color='lightgray', lw=1)

    for i, npas in enumerate(NPAS_VALUES):
        box = ax.boxplot([aupr_global[npas]], positions=[i + 1],
                         patch_artist=True, widths=0.5)
        configure_box(box, ('black', 'grey'))

    ax.set_xlim(0.5, n_npas + 0.5)
    ax.set_xticks(range(1, n_npas + 1))
    ax.set_xticklabels(NPAS_LABELS, rotation=30, ha='right', fontsize=5)
    ax.set_ylabel('AUPR', fontsize=6)
    optn = dict(fontsize=8, transform=ax.transAxes, ha='right')
    ax.text(xn, yn, 'Global (avg.)', **optn)
    ax.text(xn, yn + 0.01, 'Global (avg.)', color='none',
            bbox=dict(boxstyle='round,pad=0.2', fc='none', ec='lightgray', lw=0.8), **optn)

    fig.savefig(str(BASE_DIR / 'figure_npas.pdf'), dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    run_inference()
    aupr_per_bench, aupr_global, aupr_random = compute_aupr()
    plot_results(aupr_per_bench, aupr_global, aupr_random)
