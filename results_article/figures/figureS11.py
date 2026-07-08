# Benchmark (directed|undirected) for all test networks
import os
import runpy
import sys
import types
from contextlib import contextmanager
from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.metrics import precision_recall_curve, auc

algoD = ['CardamomOT', 'Reference Fitting', 'GENIE3', 'SINCERITIES']

# Map display names to folder names on disk
algo_file = {'CardamomOT': 'CARDAMOM2', 'CARDAMOM': 'CARDAMOM1', 'Reference Fitting': 'REFERENCE_FITTING'}
def file_name(algo): return algo_file.get(algo, algo)

benchmarks = ['FN4', 'CN5', 'BN8', 'FN8',
    'Trees5', 'Trees10', 'Trees20', 'Trees50', 'Trees100']
N = 10

BASE_DIR = Path(__file__).resolve().parent
FIGURES11_DIR = BASE_DIR / 'figureS11'
SOURCE_BENCHMARKS = [BASE_DIR / benchmark for benchmark in benchmarks]
REPO_ROOT = BASE_DIR.parents[1]
MODE_ROOTS = {
    'dropout': FIGURES11_DIR / 'dropout',
    'library_size': FIGURES11_DIR / 'library_size',
}
MODE_PANELS = [
    ('dropout', 'Dropout', 0),
    ('library_size', 'Library size', 2),
]
INFERENCE_SCRIPTS = [
    'infer_CARDAMOM2.py',
    'infer_reference_fitting.py',
    'infer_GENIE3.py',
    'infer_SINCERITIES.py',
]


def copy_benchmark_data(src_benchmark, dst_benchmark):
    if dst_benchmark.exists():
        shutil.rmtree(dst_benchmark)
    dst_benchmark.mkdir(parents=True, exist_ok=True)
    for subdir in ['Data', 'True']:
        shutil.copytree(src_benchmark / subdir, dst_benchmark / subdir)
    # Create algorithm output directories so np.save doesn't fail.
    for algo in algoD:
        (dst_benchmark / file_name(algo)).mkdir(parents=True, exist_ok=True)


def ensure_helper_scripts(mode_root):
    helper_src = BASE_DIR / '_scripts'
    helper_dst = mode_root / '_scripts'
    if helper_dst.exists() or helper_dst.is_symlink():
        return
    try:
        helper_dst.symlink_to(helper_src, target_is_directory=True)
    except OSError:
        shutil.copytree(helper_src, helper_dst)


def apply_random_dropout(data_file, rng):
    data = np.loadtxt(data_file, dtype=int, delimiter='\t')
    # Thin each count independently with 50% retention.
    data[2:, :] = rng.binomial(data[2:, :], 0.5)
    np.savetxt(data_file, data, fmt='%d', delimiter='\t')


def apply_library_size_dropout(data_file, rng):
    data = np.loadtxt(data_file, dtype=int, delimiter='\t')
    counts = data[2:, :]
    target = int(np.median(counts.sum(axis=0)))
    # Downsample each cell to the median library size when it exceeds that depth.
    for cell_idx in range(counts.shape[1]):
        total = int(counts[:, cell_idx].sum())
        if total <= target or total == 0:
            continue
        probs = counts[:, cell_idx] / total
        counts[:, cell_idx] = rng.multinomial(target, probs)
    np.savetxt(data_file, data, fmt='%d', delimiter='\t')


@contextmanager
def temporary_cwd(path):
    previous_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


@contextmanager
def temporary_sys_path(paths):
    previous_sys_path = sys.path[:]
    sys.path[:0] = [str(path) for path in paths]
    try:
        yield
    finally:
        sys.path[:] = previous_sys_path


def run_inference_suite(mode_root):
    ensure_helper_scripts(mode_root)

    import CardamomOT

    cardamom_alias = types.ModuleType('cardamom')
    cardamom_alias.NetworkModel = CardamomOT.NetworkModel

    previous_cardamom = sys.modules.get('cardamom')
    sys.modules['cardamom'] = cardamom_alias
    try:
        with temporary_cwd(mode_root), temporary_sys_path([REPO_ROOT, BASE_DIR]):
            for script_name in INFERENCE_SCRIPTS:
                runpy.run_path(str(BASE_DIR / script_name), run_name='__main__')
    finally:
        if previous_cardamom is None:
            sys.modules.pop('cardamom', None)
        else:
            sys.modules['cardamom'] = previous_cardamom


def prepare_subsampled_benchmark_copies():
    dropout_dir = FIGURES11_DIR / 'dropout'
    library_size_dir = FIGURES11_DIR / 'library_size'
    dropout_rng = np.random.default_rng(0)
    library_rng = np.random.default_rng(1)

    for src_benchmark in SOURCE_BENCHMARKS:
        for mode_dir, transformer, rng in [
            (dropout_dir, apply_random_dropout, dropout_rng),
            (library_size_dir, apply_library_size_dropout, library_rng),
        ]:
            dst_benchmark = mode_dir / src_benchmark.name
            copy_benchmark_data(src_benchmark, dst_benchmark)
            for data_file in sorted((dst_benchmark / 'Data').glob('data_*.txt')):
                transformer(data_file, rng)


prepare_subsampled_benchmark_copies()

for mode_name, mode_root in MODE_ROOTS.items():
    run_inference_suite(mode_root)

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
grid = gs.GridSpec(5, 3, hspace=0.55, wspace=0.13,
    height_ratios=[1,1,1,1,1], width_ratios=[6,1,7])

def configure(ax):
    w = 0.7
    ax.tick_params(direction='out', length=3, width=w)
    ax.tick_params(axis='x', pad=2, labelsize=5.5)
    ax.tick_params(axis='y', pad=0.5, labelsize=5.5)
    for side in ['top','bottom','left','right']: ax.spines[side].set_linewidth(w)
    ax.set_ylim(0,1.1)

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

def compute_aupr_scores(results_root):
    auprTreesD = {algo: [] for algo in algoD+['Random']}
    aupr_by_benchmark = {}

    for benchmark in benchmarks:
        auprD = {algo: [] for algo in algoD+['Random']}

        for r in range(1, N+1):
            inter = abs(np.load(results_root / benchmark / 'True' / f'inter_{r}.npy'))
            G = inter.shape[0]

            edges = [(i,j) for i in range(G) for j in set(range(1,G))-{i}]
            y0 = np.array([inter[i,j] for (i,j) in edges])
            auprD['Random'].append(np.mean(y0))
            for algo in algoD:
                score = abs(np.load(results_root / benchmark / file_name(algo) / f'score_{r}.npy'))
                if algo == 'GENIE3':
                    score = score.T
                y1 = np.array([score[i,j] for (i,j) in edges])
                precision, recall, _ = precision_recall_curve(y0, y1)
                auprD[algo].append(auc(recall, precision))

        if benchmark[:5] == 'Trees':
            for algo in algoD+['Random']:
                auprTreesD[algo].append(np.mean(auprD[algo]))
        aupr_by_benchmark[benchmark] = auprD

    return aupr_by_benchmark, auprTreesD


mode_results = {}
mode_tree_scores = {}
for mode_name, mode_root in MODE_ROOTS.items():
    mode_results[mode_name], mode_tree_scores[mode_name] = compute_aupr_scores(mode_root)

name = ['FN4', 'CN5', 'BN8', 'FN8']
for n, benchmark in enumerate(benchmarks):
    if n < 4:
        for mode_name, mode_title, col in MODE_PANELS:
            auprD = mode_results[mode_name][benchmark]
            ax = plt.subplot(grid[n,col])
            configure(ax)
            if n == 0:
                panel_letter = 'A' if mode_name == 'dropout' else 'B'
                ax.annotate(panel_letter, xytext=(x + (0.2 if mode_name == 'dropout' else 0), y),
                            fontweight='bold', **opt)
                ax.annotate(mode_title, xytext=(x+14,y), **opt)
            b = np.mean(auprD['Random'])
            ax.plot([0,4],[b,b], color='lightgray', ls='--', lw=0.8, zorder=0)
            ax.set_xlim(0.8, 4.5)
            for i, algo in enumerate(algoD):
                box = ax.boxplot([auprD[algo]], positions=[i+1], **opt_box)
                configure_box(box, c[algo])
            ax.set_xticks(range(1, len(algoD)+1))
            ax.set_xticklabels(algoD, fontsize=4.5, rotation=30, ha='right')
            if col == 0:
                ax.set_ylabel('AUPR', fontsize=6)
            optn = {'fontsize': 9, 'transform': ax.transAxes, 'ha': 'right'}
            ax.text(xn, yn, name[n], **optn)
            ax.text(xn, yn+0.01, name[n], color='none', zorder=0,
                    bbox=dict(boxstyle='round,pad=0.2',fc='none',ec='lightgray',lw=0.8), **optn)

# Trees panels
s = {'ls': '--', 'lw': 0.85, 'marker': '.', 'ms': 4}
p = {'borderaxespad': 0, 'frameon': False, 'fontsize': 4.5,
     'handlelength': 1.2, 'handletextpad': 0.5}
size = [5, 10, 20, 50, 100]

for mode_name, mode_title, col in MODE_PANELS:
    ax = plt.subplot(grid[4,col])
    configure(ax)
    for algo in algoD+['Random']:
        ax.plot(size, mode_tree_scores[mode_name][algo], color=c[algo][0], label=algo, **s)
    ax.legend(loc='upper right', **p)
    ax.set_xticks(size)
    ax.set_xlabel('No. of genes', fontsize=6, labelpad=1.8)
    if col == 0:
        ax.set_ylabel('AUPR', fontsize=6)
    optn = {'fontsize': 9, 'transform': ax.transAxes, 'ha': 'right'}
    ax.text(xn, yn, 'Trees', **optn)
    ax.text(xn, yn+0.01, 'Trees', color='none', zorder=0,
            bbox=dict(boxstyle='round,pad=0.2',fc='none',ec='lightgray',lw=0.8), **optn)

fig.savefig(BASE_DIR / 'figure_S11.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
#plt.show()
