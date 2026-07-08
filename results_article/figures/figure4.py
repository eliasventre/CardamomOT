# ============================================================
# Figure: Directed benchmarks + Trees + Global boxplot
# ============================================================
import os, sys
from contextlib import contextmanager
from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.metrics import precision_recall_curve, auc

# ---- Paths and constants ----
BASE_DIR = Path(__file__).resolve().parent
FIGURE4_DROPOUT_DIR = BASE_DIR / 'figure4_dropout'
DROPOUT_LEVELS = [0, 5, 10, 20, 35, 50, 70, 90]  # 0 = real (no dropout), rest = % dropout
DROPOUT_LABELS = ['real', '5%', '10%', '20%', '35%', '50%', '70%', '90%']
REPO_ROOT = BASE_DIR.parents[1]

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

# ---- Timescale robustness constants (panel H) ----
FIGURE4_TIMESCALE_DIR = BASE_DIR / 'figure4_timescale'
TIMESCALE_SIGMAS = [0, 0.2, 0.5, 1, 1.5, 2, 3, 5]  # 0 = real (no jitter)
TIMESCALE_LABELS = ['real', 'σ=0.2', 'σ=0.5', 'σ=1', 'σ=1.5', 'σ=2', 'σ=3', 'σ=5']

# ============================================================
# Dropout robustness data preparation & inference (for panel G)
# ============================================================

@contextmanager
def temporary_cwd(path):
    previous_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def copy_benchmark_data(src_benchmark, dst_benchmark):
    """Copy Data/ and True/ folders to the dropout directory.
    Preserves any existing CARDAMOM2/ scores to avoid recomputation."""
    # Remove Data/ and True/ subdirectories if they exist, but keep CARDAMOM2/
    for subdir in ['Data', 'True']:
        subpath = dst_benchmark / subdir
        if subpath.exists():
            shutil.rmtree(subpath)
    dst_benchmark.mkdir(parents=True, exist_ok=True)
    for subdir in ['Data', 'True']:
        shutil.copytree(src_benchmark / subdir, dst_benchmark / subdir)
    # Create CARDAMOM2 output directory for scores (if not already present)
    (dst_benchmark / 'CARDAMOM2').mkdir(parents=True, exist_ok=True)


def apply_dropout(data_file, rng, dropout_pct):
    """
    Thin counts using binomial dropout at the given percentage.
    Each count is kept with probability (1 - dropout_pct/100).
    """
    data = np.loadtxt(data_file, dtype=int, delimiter='\t')
    counts = data[2:, :]
    retention = 1.0 - dropout_pct / 100.0
    for gene_idx in range(counts.shape[0]):
        counts[gene_idx, :] = rng.binomial(counts[gene_idx, :].astype(int), retention)
    np.savetxt(data_file, data, fmt='%d', delimiter='\t')


def prepare_dropout_data():
    """Create figure4_dropout/ with dropout copies of the 4 directed benchmarks
    at 7 dropout levels (5%, 10%, 20%, 35%, 50%, 70%, 90%).
    Level 0 = real (original data, no dropout).
    Skips entirely if all output scores already exist."""
    do_dir = FIGURE4_DROPOUT_DIR
    # Check if all scores already exist — if so, skip everything
    all_done = True
    for bench in benchmarks:
        for li, level in enumerate(DROPOUT_LEVELS):
            for r in range(1, N + 1):
                if not (do_dir / bench / 'CARDAMOM2' / f'score_l{li}_r{r}.npy').exists():
                    all_done = False
                    break
            if not all_done:
                break
        if not all_done:
            break
    if all_done:
        print('Dropout data already prepared, skipping.')
        return

    rng = np.random.default_rng(42)
    for bench in benchmarks:
        src = BASE_DIR / bench
        for li, level in enumerate(DROPOUT_LEVELS):
            dst = do_dir / bench / f'level_{li}'
            copy_benchmark_data(src, dst)
            if level > 0:
                for data_file in sorted((dst / 'Data').glob('data_*.txt')):
                    apply_dropout(data_file, rng, level)


def run_dropout_inference():
    """Run CARDAMOM2 inference on dropout benchmarks (skips if already done)."""
    do_dir = FIGURE4_DROPOUT_DIR
    # Check whether scores already exist
    all_done = True
    for bench in benchmarks:
        for li in range(len(DROPOUT_LEVELS)):
            for r in range(1, N + 1):
                if not (do_dir / bench / 'CARDAMOM2' / f'score_l{li}_r{r}.npy').exists():
                    all_done = False
                    break
            if not all_done:
                break
        if not all_done:
            break
    if all_done:
        print('Dropout inference already complete, skipping.')
        return

    sys.path.insert(0, str(REPO_ROOT))
    from CardamomOT import NetworkModel

    n_repet = 4   # number of initial couplings (matching infer_CARDAMOM2.py)
    verb = 1

    with temporary_cwd(do_dir):
        for bench in benchmarks:
            for li in range(len(DROPOUT_LEVELS)):
                for r in range(1, N + 1):
                    print(f'  Dropout inference: {bench} level {DROPOUT_LABELS[li]} run {r}/{N}')
                    fname = f'{bench}/level_{li}/Data/data_{r}.txt'
                    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:, 1:]
                    time = np.loadtxt(fname, dtype=int, delimiter='\t')[0, 1:]
                    x = data.T
                    x[:, 0] = time
                    G = x.shape[1]
                    model = NetworkModel(G - 1)
                    model.d = np.loadtxt(
                        f'{bench}/level_{li}/Data/Rates/degradation_rates.txt',
                        dtype=float, delimiter='\t').T
                    model.fit(x, verb=verb)
                    score = model.inter
                    for _ in range(n_repet):
                        model = NetworkModel(G - 1)
                        model.d = np.loadtxt(
                            f'{bench}/level_{li}/Data/Rates/degradation_rates.txt',
                            dtype=float, delimiter='\t').T
                        model.fit(x, verb=verb)
                        score += model.inter
                    os.makedirs(f'{bench}/CARDAMOM2', exist_ok=True)
                    np.save(f'{bench}/CARDAMOM2/score_l{li}_r{r}', score)


# ============================================================
# Timescale robustness data preparation & inference (panel H)
# ============================================================

def prepare_timescale_data():
    """
    Simulate each of the 4 benchmarks with harissa at varying levels
    of timepoint jitter (sigma). 8 sigma levels × N replicates each.
    Saves data in figure4_timescale/BENCHMARK/.
    Skips entirely if all output scores already exist.
    """
    from harissa import NetworkModel as HarissaNetworkModel

    ts_dir = FIGURE4_TIMESCALE_DIR
    n_sigmas = len(TIMESCALE_SIGMAS)
    n_reps = N
    C = 1000  # total cells per simulation

    # Check if all scores already exist — if so, skip everything
    all_done = True
    for bench in benchmarks:
        for si in range(n_sigmas):
            for r in range(1, n_reps + 1):
                if not (ts_dir / bench / 'CARDAMOM2' / f'score_s{si}_r{r}.npy').exists():
                    all_done = False
                    break
            if not all_done:
                break
        if not all_done:
            break
    if all_done:
        print('Timescale data already prepared, skipping.')
        return

    # ---- Harissa model builders for each benchmark ----
    def _make_BN8():
        mh = HarissaNetworkModel(8)
        mh.d[0] = 0.25; mh.d[1] = 0.05
        mh.basal[1:] = [-4] * 8
        for (i, j, v) in [(0, 1, 10), (1, 2, 10), (1, 3, 10), (3, 2, -10),
                          (2, 3, -10), (2, 2, 5), (3, 3, 5),
                          (2, 4, 10), (3, 5, 10), (2, 5, -10), (3, 4, -10),
                          (4, 7, -10), (5, 6, -10),
                          (4, 6, 10), (5, 7, 10), (7, 8, 10), (6, 8, -10)]:
            mh.inter[i, j] = v
        return mh

    def _make_FN8():
        mh = HarissaNetworkModel(8)
        mh.d[0] = 0.4; mh.d[1] = 0.08
        mh.basal[1:] = [-5] * 8
        for (i, j, v) in [(0, 1, 10), (1, 2, 10), (2, 3, 10), (3, 4, 10),
                          (3, 5, 10), (3, 6, 10),
                          (4, 1, -10), (5, 1, -10), (6, 1, -10),
                          (4, 4, 10), (5, 5, 10), (6, 6, 10),
                          (4, 8, -10), (4, 7, -10), (6, 7, 10),
                          (7, 6, 10), (8, 8, 10)]:
            mh.inter[i, j] = v
        return mh

    def _make_CN5():
        mh = HarissaNetworkModel(5)
        mh.d[0] = 0.5; mh.d[1] = 0.1
        mh.basal[1:] = [-5, 4, 4, -5, -5]
        for (i, j, v) in [(0, 1, 10), (1, 2, -10), (2, 3, -10),
                          (3, 4, 10), (4, 5, 10), (5, 1, -10)]:
            mh.inter[i, j] = v
        return mh

    def _make_FN4():
        mh = HarissaNetworkModel(4)
        mh.d[0] = 1 / 5; mh.d[1] = 0.2 / 5
        mh.basal[1:] = [-5] * 4
        for (i, j, v) in [(0, 1, 10), (1, 2, 10), (1, 3, 10),
                          (3, 4, 10), (4, 1, -10), (2, 2, 10), (3, 3, 10)]:
            mh.inter[i, j] = v
        return mh

    make_bench = {'BN8': _make_BN8, 'FN8': _make_FN8,
                  'CN5': _make_CN5, 'FN4': _make_FN4}

    # Master RNG for reproducibility
    master_rng = np.random.default_rng(42)
    all_seeds = master_rng.integers(0, 2**31, size=(len(benchmarks), n_sigmas, n_reps))

    for bi, bench in enumerate(benchmarks):
        dst = ts_dir / bench
        # Remove Data/ and True/ subdirectories; preserve CARDAMOM2/ scores
        for sub in ['Data', 'True']:
            subpath = dst / sub
            if subpath.exists():
                shutil.rmtree(subpath)
        for sub in ['Data', 'True', 'CARDAMOM2', 'Data/Rates']:
            (dst / sub).mkdir(parents=True, exist_ok=True)

        # Get original timepoints from the benchmark data
        src = BASE_DIR / bench
        data_ref = np.loadtxt(src / 'Data' / 'data_1.txt', dtype=int, delimiter='\t')
        t_orig = np.unique(data_ref[0, 1:]).astype(float)
        n_tp = len(t_orig)
        t_min, t_max = t_orig[0], t_orig[-1]
        cells_per_tp = C // n_tp

        # Build the harissa model once per benchmark
        mh = make_bench[bench]()
        G_h = mh.inter.shape[0] - 1  # number of non-stimulus genes

        # Save degradation rates (harissa format: 2 × (G_h+1))
        np.savetxt(dst / 'Data/Rates/degradation_rates.txt',
                   mh.d.T, fmt='%.6f', delimiter='\t')

        # Save ground truth (same for all sigma and replicates)
        inter = 1 * (abs(mh.inter) > 0)
        for r in range(1, n_reps + 1):
            np.save(dst / 'True' / f'inter_{r}.npy', inter)

        for si, sigma in enumerate(TIMESCALE_SIGMAS):
            for r in range(1, n_reps + 1):
                print(f'  Timescale simulation: {bench} sigma={sigma} rep {r}/{n_reps}')
                rep_rng = np.random.default_rng(all_seeds[bi, si, r - 1])

                # Jitter timepoints
                if sigma == 0:
                    t = t_orig.copy()
                else:
                    t = np.sort(np.clip(
                        rep_rng.normal(t_orig, sigma), t_min, t_max))
                    t[0] = t_min
                    t[-1] = t_max

                np.save(dst / 'Data' / f'timepoints_s{si}_r{r}.npy', t)

                # Assign cells to timepoints (equal per timepoint)
                time_int = np.zeros(C, dtype='int')
                for i in range(n_tp):
                    time_int[i * cells_per_tp:(i + 1) * cells_per_tp] = int(round(t[i]))

                # Build data array: rows=cells+1, cols=G_h+2 (time, stimulus, G_h genes)
                data = np.zeros((C + 1, G_h + 2), dtype='int')
                data[0, 1:] = np.arange(G_h + 1)
                data[1:, 0] = time_int
                data[1:, 1] = 100 * (time_int > 0)  # stimulus indicator

                # Simulate using actual jittered time for each cell
                sim_rng = np.random.default_rng(all_seeds[bi, si, r - 1] + 1)
                for k in range(C):
                    sim = mh.simulate(t[k // cells_per_tp], burnin=5)
                    data[k + 1, 2:] = sim_rng.poisson(sim.m[-1])

                # Save data
                np.savetxt(dst / 'Data' / f'data_s{si}_r{r}.txt',
                           data.T, fmt='%d', delimiter='\t')


def run_timescale_inference():
    """Run CARDAMOM2 inference on timescale-perturbed benchmarks (skips if done)."""
    ts_dir = FIGURE4_TIMESCALE_DIR
    n_sigmas = len(TIMESCALE_SIGMAS)
    n_reps = N

    # Check if all scores exist
    all_done = True
    for bench in benchmarks:
        for si in range(n_sigmas):
            for r in range(1, n_reps + 1):
                if not (ts_dir / bench / 'CARDAMOM2' / f'score_s{si}_r{r}.npy').exists():
                    all_done = False
                    break
            if not all_done:
                break
        if not all_done:
            break
    if all_done:
        print('Timescale inference already complete, skipping.')
        return

    sys.path.insert(0, str(REPO_ROOT))
    from CardamomOT import NetworkModel

    n_repet = 4
    verb = 1

    with temporary_cwd(ts_dir):
        for bench in benchmarks:
            # Read degradation rates once per benchmark
            d = np.loadtxt(f'{bench}/Data/Rates/degradation_rates.txt',
                           dtype=float, delimiter='\t').T
            for si in range(n_sigmas):
                for r in range(1, n_reps + 1):
                    print(f'  Timescale inference: {bench} sigma={TIMESCALE_SIGMAS[si]} '
                          f'rep {r}/{n_reps}')
                    fname = f'{bench}/Data/data_s{si}_r{r}.txt'
                    t_jittered = np.load(f'{bench}/Data/timepoints_s{si}_r{r}.npy')
                    data = np.loadtxt(fname, dtype=int, delimiter='\t')[1:, 1:]
                    x = data.T
                    # Assign jittered time values to cells (equal per timepoint)
                    C_cells = x.shape[0]
                    n_tp = len(t_jittered)
                    cells_per_tp = C_cells // n_tp
                    time = np.zeros(C_cells)
                    for i in range(n_tp):
                        time[i * cells_per_tp:(i + 1) * cells_per_tp] = t_jittered[i]
                    x[:, 0] = time
                    G = x.shape[1]
                    model = NetworkModel(G - 1)
                    model.d = d
                    model.fit(x, verb=verb)
                    score = model.inter
                    for _ in range(n_repet):
                        model = NetworkModel(G - 1)
                        model.d = d
                        model.fit(x, verb=verb)
                        score += model.inter
                    np.save(f'{bench}/CARDAMOM2/score_s{si}_r{r}', score)


# ---- Prepare data and run inference ----
prepare_dropout_data()
run_dropout_inference()
prepare_timescale_data()
run_timescale_inference()

# ---- Compute AUPR for dropout robustness (panel G, per dropout level) ----
do_dir = FIGURE4_DROPOUT_DIR
n_do_levels = len(DROPOUT_LEVELS)
dropout_aupr = {li: [] for li in range(n_do_levels)}
for bench in benchmarks:
    for r in range(1, N + 1):
        inter_real = abs(np.load(f'{path}{bench}/True/inter_{r}.npy'))
        G = inter_real.shape[0]
        edges = [(i, j) for i in range(G) for j in set(range(1, G)) - {i}]
        for li in range(n_do_levels):
            if li == 0:
                inter = inter_real
                score = abs(np.load(f'{path}{bench}/CARDAMOM2/score_{r}.npy'))
            else:
                inter = abs(np.load(do_dir / bench / f'level_{li}' / 'True' / f'inter_{r}.npy'))
                score = abs(np.load(do_dir / bench / 'CARDAMOM2' / f'score_l{li}_r{r}.npy'))
            y0 = np.array([inter[i, j] for i, j in edges])
            y1 = np.array([score[i, j] for i, j in edges])
            p, rcl, _ = precision_recall_curve(y0, y1)
            dropout_aupr[li].append(auc(rcl, p))

# ---- Compute AUPR for timescale robustness (panel H, pooled across 4 benchmarks) ----
ts_dir = FIGURE4_TIMESCALE_DIR
n_sigmas = len(TIMESCALE_SIGMAS)
timescale_aupr = {si: [] for si in range(n_sigmas)}
for bench in benchmarks:
    for r in range(1, N + 1):
        inter = abs(np.load(ts_dir / bench / 'True' / f'inter_{r}.npy'))
        G = inter.shape[0]
        edges = [(i, j) for i in range(G) for j in set(range(1, G)) - {i}]
        for si in range(n_sigmas):
            score = abs(np.load(ts_dir / bench / 'CARDAMOM2' / f'score_s{si}_r{r}.npy'))
            y0 = np.array([inter[i, j] for i, j in edges])
            y1 = np.array([score[i, j] for i, j in edges])
            p, rcl, _ = precision_recall_curve(y0, y1)
            timescale_aupr[si].append(auc(rcl, p))

# Random baseline for timescale (pooled across all 4 benchmarks)
ts_aupr_random = []
for bench in benchmarks:
    for r in range(1, N + 1):
        inter = abs(np.load(f'{path}{bench}/True/inter_{r}.npy'))
        G = inter.shape[0]
        edges = [(i, j) for i in range(G) for j in set(range(1, G)) - {i}]
        ts_aupr_random.append(np.mean([inter[i, j] for i, j in edges]))

# ============================================================
# Plotting
# ============================================================

cmap = plt.get_cmap('tab20')
c = {
    'CardamomOT':      (cmap(6),  cmap(7)),
    'random init':     (cmap(8),  cmap(9)),
    'noloop':          (cmap(0),  cmap(1)),
    'random + noloop': (cmap(2),  cmap(3)),
    'Random':          2*('lightgray',)
}
for a in algoG: c[a] = ('black', 'grey')

# Figure: same width as figure2.py (6.85 in = 174 mm), now 7.8 in tall to fit G/H
fig = plt.figure(figsize=(6.85, 7.8))
grid = gs.GridSpec(4, 2, hspace=0.55, wspace=0.50, height_ratios=[1, 1, 1, 1])

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
    kw = dict(fontweight='bold', **opt_panel) 
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
ax.set_xlim(0.5, len(algoG) + 0.5)
ax.set_xticks(range(1, len(labelG)+1))
ax.set_xticklabels(labelG, rotation=45, ha='right', fontsize=5)
ax.set_ylabel('AUPR', fontsize=6)
optn = dict(fontsize=9, transform=ax.transAxes, ha='right')
ax.text(xn, yn, 'Protein\nDegrad', **optn)
ax.text(xn, yn + 0.01, 'Protein\nDegrad', color='none',
        bbox=dict(boxstyle='round,pad=0.2', fc='none', ec='lightgray', lw=0.8), **optn)

# ---- Panel G: Dropout robustness (CardamomOT with binomial dropout) ----
ax = plt.subplot(grid[3, 0])
configure(ax)
ax.annotate('G', xytext=(x, y), fontweight='bold', **opt_panel)

# Compute random baseline from original (real) data across all benchmarks
aupr_dropout_random = []
for bench in benchmarks:
    for r in range(1, N + 1):
        inter = abs(np.load(f'{path}{bench}/True/inter_{r}.npy'))
        G = inter.shape[0]
        edges = [(i, j) for i in range(G) for j in set(range(1, G)) - {i}]
        aupr_dropout_random.append(np.mean([inter[i, j] for i, j in edges]))

for li in range(n_do_levels):
    box = ax.boxplot([dropout_aupr[li]], positions=[li + 1],
                     patch_artist=True, widths=0.5)
    configure_box(box, ('black', 'grey'))

ax.plot([0, n_do_levels + 1], [np.mean(aupr_dropout_random)] * 2,
        '--', color='lightgray', lw=1)
ax.set_xlim(0.5, n_do_levels + 0.5)
ax.set_xticks(range(1, n_do_levels + 1))
ax.set_xticklabels(DROPOUT_LABELS, rotation=45, ha='right', fontsize=5)
ax.set_ylabel('AUPR', fontsize=6)
optn = dict(fontsize=9, transform=ax.transAxes, ha='right')
ax.text(xn, yn, 'Dropout', **optn)
ax.text(xn, yn + 0.01, 'Dropout', color='none',
        bbox=dict(boxstyle='round,pad=0.2', fc='none', ec='lightgray', lw=0.8), **optn)

# ---- Panel H: Timescale robustness (AUPR vs timepoint jitter) ----
ax = plt.subplot(grid[3, 1])
configure(ax)
ax.annotate('H', xytext=(x, y), fontweight='bold', **opt_panel)

for si in range(n_sigmas):
    box = ax.boxplot([timescale_aupr[si]], positions=[si + 1],
                     patch_artist=True, widths=0.5)
    configure_box(box, ('black', 'grey'))

ax.plot([0, n_sigmas + 1], [np.mean(ts_aupr_random)] * 2, '--', color='lightgray', lw=1)
ax.set_xlim(0.5, n_sigmas + 0.5)
ax.set_xticks(range(1, n_sigmas + 1))
ax.set_xticklabels(TIMESCALE_LABELS, rotation=45, ha='right', fontsize=5)
ax.set_ylabel('AUPR', fontsize=6)
optn = dict(fontsize=9, transform=ax.transAxes, ha='right')
ax.text(xn, yn, 'Timescale', **optn)
ax.text(xn, yn + 0.01, 'Timescale', color='none',
        bbox=dict(boxstyle='round,pad=0.2', fc='none', ec='lightgray', lw=0.8), **optn)

fig.savefig('figure_4.pdf', dpi=300, bbox_inches='tight', pad_inches=0.15)
plt.show()
plt.show()