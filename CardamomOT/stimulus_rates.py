"""
Effects of the inference stimuli on the net proliferation rate.

Given in the RATEk columns of the row sample_id = all of the perturbation_inference sheet, as
TARGET:delta entries (per hour):
- TARGET = a cell type of cell_type_proliferation: delta is added to the rate of this cell type in
  proliferation_rates, which is then the rate WITHOUT the stimulus;
- otherwise a gene list of gene_lists, GENE1+GENE2..., or a gene: delta x signature score, the score
  being computed on mRNA counts (mean over the genes of log1p(x) / q99, clipped to [0, 1]).
A cell at time t receives u_k * delta, u_k being the value of inference stimulus k over the interval
that starts at t (that of the next timepoint, as in the simulations), in the schedule of its sample.

The effects enter the prior rates of the observed cells (get_proliferation_rates, on mRNA: all genes, counts
normalised by the library size if the transcriptome is available) and the anchors of fit_population_anchors. They
stay in the growth targets of the inference: the proliferation MLP learns them from its stimulus inputs, and the
simulations only feed it their schedule.
"""
import json
import os

import numpy as np

from .inputs import input_dir

# Genes of the scored data above which the counts are normalised by the library size (whole transcriptome)
MIN_GENES_SCORING = 10000


def load_effects(project):
    """{k (1-based inference stimulus): [(target, delta)]} of the project, {} if none."""
    path = os.path.join(input_dir(project), 'stimulus_rates.json')
    if not os.path.exists(path):
        return {}
    return {int(k): [(str(t), float(d)) for t, d in v] for k, v in json.load(open(path)).items()}


def schedule_values(project, times_sorted, n_stimuli=None):
    """(T, n_stimuli) values of the inference stimuli at the sorted timepoints (stimulus_inference_schedule;
    default 0 at the first time, 1 after)."""
    from .config import inference_schedule
    T = len(times_sorted)
    U = inference_schedule(input_dir(project), times_sorted)  # by time when the workbook gives the times of its rows
    if U is not None:
        U = U.astype(float)
        if len(U) < T:
            U = np.vstack([U, np.tile(U[-1], (T - len(U), 1))])
        return U[:T]
    U = np.ones((T, n_stimuli or 1))
    U[0] = 0.0
    return U


def cell_values(project, cell_times, cell_samples, times_sorted, n_stimuli=None, names=None):
    """
    (n, n_stimuli) values at cell_times of the inference schedule of each cell's sample (per-sample overrides
    of stimulus_inference_schedule; samples are dataset_id labels, or indices into `names`).
    """
    from .schedules import StimulusSchedule, keep_present, load_overrides
    U = schedule_values(project, times_sorted, n_stimuli)
    tu = [float(t) for t in times_sorted]
    present = names if names is not None else np.unique(np.asarray(cell_samples).astype(str))
    ov = keep_present(load_overrides(project, 'inference', U.shape[1]), present, 'stimulus schedule')
    sched = StimulusSchedule(dict(zip(tu, U)), ov, names)
    cell_times = np.asarray(cell_times, dtype=float)
    cell_samples = np.asarray(cell_samples)
    out = np.zeros((len(cell_times), U.shape[1]))
    for t in np.unique(cell_times):
        m = cell_times == t
        out[m] = sched.per_cell(float(t), cell_samples[m])
    return out


def split_effects(effects, cell_types):
    """Per stimulus: ({cell type: delta}, [(signature target, delta)]), cell types matched case-insensitively."""
    lower = {str(c).lower(): str(c) for c in cell_types}
    out = {}
    for k, entries in effects.items():
        ct, sig = {}, []
        for target, delta in entries:
            if target.lower() in lower:
                ct[lower[target.lower()]] = ct.get(lower[target.lower()], 0.0) + delta
            else:
                sig.append((target, delta))
        out[k] = (ct, sig)
    return out


def log_counts(X, normalize=False):
    """log1p of the counts (dense), normalised by the library size (median library) if requested."""
    import scipy.sparse
    X = X.toarray() if scipy.sparse.issparse(X) else np.asarray(X, dtype=float)
    if normalize:
        lib = np.maximum(X.sum(axis=1, keepdims=True), 1e-12)
        X = X / lib * np.median(lib)
    return np.log1p(X)


def signature_score(L, idx, q99):
    """Mean over the genes idx of log1p counts / q99, clipped to [0, 1]."""
    return np.clip(L[:, idx] / q99, 0.0, 1.0).mean(axis=1)


def signature_scores(X, genes, targets, project, normalize=False):
    """
    {target: (n,) score of each cell} of the signature targets of the RATE effects: mean over the genes of the target
    of log1p(counts) (normalised by the library size if normalize) / its 99th percentile over the cells, clipped to
    [0, 1]. Only the genes of the targets are densified (the library size is summed on the sparse counts).
    """
    import scipy.sparse
    from .tools.perturbations import rate_target_genes
    genes = [str(g) for g in genes]
    lib = None
    if normalize:
        lib = np.asarray(X.sum(axis=1)).ravel().astype(float)
        lib = np.maximum(lib, 1e-12) / np.median(np.maximum(lib, 1e-12))
    out = {}
    for target in dict.fromkeys(targets):
        tg = rate_target_genes(target, genes, input_dir(project))
        idx = np.arange(len(genes)) if tg is None else np.array([genes.index(g) for g in tg])
        Xg = X[:, idx]
        Xg = Xg.toarray() if scipy.sparse.issparse(Xg) else np.asarray(Xg, dtype=float)
        L = np.log1p(Xg if lib is None else Xg / lib[:, None])
        out[target] = signature_score(L, np.arange(len(idx)), np.maximum(np.percentile(L, 99, axis=0), 1e-6))
    return out
