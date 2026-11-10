"""
Effects of the inference stimuli on the net proliferation rate.

Given in the RATEk columns of the row sample_id = all of the perturbation_inference sheet (the row of a sample: that
sample only, replacing the row all for that stimulus), as TARGET:delta entries (per hour):
- TARGET = a cell type of cell_type_proliferation: delta is added to the rate of this cell type in
  proliferation_rates, which is then the rate WITHOUT the stimulus;
- otherwise a gene list of gene_lists, GENE1+GENE2..., or a gene: delta x signature score, the score
  being computed on mRNA counts (mean over the genes of log1p(x) / q99, clipped to [0, 1]).
The rates of a cell are those of its state at its time t: it receives u_k(t) * delta, u_k(t) being the value of
inference stimulus k at t in the schedule of its sample (0 for the untreated cells; a cell-type delta is shared in
proportion to the rate it changes, get_proliferation_rates). Over an interval (t_i, t_i+1] (t_i+1 the next observed
time of the sample), the inference adds instead the value of the interval, u_k(t_i+1), to the cells at t_i
(NetworkModel._set_interval_rates).

The effects enter the prior rates of the observed cells (get_proliferation_rates, on mRNA: all genes, counts
normalised by the library size if the transcriptome is available). They
stay in the growth targets of the inference: the proliferation MLP learns them from its stimulus inputs, and the
simulations only feed it their schedule.
"""
import json
import os

import numpy as np

from .inputs import input_dir

# Genes of the scored data above which the counts are normalised by the library size (whole transcriptome)
MIN_GENES_SCORING = 10000


def _read_effects(project):
    """(row all {k: [(target, delta)]}, {sample: {k: [(target, delta)]}}) of stimulus_rates.json, empty if none."""
    path = os.path.join(input_dir(project), 'stimulus_rates.json')
    if not os.path.exists(path):
        return {}, {}
    d = json.load(open(path))
    parse = lambda e: {int(k): [(str(t), float(x)) for t, x in v] for k, v in e.items()}
    return parse(d.get('default', {})), {str(s): parse(e) for s, e in d.get('samples', {}).items()}


def load_effects(project, sample=None):
    """{k (1-based inference stimulus): [(target, delta)]} of the project ({} if none): the row all, with sample the
    rows of that sample replacing it stimulus by stimulus."""
    default, per = _read_effects(project)
    out = dict(default)
    if sample is not None:
        out.update(per.get(str(sample), {}))
    return out


def all_effects(project):
    """Every effect set of the project (row all and rows of the samples), [{k: [(target, delta)]}]."""
    default, per = _read_effects(project)
    return [default] + list(per.values())


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


def state_values(project, obs, n_stimuli=None):
    """
    (n, n_stimuli) values of the inference stimuli at the time of each cell (rows of obs), in the schedule of its
    sample: the inference schedule on the timepoints of the samples of the inference; for their held-out cells
    (obs['split'] = 'test') the schedule of the test pass (schedules.test_pass_schedule: stimulus_test_schedule, else
    the inference one, on the test timepoints); for the samples removed from the inference (remove_from_inference) the
    schedule of their validation (schedules.held_out_schedule).
    """
    from .inputs import removed_samples
    from .schedules import held_out_schedule, test_pass_schedule
    times = obs['time'].to_numpy(dtype=float) if 'time' in obs else np.zeros(len(obs))
    samples = obs['dataset_id'].astype(str).to_numpy() if 'dataset_id' in obs else np.array(['0'] * len(obs))
    removed = removed_samples(project, present=np.unique(samples))[0]
    held = np.isin(samples, removed)
    test = (obs['split'].astype(str).to_numpy() == 'test') & ~held if 'split' in obs else np.zeros(len(obs), bool)
    fit = ~held & ~test
    train_t = np.sort(np.unique(times[fit] if fit.any() else times[~held] if (~held).any() else times))
    ns = schedule_values(project, train_t, n_stimuli).shape[1]
    U = np.zeros((len(obs), ns))
    if fit.any():
        U[fit] = cell_values(project, times[fit], samples[fit], train_t, ns)
    if test.any():
        sched = test_pass_schedule(project, ns, times[test], train_t, removed)
        for s in np.unique(samples[test]):
            for t in np.unique(times[test & (samples == s)]):
                U[test & (samples == s) & (times == t)] = sched.at(float(t), s)
    if held.any():
        sched = held_out_schedule(project, ns, train_t, times[held])
        for s in np.unique(samples[held]):
            for t in np.unique(times[held & (samples == s)]):
                U[held & (samples == s) & (times == t)] = sched.at(float(t), s)
    return U


def next_times(times, samples):
    """Next observed time of the sample of each cell (own time at its last one): end of the OT interval that starts
    at the cell."""
    times, samples = np.asarray(times, dtype=float), np.asarray(samples).astype(str)
    out = times.copy()
    for s in np.unique(samples):
        m = samples == s
        ts = np.sort(np.unique(times[m]))
        out[m] = ts[np.minimum(np.searchsorted(ts, times[m]) + 1, len(ts) - 1)]
    return out


def interval_rates(project, obs, n_stimuli=None, times_sorted=None):
    """(birth, death) of each cell (rows of obs: the transported cells) over the interval that starts at its time:
    rates without the inference stimuli (obs['proliferation_*_rate_base']) + effects per unit
    (obs['proliferation_stimulus_*']) x the value of each stimulus at the next observed time of its sample (own value
    at the last one), in the schedule of the cell's sample (rows on times_sorted, default the times of obs; the rates
    of obs are those of the state of the cell at its time); the rates of obs without stimulus effects, None without
    birth rates."""
    if 'proliferation_birth_rate' not in obs:
        return None
    b, d = obs['proliferation_birth_rate'].to_numpy(dtype=float), obs['proliferation_death_rate'].to_numpy(dtype=float)
    ks = sorted(int(c.rsplit('_', 1)[1]) for c in obs.columns if c.startswith('proliferation_stimulus_birth_'))
    if not ks or 'proliferation_birth_rate_base' not in obs:
        return b, d
    times = obs['time'].to_numpy(dtype=float) if 'time' in obs else np.zeros(len(obs))
    tu = np.sort(np.unique(times if times_sorted is None else times_sorted))
    samples = obs['dataset_id'].astype(str).to_numpy() if 'dataset_id' in obs else np.array(['0'] * len(obs))
    t_next = next_times(times, samples)
    U = cell_values(project, t_next, samples, tu, max(ks) if n_stimuli is None else n_stimuli)
    eb = np.stack([obs[f'proliferation_stimulus_birth_{k}'].to_numpy(dtype=float) for k in ks], axis=1)
    ed = np.stack([obs[f'proliferation_stimulus_death_{k}'].to_numpy(dtype=float) for k in ks], axis=1)
    Uk = U[:, [k - 1 for k in ks]]
    return (np.maximum(obs['proliferation_birth_rate_base'].to_numpy(dtype=float) + (Uk * eb).sum(axis=1), 0.0),
            np.maximum(obs['proliferation_death_rate_base'].to_numpy(dtype=float) + (Uk * ed).sum(axis=1), 0.0))


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
