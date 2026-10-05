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

The effects are used in three places, always on mRNA:
1. get_proliferation_rates: prior rate of the observed cells (all genes; counts normalised by the
   library size if the transcriptome is available);
2. infer_network_simul: the stimulus part is removed from the growth targets, so that the
   proliferation MLP learns the rate without stimulus (StimulusRateModel, model genes);
3. simulations: the stimulus part is added back with the simulated schedule, on the mRNA drawn from
   the simulated proteins (same StimulusRateModel).
"""
import json
import os

import numpy as np

from .inputs import input_dir


def load_effects(project):
    """{k (1-based inference stimulus): [(target, delta)]} of the project, {} if none."""
    path = os.path.join(input_dir(project), 'stimulus_rates.json')
    if not os.path.exists(path):
        return {}
    return {int(k): [(str(t), float(d)) for t, d in v] for k, v in json.load(open(path)).items()}


def schedule_values(project, times_sorted, n_stimuli=None):
    """(T, n_stimuli) values of the inference stimuli at the sorted timepoints (stimulus_inference_schedule;
    default 0 at the first time, 1 after)."""
    from .config import find_stimulus_schedule
    T = len(times_sorted)
    path = find_stimulus_schedule(input_dir(project))
    if path is not None:
        U = np.loadtxt(path, ndmin=2).astype(float)
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


def slot_offsets(project, S, tu, slot_samples, names):
    """
    Stimulus part of the net rate along trajectories: S (T, N, K) stimulus parts at the states, slot_samples
    (N,) sample index of each slot. Interval k: u(t_k+1) of the slot's sample x mean of S at both ends. (T, N).
    """
    T, N, K = S.shape
    U = cell_values(project, np.repeat(np.asarray(tu, dtype=float)[1:], N), np.tile(slot_samples, T - 1), tu,
                    K, names)[:, :K].reshape(T - 1, N, K)
    off = np.zeros((T, N))
    off[:-1] = np.sum((S[:-1] + S[1:]) / 2 * U, axis=-1)
    return off


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


class StimulusRateModel:
    """
    Stimulus part of the net rate of cells from their mRNA counts on the model genes:
    S_k(x) = sum_c p(c | x) delta_{k,c} + sum_s delta_{k,s} score_s(x), p(c | x) by a logistic regression
    trained on the observed cells (log1p counts, labels cell_type_proliferation). The rate of a cell
    over an interval is then sum_k u_k S_k(x).
    """

    def __init__(self, project, effects, genes, X, cell_types=None, verb=True):
        from .tools.perturbations import rate_target_genes
        self.n_stimuli = max(effects) if effects else 0
        self.genes = list(genes)
        L = log_counts(X)
        labels = np.asarray(cell_types).astype(str) if cell_types is not None else None
        parts = split_effects(effects, np.unique(labels) if labels is not None else [])
        self.ct_delta, self.sig = {}, []
        for k, (ct, sig) in parts.items():
            for c, d in ct.items():
                self.ct_delta.setdefault(c, np.zeros(self.n_stimuli))[k - 1] += d
            for target, d in sig:
                idx = [self.genes.index(g) for g in rate_target_genes(target, self.genes, input_dir(project))]
                q99 = np.maximum(np.percentile(L[:, idx], 99, axis=0), 1e-6)
                self.sig.append((k, target, float(d), np.array(idx), q99))
        self.clf = None
        if self.ct_delta:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            self.clf.fit(L, labels)
            # Effect per class, in the order of the classifier (0 for the cell types without effect)
            self.class_delta = np.array([self.ct_delta.get(c, np.zeros(self.n_stimuli)) for c in self.clf.classes_])
            if verb:
                acc = float(np.mean(self.clf.predict(L) == labels))
                print(f"[stimulus rates] cell types {list(self.clf.classes_)} from mRNA (logistic regression, "
                      f"training accuracy {acc:.2f}); effects per type and stimulus: "
                      + ', '.join(f'{c}: {np.round(v, 5).tolist()}' for c, v in self.ct_delta.items()))
        if verb:
            for k, target, d, idx, _ in self.sig:
                print(f"[stimulus rates] stimulus {k}: {target} ({len(idx)} model genes) {d:+g} at maximal score")

    def effect(self, X):
        """(n, n_stimuli) stimulus part S_k of the net rate for mRNA counts X (n, model genes)."""
        L = log_counts(X)
        S = np.zeros((len(L), self.n_stimuli))
        if self.clf is not None:
            S += self.clf.predict_proba(L) @ self.class_delta
        for k, _, d, idx, q99 in self.sig:
            S[:, k - 1] += d * signature_score(L, idx, q99)
        return S
