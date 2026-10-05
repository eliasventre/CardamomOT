"""
Stimulus schedules with per-sample overrides.

The schedule sheets of Data/CardamomOT_inputs.xlsx (stimulus_inference_schedule,
stimulus_simulation_schedule, stimulus_test_schedule) take an optional sample_id column: rows without
sample_id (or "all") give the default schedule of every sample, rows of a sample_id replace it for
that sample. The default schedule keeps its former files (stimulus_schedule_*.txt); the overrides
are exported to cardamomOT/inputs/stimulus_schedules.json, {kind: {sample_id: {"time": [...] or null,
"values": [[...]]}}}, kind in inference / simulation / test. Without a time column, the rows of an
override follow the sorted timepoints of the run, as the default schedule.
"""
import json
import os

import numpy as np

KINDS = ('inference', 'simulation', 'test')


class StimulusSchedule(dict):
    """
    Default schedule {t: values (n_stimuli,)} (the former dict, unchanged for code that indexes it)
    with per-sample overrides {sample_id: (times or None, values (n_rows, n_stimuli))}. at(t, sample)
    and per_cell(t, samples) give the values of a sample / of each cell; samples are dataset_id
    labels, or indices into sample_names (sorted dataset_id of the run).
    """

    def __init__(self, default, overrides=None, sample_names=None):
        super().__init__(default)
        self.overrides = dict(overrides or {})
        self.sample_names = [str(s) for s in sample_names] if sample_names is not None else None
        self._times = np.sort(np.array(list(default.keys()), dtype=float))

    def name(self, sample):
        if sample is None:
            return None
        if isinstance(sample, (int, np.integer)) and self.sample_names is not None \
                and 0 <= int(sample) < len(self.sample_names):
            return self.sample_names[int(sample)]
        return str(sample)

    def has_overrides(self):
        return bool(self.overrides)

    def at(self, t, sample=None):
        """Values at time t for a sample (default schedule if it has no override)."""
        name = self.name(sample)
        if name is None or name not in self.overrides:
            return np.asarray(self[t], dtype=float)
        times, vals = self.overrides[name]
        if times is None:
            i = int(np.searchsorted(self._times, float(t) - 1e-9))
        else:
            i = int(np.searchsorted(np.asarray(times, dtype=float), float(t) + 1e-9, side='right')) - 1
        return np.asarray(vals[min(max(i, 0), len(vals) - 1)], dtype=float)

    def per_cell(self, t, samples):
        """(n, n_stimuli) values at time t for cells of the given samples."""
        samples = np.asarray(samples)
        out = np.tile(np.asarray(self[t], dtype=float), (len(samples), 1))
        if self.overrides:
            for s in np.unique(samples):
                if self.name(s) in self.overrides:
                    out[samples == s] = self.at(t, s)
        return out


def load_overrides(project, kind, n_columns=None, columns=slice(None)):
    """{sample_id: (times or None, values)} of a schedule sheet (columns: slice of the value columns)."""
    from .inputs import input_dir
    path = os.path.join(input_dir(project), 'stimulus_schedules.json')
    if not os.path.exists(path):
        return {}
    spec = json.load(open(path)).get(kind, {})
    out = {}
    for sid, d in spec.items():
        vals = np.atleast_2d(np.asarray(d['values'], dtype=float))[:, columns]
        if n_columns is not None and vals.shape[1] != n_columns:
            print(f"[CardamomOT] Warning: {kind} schedule of sample {sid}: {vals.shape[1]} column(s) instead of "
                  f"{n_columns}: ignored")
            continue
        out[str(sid)] = (None if d.get('time') is None else np.asarray(d['time'], dtype=float), vals)
    return out


def keep_present(overrides, sample_names, what='schedule'):
    """Overrides of samples present in the data; a warning for the others (treated as absent)."""
    names = {str(s) for s in sample_names}
    absent = sorted(s for s in overrides if s not in names)
    if absent:
        print(f"[CardamomOT] Warning: {what} given for sample(s) {absent} absent from the data: ignored")
    return {s: v for s, v in overrides.items() if s in names}


def sample_names(adata):
    """Sorted dataset_id of an AnnData (labels of the sample indices of the run); ['0'] without dataset_id."""
    obs = getattr(adata, 'obs', None)
    if obs is None or 'dataset_id' not in obs:
        return ['0']
    return [str(s) for s in np.sort(np.unique(obs['dataset_id'].values))]


def simulation_overrides(project, n_stimuli):
    """
    Per-sample overrides of the simulations (stimulus_simulation_schedule): (inference stimuli, perturbation
    stimuli). Without a default simulation schedule, the samples keep their inference overrides.
    """
    from .inputs import input_dir
    inf, pert = {}, {}
    for sid, (t, vals) in load_overrides(project, 'simulation').items():
        if vals.shape[1] < n_stimuli:
            print(f"[CardamomOT] Warning: simulation schedule of sample {sid}: {vals.shape[1]} column(s), fewer "
                  f"than the {n_stimuli} inference stimuli: ignored")
            continue
        inf[sid] = (t, vals[:, :n_stimuli])
        if vals.shape[1] > n_stimuli:
            pert[sid] = (t, vals[:, n_stimuli:])
    if not os.path.exists(os.path.join(input_dir(project), 'stimulus_schedule_simulate.txt')):
        inf = {**load_overrides(project, 'inference', n_stimuli), **inf}
    return inf, pert


def override_function(override, times, k):
    """t -> value of column k (1-based) of an override (rows on the sorted `times` without time column)."""
    t_ov, vals = override
    if vals.shape[1] < k:
        return None
    col = np.asarray(vals, dtype=float)[:, k - 1]
    tt = np.sort(np.asarray(times if t_ov is None else t_ov, dtype=float))
    return lambda t: float(col[min(max(int(np.searchsorted(tt, float(t) + 1e-9, side='right')) - 1, 0), len(col) - 1)])


def test_schedule(project, n_stimuli):
    """
    (default matrix or None, per-sample overrides) of the held-out cells (infer_test): stimulus_test_schedule,
    else the inference schedule; without a default test schedule, the samples keep their inference overrides.
    """
    from .config import find_stimulus_schedule
    from .inputs import input_dir
    d = input_dir(project)
    over = load_overrides(project, 'test', n_stimuli)
    path = os.path.join(d, 'stimulus_schedule_test.txt')
    if os.path.exists(path):
        default = np.loadtxt(path, ndmin=2)
        if default.shape[1] != n_stimuli:
            print(f"[CardamomOT] Warning: stimulus_test_schedule has {default.shape[1]} column(s) instead of "
                  f"{n_stimuli}: inference schedule used")
        else:
            return default, over
    inf = find_stimulus_schedule(d)
    return (np.loadtxt(inf, ndmin=2) if inf is not None else None), {**load_overrides(project, 'inference', n_stimuli), **over}
