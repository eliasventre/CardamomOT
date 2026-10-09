"""
fit_population_anchors.py
-------------------------
Makes the population constraints of each sample consistent with its data, before get_proliferation_rates.py and the
inference use them.

Population constraints (workbook), each for all samples (rows without sample_id) or for one sample (sample_id):
- net proliferation rates per cell type of the proliferation grouping (sheet proliferation_rates, obs
  cell_type_proliferation);
- transition rates between the cell types of the transition grouping (sheet transition_rates, obs cell_type_transition);
- population sizes per time (sheet population_sizes).
A rate that does not reproduce the evolution of the proportions of the types (and of the population size, if given) of
its sample is not coherent with the data, so each sample is fitted with a small model of growth and transitions between
types (CardamomOT.inference.population_fit):
- the net rates without stimulus are fitted around the prescribed ones, with the effects of the inference stimuli
  (RATEk of perturbation_inference, on cell types) and the inference schedule of the sample (value over each interval =
  value at its end, as in get_proliferation_rates); the population sizes constrain the absolute growth; transitions
  between the types are free (small prior): they absorb the conversion between types and are NOT kept;
- prescribed transition rates (for the types of the transition grouping) are fitted around the prescribed ones, the
  growth of each type being that of the corrected rates (composition of the type in cell_type_proliferation).
Only what was prescribed is kept, corrected. A sample without rates takes, from the samples with rates, the one whose
rates fitted on its proportions (and sizes) stay the closest to the data and to the prescribed ones (lowest cost); with
population sizes but no sample with rates, its rates are fitted from its proportions and sizes alone (weak prior: its
mean growth). Without any constraint in the workbook nothing is done (no anchor, the stale output is removed). A sample
observed at a single timepoint cannot be fitted: its rates are kept as prescribed, or taken from the first reference.

Usage:
    python fit_population_anchors.py -i <project_path>

Required input files:
    - Data/data.h5ad: obs['cell_type_proliferation'] (else cell_type_transition, else cell_type), obs['time'],
      obs['dataset_id']
    - sheets proliferation_rates, transition_rates, population_sizes of Data/CardamomOT_inputs.xlsx (at least one row)

Output files:
    - cardamomOT/population_anchors.json: per sample, the corrected 'rates' (net rate without stimulus per type), their
      'source' (specified / borrowed from <sample> / population sizes) and the fit; and the corrected 'transitions' per
      sample. get_proliferation_rates.py and the inference read it in place of the sheets.
"""
import sys; sys.path += ['../']
import json
import os

import anndata as ad
import numpy as np
import pandas as pd

from CardamomOT import find_data_file, resolve_cell_type_obs
from CardamomOT.inference.population_fit import fit_population, DEFAULTS as FIT_DEFAULTS
from CardamomOT.inputs import input_dir, proliferation_sample_anchors, load_transition_rates, population_sizes
from CardamomOT.run_options import parse_step_options
from CardamomOT.stimulus_rates import load_effects, cell_values, split_effects

TAG = "[fit_population_anchors]"
log = lambda *a: print(TAG, *a, flush=True)


def default_rates(path, types):
    """{type: rate} of the default rows of proliferation_rates (cell types matched case-insensitively), None if absent."""
    if path is None:
        return None
    s = pd.read_csv(path, sep=None, engine='python', header=None, index_col=0).iloc[:, 0]
    return match_types({str(k).strip(): float(v) for k, v in s.items()}, types, 'default proliferation_rates')


def match_types(rates, types, what):
    """rates {type: value} for every type of `types` (case-insensitive), else None with a warning (partial anchoring is worse than none)."""
    low = {k.lower(): v for k, v in rates.items()}
    missing = [t for t in types if t.lower() not in low]
    if missing:
        log(f"Warning: {what}: no rate for the cell type(s) {missing}: ignored")
        return None
    return {t: low[t.lower()] for t in types}


def proportions(obs, sid, col, types):
    """(times, proportions (T, K), cells (T,)) of the types in the cells of the sample."""
    t = obs[obs['_sid'] == sid]
    c = pd.crosstab(t['time'].astype(float), t[col].astype(str)).reindex(columns=types, fill_value=0)
    c = c[c.sum(axis=1) > 0]
    return c.index.values.astype(float), c.div(c.sum(axis=1), axis=0).values, c.sum(axis=1).values


def sizes_at(sizes, times, sid):
    """Population sizes of a sample at `times` (NaN where not given); sizes at unobserved times are ignored."""
    if not sizes:
        return None
    out = np.array([sizes.get(float(t), np.nan) for t in times], dtype=float)
    extra = sorted(set(sizes) - {float(t) for t in times})
    if extra:
        log(f"Warning: {sid}: population sizes at times without cells {extra}: ignored")
    return out if np.isfinite(out).sum() >= 2 else None


def stimulus_inputs(p, effects, types, times, sid, tu, all_samples):
    """(effects (K, S), u (T - 1, S)): RATEk deltas on the cell types and the schedule of the sample over each interval."""
    if not effects:
        return np.zeros((len(types), 0)), None
    S = max(effects)
    E = np.zeros((len(types), S))
    for k, (ct, _) in split_effects(effects, types).items():
        for c, d in ct.items():
            E[types.index(c), k - 1] += d
    u = (cell_values(p, times[1:], np.array([sid] * (len(times) - 1)), tu, S, names=sorted(all_samples))
         if len(times) > 1 else np.zeros((0, S)))
    return E, u


def fit_report(res):
    txt = f"rmse {res['rmse'] * 100:.2f} points"
    return txt + (f", log population sizes rmse {res['size_rmse']:.3f}" if res.get('size_rmse') is not None else '')


def main(argv):
    opts = parse_step_options(argv, 'fit_population_anchors', __doc__)
    p = opts.p
    out_path = os.path.join(p, 'cardamomOT', 'population_anchors.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stale = lambda: os.path.exists(out_path) and os.remove(out_path)

    own_rates = proliferation_sample_anchors(p)
    prolif_path = find_data_file(input_dir(p), 'proliferation_rates')
    raw_transitions = load_transition_rates(p, fitted=False)
    sizes_default, sizes_own = population_sizes(p)
    if not (own_rates or prolif_path is not None or raw_transitions is not None or sizes_default or sizes_own):
        log("No proliferation rate, transition rate nor population size in the workbook: nothing to fit, no anchor")
        stale()
        return

    data_path = os.path.join(p, 'Data', 'data.h5ad')
    if not os.path.exists(data_path):
        log(f"Error: {data_path} not found")
        sys.exit(1)
    A = ad.read_h5ad(data_path, backed='r')
    obs = A.obs.copy()
    obs['_sid'] = obs['dataset_id'].astype(str) if 'dataset_id' in obs else '0'
    samples = sorted(obs['_sid'].unique())
    tu = np.sort(obs['time'].astype(float).unique())
    effects = load_effects(p)
    for sid in sorted((set(own_rates) | set(sizes_own)) - set(samples)):
        log(f"Warning: constraints given for sample {sid} absent from the data: ignored")
    size_of = {sid: sizes_own.get(sid) or sizes_default for sid in samples}

    # ---------------------------------------------------------------- proliferation rates
    col = resolve_cell_type_obs(A, 'proliferation')
    result = dict(samples={}, transitions={})
    corrected = {}          # sample -> {'rates': array, 'source': .., 'prior': array, 'fit': dict}
    if col is None:
        log("Warning: obs has none of the cell type columns of the proliferation grouping: proliferation rates not fitted")
    else:
        types = sorted(obs[col].astype(str).unique())
        default = default_rates(prolif_path, types)
        spec = {sid: (match_types(own_rates[sid], types, f'proliferation_rates of {sid}') if sid in own_rates else default)
                for sid in samples}

        def fit_sample(sid, prior, r_sd=None):
            times, f, n = proportions(obs, sid, col, types)
            if len(times) < 2:
                return None
            E, u = stimulus_inputs(p, effects, types, times, sid, tu, samples)
            return fit_population(f, n, times, E, u, r_prior=prior, sizes=sizes_at(size_of[sid], times, sid), r_sd=r_sd)

        # 1. samples with prescribed rates: corrected
        for sid in samples:
            if spec[sid] is None:
                continue
            prior = np.array([spec[sid][t] for t in types])
            res = fit_sample(sid, prior)
            if res is None:
                log(f"{sid}: observed at a single timepoint, its prescribed rates are kept as they are")
                res = dict(r0=prior, rmse=None, cost=None, size_rmse=None)
            else:
                log(f"{sid}: prescribed rates corrected " + ', '.join(f"{t} {a:+.5f} -> {b:+.5f}" for t, a, b in zip(types, prior, res['r0']))
                    + f" h^-1 ({fit_report(res)}; transitions fitted, not kept: "
                    + ', '.join(f"{types[i]}->{types[j]} {res['Q'][i, j]:.5f}" for i in range(len(types)) for j in range(len(types))
                                if i != j and res['Q'][i, j] > 1e-5) + ")")
            corrected[sid] = dict(rates=res['r0'], source='specified', prior=prior, fit=res)
        # 2. the other samples: the best reference among the corrected ones, else their population sizes alone
        refs = list(corrected)
        for sid in samples:
            if sid in corrected:
                continue
            if refs:
                best = None
                for ref in refs:
                    res = fit_sample(sid, corrected[ref]['rates'])
                    if res is None:   # nothing to compare: the first reference
                        res = dict(r0=corrected[ref]['rates'], rmse=None, cost=0.0 if best is None else np.inf, size_rmse=None)
                    else:
                        log(f"{sid}: rates of {ref} as prior -> cost {res['cost']:.2f}, {fit_report(res)}")
                    if best is None or res['cost'] < best[1]['cost']:
                        best = (ref, res)
                ref, res = best
                corrected[sid] = dict(rates=res['r0'], source=f'borrowed from {ref}', prior=corrected[ref]['rates'], fit=res)
                log(f"{sid}: no prescribed rate, fitted around those of {ref} (closest): "
                    + ', '.join(f"{t} {b:+.5f}" for t, b in zip(types, res['r0'])) + " h^-1")
                continue
            times, _, _ = proportions(obs, sid, col, types)
            s = sizes_at(size_of[sid], times, sid)
            if s is None:
                continue
            k = np.flatnonzero(np.isfinite(s))
            g = (np.log(s[k[-1]]) - np.log(s[k[0]])) / (times[k[-1]] - times[k[0]])
            prior = np.full(len(types), g)
            res = fit_sample(sid, prior, r_sd=FIT_DEFAULTS['sd_r_free'])
            corrected[sid] = dict(rates=res['r0'], source='population sizes', prior=prior, fit=res)
            log(f"{sid}: rates fitted on its proportions and population sizes (mean growth {g:+.5f} h^-1): "
                + ', '.join(f"{t} {b:+.5f}" for t, b in zip(types, res['r0'])) + f" h^-1 ({fit_report(res)})")
        for sid, c in corrected.items():
            fit = c['fit']
            result['samples'][sid] = dict(
                rates={t: float(r) for t, r in zip(types, c['rates'])}, source=c['source'], types=types,
                fit=dict(rmse_points=None if fit.get('rmse') is None else float(fit['rmse'] * 100),
                         size_log_rmse=fit.get('size_rmse'),
                         prescribed={t: float(r) for t, r in zip(types, c['prior'])}))

    # ---------------------------------------------------------------- transition rates
    tcol = resolve_cell_type_obs(A, 'transition')
    if raw_transitions is not None and tcol is None:
        log("Warning: transition rates given but obs has none of the transition cell type columns: not fitted")
    elif raw_transitions is not None:
        ttypes = sorted(obs[tcol].astype(str).unique())
        raw = raw_transitions if isinstance(raw_transitions, dict) else {'default': raw_transitions}
        mats = {}
        for sid in samples:
            df = raw.get(sid, raw.get('default'))
            if df is None:
                continue
            df = df.rename(index=str, columns=str)
            if set(ttypes) - set(df.index) or set(ttypes) - set(df.columns):
                log(f"Warning: transition rates of {sid} lack some of the cell types {ttypes}: ignored")
                continue
            mats[sid] = df.loc[ttypes, ttypes].to_numpy(dtype=float)
        fitted_T = {}

        def growth_of(sid):
            """(g (K',), effects (K', S)) of the transition types of a sample: composition in the proliferation types."""
            K = len(ttypes)
            if col is None or sid not in corrected:
                return np.zeros(K), np.zeros((K, 0)) if not effects else np.zeros((K, max(effects)))
            pt = sorted(obs[col].astype(str).unique())
            t = obs[obs['_sid'] == sid]
            comp = pd.crosstab(t[tcol].astype(str), t[col].astype(str)).reindex(index=ttypes, columns=pt, fill_value=0)
            P = comp.div(comp.sum(axis=1).replace(0, 1), axis=0).values
            E, _ = stimulus_inputs(p, effects, pt, np.array([0.0]), sid, tu, samples)
            return P @ corrected[sid]['rates'], P @ E

        def fit_T(sid, Q0):
            times, f, n = proportions(obs, sid, tcol, ttypes)
            if len(times) < 2:
                return None
            g, E = growth_of(sid)
            _, u = stimulus_inputs(p, effects, ttypes, times, sid, tu, samples)
            return fit_population(f, n, times, E, u, r_fixed=g, q_prior=Q0, sizes=sizes_at(size_of[sid], times, sid))

        for sid, Q0 in mats.items():
            res = fit_T(sid, Q0)
            if res is None:
                log(f"{sid}: observed at a single timepoint, its prescribed transition rates are kept as they are")
                fitted_T[sid] = (Q0, 'specified', None)
                continue
            fitted_T[sid] = (res['Q'], 'specified', res['rmse'])
            log(f"{sid}: prescribed transition rates corrected on the proportions of '{tcol}' ({fit_report(res)})")
        for sid in samples:
            if sid in fitted_T or not mats:
                continue
            best = None
            for ref in mats:
                res = fit_T(sid, fitted_T[ref][0])
                if res is None:
                    best = best or (ref, fitted_T[ref][0], 0.0, None)
                    continue
                if best is None or res['cost'] < best[2]:
                    best = (ref, res['Q'], res['cost'], res['rmse'])
            fitted_T[sid] = (best[1], f'borrowed from {best[0]}', best[3])
            log(f"{sid}: no prescribed transition rates, fitted around those of {best[0]} (closest)")
        for sid, (Q, src, rmse) in fitted_T.items():
            result['transitions'][sid] = dict(types=ttypes, source=src, rmse_points=None if rmse is None else float(rmse * 100),
                                              matrix={a: {b: float(Q[i, j]) for j, b in enumerate(ttypes) if i != j}
                                                      for i, a in enumerate(ttypes)})

    if not result['samples'] and not result['transitions']:
        log("Nothing could be fitted: the prescribed values of the workbook are used as they are")
        stale()
        return
    json.dump(result, open(out_path, 'w'), indent=1)
    log(f"Saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
