"""
get_proliferation_rates.py
----------------------------
Estimate per-cell net proliferation rates (birth - death) from literature
proliferation/death/senescence gene signatures.

Runs BEFORE differential gene selection (select_genes.py), on the
full unfiltered dataset — DE gene selection can otherwise discard many of the
literature marker genes used to score proliferation/death, so scoring happens
first while the complete gene set is still available. Because the estimate
is written to `adata.obs`, it is preserved automatically when
split_dataset.py / select_genes.py later split cells into train/test and
subset genes (obs columns are untouched by that subsetting); they
only ever read Data/data.h5ad, never Data/data_complete.h5ad.

If `Data/data.h5ad` was already prepared with a pre-filtered gene set, the
literature marker genes may be missing from it. In that case, place the
unfiltered dataset alongside it as `Data/data_complete.h5ad`: only its counts
are used, to score the proliferation/death/senescence signatures (it is never
modified or written back), on the cells of `Data/data.h5ad` (matched by cell
name; every cell in Data/data.h5ad must be present in Data/data_complete.h5ad,
the converse need not hold) and with their annotations of Data/data.h5ad
(time, dataset_id, cell types), so that the anchoring is done on the analysed
cells; the resulting per-cell rates are written to Data/data.h5ad. If `Data/data_complete.h5ad` is absent,
`Data/data.h5ad` is used directly for both scoring and output if it has at
least MIN_GENES_SCORING (10,000) genes. With fewer, the signatures are not
scored (warning): the reference rates of the proliferation_rates sheet,
if given, are assigned uniformly per cell type; otherwise no rate is assigned.

Usage:
    python get_proliferation_rates.py -i <project_path>

    The parameter species (model_parameters sheet) selects the built-in marker gene
    lists. By default ('auto') it is detected from the gene nomenclature (mouse "Mki67"
    vs human "MKI67"), as in get_degradation_rates.py; set it to override the detection.
    senescence_gating = False disables the senescence gating of the proliferation score.

Required input files:
    - Data/data.h5ad: count matrix (all genes, pre-selection, unless
      Data/data_complete.h5ad is provided)

Optional input files:
    - Data/data_complete.h5ad: unfiltered count matrix used only to score the
      gene signatures when Data/data.h5ad has already been gene-filtered;
      never modified
    - Data/proliferation_signatures.csv|txt: proliferation marker genes,
      one per line or comma-separated (overrides the built-in default list
      for the chosen species)
    - Data/death_signatures.csv|txt: death marker genes, one per line or
      comma-separated (overrides the built-in default list for the chosen
      species)
    - Data/senescence_signatures.csv|txt: senescence/cell-cycle-arrest/
      quiescence marker genes, one per line or comma-separated (overrides
      the built-in default list for the chosen species). Used to gate the
      birth rate to ~0 for cells that are arrested/senescent rather than
      actively dying, even if they still carry residual cell-cycle gene
      expression — see CardamomOT.tools.estimate_proliferation. Pass
      an empty file (or set senescence_gating = False) to disable this and
      recover the plain birth - death estimate.
    - sheet proliferation_rates of Data/CardamomOT_inputs.xlsx: cell_type, net_rate_per_hour, and optionally
      birth_rate_per_hour / death_rate_per_hour, anchors the literature estimate's per-cell-type
      mean to a trusted population-level rate (birth and death separately where they are given,
      each scaled to its reference mean; otherwise the net rate is shifted). Grouping uses
      `adata.obs['cell_type_proliferation']` if present, else
      `cell_type_transition`, else `cell_type`; each sample is anchored on its own cells (its own rows, else the
      default ones); if no grouping is found or any cell type of a sample is missing from its rows, that sample
      keeps its unanchored estimate. `net_rate` must be in the same
      time unit as `adata.obs['time']` (hours, not moscot/WOT's day^-1
      convention — see CardamomOT.tools.estimate_proliferation), since it is
      blended directly with the (already hour^-1) literature estimate

Output files:
    - Data/data.h5ad: updated in place with obs['proliferation_net_rate'] (birth - death), and the birth and death
      rates obs['proliferation_birth_rate'], obs['proliferation_death_rate'] (birth: dilution of the proteins);
      without stimulus: obs['proliferation_*_rate_base']. Anchoring: the rates per cell type of the
      proliferation_rates sheet (as given: their consistency with the data is established upstream), done time by
      time (rates constant in time apart from the stimulus effects; all times together in CardamomOT-stat or without
      obs['time']); types with reference birth and death get both (estimate_proliferation.anchor_birth_death), the
      others their net rate, each change of the net rate being shared between birth and death by
      estimate_proliferation.split_net_change (w = birth / (birth + death)). The rates are those of the state of each
      cell at its time: the stimulus effects (RATEk) are added with the value of the stimulus at that time (untreated
      cells: natural rates; samples removed from the inference: their stimulus_test_schedule), a cell-type effect
      being shared in proportion to the rate it changes; the effects per unit of stimulus are kept
      (obs['proliferation_stimulus_*']) for the intervals of the inference.
      (Data/data_complete.h5ad, if used, is never written to)
"""
import sys; sys.path += ['../']
import os
from CardamomOT.run_options import parse_step_options, settings
import anndata as ad
import numpy as np
import pandas as pd

from CardamomOT import find_data_file, read_gene_list, resolve_cell_type_obs, ensure_raw_counts
from CardamomOT.inputs import input_dir, read_proliferation_rates
from CardamomOT.config import CELL_TYPE_OBS_KEYS
from CardamomOT.tools.estimate_proliferation import (
    estimate_growth_rates, combine_growth_rates_with_reference, split_net_change, anchor_birth_death,
)
from CardamomOT.inference.halflife_db import detect_species


# Genes needed in Data/data.h5ad to score the signatures without Data/data_complete.h5ad
from CardamomOT.stimulus_rates import MIN_GENES_SCORING  # whole transcriptome: library-size normalisation

def set_rates(obs, birth, death, suffix=''):
    """obs['proliferation_birth_rate'], ['proliferation_death_rate'] and ['proliferation_net_rate'] (= birth - death)."""
    obs['proliferation_birth_rate' + suffix] = np.asarray(birth, dtype=float)
    obs['proliferation_death_rate' + suffix] = np.asarray(death, dtype=float)
    obs['proliferation_net_rate' + suffix] = obs['proliferation_birth_rate' + suffix] - obs['proliferation_death_rate' + suffix]


def assign_proliferation_rates(adata, prolif_path, species='human', proliferation_genes=None,
                                death_genes=None, senescence_genes=None, senescence_gating=True, sample_specs=None,
                                times=None, induced=None):
    """
    Compute and assign adata.obs['proliferation_net_rate'] (birth − death) in place.

    Estimated per-cell from literature proliferation/death/senescence gene
    signatures (tools.estimate_proliferation), scored on the cells of each sample (dataset_id) alone. A senescence/arrest score
    gates the birth term toward 0 for cells that look arrested/senescent
    rather than actively proliferating, without touching the death term
    (see estimate_proliferation.estimate_growth_rates); set
    `senescence_gating=False` to disable this and recover the plain
    birth - death estimate.

    Birth and death are kept separately (obs['proliferation_birth_rate'], obs['proliferation_death_rate']).
    If `prolif_path` points to a per-cell-type reference table
    (sheet proliferation_rates: cell_type, rate), the
    literature estimate is anchored so its mean matches the reference rate
    within each cell type, keeping per-cell heterogeneity from the
    signature. Grouping uses `adata.obs['cell_type_proliferation']` if
    present, else `cell_type_transition`, else `cell_type`.
    `rate` must be expressed per hour, matching `adata.obs['time']`
    and the (already hour^-1) literature estimate it is blended with — see
    estimate_proliferation.estimate_growth_rates for why that estimate is in
    hours rather than moscot/WOT's native day^-1.

    sample_specs : {dataset_id: {'rates': {cell type: {'net', 'birth', 'death'}}}} anchors of the samples with their
    own (rows of proliferation_rates with a sample_id). Every sample is anchored on its own cells, on its own
    anchors, else on the default ones (prolif_path). A sample with neither keeps its literature estimate.

    times : (N,) time of each cell (NaN: unknown), or None. Given (temporal method), the anchoring is done time by
    time: the reference rates (without stimulus) are the mean rates of each cell type at every time, the rates being
    constant in time apart from the stimulus effects added afterwards (add_stimulus_effects). None (CardamomOT-stat,
    or no obs['time']): all the cells of a sample are anchored together.
    The literature estimate is the rate with the stimulus at the time of the cell: wherever it is kept (no anchors,
    partial anchors, no cell-type grouping), its natural part is used, the mean effect `induced` ((birth part, death
    part) (N,), induced_rates) removed in proportion within the cells of each type at each time of each sample (inverse
    of the sharing of add_stimulus_effects, which adds it back). A sample without anchors is then anchored time by time
    on its own mean natural birth and death rates over all times per cell type; with times None, it keeps its
    natural literature estimate.
    """
    prefix = "[get_proliferation_rates]"
    kwargs = {'species': species, 'senescence_gating': senescence_gating}
    if proliferation_genes is not None:
        kwargs['proliferation_genes'] = proliferation_genes
    if death_genes is not None:
        kwargs['death_genes'] = death_genes
    if senescence_genes is not None:
        kwargs['senescence_genes'] = senescence_genes

    # Signatures scored on the cells of each sample alone (control genes of score_genes binned on its own cells): the
    # rates of a sample do not depend on the other samples of the data
    samples = adata.obs['dataset_id'].astype(str).to_numpy() if 'dataset_id' in adata.obs else np.array(['0'] * adata.n_obs)
    net_lit, birth_lit, death_lit = (np.zeros(adata.n_obs) for _ in range(3))
    for sid in np.unique(samples):
        m = samples == sid
        net_lit[m], birth_lit[m], death_lit[m] = estimate_growth_rates(adata[m], return_components=True, **kwargs)
    print(f"{prefix} Estimated literature-based proliferation rates for "
          f"{len(net_lit)} cells (mean net={net_lit.mean():.4f}, birth={birth_lit.mean():.4f}, "
          f"death={death_lit.mean():.4f} h^-1)")
    set_rates(adata.obs, birth_lit, death_lit)
    # Natural part of the literature estimate (stimulus effects at the time of the cells removed), kept where no anchor
    ct_key = resolve_cell_type_obs(adata, 'proliferation')
    labels_all = adata.obs[ct_key].astype(str).to_numpy() if ct_key else np.full(adata.n_obs, '')
    t_cell = times if times is not None else adata.obs['time'] if 'time' in adata.obs else np.zeros(adata.n_obs)
    t_grp = np.nan_to_num(pd.to_numeric(pd.Series(np.asarray(t_cell)), errors='coerce').to_numpy(dtype=float), nan=-1.0)
    birth_nat, death_nat = birth_lit.copy(), death_lit.copy()
    if induced is not None:
        group = np.array([f'{s_}|{t_:g}|{c_}' for s_, t_, c_ in zip(samples, t_grp, labels_all)])
        for g_ in np.unique(group):
            mg = group == g_
            for nat, lit, ind in ((birth_nat, birth_lit, induced[0]), (death_nat, death_lit, induced[1])):
                mean = lit[mg].mean()
                if mean > 1e-12 and np.any(ind[mg]):
                    nat[mg] = lit[mg] * max(mean - ind[mg].mean(), 0.0) / mean
    specs = {s: v for s, v in (sample_specs or {}).items() if s in set(samples)}
    for s_ in sorted(set(sample_specs or {}) - set(specs)):
        print(f"{prefix} Warning: anchors given for sample {s_} absent from the data: ignored")
    # Groups: each sample on its own cells, with its own specification, else the default rows
    groups = [(sid, samples == sid, specs[sid]['rates'] if specs.get(sid, {}).get('rates') else prolif_path)
              for sid in np.unique(samples)]
    birth, death = birth_lit.copy(), death_lit.copy()
    t_key = None if times is None else np.where(np.isfinite(np.asarray(times, dtype=float)), np.asarray(times, dtype=float), -np.inf)
    # Without anchors (temporal method): each sample is anchored time by time on its own mean natural rates over all
    # times per type
    if t_key is not None and ct_key is not None:
        expanded = []
        for name, m, rates_src in groups:
            if rates_src is not None:
                expanded.append((name, m, rates_src))
                continue
            spec = {}
            for c in np.unique(labels_all[m]):
                mc = m & (labels_all == c)
                b_c, d_c = float(np.mean(birth_nat[mc])), float(np.mean(death_nat[mc]))
                spec[c] = dict(net=b_c - d_c, birth=b_c, death=d_c)
            print(f"{prefix} [{name}] No anchors: re-anchored time by time on its mean natural rates over all times "
                  f"per '{ct_key}'" + (" (stimulus effects at the time of the cells subtracted)" if induced is not None
                                       and (np.any(induced[0][m]) or np.any(induced[1][m])) else ''))
            expanded.append((name, m, spec))
        groups = expanded
    for name, m, rates_src in groups:
        # Time by time (temporal method), else all the cells of the group together
        blocks = [m] if t_key is None else [m & (t_key == t) for t in np.unique(t_key[m])]
        for k, mt in enumerate(blocks):
            label = name if len(groups) > 1 else None
            if t_key is not None:
                t = t_key[mt][0]
                label = f"{label + ', ' if label else ''}t={'?' if np.isinf(t) else f'{t:g}'}"
            b_g, d_g = anchor_group(prefix, label, adata[mt], net_lit[mt], birth_lit[mt], death_lit[mt], rates_src,
                                    quiet=k > 0, natural=(birth_nat[mt], death_nat[mt]))
            birth[mt], death[mt] = b_g, d_g
        if t_key is not None and len(blocks) > 1:
            print(f"{prefix}{' [' + name + ']' if len(groups) > 1 else ''} Anchored time by time ({len(blocks)} times)")
    set_rates(adata.obs, birth, death)
    print(f"{prefix} Rates: mean birth {adata.obs['proliferation_birth_rate'].mean():.4f}, death "
          f"{adata.obs['proliferation_death_rate'].mean():.4f}, net {adata.obs['proliferation_net_rate'].mean():.4f} h^-1")


def anchor_group(prefix, name, adata, net_lit, birth_lit, death_lit, rates_src, quiet=False, natural=None):
    """
    Birth and death of the cells of one group (`adata`: its cells; name: sample, None if single group) anchored on the
    reference rates `rates_src` (path of a rates table, or {cell type: {'net', 'birth', 'death'}}): in a cell type
    with reference birth and death, each is scaled to its reference mean (anchor_birth_death); in the others, the mean
    net rate is shifted to its reference, the change shared between birth and death (split_net_change). Without
    anchors, `natural` (birth, death): the natural part of the literature estimate (default: the estimate itself).
    quiet: no summary line (blocks after the first time of a group).
    """
    tag = prefix if name is None else f"{prefix} [{name}]"
    natural = (birth_lit, death_lit) if natural is None else natural
    anchors = read_anchors(adata, rates_src, "using unanchored literature estimate") if rates_src is not None else None
    if rates_src is None:
        print(f"{tag} No anchors: literature-only proliferation rate estimate")
        return natural
    if anchors is None:
        return natural
    labels, spec, celltype_col = anchors
    rates = {c: v['net'] for c, v in spec.items()}
    bd = {c: (v['birth'], v['death']) for c, v in spec.items() if v.get('birth') is not None and v.get('death') is not None}
    change = combine_growth_rates_with_reference(net_lit, labels, rates) - net_lit
    birth, death = split_net_change(birth_lit, death_lit, change)
    # Types with birth / death references: scaled from the literature estimate, not from the net-shifted one
    m = np.isin(labels, list(bd))
    birth[m], death[m] = anchor_birth_death(birth_lit[m], death_lit[m], labels[m], bd)
    if not quiet:
        print(f"{tag} Anchored literature proliferation rates per '{celltype_col}' ({len(rates)} types)"
              + (f" ({rates_src})" if isinstance(rates_src, str) else '') + ': '
              + ', '.join(f'{c} net {v["net"]:+.4f}' + (f' (birth {bd[c][0]:.4f}, death {bd[c][1]:.4f})' if c in bd else '')
                          for c, v in spec.items()) + ' h^-1')
    return birth, death


def stimulus_parts(adata_target, adata_score, p):
    """
    Effects of the inference stimuli (RATEk) on the cells of adata_target, None without any: (n_k, values of the
    stimuli at the time of each cell (n, n_k) in the schedule of its sample (stimulus_rates.state_values: test
    schedule for the samples removed from the inference), cell types, their obs key, {sample: {k: ({type: delta},
    [(signature, delta)])}} (effects of the sample: row all, or its own rows), {signature: (n,) score on the mRNA of
    the scored cells}, sample of each cell (n,)).
    """
    from CardamomOT.stimulus_rates import load_effects, state_values, split_effects, signature_scores
    obs = adata_target.obs
    samples = obs['dataset_id'].astype(str).to_numpy() if 'dataset_id' in obs else np.array(['0'] * len(obs))
    effects = {s: load_effects(p, s) for s in np.unique(samples)}
    if not any(effects.values()):
        return None
    n_k = max(k for e in effects.values() for k in e)
    U = state_values(p, obs, n_k)[:, :n_k]
    ct_key = resolve_cell_type_obs(adata_target, 'proliferation')
    labels = obs[ct_key].astype(str).to_numpy() if ct_key else np.array([''] * len(obs))
    parts = {s: split_effects(e, np.unique(labels) if ct_key else []) for s, e in effects.items()}
    targets = [target for ps in parts.values() for _, sig in ps.values() for target, _ in sig]
    scores = {}
    if targets:
        score_cells = adata_score[adata_target.obs_names]
        scores = signature_scores(score_cells.X, list(adata_score.var_names), targets, p,
                                  normalize=score_cells.n_vars >= MIN_GENES_SCORING)
    return n_k, U, labels, ct_key, parts, scores, samples


def induced_rates(adata, p):
    """(birth part, death part) (N,) of the mean effect of the inference stimuli at the time of each cell: u(t) |delta|
    for a cell-type target, u(t) |delta| score for a signature (as add_stimulus_effects, before the sharing within the
    cell types, which keeps their means); zeros without effects."""
    ind_b, ind_d = np.zeros(adata.n_obs), np.zeros(adata.n_obs)
    sp = stimulus_parts(adata, adata, p)
    if sp is None:
        return ind_b, ind_d
    _, U, labels, _, parts, scores, samples = sp
    for s, ps in parts.items():
        ms = samples == s
        for k, (ct, sig) in ps.items():
            for c, d in ct.items():
                (ind_b if d > 0 else ind_d)[:] += U[:, k - 1] * abs(d) * ((labels == c) & ms)
            for target, d in sig:
                (ind_b if d > 0 else ind_d)[:] += U[:, k - 1] * abs(d) * scores[target] * ms
    return ind_b, ind_d


def add_stimulus_effects(adata_target, adata_score, p):
    """
    Effects of the inference stimuli on the rates (RATEk of perturbation_inference: row all, or the rows of the sample
    of each cell): the rates computed so
    far (anchors of proliferation_rates) are those without stimulus. The rates of a cell are those of its state at its
    own time t: it gets u_k(t) * effect, u_k(t) being the value of stimulus k at t in the schedule of its sample (0 for
    the untreated cells; for the samples removed from the inference, the schedule of their validation,
    stimulus_test_schedule). Effect of a cell-type target (cell_type_proliferation): delta shared in proportion to the
    rate it adds to (birth if delta > 0, death if < 0) within the cells of the type at the same time of the same
    sample, so that the mean of the type is its rate + u delta (its natural plus induced rate) and its signatures
    give the heterogeneity of both; signature targets: delta x score on the mRNA (log1p counts, normalised by the
    library size if the transcriptome is scored). The rates without stimulus are kept in obs['proliferation_*_rate_base'],
    the effect of each stimulus k per unit of its value in obs['proliferation_stimulus_birth_<k>'] and
    obs['proliferation_stimulus_death_<k>'] (also for the untreated cells): the inference adds, over each interval,
    the value of the stimulus of the interval (NetworkModel._set_interval_rates).
    """
    prefix = "[get_proliferation_rates]"
    obs = adata_target.obs
    stale = [c for c in obs.columns if c.startswith('proliferation_stimulus_') or
             (c.startswith('proliferation_') and c.endswith('_rate_base'))]
    if stale:
        adata_target.obs = obs.drop(columns=stale)
        obs = adata_target.obs
    sp = stimulus_parts(adata_target, adata_score, p)
    if sp is None or 'proliferation_net_rate' not in adata_target.obs:
        return
    n_k, U, labels, ct_key, parts, scores, samples = sp
    base = obs['proliferation_net_rate'].to_numpy(dtype=float)
    times = obs['time'].to_numpy(dtype=float) if 'time' in obs else np.zeros(len(obs))
    tu = np.sort(np.unique(times))
    # Effect of each stimulus per unit of its value on each cell: birth part (positive deltas) and death part
    # (negative deltas, pure death)
    if 'proliferation_birth_rate' not in obs:  # no signatures: birth = max(net, 0) (negligible death)
        set_rates(obs, np.maximum(base, 0.0), np.maximum(-base, 0.0))
    b0, d0 = obs['proliferation_birth_rate'].to_numpy(dtype=float), obs['proliferation_death_rate'].to_numpy(dtype=float)
    # Share of each cell in the rate of its type at its time in its sample (mean 1; uniform if the type has none)
    group = np.array([f'{s}|{t:g}|{lab}' for s, t, lab in zip(samples, times, labels)])

    def share(x):
        out = np.ones(len(x))
        for g in np.unique(group):
            m = group == g
            mean = x[m].mean()
            if mean > 1e-12:
                out[m] = x[m] / mean
        return out

    sh_b, sh_d = share(b0), share(d0)
    eb, ed = np.zeros((len(obs), n_k)), np.zeros((len(obs), n_k))
    kind = lambda d: 'birth' if d > 0 else 'death' if d < 0 else 'no effect'
    for s, ps in parts.items():
        ms = samples == s
        tag = f" in {s}" if len(parts) > 1 else ""
        for k, (ct, sig) in ps.items():
            for c, d in ct.items():
                if d > 0:
                    eb[:, k - 1] += d * ((labels == c) & ms) * sh_b
                else:
                    ed[:, k - 1] += -d * ((labels == c) & ms) * sh_d
                print(f"{prefix} Stimulus {k}{tag}: {d:+g} h^-1 on '{c}' ({ct_key}), scaled by its schedule ({kind(d)}"
                      + (f", shared in proportion to the {kind(d)} rate of the cells)" if d else ")"))
            for target, d in sig:
                (eb if d > 0 else ed)[:, k - 1] += abs(d) * scores[target] * ms
                print(f"{prefix} Stimulus {k}{tag}: {target} {d:+g} h^-1 at maximal mRNA score ({kind(d)})")
    for k in range(n_k):
        obs[f'proliferation_stimulus_birth_{k + 1}'] = eb[:, k]
        obs[f'proliferation_stimulus_death_{k + 1}'] = ed[:, k]
    add_b, add_d = (U[:, :n_k] * eb).sum(axis=1), (U[:, :n_k] * ed).sum(axis=1)
    add = add_b - add_d
    set_rates(obs, b0, d0, suffix='_base')
    set_rates(obs, b0 + add_b, d0 + add_d)
    if ct_key:
        for c in np.unique(labels):
            for t in tu:
                m = (labels == c) & (times == t)
                if m.any():
                    print(f"{prefix}   {c} t={t:g}: mean rate without stimulus {base[m].mean():.5f}, with the stimulus "
                          f"at its time {(base + add)[m].mean():.5f} h^-1 (death {(d0 + add_d)[m].mean():.5f})")


def read_anchors(adata, prolif_path, fallback):
    """
    Reference rates per cell type (sheet proliferation_rates: net, and optionally birth and death, per hour),
    grouped by adata.obs['cell_type_proliferation'], else 'cell_type_transition', else 'cell_type'.
    prolif_path: path of the exported table, or {cell type: {'net', 'birth', 'death'}}.
    Returns (labels per cell, {cell type: {'net', 'birth', 'death'}}, column) or None (message ending with fallback)
    when no grouping exists or a cell type has no net rate (partial anchoring is worse than none).
    """
    prefix = "[get_proliferation_rates]"
    celltype_col = resolve_cell_type_obs(adata, 'proliferation')
    if celltype_col is None:
        print(f"{prefix} Warning: found {prolif_path} but adata.obs has none of "
              f"{list(CELL_TYPE_OBS_KEYS['proliferation'])}; {fallback}")
        return None
    user_rates = read_proliferation_rates(prolif_path)
    if isinstance(prolif_path, dict):
        prolif_path = 'the anchors of the sample'
    labels = adata.obs[celltype_col].astype(str).to_numpy(dtype=object)
    # Cell types matched case-insensitively
    by_lower = {k.lower(): v for k, v in user_rates.items() if v.get('net') is not None}
    user_rates = {c: by_lower[c.lower()] for c in sorted(set(labels)) if c.lower() in by_lower}
    missing = sorted(set(labels) - set(user_rates))
    if missing:
        print(f"{prefix} Warning: cell type(s) {missing} of adata.obs['{celltype_col}'] "
              f"not found in {prolif_path}; {fallback}")
        return None
    return labels, user_rates, celltype_col


def main(argv):
    """
    Estimate and assign per-cell net proliferation rates.

    Args:
        argv: Command-line arguments (-i <project>; species, senescence_gating: model_parameters sheet).

    Returns:
        None. Updates Data/data.h5ad in place with obs['proliferation_net_rate'].
    """
    opts = parse_step_options(argv, 'get_proliferation_rates', __doc__)
    cfg = settings(opts)
    species, senescence_gating = str(cfg.species).strip().lower(), bool(cfg.senescence_gating)
    if species not in ("auto", "human", "mouse"):
        print(f"[get_proliferation_rates] Error: species must be auto, human or mouse (got '{species}')")
        sys.exit(1)

    p = opts.p
    data_dir = os.path.join(p, 'Data')  # data files; the run inputs are read from input_dir(p)

    # Data/data.h5ad is always the file that gets updated.
    data_path = os.path.join(data_dir, 'data.h5ad')
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        adata_target = ad.read_h5ad(data_path)
        print(f"[get_proliferation_rates] Loaded dataset from {data_path}")
        print(f"[get_proliferation_rates] Dataset contains "
              f"{adata_target.shape[0]} cells and {adata_target.shape[1]} genes")
    except FileNotFoundError as e:
        print(f"[get_proliferation_rates] Error: {e}")
        print(f"[get_proliferation_rates] Please ensure Data/data.h5ad exists in {p}")
        sys.exit(1)

    if 'proliferation_net_rate' in adata_target.obs.columns:
        print("[get_proliferation_rates] adata.obs['proliferation_net_rate'] "
              "already present in Data/data.h5ad; recomputing and overwriting it")

    # Data/data_complete.h5ad, if present, is used only to score the gene
    # signatures (e.g. when Data/data.h5ad was already gene-filtered) and is
    # never written to.
    complete_path = os.path.join(data_dir, 'data_complete.h5ad')
    using_complete = os.path.exists(complete_path)
    # Scoring needs raw counts (raw-count layer used if X is log-normalised; files are never modified)
    try:
        if using_complete:
            adata_score = ensure_raw_counts(ad.read_h5ad(complete_path), complete_path)
            missing = adata_target.obs_names.difference(adata_score.obs_names)
            if len(missing) > 0:
                print(f"[get_proliferation_rates] Error: {len(missing)} cell(s) in Data/data.h5ad "
                      f"were not found in Data/data_complete.h5ad (first missing: {list(missing[:5])})")
                sys.exit(1)
            # Counts of data_complete.h5ad on the cells of Data/data.h5ad, with their annotations (time, sample,
            # cell types): the anchoring is done on the analysed cells
            adata_score = adata_score[adata_target.obs_names].copy()
            adata_score.obs = adata_target.obs.copy()
            print(f"[get_proliferation_rates] Found {complete_path}; scoring gene signatures on its counts "
                  f"({adata_score.shape[0]} cells of Data/data.h5ad, {adata_score.shape[1]} genes)")
        else:
            adata_score = ensure_raw_counts(adata_target, data_path)
    except ValueError as e:
        print(f"[get_proliferation_rates] Error: {e}")
        sys.exit(1)

    prolif_path = find_data_file(input_dir(p), 'proliferation_rates')
    # Anchors per sample (dataset_id); the default rows apply to the others
    from CardamomOT.inputs import proliferation_sample_anchors
    sample_specs = {}
    for sid, rates in proliferation_sample_anchors(p).items():
        sample_specs.setdefault(sid, {})['rates'] = rates

    # Too few genes to score the signatures: no estimate, except the per-cell-type anchors if given
    if not using_complete and adata_target.shape[1] < MIN_GENES_SCORING:
        print(f"[get_proliferation_rates] Warning: Data/data.h5ad has only {adata_target.shape[1]} genes "
              f"(< {MIN_GENES_SCORING}) and there is no Data/data_complete.h5ad: the gene signatures "
              f"cannot be scored reliably, no literature estimate")
        samples = (adata_target.obs['dataset_id'].astype(str).to_numpy() if 'dataset_id' in adata_target.obs
                   else np.array(['0'] * adata_target.n_obs))
        net, birth, death = (np.full(adata_target.n_obs, np.nan) for _ in range(3))
        # Per sample: its own anchors, else the default ones
        for sid in np.unique(samples):
            m = samples == sid
            own = sample_specs.get(sid, {})
            src = own.get('rates') or prolif_path
            anchors = read_anchors(adata_target[m], src, "no proliferation rate assigned") if src is not None else None
            if anchors is not None:
                labels, spec, celltype_col = anchors
                get = lambda key: np.array([np.nan if spec[l].get(key) is None else spec[l][key] for l in labels], dtype=float)
                net[m], birth[m], death[m] = get('net'), get('birth'), get('death')
                print(f"[get_proliferation_rates] Rates of {'the sample ' + sid if isinstance(src, dict) else src} assigned "
                      f"uniformly per '{celltype_col}' to {sid}")
        if np.isfinite(net).any():
            net = np.where(np.isfinite(net), net, 0.0)
            # Reference birth / death where given, else negligible death
            given = np.isfinite(birth) & np.isfinite(death)
            set_rates(adata_target.obs, np.where(given, birth, np.maximum(net, 0.0)),
                      np.where(given, death, np.maximum(-net, 0.0)))
            add_stimulus_effects(adata_target, adata_score, p)
        elif 'proliferation_net_rate' in adata_target.obs.columns:
            print("[get_proliferation_rates] Warning: keeping the existing adata.obs['proliferation_net_rate'] "
                  "(computed beforehand, e.g. by another method)")
        else:
            print("[get_proliferation_rates] No proliferation_rates sheet filled: no rate assigned")
        adata_target.write(data_path)
        print(f"[get_proliferation_rates] Saved updated dataset to {data_path}")
        return
    if not using_complete:
        print(f"[get_proliferation_rates] No Data/data_complete.h5ad; scoring on Data/data.h5ad "
              f"({adata_target.shape[1]} genes >= {MIN_GENES_SCORING})")

    if species == "auto":
        species, hits = detect_species(adata_score.var_names)
        print(f"[get_proliferation_rates] Detected species='{species}' from gene names "
              f"(official-name matches: {hits}); set the parameter species to override")
    else:
        print(f"[get_proliferation_rates] Using species='{species}'")
    print(f"[get_proliferation_rates] Senescence gating: "
          f"{'enabled' if senescence_gating else 'disabled (plain birth - death)'}")

    proliferation_genes = None
    proliferation_genes_path = find_data_file(input_dir(p), 'proliferation_signatures')
    if proliferation_genes_path is not None:
        proliferation_genes = read_gene_list(proliferation_genes_path)
        print(f"[get_proliferation_rates] Loaded {len(proliferation_genes)} proliferation marker "
              f"genes from {proliferation_genes_path}")

    death_genes = None
    death_genes_path = find_data_file(input_dir(p), 'death_signatures')
    if death_genes_path is not None:
        death_genes = read_gene_list(death_genes_path)
        print(f"[get_proliferation_rates] Loaded {len(death_genes)} death marker genes "
              f"from {death_genes_path}")

    senescence_genes = None
    senescence_genes_path = find_data_file(input_dir(p), 'senescence_signatures')
    if senescence_genes_path is not None:
        senescence_genes = read_gene_list(senescence_genes_path)
        print(f"[get_proliferation_rates] Loaded {len(senescence_genes)} senescence/arrest marker "
              f"genes from {senescence_genes_path}")

    # Anchoring time by time (temporal method); all times together in CardamomOT-stat or without obs['time']
    stat = str(cfg.CardamomOT_method).strip().lower() == 'stat' or 'time' not in adata_target.obs
    times = None
    if not stat:
        t_src = adata_score.obs['time'] if 'time' in adata_score.obs else adata_target.obs['time'].reindex(adata_score.obs_names)
        times = pd.to_numeric(t_src, errors='coerce').to_numpy(dtype=float)
    print(f"[get_proliferation_rates] Anchoring {'of all the cells of each sample together (CardamomOT-stat)' if stat else 'time by time'}")
    assign_proliferation_rates(adata_score, prolif_path, species=species,
                                proliferation_genes=proliferation_genes, death_genes=death_genes,
                                senescence_genes=senescence_genes, senescence_gating=senescence_gating,
                                sample_specs=sample_specs, times=times,
                                induced=induced_rates(adata_score, p))

    if using_complete:
        missing = adata_target.obs_names.difference(adata_score.obs_names)
        if len(missing) > 0:
            print(f"[get_proliferation_rates] Error: {len(missing)} cell(s) in Data/data.h5ad "
                  f"were not found in Data/data_complete.h5ad; cannot map proliferation rates "
                  f"(first missing: {list(missing[:5])})")
            sys.exit(1)
        for c in ('proliferation_net_rate', 'proliferation_birth_rate', 'proliferation_death_rate'):
            adata_target.obs[c] = adata_score.obs.loc[adata_target.obs_names, c].to_numpy()
        print("[get_proliferation_rates] Mapped proliferation rates from Data/data_complete.h5ad "
              "onto Data/data.h5ad cells")
    elif adata_score is not adata_target:
        # Scored on the raw-count layer: only the rates go back to Data/data.h5ad (X untouched)
        for c in ('proliferation_net_rate', 'proliferation_birth_rate', 'proliferation_death_rate'):
            adata_target.obs[c] = adata_score.obs[c].to_numpy()
    add_stimulus_effects(adata_target, adata_score, p)

    try:
        adata_target.write(data_path)
        print(f"[get_proliferation_rates] Saved updated dataset to {data_path}")
    except Exception as e:
        print(f"[get_proliferation_rates] Error saving dataset: {e}")
        sys.exit(1)

    print("[get_proliferation_rates] Proliferation rate assignment completed successfully")


if __name__ == "__main__":
    main(sys.argv[1:])