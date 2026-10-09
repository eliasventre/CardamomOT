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
unfiltered dataset alongside it as `Data/data_complete.h5ad`: it is used ONLY
to score the proliferation/death/senescence signatures (and is never
modified or written back), and the resulting per-cell rates are mapped onto
the matching cells of `Data/data.h5ad` by cell name (every cell in
Data/data.h5ad is assumed to also be present in Data/data_complete.h5ad; the
converse need not hold). If `Data/data_complete.h5ad` is absent,
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
    - sheet proliferation_rates of Data/CardamomOT_inputs.xlsx: cell_type, net_rate_per_hour
      anchors the literature estimate's per-cell-type
      mean to a trusted population-level rate. Grouping uses
      `adata.obs['cell_type_proliferation']` if present, else
      `cell_type_transition`, else `cell_type`; if no grouping is found or
      any cell type is missing from the file, the unanchored estimate is
      kept for all cells. `net_rate` must be in the same
      time unit as `adata.obs['time']` (hours, not moscot/WOT's day^-1
      convention — see CardamomOT.tools.estimate_proliferation), since it is
      blended directly with the (already hour^-1) literature estimate

Output files:
    - Data/data.h5ad: updated in place with obs['proliferation_net_rate'] (birth - death), and the birth and death
      rates obs['proliferation_birth_rate'], obs['proliferation_death_rate'] (birth: dilution of the proteins);
      without stimulus: obs['proliferation_*_rate_base']. Anchoring: the net rates per cell type of the
      proliferation_rates sheet, as corrected per sample by fit_population_anchors.py (cardamomOT/population_anchors.json)
      if it was run; each change of the net rate (anchoring, stimulus) is shared between birth and death by
      estimate_proliferation.split_net_change (w = birth / (birth + death)).
      (Data/data_complete.h5ad, if used, is never written to)
"""
import sys; sys.path += ['../']
import os
from CardamomOT.run_options import parse_step_options, settings
import anndata as ad
import numpy as np
import pandas as pd

from CardamomOT import find_data_file, read_gene_list, resolve_cell_type_obs, ensure_raw_counts
from CardamomOT.inputs import input_dir
from CardamomOT.config import CELL_TYPE_OBS_KEYS
from CardamomOT.tools.estimate_proliferation import (
    estimate_growth_rates, combine_growth_rates_with_reference, split_net_change,
)
from CardamomOT.inference.halflife_db import detect_species


# Genes needed in Data/data.h5ad to score the signatures without Data/data_complete.h5ad
MIN_GENES_SCORING = 10000

def set_rates(obs, birth, death, suffix=''):
    """obs['proliferation_birth_rate'], ['proliferation_death_rate'] and ['proliferation_net_rate'] (= birth - death)."""
    obs['proliferation_birth_rate' + suffix] = np.asarray(birth, dtype=float)
    obs['proliferation_death_rate' + suffix] = np.asarray(death, dtype=float)
    obs['proliferation_net_rate' + suffix] = obs['proliferation_birth_rate' + suffix] - obs['proliferation_death_rate' + suffix]


def assign_proliferation_rates(adata, prolif_path, species='human', proliferation_genes=None,
                                death_genes=None, senescence_genes=None, senescence_gating=True, sample_specs=None):
    """
    Compute and assign adata.obs['proliferation_net_rate'] (birth − death) in place.

    Estimated per-cell from literature proliferation/death/senescence gene
    signatures (tools.estimate_proliferation). A senescence/arrest score
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

    sample_specs : {dataset_id: {'rates': {cell type: rate}}} anchors of the samples with their own (rows of
    proliferation_rates with a sample_id, or corrected by fit_population_anchors.py); each of them is anchored on its
    own cells, the other samples together on the default (prolif_path). A sample with neither own nor default anchors
    keeps its literature estimate.
    """
    prefix = "[get_proliferation_rates]"
    kwargs = {'species': species, 'senescence_gating': senescence_gating}
    if proliferation_genes is not None:
        kwargs['proliferation_genes'] = proliferation_genes
    if death_genes is not None:
        kwargs['death_genes'] = death_genes
    if senescence_genes is not None:
        kwargs['senescence_genes'] = senescence_genes

    net_lit, birth_lit, death_lit = estimate_growth_rates(adata, return_components=True, **kwargs)
    print(f"{prefix} Estimated literature-based proliferation rates for "
          f"{len(net_lit)} cells (mean net={net_lit.mean():.4f}, birth={birth_lit.mean():.4f}, "
          f"death={death_lit.mean():.4f} h^-1)")
    set_rates(adata.obs, birth_lit, death_lit)
    samples = adata.obs['dataset_id'].astype(str).to_numpy() if 'dataset_id' in adata.obs else np.array(['0'] * len(net_lit))
    specs = {s: v for s, v in (sample_specs or {}).items() if s in set(samples)}
    for s_ in sorted(set(sample_specs or {}) - set(specs)):
        print(f"{prefix} Warning: anchors given for sample {s_} absent from the data: ignored")
    own = np.isin(samples, list(specs))
    # Groups: each sample with its own specification, and the other samples together (the default)
    groups = [('default', ~own, prolif_path)] if (~own).any() else []
    for sid, v in specs.items():
        groups.append((sid, samples == sid, v.get('rates') if v.get('rates') else prolif_path))
    birth, death = birth_lit.copy(), death_lit.copy()
    for name, m, rates_src in groups:
        b_g, d_g = anchor_group(prefix, name if len(groups) > 1 else None, adata[m], net_lit[m], birth_lit[m], death_lit[m],
                                rates_src)
        birth[m], death[m] = b_g, d_g
    set_rates(adata.obs, birth, death)
    print(f"{prefix} Rates: mean birth {adata.obs['proliferation_birth_rate'].mean():.4f}, death "
          f"{adata.obs['proliferation_death_rate'].mean():.4f}, net {adata.obs['proliferation_net_rate'].mean():.4f} h^-1")


def anchor_group(prefix, name, adata, net_lit, birth_lit, death_lit, rates_src):
    """
    Birth and death of the cells of one group (`adata`: its cells; name: sample, None if single group) anchored on the
    reference net rates `rates_src` (path of a rates table, or {cell type: rate}): the mean net rate of each cell type
    is its reference, the change shared between birth and death (split_net_change); the literature estimate without.
    """
    tag = prefix if name is None else f"{prefix} [{name}]"
    anchors = read_anchors(adata, rates_src, "using unanchored literature estimate") if rates_src is not None else None
    if rates_src is None:
        print(f"{tag} No anchors: literature-only proliferation rate estimate")
        return birth_lit, death_lit
    if anchors is None:
        return birth_lit, death_lit
    labels, rates, celltype_col = anchors
    change = combine_growth_rates_with_reference(net_lit, labels, rates) - net_lit
    print(f"{tag} Anchored literature proliferation rates per '{celltype_col}' ({len(rates)} types)"
          + (f": {rates_src}" if isinstance(rates_src, str) else ': ' + ', '.join(f'{c} {r:+.4f}' for c, r in rates.items())))
    return split_net_change(birth_lit, death_lit, change)


def add_stimulus_effects(adata_target, adata_score, p):
    """
    Effects of the inference stimuli on the net rate (RATEk of perturbation_inference, row all): the
    rates computed so far (anchors of proliferation_rates) are those without stimulus; each cell gets
    u_k(t) * delta for its cell type (cell_type_proliferation) and for the signatures (score on its mRNA:
    log1p counts, normalised by the library size if the transcriptome is scored). The rate without
    stimulus is kept in obs['proliferation_net_rate_base'] (and birth / death in obs['proliferation_*_rate_base']).
    The change of the net rate is shared between birth and death by split_net_change.
    """
    from CardamomOT.stimulus_rates import load_effects, cell_values, split_effects, log_counts, signature_score
    from CardamomOT.tools.perturbations import rate_target_genes
    prefix = "[get_proliferation_rates]"
    effects = load_effects(p)
    if not effects or 'proliferation_net_rate' not in adata_target.obs:
        return
    obs = adata_target.obs
    base = obs['proliferation_net_rate'].to_numpy(dtype=float)
    times = obs['time'].to_numpy(dtype=float) if 'time' in obs else np.zeros(len(obs))
    tu = np.sort(np.unique(times))
    # Value over the interval that starts at each cell's time (next timepoint), schedule of its sample
    t_next = tu[np.minimum(np.searchsorted(tu, times) + 1, len(tu) - 1)]
    samples = obs['dataset_id'].astype(str).to_numpy() if 'dataset_id' in obs else np.array(['0'] * len(obs))
    U = cell_values(p, t_next, samples, tu, max(effects))
    ct_key = resolve_cell_type_obs(adata_target, 'proliferation')
    labels = obs[ct_key].astype(str).to_numpy() if ct_key else np.array([''] * len(obs))
    add = np.zeros(len(obs))
    L = None
    for k, (ct, sig) in split_effects(effects, np.unique(labels) if ct_key else []).items():
        for c, d in ct.items():
            add += U[:, k - 1] * d * (labels == c)
            print(f"{prefix} Stimulus {k}: {d:+g} h^-1 on '{c}' ({ct_key}), scaled by its schedule")
        for target, d in sig:
            if L is None:
                score_cells = adata_score[adata_target.obs_names]
                L = log_counts(score_cells.X, normalize=score_cells.n_vars >= MIN_GENES_SCORING)
            genes = list(adata_score.var_names)
            idx = [genes.index(g) for g in rate_target_genes(target, genes, input_dir(p))]
            score = signature_score(L, idx, np.maximum(np.percentile(L[:, idx], 99, axis=0), 1e-6))
            add += U[:, k - 1] * d * score
            print(f"{prefix} Stimulus {k}: {target} ({len(idx)} genes) {d:+g} h^-1 at maximal mRNA score")
    if 'proliferation_birth_rate' not in obs:  # no signatures: birth = max(net, 0) (negligible death)
        set_rates(obs, np.maximum(base, 0.0), np.maximum(-base, 0.0))
    b0, d0 = obs['proliferation_birth_rate'].to_numpy(dtype=float), obs['proliferation_death_rate'].to_numpy(dtype=float)
    set_rates(obs, b0, d0, suffix='_base')
    set_rates(obs, *split_net_change(b0, d0, add))
    if ct_key:
        for c in np.unique(labels):
            m = labels == c
            print(f"{prefix}   {c}: mean rate without stimulus {base[m].mean():.5f}, with the inference "
                  f"schedule {(base + add)[m].mean():.5f} h^-1")


def read_anchors(adata, prolif_path, fallback):
    """
    Reference rates per cell type (sheet proliferation_rates: cell_type, rate per hour),
    grouped by adata.obs['cell_type_proliferation'], else 'cell_type_transition', else 'cell_type'.
    Returns (labels per cell, {cell type: rate}, column) or None (message ending with fallback)
    when no grouping exists or a cell type has no rate (partial anchoring is worse than none).
    """
    prefix = "[get_proliferation_rates]"
    celltype_col = resolve_cell_type_obs(adata, 'proliferation')
    if celltype_col is None:
        print(f"{prefix} Warning: found {prolif_path} but adata.obs has none of "
              f"{list(CELL_TYPE_OBS_KEYS['proliferation'])}; {fallback}")
        return None
    if isinstance(prolif_path, dict):
        user_rates = pd.Series({str(k).strip(): float(v) for k, v in prolif_path.items()})
        prolif_path = 'the anchors of the sample'
    else:
        user_rates = pd.read_csv(prolif_path, sep=None, engine='python', header=None, index_col=0).iloc[:, 0]
        user_rates.index = user_rates.index.astype(str).str.strip()
    labels = adata.obs[celltype_col].astype(str).values
    # Cell types matched case-insensitively
    by_lower = {k.lower(): float(v) for k, v in user_rates.items()}
    user_rates = pd.Series({c: by_lower[c.lower()] for c in set(labels) if c.lower() in by_lower})
    missing = sorted(set(labels) - set(user_rates.index))
    if missing:
        print(f"{prefix} Warning: cell type(s) {missing} of adata.obs['{celltype_col}'] "
              f"not found in {prolif_path}; {fallback}")
        return None
    return labels, user_rates.to_dict(), celltype_col


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
            print(f"[get_proliferation_rates] Found {complete_path}; scoring gene signatures on it "
                  f"({adata_score.shape[0]} cells, {adata_score.shape[1]} genes) instead of Data/data.h5ad")
        else:
            adata_score = ensure_raw_counts(adata_target, data_path)
    except ValueError as e:
        print(f"[get_proliferation_rates] Error: {e}")
        sys.exit(1)

    prolif_path = find_data_file(input_dir(p), 'proliferation_rates')
    # Anchors per sample (dataset_id); the default rows apply to the others
    from CardamomOT.inputs import proliferation_sample_anchors, fitted_anchors
    sample_specs = {}
    fitted = fitted_anchors(p)
    if fitted is not None and fitted.get('samples'):
        # Anchors corrected on the proportions (and population sizes) by fit_population_anchors.py: one specification
        # per sample, the sheet (default rows included) being already taken into account; the other samples keep the
        # literature estimate
        prolif_path = None
        for sid, v in fitted['samples'].items():
            if v.get('rates'):
                sample_specs[sid] = dict(rates=v['rates'])
        print(f"[get_proliferation_rates] Anchors corrected by fit_population_anchors.py for {sorted(sample_specs)}")
    else:
        for sid, rates in proliferation_sample_anchors(p).items():
            sample_specs.setdefault(sid, {})['rates'] = rates

    # Too few genes to score the signatures: no estimate, except the per-cell-type anchors if given
    if not using_complete and adata_target.shape[1] < MIN_GENES_SCORING:
        print(f"[get_proliferation_rates] Warning: Data/data.h5ad has only {adata_target.shape[1]} genes "
              f"(< {MIN_GENES_SCORING}) and there is no Data/data_complete.h5ad: the gene signatures "
              f"cannot be scored reliably, no literature estimate")
        samples = (adata_target.obs['dataset_id'].astype(str).to_numpy() if 'dataset_id' in adata_target.obs
                   else np.array(['0'] * adata_target.n_obs))
        net = np.full(adata_target.n_obs, np.nan)
        # Per sample: its own anchors, else the default ones
        for sid in np.unique(samples):
            m = samples == sid
            own = sample_specs.get(sid, {})
            src = own.get('rates') or prolif_path
            anchors = read_anchors(adata_target[m], src, "no proliferation rate assigned") if src is not None else None
            if anchors is not None:
                labels, rates, celltype_col = anchors
                net[m] = np.array([rates[l] for l in labels], dtype=float)
                print(f"[get_proliferation_rates] Rates of {'the sample ' + sid if isinstance(src, dict) else src} assigned "
                      f"uniformly per '{celltype_col}' to {sid}")
        if np.isfinite(net).any():
            net = np.where(np.isfinite(net), net, 0.0)
            set_rates(adata_target.obs, np.maximum(net, 0.0), np.maximum(-net, 0.0))  # negligible death
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

    assign_proliferation_rates(adata_score, prolif_path, species=species,
                                proliferation_genes=proliferation_genes, death_genes=death_genes,
                                senescence_genes=senescence_genes, senescence_gating=senescence_gating,
                                sample_specs=sample_specs)

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