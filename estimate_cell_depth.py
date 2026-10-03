"""
estimate_cell_depth.py
----------------------
First step of the pipeline: per-cell depth diagnostic and, if needed, depth factor s_i.

Within a sample and a timepoint, cells differ in sequencing depth (capture, and cell size). A
depth common to all genes of a cell acts like a hidden common regulator: it correlates all genes
and can move cells between NB basins. The diagnostic, on the whole transcriptome
(Data/data_complete.h5ad, else Data/data.h5ad if it has >= 10,000 genes; otherwise no estimate),
measures within homogeneous groups (sample, time, cell type) the share of the correlation of
reference genes due to depth and the share of genes whose NB modes are closer than the depth
spread. If the correction is recommended and model.allow_depth_correction is True, the depth
factor of model.depth_method (CardamomOT/inference/depth_methods) is written to
adata.obs['depth_factor'] of Data/data.h5ad (X untouched). With model.use_depth_factor (default
False), the following steps then model counts as NB(k, c / s_i) and the simulations draw counts with
the s_i of the cells they mimic; otherwise the factor is kept but ignored (s_i = 1).
With model.compute_depth_factor = False, nothing is computed: an existing obs['depth_factor'] is only
read (summary printed) and never removed, and the previous diagnostic files are left as they are.

Usage:
    python estimate_cell_depth.py -i <project_path>
(allow_depth_correction, depth_method, compute_depth_factor, use_depth_factor: Model_parameters sheet)

Outputs:
    - cardamomOT/depth_diagnostic.csv: depth per group (cells, median counts, CV, s quantiles)
    - cardamomOT/depth_diagnostic.json: global indicators and decision
    - Data/data.h5ad: obs['depth_factor'] (if applied)
"""
import sys; sys.path += ['../']
import os
import json
import numpy as np
import anndata as ad

from CardamomOT import NetworkModel, ensure_raw_counts, harmonize_obs, resolve_cell_type_obs
from CardamomOT.inference.depth import MIN_GENES_DEPTH, library, depth_groups, diagnose_depth
from CardamomOT.inference.depth_methods import compute_depth
from CardamomOT.run_options import parse_step_options, configure

TAG = "[estimate_cell_depth]"


def doublet_check(adata):
    """Cells still flagged obs['predicted_doublet'] (warning only, never filtered): dict or None."""
    if 'predicted_doublet' not in adata.obs:
        return None
    flag = adata.obs['predicted_doublet'].astype(str).str.lower().isin(['true', '1', 'yes']).values
    out = dict(n=int(flag.sum()), fraction=float(flag.mean()))
    ct_key = resolve_cell_type_obs(adata, 'selection')
    if ct_key and flag.any():
        cts = adata.obs[ct_key].astype(str).values
        out['by_cell_type'] = {c: float(flag[cts == c].mean()) for c in np.unique(cts)}
    if out['n']:
        detail = ', '.join(f"{c}: {v * 100:.1f}%" for c, v in out.get('by_cell_type', {}).items())
        print(f"{TAG} Warning: {out['n']} cells ({out['fraction'] * 100:.1f}%) are flagged obs['predicted_doublet'] "
              f"and still present{' (' + detail + ')' if detail else ''}: doublets mix two transcriptomes, which a "
              f"depth factor cannot correct and which create artificial intermediate states for the trajectories; "
              f"remove them upstream (checking that the threshold does not remove mostly large, deep cells)")
    else:
        print(f"{TAG} No cell flagged obs['predicted_doublet']")
    return out


def main(argv):
    opts = parse_step_options(argv, 'estimate_cell_depth', __doc__)
    p = opts.p
    model = configure(NetworkModel(1), opts)
    allow, method = model.allow_depth_correction, model.depth_method
    data_path = os.path.join(p, 'Data', 'data.h5ad')
    complete_path = os.path.join(p, 'Data', 'data_complete.h5ad')
    out_dir = os.path.join(p, 'cardamomOT')
    os.makedirs(out_dir, exist_ok=True)
    use = 'used' if model.use_depth_factor else 'ignored (use_depth_factor = False)'

    target = ad.read_h5ad(data_path)
    if not model.compute_depth_factor:
        # Read only: keep any existing factor and the previous diagnostic
        if 'depth_factor' in target.obs:
            s = target.obs['depth_factor'].values.astype(float)
            q05, q95 = np.percentile(s, [5, 95])
            print(f"{TAG} compute_depth_factor = False: existing obs['depth_factor'] kept (q05-q95 {q05:.2f}-{q95:.2f}, "
                  f"x{q95 / q05:.2f}), {use}")
        else:
            print(f"{TAG} compute_depth_factor = False and no obs['depth_factor'] in Data/data.h5ad: no depth factor")
        doublet_check(target)
        return
    print(f"{TAG} The depth factor will be {use} by the following steps")
    src_path = complete_path if os.path.exists(complete_path) else data_path
    src = target if src_path == data_path else ad.read_h5ad(src_path)
    if src.n_vars < MIN_GENES_DEPTH:
        print(f"{TAG} Warning: {src_path} has {src.n_vars} genes (< {MIN_GENES_DEPTH}) and there is no "
              f"Data/data_complete.h5ad: the depth cannot be estimated from the transcriptome, no correction")
        json.dump(dict(estimated=False, applied=False, reason=f'{src.n_vars} genes < {MIN_GENES_DEPTH}',
                       predicted_doublets=doublet_check(target)),
                  open(os.path.join(out_dir, 'depth_diagnostic.json'), 'w'), indent=1)
        return
    try:
        src = ensure_raw_counts(src, src_path)
    except ValueError as e:
        print(f"{TAG} Error: {e}")
        sys.exit(1)
    harmonize_obs(src)
    missing = target.obs_names.difference(src.obs_names)
    if len(missing):
        print(f"{TAG} Error: {len(missing)} cells of Data/data.h5ad absent from {src_path} (first: {list(missing[:5])})")
        sys.exit(1)
    if src is not target:
        src = src[target.obs_names]

    times = src.obs['time'].values.astype(float) if 'time' in src.obs else np.zeros(src.n_obs)
    samples = src.obs['dataset_id'].values if 'dataset_id' in src.obs else None
    ct_key = resolve_cell_type_obs(src, 'selection')
    cts = src.obs[ct_key].astype(str).values if ct_key else None
    lib = library(src.X)
    # s_i estimated within (sample, time) groups, or (sample, time, cell type) if depth_by_cell_type
    # (circular: cell types derive from expression); the diagnostic always uses the finest groups
    groups = depth_groups(times, samples, cts)
    groups_est = groups if model.depth_by_cell_type else depth_groups(times, samples, None)
    s = compute_depth(method, src.X, lib, groups_est, model.depth_method_params, p)
    print(f"{TAG} Depth on {src.n_vars} genes ({os.path.basename(src_path)}), method '{method}' within "
          f"{len(np.unique(groups_est))} (sample, time{', ' + ct_key if (ct_key and model.depth_by_cell_type) else ''}) "
          f"groups; diagnostic within {len(np.unique(groups))} groups (sample, time{', ' + ct_key if ct_key else ''})")
    doublet = src.obs['doublet_score'].values if 'doublet_score' in src.obs else None
    per_group, glob = diagnose_depth(src.X, lib, groups, times, cts, s, doublet, n_cells=model.batch_size_mixture,
                                     seuil=model.seuil, seed=model.seed)
    applied = bool(glob['recommended'] and allow)
    glob.update(estimated=True, method=method, allow_depth_correction=allow, applied=applied, source=src_path,
                depth_by_cell_type=bool(model.depth_by_cell_type),
                predicted_doublets=doublet_check(src))
    per_group.to_csv(os.path.join(out_dir, 'depth_diagnostic.csv'), index=False)
    json.dump(glob, open(os.path.join(out_dir, 'depth_diagnostic.json'), 'w'), indent=1)
    print(f"{TAG} Depth spread within groups (q95/q05 of s): x{glob['depth_spread_q95_q05']:.2f}; "
          f"{glob['corr_share_depth'] * 100:.0f}% of the within-group correlation of {glob['n_reference_genes']} "
          f"reference genes due to depth; {glob['share_modes_closer_than_spread'] * 100:.0f}% of them with NB modes "
          f"closer than the depth spread; extrinsic noise Var[c]/E[c]^2 per group: "
          f"{per_group.extrinsic_noise.min():.2f}-{per_group.extrinsic_noise.max():.2f} "
          f"(median {glob['median_extrinsic_noise']:.2f})")
    if 'doublet_depth_corr' in glob and glob['doublet_depth_corr'] > 0.3:
        print(f"{TAG} Warning: depth correlates with the doublet score ({glob['doublet_depth_corr']:.2f}): "
              f"doublets should be filtered upstream (a depth factor does not correct them)")

    ours = 'depth_factor' in target.obs and target.uns.get('depth_factor_info', {}).get('written_by') == 'estimate_cell_depth'
    if applied:
        target.obs['depth_factor'] = s
        target.uns['depth_factor_info'] = dict(written_by='estimate_cell_depth', method=method)
        print(f"{TAG} Depth correction recommended: obs['depth_factor'] written to Data/data.h5ad")
    elif glob['recommended']:
        print(f"{TAG} Warning: depth correction recommended but not allowed (allow_depth_correction = False)")
    else:
        print(f"{TAG} Depth correction not needed")
    if not applied and ours:
        del target.obs['depth_factor']
        target.uns.pop('depth_factor_info', None)
        print(f"{TAG} Previous obs['depth_factor'] of this step removed")
    elif not applied and 'depth_factor' in target.obs:
        print(f"{TAG} Warning: keeping the existing obs['depth_factor'] (not written by this step)")
    if applied or ours:
        target.write(data_path)
        print(f"{TAG} Saved {data_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
