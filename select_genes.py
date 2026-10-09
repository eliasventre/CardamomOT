"""
select_genes.py
---------------
Select the genes of the model on the train cells of split_dataset.py, and write the train/test datasets.

Usage:
    python select_genes.py -i <project_path> [--prior <float>]
(select_genes, build_prior_network, selection_edge_prob: model_parameters sheet)

The selection only sees the cells with obs['split'] = 'train' (or 'full'), written by split_dataset.py:
the test cells do not leak into it.
With select_genes = True, genes are selected from the whole transcriptome of Data/data.h5ad in two steps
(CardamomOT.inference.gene_selection): a coarse global GRN (network_method, OTVelo-Granger by default) on log(x+1) counts
(highly variable, protein-coding, non mito/ribo genes), then a directed Steiner tree linking the
stimulus to the genes of genes_queries (gene_lists sheet), to the fate drivers of run_classical_OT (n_driver_genes,
cardamomOT/classical_OT/fate_drivers.csv) and to the highest-entropy genes, within a budget
of model.num_max_genes genes. With select_genes = False all genes are kept. With literature_selection, the
edge probabilities are reweighted by the literature (OmniPath / CollecTRI), whatever the prior; otherwise the
literature is not queried. With build_prior_network and a hard prior (--prior 0, or model.prior_network_pen = 0
if --prior is not given), the literature prior is built along: the gene budget is set so that it leaves
model.max_free_params free network parameters, and it is written to cardamomOT/ref_network.csv (otherwise
build_reference_network builds the prior after the selection).

Required input files:
    - Data/data.h5ad: input count matrix, with obs['time'] and obs['split'] (split_dataset.py)
Optional:
    - genes_queries (gene_lists sheet): genes of interest (one per line or comma-separated)
    - stimulus_inference_schedule sheet: stimulus values per timepoint (default 0 at the first one, 1 after)

Output files (selected genes only):
    - Data/data_full.h5ad: every cell of the samples used for the inference
    - Data/data_train.h5ad, data_test.h5ad: cells with obs['split'] = 'train' / 'test' (data_train with split = 'train';
      data_test whenever there are test cells: split = 'train' or samples with remove_from_inference)
    - cardamomOT/gene_selection_report.csv: role of each selected gene (query, entropy, Steiner, regulator), its
      Steiner parent, and its regulators (is_regulated_by) and targets (regulates) inside the selection: edges of
      the global network with probability >= selection_edge_prob, feasible in the literature (path of at most
      literature_depth edges; '*': pair not covered by the literature, kept as in the prior)
    - cardamomOT/global_network.npz: global network C, its null C_null, edge probabilities W, genes (stimuli first)
    - cardamomOT/ref_network.csv: literature prior of the selected genes
    - cardamomOT/selection_preservation.json (if run_classical_OT was run): trajectory preservation of the
      selection (classical OT on the selected genes vs every gene; velocity cosine, fate Jensen-Shannon
      distance, random gene sets of the same size as baseline)
"""
import sys; sys.path += ['../']
import os
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta
from CardamomOT.inputs import input_dir, removed_samples
from CardamomOT.inference.gene_selection import select_genes
from CardamomOT import check_stationary, read_gene_list, resolve_cell_type_obs, ensure_raw_counts, harmonize_obs, read_stimulus_targets
import anndata as ad
from CardamomOT.run_options import parse_step_options, settings, configure
from CardamomOT.config import find_stimulus_schedule, n_inference_stimuli
import pandas as pd



def load_queries(p):
    """Genes of interest: genes_queries (gene_lists sheet) (one per line or comma-separated)."""
    path = os.path.join(input_dir(p), 'genes_queries.txt')
    if os.path.isfile(path):
        return read_gene_list(path)
    print("[select_genes] No genes_queries (gene_lists sheet): the selection only uses entropy genes")
    return []

def perturbed_genes(p, names=None):
    """Genes always selected: those perturbed in perturbation_simulation (KO, OV, stimulus targets, and
    the genes of the RATE signatures, needed to score them) and in perturbation_inference (KO, OV, and
    the genes of the RATE signatures of the inference stimuli)."""
    from CardamomOT.tools.perturbations import (find_perturbation_file, load_perturbations, combo_genes,
                                                rate_target_genes)
    from CardamomOT.stimulus_rates import load_effects
    genes = []

    def signature_genes(target):
        # Gene list / GENE1+GENE2 / gene of a RATE entry ('all' and cell types have no genes)
        if names is None or target.lower() == 'all':
            return []
        try:
            return rate_target_genes(target, names, input_dir(p))
        except ValueError:
            return []  # e.g. a cell type of cell_type_proliferation

    path = find_perturbation_file(input_dir(p))
    if path is not None:
        for combo in load_perturbations(path, names):
            genes += combo_genes(combo)
            for effects in combo.get('RATE', {}).values():
                for target, _ in effects:
                    genes += signature_genes(target)
    for effects in load_effects(p).values():
        for target, _ in effects:
            genes += signature_genes(target)
    path = os.path.join(input_dir(p), 'KO_OV_inference.txt')
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t', dtype=str).fillna('')
        for col in [c for c in df.columns if c.strip().upper() in ('KO', 'OV')]:
            for cell in df[col]:
                genes += [g.strip() for g in str(cell).split(',') if g.strip() and g.strip() not in ('0', 'nan')]
    genes = list(dict.fromkeys(genes))
    if genes:
        print(f"[select_genes] Perturbed genes always selected (KO_OV files): {genes}")
    return genes


def main(argv):
    """Select the genes on the train cells and write data_full/train/test (-i <project>, --prior)."""
    opts = parse_step_options(argv, 'select_genes', __doc__)
    p = opts.p
    cfg = settings(opts)
    change, ref = cfg.select_genes, cfg.build_prior_network
    print(f"[select_genes] select_genes={change}, build_prior_network={ref}")

    data_path = os.path.join(p, 'Data', 'data.h5ad')
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Input data file not found at {data_path}")
        adata = ensure_raw_counts(ad.read_h5ad(data_path), data_path)
        harmonize_obs(adata)  # obs names close to the expected ones (written as such to data_full/train/test)
        if 'split' not in adata.obs:
            raise ValueError("no obs['split'] in Data/data.h5ad: run split_dataset.py first")
        print(f"[select_genes] Loaded {data_path}: {adata.shape[0]} cells and {adata.shape[1]} genes")
    except (FileNotFoundError, ValueError) as e:
        print(f"[select_genes] Error: {e}")
        sys.exit(1)
    split = adata.obs['split'].astype(str).values
    if (cfg.split == 'full') != ('full' in set(split)):
        print(f"[select_genes] Error: obs['split'] does not match split = '{cfg.split}': rerun split_dataset.py")
        sys.exit(1)

    # Samples held out of the inference (remove_from_inference): excluded from data_full
    inference = np.ones(adata.n_obs, bool)
    if 'dataset_id' in adata.obs:
        removed, _ = removed_samples(p, present=adata.obs['dataset_id'].astype(str).unique())
        inference = ~adata.obs['dataset_id'].astype(str).isin(removed).to_numpy()

    # Selection on the train cells only (all the cells used for the inference with split = 'full')
    fit = np.isin(split, ['train', 'full'])
    ad_fit = adata[fit].copy()
    print(f"[select_genes] Gene selection on the {ad_fit.n_obs} {cfg.split} cells")

    # Temporal information (absent or single timepoint = stationary setting)
    stationary = check_stationary(ad_fit)
    times = ad_fit.obs['time'].values
    if stationary:
        print("[select_genes] Stationary data (no or single timepoint): "
              "no OTVelo network, gene selection uses queries and entropy genes only")
    else:
        print(f"[select_genes] Found {len(np.unique(times))} unique timepoints: {sorted(np.unique(times))}")

    genes_list_init = sorted(adata.var_names.values)
    genes_list_final = genes_list_init

    if change:
        model = configure(NetworkModel_beta(adata.shape[1], n_stimuli=n_inference_stimuli(input_dir(p))), opts)
        queries = load_queries(p)
        # Literature-reweighted selection with literature_selection, whatever the prior; the prior network is
        # built along (budget in free network parameters) only for build_prior_network with a hard prior (0)
        prior_pen = model.prior_network_pen
        use_literature = bool(model.literature_selection)
        build_prior = bool(use_literature and ref and prior_pen == 0)
        max_free = model.max_free_params if build_prior else None
        budget = f"{max_free} free network parameters" if max_free else f"{model.num_max_genes} genes"
        print(f"[select_genes] Gene selection: {model.n_query_genes} queries (of {len(queries)}) + "
              f"{model.n_entropy_genes} entropy genes, '{model.network_method}' network and directed Steiner tree "
              f"from the stimulus, budget {budget}")

        # Stimulus schedule per sorted timepoint (default: 0 at the first time, 1 after)
        tu = np.sort(np.unique(times))
        sched_path = (find_stimulus_schedule(input_dir(p))
                      or os.path.join(input_dir(p), 'stimulus_schedule_inference.txt'))
        stim = np.loadtxt(sched_path) if os.path.exists(sched_path) else None
        model._stim_schedule = model._build_stimulus_schedule(tu, stim)
        stim = np.array([model._stim_schedule[t] for t in tu])
        # Per-sample schedules (sample_id rows of stimulus_inference_schedule): one schedule per sample
        from CardamomOT.schedules import sample_names
        names_s = sample_names(ad_fit)
        model.set_sample_names(names_s)
        if model._stim_schedule.has_overrides():
            stim = {None: stim, **{s: np.array([model._stim_schedule.at(t, s) for t in tu]) for s in names_s}}

        # Fate drivers of the classical OT (module representatives, round robin over the fates)
        drivers = []
        drv_path = os.path.join(p, 'cardamomOT', 'classical_OT', 'fate_drivers.csv')
        if model.n_driver_genes and os.path.exists(drv_path):
            from CardamomOT.tools.classical_ot import driver_order
            drivers = driver_order(pd.read_csv(drv_path))
            print(f"[select_genes] {len(drivers)} fate drivers (q <= 0.05, module representatives) from {drv_path}")
        elif model.n_driver_genes:
            print("[select_genes] No cardamomOT/classical_OT/fate_drivers.csv (run_classical_OT): no fate drivers")

        # Combination of the per-sample networks: 'any' for independent condition networks
        combo = str(model.sample_network_combination).lower()
        if combo == 'auto':
            n_cond = ad_fit.obs['network_condition'].nunique() if 'network_condition' in ad_fit.obs else 1
            combo = 'any' if (n_cond >= 2 and float(model.network_condition_pen) == 0) else 'consensus'
        if combo not in ('any', 'consensus'):
            raise ValueError(f"sample_network_combination '{combo}': use 'auto', 'consensus' or 'any'")
        selected, df_report, net, prior = select_genes(
            ad_fit, queries, model.num_max_genes, n_query=model.n_query_genes, n_entropy=model.n_entropy_genes,
            stim=stim, cell_type_key=resolve_cell_type_obs(ad_fit, 'selection'),
            n_hvg=model.n_hvg_selection, n_top_entropy=model.n_top_entropy, n_cells_entropy=model.n_cells_entropy,
            network_method=model.network_method, network_params=model.network_method_params,
            project_path=p, k_in=model.k_in_steiner, k_stim=model.k_stim_steiner,
            min_edge_prob=model.min_edge_prob, edge_prior=model.edge_prior, null_network=model.null_network,
            closure_min=model.closure_min, literature=use_literature,
            literature_depth=model.literature_depth, literature_weight=model.literature_weight,
            literature_resources=model.literature_resources, max_free_params=max_free,
            min_entropy_change=model.min_entropy_change, min_nb_separation=model.min_nb_separation,
            n_cells_mixture=model.batch_size_mixture, seuil_mixture=model.seuil, n_cells_network=model.batch_size_mixture,
            forced_genes=perturbed_genes(p, list(adata.var_names)), stimulus_targets=read_stimulus_targets(input_dir(p)),
            drivers=drivers, n_driver=int(model.n_driver_genes), entropy_preselection=bool(model.entropy_preselection),
            use_depth_factor=model.use_depth_factor, report_edge_prob=model.selection_edge_prob,
            sample_combination=combo, seed=model.seed)

        out_dir = os.path.join(p, 'cardamomOT')
        os.makedirs(out_dir, exist_ok=True)
        df_report.to_csv(os.path.join(out_dir, 'gene_selection_report.csv'), index=False)
        if net is not None:
            np.savez_compressed(os.path.join(out_dir, 'global_network.npz'), **net)
        print(f"[select_genes] Saved gene_selection_report.csv and global_network.npz to {out_dir}")
        genes_list_final = [g for g in genes_list_init if g in set(selected)]
        ref_path = os.path.join(out_dir, 'ref_network.csv')
        if prior is not None and build_prior:
            # Literature prior of the selection, in the order of the saved genes
            pos = [selected.index(g) for g in genes_list_final]
            up = [g.upper() for g in genes_list_final]  # infer_network_structure matches upper-case names
            pd.DataFrame(prior[np.ix_(pos, pos)], index=up, columns=up).to_csv(ref_path)
            print(f"[select_genes] Saved the literature prior ref_network.csv to {out_dir}")
        elif not ref and os.path.exists(ref_path):
            # Prior of an earlier gene list: would constrain the new network (build_prior_network = False)
            os.remove(ref_path)
            print("[select_genes] Removed the ref_network.csv of an earlier run (build_prior_network = False)")

    # How much the selection preserves the trajectories of the classical OT on every gene (if it was run)
    vel = os.path.join(p, 'cardamomOT', 'classical_OT', 'velocity.npz')
    if change and os.path.exists(vel):
        import json
        from CardamomOT.tools.velocity import coupling_blocks
        from CardamomOT.tools.classical_ot import trajectory_preservation, dense_blocks
        d = np.load(vel, allow_pickle=True)
        if np.array_equal(d['obs_names'].astype(str), adata.obs_names.values.astype(str)):
            full = dense_blocks(coupling_blocks(os.path.join(p, 'cardamomOT', 'classical_OT', 'couplings.npz'),
                                                adata.n_obs))
            res = trajectory_preservation(adata, genes_list_final, d['Z'].astype(float), full, d['transported'],
                                          bool(model.use_depth_factor), int(model.classical_ot_max_cells),
                                          0 if model.seed is None else int(model.seed))
            res['network_method'] = model.network_method
            # Capacity of the selection to predict the fates of the classical OT (cells before the last time)
            fate_path = os.path.join(p, 'cardamomOT', 'classical_OT', 'fate.npz')
            if os.path.exists(fate_path):
                from CardamomOT.tools.classical_ot import fate_prediction
                from CardamomOT.tools.embedding import log_normalised as _ln
                fz = np.load(fate_path, allow_pickle=True)
                if np.array_equal(fz['obs_names'].astype(str), adata.obs_names.values.astype(str)):
                    t_all = adata.obs['time'].astype(float).values
                    s_all = (adata.obs['dataset_id'].astype(str).values if 'dataset_id' in adata.obs
                             else np.full(adata.n_obs, '0'))
                    tr = d['transported'].astype(bool)
                    last = np.zeros(adata.n_obs, bool)
                    for s_ in np.unique(s_all[tr]):
                        m_ = tr & (s_all == s_)
                        last |= m_ & (t_all == t_all[m_].max())
                    rows = np.flatnonzero(tr & ~last & np.isfinite(fz['F']).all(axis=1))
                    Xl = _ln(adata, bool(model.use_depth_factor))
                    expressed = adata.var_names.values[np.asarray((Xl[tr] > 0).sum(axis=0)).ravel() >= 10]
                    res.update(fate_prediction(Xl, adata.var_names.values.astype(str), genes_list_final, fz['F'], rows,
                                               t_all, Z_full=d['Z'].astype(float), pool=list(expressed),
                                               seed=0 if model.seed is None else int(model.seed)))
                    print(f"[select_genes] Fate prediction (cross-validated R² of the classical OT fates from the "
                          f"expression at t, + time): {res['fate_r2']:.3f} with the selection, {res['fate_r2_time']:.3f} "
                          f"time only, {res['fate_r2_random']:.3f} random genes, {res['fate_r2_all_genes']:.3f} every gene")
            with open(os.path.join(p, 'cardamomOT', 'selection_preservation.json'), 'w') as f:
                json.dump(res, f, indent=1)
            print(f"[select_genes] Trajectory preservation (classical OT on the {len(genes_list_final)} genes vs every "
                  f"gene): velocity cosine {res['velocity_cosine']:.3f} (random genes {res['random_velocity_cosine']:.3f}), "
                  f"fate Jensen-Shannon distance {res['fate_js']:.3f} (random genes {res['random_fate_js']:.3f})")
        else:
            print("[select_genes] Classical OT outputs of other cells: trajectory preservation not computed")

    # Datasets restricted to the selected genes
    outputs = {'full': inference, 'train': split == 'train', 'test': split == 'test'}
    try:
        for name, mask in outputs.items():
            path = os.path.join(p, 'Data', f'data_{name}.h5ad')
            if name != 'full' and not mask.any():
                if name == 'test' and os.path.exists(path):
                    os.remove(path)  # test cells of an earlier split
                continue
            sub = adata[mask, genes_list_final].copy()
            sub.write(path)
            print(f"[select_genes] Saved {path}: {sub.n_obs} cells, {sub.n_vars} genes")
    except Exception as e:
        print(f"[select_genes] Error saving the datasets: {e}")
        sys.exit(1)
    print("[select_genes] Gene selection completed successfully")


if __name__ == "__main__":
    main(sys.argv[1:])
