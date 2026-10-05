"""
select_genes_and_split.py
---------------------------
Select the genes of the model and split data into train/test sets.

Usage:
    python select_genes_and_split.py -i <project_path> [--prior <float>]
(split, train_rate, select_genes, build_prior_network: model_parameters sheet)

With select_genes = True, genes are selected from the whole transcriptome of Data/data.h5ad in two steps
(CardamomOT.inference.gene_selection): a coarse global GRN with OTVelo-Corr on log(x+1) counts
(highly variable, protein-coding, non mito/ribo genes), then a directed Steiner tree linking the
stimulus to the genes of genes_queries (gene_lists sheet) and to the highest-entropy genes, within a budget
of model.num_max_genes genes. With select_genes = False all genes are kept. With literature_selection, the
edge probabilities are reweighted by the literature (OmniPath / CollecTRI), whatever the prior; otherwise the
literature is not queried. With build_prior_network and a hard prior (--prior 0, or model.prior_network_pen = 0
if --prior is not given), the literature prior is built along: the gene budget is set so that it leaves
model.max_free_params free network parameters, and it is written to cardamomOT/ref_network.csv (otherwise
build_reference_network builds the prior after the selection).

Required input files:
    - Data/data.h5ad: input count matrix, with temporal information in obs['time']
Optional:
    - genes_queries (gene_lists sheet): genes of interest (one per line or comma-separated)
    - stimulus_inference_schedule sheet: stimulus values per timepoint (default 0 at the first one, 1 after)

Output files:
    - Data/data_full.h5ad: dataset restricted to the selected genes
    - Data/data_train.h5ad, data_test.h5ad: train/test split (if split = 'train', train_rate per sample and time)
      Samples with remove_from_inference (perturbation_inference) are excluded from the selection, data_full and
      data_train: all their cells go to data_test (also written with split = 'full').
    - cardamomOT/gene_selection_report.csv: role of each selected gene (query, entropy, Steiner) and its parent
    - cardamomOT/global_network.npz: global network C, its null C_null, edge probabilities W, genes (stimuli first)
    - cardamomOT/ref_network.csv: literature prior of the selected genes
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
from CardamomOT.config import find_stimulus_schedule
import pandas as pd



def load_queries(p):
    """Genes of interest: genes_queries (gene_lists sheet) (one per line or comma-separated)."""
    path = os.path.join(input_dir(p), 'genes_queries.txt')
    if os.path.isfile(path):
        return read_gene_list(path)
    print("[select_genes_and_split] No genes_queries (gene_lists sheet): the selection only uses entropy genes")
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
        print(f"[select_genes_and_split] Perturbed genes always selected (KO_OV files): {genes}")
    return genes


def main(argv):
    """
    Select differentially expressed genes and split data.

    Args:
        argv: Command-line arguments (-i <project>, --prior).
    """
    opts = parse_step_options(argv, 'select_genes_and_split', __doc__)
    p = opts.p
    cfg = settings(opts)
    change, rate, split, ref = cfg.select_genes, cfg.train_rate, cfg.split, cfg.build_prior_network
    print(f"[select_genes_and_split] select_genes={change}, split={split}, train_rate={rate}, "
          f"build_prior_network={ref}")

    # Load input dataset
    data_path = os.path.join(p, 'Data', 'data.h5ad')
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Input data file not found at {data_path}")
        adata = ensure_raw_counts(ad.read_h5ad(data_path), data_path)
        harmonize_obs(adata)  # obs names close to the expected ones (written as such to data_full/train/test)
        print(adata)
        print(f"[select_genes_and_split] Loaded input dataset from {data_path}")
        print(f"[select_genes_and_split] Dataset contains {adata.shape[0]} cells and {adata.shape[1]} genes")
    except (FileNotFoundError, ValueError) as e:
        print(f"[select_genes_and_split] Error: {e}")
        sys.exit(1)

    # Samples held out of the inference (remove_from_inference): every cell goes to data_test
    adata_removed = None
    if 'dataset_id' in adata.obs:
        removed, _ = removed_samples(p, present=adata.obs['dataset_id'].astype(str).unique())
        if removed:
            is_removed = adata.obs['dataset_id'].astype(str).isin(removed).to_numpy()
            if is_removed.all():
                print("[select_genes_and_split] Error: every sample is removed from the inference")
                sys.exit(1)
            adata_removed = adata[is_removed].copy()
            adata = adata[~is_removed].copy()
            print(f"[select_genes_and_split] Samples removed from the inference (all their {adata_removed.n_obs} "
                  f"cells in data_test): {removed}")

    # Temporal information (absent or single timepoint = stationary setting)
    stationary = check_stationary(adata)
    times = adata.obs['time'].values
    if stationary:
        print("[select_genes_and_split] Stationary data (no or single timepoint): "
              "no OTVelo network, gene selection uses queries and entropy genes only")
    else:
        print(f"[select_genes_and_split] Found {len(np.unique(times))} unique timepoints: {sorted(np.unique(times))}")

    def _make_model(n_genes):
        return configure(NetworkModel_beta(n_genes), opts)

    genes_list_init = list(adata.var_names.values)
    genes_list_init.sort()
    genes_list_final = genes_list_init

    if change:
        model = _make_model(adata.shape[1])
        queries = load_queries(p)
        # Literature-reweighted selection with literature_selection, whatever the prior; the prior network is
        # built along (budget in free network parameters) only for build_prior_network with a hard prior (0)
        prior_pen = model.prior_network_pen
        use_literature = bool(model.literature_selection)
        build_prior = bool(use_literature and ref and prior_pen == 0)
        max_free = model.max_free_params if build_prior else None
        budget = f"{max_free} free network parameters" if max_free else f"{model.num_max_genes} genes"
        print(f"[select_genes_and_split] Gene selection: {model.n_query_genes} queries (of {len(queries)}) + "
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
        names_s = sample_names(adata)
        model.set_sample_names(names_s)
        if model._stim_schedule.has_overrides():
            stim = {None: stim, **{s: np.array([model._stim_schedule.at(t, s) for t in tu]) for s in names_s}}

        selected, df_report, net, prior = select_genes(
            adata, queries, model.num_max_genes, n_query=model.n_query_genes, n_entropy=model.n_entropy_genes,
            stim=stim, cell_type_key=resolve_cell_type_obs(adata, 'selection'),
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
            use_depth_factor=model.use_depth_factor, seed=model.seed)

        out_dir = os.path.join(p, 'cardamomOT')
        os.makedirs(out_dir, exist_ok=True)
        df_report.to_csv(os.path.join(out_dir, 'gene_selection_report.csv'), index=False)
        if net is not None:
            np.savez_compressed(os.path.join(out_dir, 'global_network.npz'), **net)
        print(f"[select_genes_and_split] Saved gene_selection_report.csv and global_network.npz to {out_dir}")
        genes_list_final = [g for g in genes_list_init if g in set(selected)]
        ref_path = os.path.join(out_dir, 'ref_network.csv')
        if prior is not None and build_prior:
            # Literature prior of the selection, in the order of the saved genes
            pos = [selected.index(g) for g in genes_list_final]
            up = [g.upper() for g in genes_list_final]  # infer_network_structure matches upper-case names
            pd.DataFrame(prior[np.ix_(pos, pos)], index=up, columns=up).to_csv(ref_path)
            print(f"[select_genes_and_split] Saved the literature prior ref_network.csv to {out_dir}")
        elif not ref and os.path.exists(ref_path):
            # Prior of an earlier gene list: would constrain the new network (build_prior_network = False)
            os.remove(ref_path)
            print("[select_genes_and_split] Removed the ref_network.csv of an earlier run (build_prior_network = False)")

    adata = adata[:, genes_list_final]

    try:
        adata.write(os.path.join(p, 'Data', 'data_full.h5ad'))
        print(f"[select_genes_and_split] Saved filtered dataset to {os.path.join(p, 'Data', 'data_full.h5ad')}")
    except Exception as e:
        print(f"[select_genes_and_split] Error saving filtered dataset: {e}")
        sys.exit(1)

    if split == "train":
        print(f"[select_genes_and_split] Creating train/test split with rate: {rate}")

        train_idx = []
        test_idx = []
        try:
            samples_id = adata.obs['dataset_id'].values
        except KeyError:
            samples_id = np.zeros_like(times)

        times = adata.obs['time'].values if 'time' in adata.obs else np.zeros(adata.n_obs)

        # Filter to specified inference times if provided
        times_file = os.path.join(input_dir(p), 'times_inference.txt')
        if os.path.exists(times_file):
            with open(times_file, "r") as f:
                times_unique = [float(line.strip()) for line in f if line.strip()]
            samples_id = samples_id[times <= np.max(times_unique)]
            adata = adata[times <= np.max(times_unique)]
            times = times[times <= np.max(times_unique)]
            print(f"[select_genes_and_split] Filtered to times <= {np.max(times_unique)}")

        for t in np.unique(times):
            for i in np.unique(samples_id):
                var_bool = (times == t) & (samples_id == i)
                indices = adata.obs[var_bool].index.values
                np.random.shuffle(indices)
                split_point = max(min(100, len(indices)), int(len(indices) * float(rate)))
                train_idx.extend(indices[:split_point])
                # Test cells of each (time, sample) capped at its number of train cells
                test_idx.extend(indices[split_point:2 * split_point])

        print(f"[select_genes_and_split] Split results: {len(train_idx)} train cells, {len(test_idx)} test cells")

        adata_test = adata[test_idx].copy()
        adata = adata[train_idx]

        adata = adata[:, genes_list_final]
        adata_test = adata_test[:, genes_list_final]
        if adata_removed is not None:
            adata_test = ad.concat([adata_test, adata_removed[:, genes_list_final]], merge='same')

        try:
            adata.write(os.path.join(p, 'Data', 'data_train.h5ad'))
            adata_test.write(os.path.join(p, 'Data', 'data_test.h5ad'))
            print(f"[select_genes_and_split] Saved train/test datasets")
            print(f"[select_genes_and_split] Train: {adata.shape[0]} cells, {adata.shape[1]} genes")
            print(f"[select_genes_and_split] Test: {adata_test.shape[0]} cells, {adata_test.shape[1]} genes")
        except Exception as e:
            print(f"[select_genes_and_split] Error saving train/test datasets: {e}")
            sys.exit(1)

    elif adata_removed is not None:
        # No split: data_test holds the removed samples only
        adata_removed[:, genes_list_final].copy().write(os.path.join(p, 'Data', 'data_test.h5ad'))
        print(f"[select_genes_and_split] Saved the removed samples to data_test.h5ad ({adata_removed.n_obs} cells)")

    print("[select_genes_and_split] Gene selection and splitting completed successfully")

if __name__ == "__main__":
   main(sys.argv[1:])
