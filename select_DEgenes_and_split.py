"""
select_DEgenes_and_split.py
---------------------------
Select differentially expressed genes and split data into train/test sets.

Usage:
    python select_DEgenes_and_split.py -i <project_path> -c <change> -r <rate> -s <split> [-m <mean_forcing>]

Required input files:
    - Data/data.h5ad: input count matrix with temporal information

Output files:
    - Data/data_full.h5ad: filtered dataset with selected genes
    - Data/data_train.h5ad, data_test.h5ad: train/test split (if split="train")
    - cardamomOT/gene_variation_report.csv: gene selection report
"""
import sys; sys.path += ['../']
import os
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta
from CardamomOT import select_DEgenes
import anndata as ad
import getopt
import scipy as scp
import pandas as pd

n_genes_tokeep_temporal = [5]
n_genes_tokeep_celltype = [3]

def main(argv):
    """
    Select differentially expressed genes and split data.

    Args:
        argv: Command-line arguments (--input, --change, --rate, --split, --mean).
    """
    inputfile = ''
    change = ''
    rate = ''
    split = ''
    mean_forcing = -1
    try:
        opts, args = getopt.getopt(argv, "hi:c:r:s:m:", ["input=", "change=", "rate=", "split=", "mean-forcing="])
    except getopt.GetoptError:
        print("[select_DEgenes_and_split] Error: Invalid command-line arguments")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-c", "--change"):
            change = '{}'.format(arg)
        elif opt in ("-r", "--rate"):
            rate = '{}'.format(arg)
        elif opt in ("-s", "--split"):
            split = '{}'.format(arg)
        elif opt in ("-m", "--mean-forcing"):
            mean_forcing = float(arg)
        elif opt == "-h":
            print(__doc__)
            sys.exit(0)

    if not inputfile:
        print("[select_DEgenes_and_split] Error: Missing required argument --input")
        sys.exit(1)

    p = '{}/'.format(inputfile)

    # Load input dataset
    data_path = os.path.join(p, 'Data', 'data.h5ad')
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Input data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[select_DEgenes_and_split] Loaded input dataset from {data_path}")
        print(f"[select_DEgenes_and_split] Dataset contains {adata.shape[0]} cells and {adata.shape[1]} genes")

        if 'copycat' in p:
            adata_all_path = os.path.join('collaborations', 'copycat', 'RMS_all', 'Data', 'data.h5ad')
            if os.path.exists(adata_all_path):
                adata_all = ad.read_h5ad(adata_all_path)
                adata = adata[:, [g for g in adata.var_names.values if g in adata_all.var_names.values]]
    except FileNotFoundError as e:
        print(f"[select_DEgenes_and_split] Error: {e}")
        sys.exit(1)

    # Validate temporal information
    try:
        times = adata.obs['time'].values
        if len(np.unique(times)) <= 1:
            raise ValueError("Dataset must contain temporal information with multiple timepoints")
        print(f"[select_DEgenes_and_split] Found {len(np.unique(times))} unique timepoints: {sorted(np.unique(times))}")
    except (KeyError, ValueError) as e:
        print(f"[select_DEgenes_and_split] Error: {e}")
        sys.exit(1)

    vect_samples_id = np.zeros(adata.n_obs)
    vect_celltype_id = adata.obs['cell_type'].values if 'cell_type' in adata.obs else np.zeros(adata.n_obs)

    genes_list_init = list(adata.var_names.values)
    genes_to_keep = []

    if int(change):
        print(f"[select_DEgenes_and_split] Performing gene selection with change parameter: {change}")
        print("[select_DEgenes_and_split] Fitting mixture model for gene selection")

        model = NetworkModel_beta(adata.shape[1])
        if mean_forcing >= 0:
            model.mean_forcing_em = mean_forcing

        # Subsample cells if dataset is too large
        cells_to_use = []
        sub_N = min(adata.n_obs, int(10000/len(np.unique(times))))
        print(f"[select_DEgenes_and_split] Subsampling to {sub_N} cells per timepoint")

        for time in np.unique(times):
            for sample in np.unique(vect_samples_id):
                idx = np.where((times == time) & (vect_samples_id == sample))[0]
                n_select = min(sub_N, len(idx))
                if len(idx) > 0:
                    cells_to_use.extend(np.random.choice(idx, n_select, replace=False))

        model.hard_em = 0
        try:
            model.fit_mixture(adata[cells_to_use], gene_names=genes_list_init,
                              min_components=2, max_components=2,
                              max_iter_kinetics=0, refilter=1.0)
            print(f"[select_DEgenes_and_split] Fitted mixture model on {len(cells_to_use)} cells")

            genes_to_keep, temporal_variations, cell_type_variations, df_report = select_DEgenes(
                    times[cells_to_use], vect_samples_id[cells_to_use], vect_celltype_id[cells_to_use],
                    model.proba, genes_list_init, n_genes_tokeep_temporal=n_genes_tokeep_temporal,
                    n_genes_tokeep_celltype=n_genes_tokeep_celltype, limit_min=.001)

            out_dir = os.path.join(p, 'cardamomOT')
            os.makedirs(out_dir, exist_ok=True)
            report_path = os.path.join(p, 'cardamomOT', 'gene_variation_report.csv')
            df_report.to_csv(report_path, index=False)
            print(f"[select_DEgenes_and_split] Saved gene variation report to {report_path}")
            print(f"[select_DEgenes_and_split] Selected {len(genes_to_keep)} genes based on variation criteria")

        except Exception as e:
            print(f"[select_DEgenes_and_split] Error during mixture model fitting: {e}")
            sys.exit(1)

        # Add genes of biological interest
        genes_list_path = os.path.join(p, 'Data', 'genes_list.txt')
        if os.path.exists(genes_list_path):
            with open(genes_list_path, "r") as f:
                genes_list = [line.strip() for line in f if line.strip()]
            print(f"[select_DEgenes_and_split] Loaded {len(genes_list)} genes of biological interest")
            for gene in genes_list:
                if gene in genes_list_init:
                    idx_g = genes_list_init.index(gene)
                    if temporal_variations[idx_g] > 0.01:
                        genes_to_keep.append(gene)

        genes_list_tokeep = list(set(genes_to_keep))
        adata = adata[:, [g for g in genes_list_tokeep if g in genes_list_init]]
        print(f"[select_DEgenes_and_split] Final gene selection: {len(genes_list_tokeep)} genes retained")

    print("[select_DEgenes_and_split] Re-filtering genes to ensure only variable genes are retained")

    # Refresh metadata after potential gene filtering
    times = adata.obs['time'].values if 'time' in adata.obs else np.zeros(adata.n_obs)
    vect_samples_id = adata.obs['dataset_id'].values if 'dataset_id' in adata.obs else np.zeros(adata.n_obs)
    vect_celltype_id = adata.obs['cell_type'].values if 'cell_type' in adata.obs else np.zeros(adata.n_obs)

    model = NetworkModel_beta(adata.shape[1])
    if mean_forcing >= 0:
        model.mean_forcing_em = mean_forcing

    try:
        model.fit_mixture(adata, gene_names=list(adata.var_names),
                          min_components=2, max_components=2,
                          refilter=.001, max_iter_kinetics=0)
        print("[select_DEgenes_and_split] Re-fitted mixture model for final gene filtering")

        genes_tokeep_final, temporal_variations, cell_type_variations, df_report = select_DEgenes(
                times, vect_samples_id, vect_celltype_id,
                model.proba, adata.var_names,
                n_genes_tokeep_celltype=[100000], n_genes_tokeep_temporal=[100000], limit_min=.001)

        print(f"[select_DEgenes_and_split] Final filtering retained {len(genes_tokeep_final)} genes")

    except Exception as e:
        print(f"[select_DEgenes_and_split] Error during final gene filtering: {e}")
        sys.exit(1)

    genes_list_init = list(adata.var_names)
    genes_list_init.sort()
    adata = adata[:, [g for g in genes_list_init if g in genes_tokeep_final]]

    try:
        adata.write(os.path.join(p, 'Data', 'data_full.h5ad'))
        print(f"[select_DEgenes_and_split] Saved filtered dataset to {os.path.join(p, 'Data', 'data_full.h5ad')}")
    except Exception as e:
        print(f"[select_DEgenes_and_split] Error saving filtered dataset: {e}")
        sys.exit(1)

    if split == "train":
        print(f"[select_DEgenes_and_split] Creating train/test split with rate: {rate}")

        train_idx = []
        test_idx = []
        try:
            samples_id = adata.obs['dataset_id'].values
        except KeyError:
            samples_id = np.zeros_like(times)

        times = adata.obs['time'].values if 'time' in adata.obs else np.zeros(adata.n_obs)

        # Filter to specified inference times if provided
        times_file = os.path.join(p, 'Data', 'times_to_inference.txt')
        if os.path.exists(times_file):
            with open(times_file, "r") as f:
                times_unique = [float(line.strip()) for line in f if line.strip()]
            samples_id = samples_id[times <= np.max(times_unique)]
            adata = adata[times <= np.max(times_unique)]
            times = times[times <= np.max(times_unique)]
            print(f"[select_DEgenes_and_split] Filtered to times <= {np.max(times_unique)}")

        for t in np.unique(times):
            for i in np.unique(samples_id):
                var_bool = (times == t) & (samples_id == i)
                indices = adata.obs[var_bool].index.values
                np.random.shuffle(indices)
                split_point = int(len(indices) * float(rate))
                train_idx.extend(indices[:split_point])
                test_idx.extend(indices[split_point:])

        print(f"[select_DEgenes_and_split] Split results: {len(train_idx)} train cells, {len(test_idx)} test cells")

        adata_test = adata[test_idx].copy()
        adata = adata[train_idx]
        genes_list_tokeep = genes_list_init.copy()

        if not ('copycat' in p or 'Schiebinger' in p):
            print("[select_DEgenes_and_split] Re-filtering genes on training set")
            times_train = adata.obs['time'].values if 'time' in adata.obs else np.zeros(adata.n_obs)
            vect_samples_id_train = adata.obs['dataset_id'].values if 'dataset_id' in adata.obs else np.zeros(adata.n_obs)
            vect_celltype_id_train = adata.obs['cell_type'].values if 'cell_type' in adata.obs else np.zeros(adata.n_obs)

            model = NetworkModel_beta(adata.shape[1])
            if mean_forcing >= 0:
                model.mean_forcing_em = mean_forcing

            try:
                model.fit_mixture(adata, gene_names=list(adata.var_names),
                                  min_components=2, max_components=2,
                                  max_iter_kinetics=0, refilter=.001)
                genes_list_tokeep, temporal_variations, cell_type_variations, df_report = select_DEgenes(
                        times_train, vect_samples_id_train, vect_celltype_id_train,
                        model.proba, adata.var_names,
                        n_genes_tokeep_celltype=[10000], n_genes_tokeep_temporal=[10000], limit_min=.001)
                print(f"[select_DEgenes_and_split] Training set filtering retained {len(genes_list_tokeep)} genes")

            except Exception as e:
                print(f"[select_DEgenes_and_split] Error during training set gene filtering: {e}")
                sys.exit(1)

        genes_list_init.sort()
        genes_list_final = [g for g in genes_list_init if g in genes_list_tokeep]
        adata = adata[:, genes_list_final]
        adata_test = adata_test[:, genes_list_final]

        try:
            adata.write(os.path.join(p, 'Data', 'data_train.h5ad'))
            adata_test.write(os.path.join(p, 'Data', 'data_test.h5ad'))
            print(f"[select_DEgenes_and_split] Saved train/test datasets")
            print(f"[select_DEgenes_and_split] Train: {adata.shape[0]} cells, {adata.shape[1]} genes")
            print(f"[select_DEgenes_and_split] Test: {adata_test.shape[0]} cells, {adata_test.shape[1]} genes")
        except Exception as e:
            print(f"[select_DEgenes_and_split] Error saving train/test datasets: {e}")
            sys.exit(1)

    print("[select_DEgenes_and_split] Gene selection and splitting completed successfully")

if __name__ == "__main__":
   main(sys.argv[1:])
