"""
get_kinetic_rates.py
--------------------
Extract and assign kinetic degradation rates to genes.

Loads mammalian half-life data and extracts degradation rates for mRNA
and protein. Assigns these rates to genes in the dataset, with bounds
checking to ensure reasonable values for downstream modeling.

Usage:
    python get_kinetic_rates.py -i <project_path> -s <split>

Required input files:
    - Data/data_full.h5ad: full count matrix
    - halflife/table_halflife_mamalian.csv: mammalian half-life data

Output files:
    - Data/data_full.h5ad: updated with d0 (mRNA) and d1 (protein) degradation rates
    - Data/data_train.h5ad, data_test.h5ad: updated with degradation rates (if split != "full")
"""
import sys; sys.path += ['../']
import os
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta
from CardamomOT import select_DEgenes, extract_degradation_rates
import anndata as ad
import getopt
import scipy.sparse
import json
import pandas as pd

verb = 1

def main(argv):
    """
    Extract and assign kinetic degradation rates to genes.

    Loads mammalian half-life data from literature and extracts degradation
    rates for mRNA (d0) and protein (d1) for each gene. Applies bounds checking
    to ensure rates are within reasonable ranges for modeling.

    Args:
        argv: Command-line arguments (--input, --split).
    
    Returns:
        None. Updates AnnData files with degradation rates.
    """
    inputfile = ''
    split = ''
    try:
        opts, args = getopt.getopt(argv, "hi:s:", ["input=", "split="])
    except getopt.GetoptError:
        print("[get_kinetic_rates] Error: Invalid command-line arguments")
        print("[get_kinetic_rates] Usage: python get_kinetic_rates.py -i <project_path> -s <split>")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-s", "--split"):
            split = arg
        elif opt == "-h":
            print(__doc__)
            sys.exit(0)

    if not inputfile:
        print("[get_kinetic_rates] Error: Missing required argument --input")
        sys.exit(1)

    p = '{}/'.format(inputfile)

    # Load full dataset
    data_path = os.path.join(p, 'Data', 'data_full.h5ad')
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Full data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[get_kinetic_rates] Loaded full dataset from {data_path}")
        print(f"[get_kinetic_rates] Dataset contains {adata.shape[0]} cells and {adata.shape[1]} genes")
    except FileNotFoundError as e:
        print(f"[get_kinetic_rates] Error: {e}")
        print(f"[get_kinetic_rates] Please ensure Data/data_full.h5ad exists in {p}")
        sys.exit(1)

    # Load degradation rates from mammalian half-life data
    csv_path = os.path.join("halflife", "table_halflife_mamalian.csv")
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Half-life data not found at {csv_path}")
        df = pd.read_csv(csv_path, sep=',')
        print(f"[get_kinetic_rates] Loaded half-life data from {csv_path}")
        print(f"[get_kinetic_rates] Half-life table contains {len(df)} entries")
    except FileNotFoundError as e:
        print(f"[get_kinetic_rates] Error: {e}")
        print("[get_kinetic_rates] Please ensure halflife/table_halflife_mamalian.csv exists")
        sys.exit(1)
    except Exception as e:
        print(f"[get_kinetic_rates] Error loading half-life data: {e}")
        sys.exit(1)

    # Extract degradation rates for genes in dataset
    try:
        deg = extract_degradation_rates(df, adata.var_names)
        print(f"[get_kinetic_rates] Extracted degradation rates for {len(deg[0])} genes")
    except Exception as e:
        print(f"[get_kinetic_rates] Error extracting degradation rates: {e}")
        sys.exit(1)

    # Assign mRNA degradation rates (d0) with bounds checking
    if 'd0' not in adata.var.columns:
        adata.var['d0'] = deg[0]
        # Apply bounds: 0.1x to 10x mean
        mean_d0 = np.mean(deg[0])
        adata.var['d0'] = np.minimum(adata.var['d0'], 10 * mean_d0)
        adata.var['d0'] = np.maximum(adata.var['d0'], mean_d0 / 10)
        print(f"[get_kinetic_rates] Assigned mRNA degradation rates (d0), mean: {mean_d0:.4f}")
    else:
        print("[get_kinetic_rates] mRNA degradation rates (d0) already present, skipping")

    # Assign protein degradation rates (d1) with bounds checking
    if 'd1' not in adata.var.columns:
        adata.var['d1'] = deg[1]
        # Apply bounds: 0.1x to 10x mean
        mean_d1 = np.mean(deg[1])
        adata.var['d1'] = np.minimum(adata.var['d1'], 10 * mean_d1)
        adata.var['d1'] = np.maximum(adata.var['d1'], mean_d1 / 10)
        print(f"[get_kinetic_rates] Assigned protein degradation rates (d1), mean: {mean_d1:.4f}")
    else:
        print("[get_kinetic_rates] Protein degradation rates (d1) already present, skipping")

    # Save updated full dataset
    try:
        adata.write(os.path.join(p, 'Data', 'data_full.h5ad'))
        print(f"[get_kinetic_rates] Saved updated full dataset to {os.path.join(p, 'Data', 'data_full.h5ad')}")
    except Exception as e:
        print(f"[get_kinetic_rates] Error saving full dataset: {e}")
        sys.exit(1)

    # Handle train/test splits if specified
    if split != "full":
        print(f"[get_kinetic_rates] Processing train/test split: {split}")
        
        # Load train data
        train_path = os.path.join(p, 'Data', 'data_train.h5ad')
        test_path = os.path.join(p, 'Data', 'data_test.h5ad')
        
        try:
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Train data not found at {train_path}")
            adata_train = ad.read_h5ad(train_path)
            print(f"[get_kinetic_rates] Loaded train dataset from {train_path}")
        except FileNotFoundError as e:
            print(f"[get_kinetic_rates] Error: {e}")
            sys.exit(1)

        try:
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test data not found at {test_path}")
            adata_test = ad.read_h5ad(test_path)
            print(f"[get_kinetic_rates] Loaded test dataset from {test_path}")
        except FileNotFoundError as e:
            print(f"[get_kinetic_rates] Error: {e}")
            sys.exit(1)

        # Extract degradation rates for train genes
        try:
            deg_train = extract_degradation_rates(df, adata_train.var_names)
        except Exception as e:
            print(f"[get_kinetic_rates] Error extracting train degradation rates: {e}")
            sys.exit(1)

        # Assign rates to train data with bounds checking
        if 'd0' not in adata_train.var.columns:
            adata_train.var['d0'] = deg_train[0]
            mean_d0_train = np.mean(deg_train[0])
            adata_train.var['d0'] = np.minimum(adata_train.var['d0'], 10 * mean_d0_train)
            adata_train.var['d0'] = np.maximum(adata_train.var['d0'], mean_d0_train / 10)

        if 'd1' not in adata_train.var.columns:
            adata_train.var['d1'] = deg_train[1]
            mean_d1_train = np.mean(deg_train[1])
            adata_train.var['d1'] = np.minimum(adata_train.var['d1'], 10 * mean_d1_train)
            adata_train.var['d1'] = np.maximum(adata_train.var['d1'], mean_d1_train / 10)

        # Copy rates to test data
        adata_test.var['d1'] = adata_train.var['d1'].values
        adata_test.var['d0'] = adata_train.var['d0'].values

        # Save updated train/test datasets
        try:
            adata_train.write(os.path.join(p, 'Data', 'data_train.h5ad'))
            adata_test.write(os.path.join(p, 'Data', 'data_test.h5ad'))
            print(f"[get_kinetic_rates] Saved updated train/test datasets")
        except Exception as e:
            print(f"[get_kinetic_rates] Error saving train/test datasets: {e}")
            sys.exit(1)

    # Report final statistics
    mean_d1 = np.mean(adata.var['d1'].values)
    mean_d0 = np.mean(adata.var['d0'].values)
    print(f"[get_kinetic_rates] Final statistics:")
    print(f"[get_kinetic_rates]   Mean protein degradation rate (d1): {mean_d1:.4f}")
    print(f"[get_kinetic_rates]   Mean mRNA degradation rate (d0): {mean_d0:.4f}")
    print("[get_kinetic_rates] Kinetic rates assignment completed successfully")

if __name__ == "__main__":
   main(sys.argv[1:])

