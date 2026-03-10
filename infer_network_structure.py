"""
infer_network_structure.py
---------------------------
Gene regulatory network structure inference pipeline.

Estimates the regulatory interactions between genes by solving an optimal
transport problem that matches the temporal dynamics of the inferred
mixture model to observed transcriptomic data.

Usage:
    python infer_network_structure.py -i <project_path> -s <split>

Required input files:
    - Data/data_<split>.h5ad: count matrix with degradation rates
    - cardamom/modes.npy, proba.npy, mixture_parameters.npy: from infer_mixture.py

Output files:
    - cardamom/inter.npy: gene interaction matrix (G×G)
    - cardamom/basal.npy: basal expression parameters (G×1)
    - cardamom/alpha.npy: cellular scaling factors
"""
import sys; sys.path += ['../']
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta
import getopt
import anndata as ad
import pandas as pd
import scipy.sparse
import os
import pickle

def main(argv):
    """
    Infer the gene regulatory network structure from temporal scRNA-seq data.

    Loads inferred mixture parameters and fits a mechanistic gene expression
    model to deduce regulatory interactions and degradation dynamics.

    Args:
        argv: Command-line arguments (--input, --split).
    """
    inputfile = ''
    split = ''
    try:
        opts, args = getopt.getopt(argv, "hi:s:", ["input=", "split="])
    except getopt.GetoptError:
        print("Error: Invalid arguments. Use: infer_network_structure.py -i <project> -s <split>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        if opt in ("-s", "--split"):
            split = '{}'.format(arg)

    p = '{}/'.format(inputfile)

    data_path = os.path.join(p, 'Data', 'data_{}.h5ad'.format(split))
    if os.path.exists(data_path):
        adata = ad.read_h5ad(data_path)
        print(f"[infer_network_structure] Loaded data from {data_path}")
    else:
        error_msg = (
            f"Error: Data file not found at {data_path}.\n"
            f"Ensure you have created a 'data_{split}.h5ad' file in the Data/ directory."
        )
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    if scipy.sparse.issparse(adata.X):
        data_rna_extracted = adata.X.T.toarray()
    else:
        data_rna_extracted = adata.X.T

    # ─── CHECK TEMPORAL INFORMATION ──────────────────────────────────────
    try:
        times = adata.obs['time'].values 
        if len(np.unique(times)) <= 1:
            raise ValueError(
                "Data must contain temporal information with at least 2 distinct timepoints."
            )
    except KeyError as e:
        error_msg = f"Error: 'time' column not found in adata.obs. {e}"
        print(error_msg)
        raise SystemExit(error_msg)
    data_rna = np.vstack([times, data_rna_extracted]).T
    vect_samples_id = adata.obs['dataset_id'].values if 'dataset_id' in adata.obs else np.zeros(adata.n_obs)
    G = np.size(data_rna, 1)
    genes_list = ['Stimulus'] + list(adata.var_names[:])

    print(f"[infer_network_structure] Starting network inference ({G-1} genes)...")

    # ─── LOAD MIXTURE PARAMETERS ────────────────────────────────────────
    try:
        model = NetworkModel_beta(G-1)
        model.modes = np.load(os.path.join(p, 'cardamomOT', 'modes.npy'))
        with open(os.path.join(p, 'cardamomOT', 'pi_init.pkl'), "rb") as f:
            model.pi_init = pickle.load(f)
        model.proba = np.load(os.path.join(p, 'cardamomOT', 'proba.npy'))
        model.proba_init = np.load(os.path.join(p, 'cardamomOT', 'proba_init.npy'))
        model.a = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        model.n_networks = np.load(os.path.join(p, 'cardamomOT', 'n_networks.npy'))

        print(f"[infer_network_structure] Loaded mixture parameters ({model.n_networks} networks)")
    except FileNotFoundError as e:
        error_msg = (
            f"Error: Required mixture parameter files not found. "
            f"Run infer_mixture.py first. {e}"
        )
        print(error_msg)
        raise SystemExit(error_msg)
    
    model.ref_network = np.ones((G, G, model.n_networks))
    ref_path = os.path.join(p, 'cardamomOT', 'ref_network.csv')
    if os.path.exists(ref_path):
        # 1️⃣ Charger la matrice complète depuis le CSV
        ref_df = pd.read_csv(ref_path, index_col=0)
        # 2️⃣ S’assurer que les noms des colonnes sont bien des chaînes
        ref_df.columns = ref_df.columns.astype(str)
        ref_df.index = ref_df.index.astype(str)
        # 3️⃣ Filtrer les gènes présents à la fois dans ref_df et ta liste
        common_genes = [g for g in genes_list if g in ref_df.index]
        # 4️⃣ Extraire la sous-matrice dans le bon ordre (ligne + colonne)
        sub_df = ref_df.loc[common_genes, common_genes]
        # 5️⃣ Convertir en numpy array si besoin
        ref_mat = sub_df.to_numpy()
        if ref_mat.shape[0] == G:
            for n in range(0, model.n_networks):
                model.ref_network[:, :, n] = np.abs(ref_mat)

    # Infer network
    model.d = np.ones((2, G))
    model.d[1, 1:], model.d[0, 1:] = adata.var['d1'].values, adata.var['d0'].values

    model.fit_network(data_rna, intensity_prior=100, vect_samples_id=vect_samples_id, verb=1)

    # Save inferred network structure parameters
    cardamom_dir = os.path.join(p, 'cardamomOT')
    np.save(os.path.join(cardamom_dir, 'basal'), model.basal)
    np.save(os.path.join(cardamom_dir, 'inter'), model.inter)
    np.save(os.path.join(cardamom_dir, f'inter_stim{model.stimulus}_prior{model.prior_network_pen}'), model.inter)
    np.save(os.path.join(cardamom_dir, 'basal_tmp'), model.basal_tmp)
    np.save(os.path.join(cardamom_dir, 'inter_tmp'), model.inter_tmp)
    np.save(os.path.join(cardamom_dir, 'data_prot'), model.prot)
    np.save(os.path.join(cardamom_dir, 'data_rna'), model.rna)
    np.save(os.path.join(cardamom_dir, 'data_times'), model.times_data)
    np.save(os.path.join(cardamom_dir, 'data_samples'), model.samples_data)
    np.save(os.path.join(cardamom_dir, 'proba_traj'), model.proba_traj)
    np.save(os.path.join(cardamom_dir, 'data_kon_theta'), model.kon_theta)
    np.save(os.path.join(cardamom_dir, 'data_kon_beta'), model.kon_beta)
    np.save(os.path.join(cardamom_dir, 'alpha'), model.alpha)
    np.save(os.path.join(cardamom_dir, 'degradations'), model.d)
    
    print(f"[infer_network_structure] Successfully saved network inference results to {cardamom_dir}")
    

if __name__ == "__main__":
   main(sys.argv[1:])

