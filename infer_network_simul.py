"""
infer_network_simul.py
----------------------
Adapt inferred network parameters for simulation.

Loads the inferred network structure and adapts parameters to enable
simulation of gene expression dynamics. Handles reference network loading
and parameter transformation for downstream simulation steps.

Usage:
    python infer_network_simul.py -i <project_path> -s <split>

Required input files:
    - Data/data_<split>.h5ad: count matrix with temporal information
    - cardamom/inter.npy, basal.npy: inferred network parameters
    - cardamom/mixture_parameters.npy: burst kinetics parameters
    - cardamom/ref_network.csv: optional reference network (tab-separated)

Output files:
    - cardamom/data_prot_unitary.npy: adapted protein abundance
    - cardamom/data_kon_unitary.npy: adapted bursting parameters
    - cardamom/basal_simul.npy, inter_simul.npy: simulation-ready parameters
    - cardamom/basal_t_simul.npy, inter_t_simul.npy: temporal parameters
    - cardamom/ratios.npy: parameter ratios
    - cardamom/degradations_temporal.npy: temporal degradation rates
"""
import sys; sys.path += ['../']
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta
import getopt
import anndata as ad
import pandas as pd
import scipy.sparse
import os

verb = 1

def main(argv):
    """
    Adapt inferred network parameters for simulation.

    Loads the inferred gene regulatory network and transforms parameters
    to enable stochastic simulation of expression dynamics. Optionally
    incorporates reference network information for improved inference.

    Args:
        argv: Command-line arguments (--input, --split).
    
    Returns:
        None. Saves adapted parameters to cardamom/ directory.
    """
    inputfile = ''
    split = ''
    try:
        opts, args = getopt.getopt(argv, "hi:s:", ["input=", "split="])
    except getopt.GetoptError:
        print("[infer_network_simul] Error: Invalid command-line arguments")
        print("[infer_network_simul] Usage: python infer_network_simul.py -i <project_path> -s <split>")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-s", "--split"):
            split = '{}'.format(arg)
        elif opt == "-h":
            print(__doc__)
            sys.exit(0)

    if not inputfile or not split:
        print("[infer_network_simul] Error: Missing required arguments --input and --split")
        sys.exit(1)

    p = '{}/'.format(inputfile)

    # Load gene expression data
    data_path = os.path.join(p, 'Data', 'data_{}.h5ad'.format(split))
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[infer_network_simul] Loaded data from {data_path}")
    except FileNotFoundError as e:
        print(f"[infer_network_simul] Error: {e}")
        print(f"[infer_network_simul] Please ensure Data/data_{split}.h5ad exists in {p}")
        sys.exit(1)
    
    # Extract count matrix
    if scipy.sparse.issparse(adata.X):
        data_rna_extracted = adata.X.T.toarray()
    else:
        data_rna_extracted = adata.X.T
    
    # Validate temporal information
    try:
        times = adata.obs['time'].values 
        if len(np.unique(times)) <= 1:
            raise ValueError("Data must contain multiple timepoints in obs['time']")
        print(f"[infer_network_simul] Detected {len(np.unique(times))} timepoints")
    except KeyError:
        print("[infer_network_simul] Error: data.obs['time'] not found")
        sys.exit(1)
    except ValueError as e:
        print(f"[infer_network_simul] Error: {e}")
        sys.exit(1)
    
    data_rna = np.vstack([times, data_rna_extracted]).T
    G = np.size(data_rna, 1)
    genes_list = ['Stimulus'] + list(adata.var_names[:])
    print(f"[infer_network_simul] Data shape: {G} genes, {np.size(data_rna, 0)} cells")

    model = NetworkModel_beta(G-1)

    # Load inferred network parameters
    print("[infer_network_simul] Loading inferred network parameters...")
    try:
        model.d = np.load(os.path.join(p, 'cardamomOT', 'degradations.npy'))
        model.basal = np.load(os.path.join(p, 'cardamomOT', 'basal.npy'))
        model.inter = np.load(os.path.join(p, 'cardamomOT', 'inter.npy'))
        model.a = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        model.modes = np.load(os.path.join(p, 'cardamomOT', 'modes.npy'))
        model.prot = np.load(os.path.join(p, 'cardamomOT', 'data_prot.npy'))
        model.rna = np.load(os.path.join(p, 'cardamomOT', 'data_rna.npy'))
        model.times_data = np.load(os.path.join(p, 'cardamomOT', 'data_times.npy'))
        model.samples_data = np.load(os.path.join(p, 'cardamomOT', 'data_samples.npy'))
        model.kon_theta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_theta.npy'))
        model.kon_beta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_beta.npy'))
        model.alpha = np.load(os.path.join(p, 'cardamomOT', 'alpha.npy'))
        model.proba_traj = np.load(os.path.join(p, 'cardamomOT', 'proba_traj.npy'))
        model.n_networks = np.load(os.path.join(p, 'cardamomOT', 'n_networks.npy'))
        print("[infer_network_simul] Successfully loaded all network parameters")
    except FileNotFoundError as e:
        print(f"[infer_network_simul] Error: Missing parameter file: {e}")
        print("[infer_network_simul] Please ensure network inference has been completed")
        sys.exit(1)
    except Exception as e:
        print(f"[infer_network_simul] Error loading parameters: {e}")
        sys.exit(1)

    # Load reference network if available
    model.ref_network = np.ones((G, G, model.n_networks))
    ref_path = os.path.join(p, 'cardamomOT', 'ref_network.csv')
    if os.path.exists(ref_path):
        print(f"[infer_network_simul] Loading reference network from {ref_path}")
        try:
            # Load the complete matrix from CSV
            ref_df = pd.read_csv(ref_path, index_col=0)
            # Ensure column names are strings
            ref_df.columns = ref_df.columns.astype(str)
            ref_df.index = ref_df.index.astype(str)
            # Filter genes present in both ref_df and gene list
            common_genes = [g for g in genes_list if g in ref_df.index]
            # Extract submatrix in correct order
            sub_df = ref_df.loc[common_genes, common_genes]
            # Convert to numpy array
            ref_mat = sub_df.to_numpy()
            if ref_mat.shape[0] == G:
                for n in range(model.n_networks):
                    model.ref_network[:, :, n] = np.abs(ref_mat)
                print(f"[infer_network_simul] Incorporated reference network with {len(common_genes)} genes")
            else:
                print(f"[infer_network_simul] Warning: Reference network size ({ref_mat.shape[0]}) doesn't match data ({G})")
        except Exception as e:
            print(f"[infer_network_simul] Warning: Could not load reference network: {e}")
    else:
        print("[infer_network_simul] No reference network found, using inferred network only")
    
    model.ref_network *= (np.abs(model.inter) > 0) # Re-filter with null interactions

    # Adapt parameters for simulation
    print("[infer_network_simul] Adapting parameters for simulation...")
    model.adapt_to_unitary()
    print("[infer_network_simul] Parameter adaptation completed")
    
    # Save adapted parameters
    cardamom_dir = os.path.join(p, 'cardamomOT')
    try:
        np.save(os.path.join(cardamom_dir, 'data_prot_unitary'), model.prot)
        np.save(os.path.join(cardamom_dir, 'data_kon_unitary'), model.kon_theta)
        np.save(os.path.join(cardamom_dir, 'basal_simul'), model.basal)
        np.save(os.path.join(cardamom_dir, 'inter_simul'), model.inter)
        np.save(os.path.join(cardamom_dir, 'basal_t_simul'), model.basal_t)
        np.save(os.path.join(cardamom_dir, 'inter_t_simul'), model.inter_t)
        np.save(os.path.join(cardamom_dir, 'ratios'), model.ratios)
        np.save(os.path.join(cardamom_dir, 'degradations_temporal.npy'), model.d_t)
        print(f"[infer_network_simul] Successfully saved adapted parameters to {cardamom_dir}")
    except Exception as e:
        print(f"[infer_network_simul] Error saving parameters: {e}")
        sys.exit(1)

if __name__ == "__main__":
   main(sys.argv[1:])

