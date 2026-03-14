"""
check_sim_to_data.py
--------------------
Compare network simulations with observed data.

Validates the quality of the inferred gene regulatory network by comparing
simulated expression dynamics with the real data used for inference. Computes
zero-inflation adjusted distributions and generates comparison plots.

Usage:
    python check_sim_to_data.py -i <project_path> -s <split>

Required input files:
    - Data/data_<split>.h5ad: observed count matrix
    - cardamom/data_rna.npy: RNA expression from network inference
    - cardamom/data_prot_simul.npy: simulated protein abundance
    - cardamom/simulation_times.npy: timepoints used for simulation

Output files:
    - cardamom/adata_*_stim*.h5ad: generated AnnData objects for visualization
    - Check/sim_vs_data/ directory: distribution comparison plots
"""

import numpy as np
import sys, getopt
import anndata as ad
from CardamomOT import NetworkModel, plot_data_umap_altogether, plot_data_distrib
import scipy.sparse
import os

plot_in_script = 0

def main(argv):
    """
    Compare network simulation results with observed data.

    Generates synthetic data from simulated network dynamics and compares
    expression distributions with the real observed data to assess network
    inference quality. Saves comparison datasets and generates visualizations.

    Args:
        argv: Command-line arguments (--input, --split).
    
    Returns:
        None. Saves comparison datasets and comparison plots.
    """
    inputfile = ''
    split = ''
    try:
        opts, args = getopt.getopt(argv, "hi:s:", ["input=", "split="])
    except getopt.GetoptError:
        print("[check_sim_to_data] Error: Invalid command-line arguments")
        print("[check_sim_to_data] Usage: python check_sim_to_data.py -i <project_path> -s <split>")
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
        print("[check_sim_to_data] Error: Missing required arguments --input and --split")
        sys.exit(1)

    p = '{}/'.format(inputfile)

    outputfile = 'Check'
    complement1 = 'sim_vs_data'

    # Load observed expression data
    data_path = os.path.join(p, 'Data', 'data_{}.h5ad'.format(split))
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[check_sim_to_data] Loaded data from {data_path}")
    except FileNotFoundError as e:
        print(f"[check_sim_to_data] Error: {e}")
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
        print(f"[check_sim_to_data] Detected {len(np.unique(times))} observable timepoints")
    except KeyError:
        print("[check_sim_to_data] Error: data.obs['time'] not found")
        sys.exit(1)
    except ValueError as e:
        print(f"[check_sim_to_data] Error: {e}")
        sys.exit(1)
    
    data_real = np.vstack([times, data_rna_extracted]).astype(float)

    # Load mixture and simulation parameters
    print("[check_sim_to_data] Loading mixture and simulation parameters...")
    try:
        mixture_parameters = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        c = mixture_parameters[-1, :]
        kz = mixture_parameters[:-1, :] + 1e-6
        pi_zinb = np.load(os.path.join(p, 'cardamomOT', 'pi_zinb.npy'))
        
        vect_kon_beta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_beta.npy')) + 1e-6
        vect_kon_theta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_theta.npy')) + 1e-6
        vect_kon_sim = np.load(os.path.join(p, 'cardamomOT', 'data_kon_simul.npy')) + 1e-6
        times_data = np.load(os.path.join(p, 'cardamomOT', 'data_times.npy'))
        times_simulation = np.load(os.path.join(p, 'cardamomOT', 'simulation_times.npy'))
        rna_ref = np.load(os.path.join(p, 'cardamomOT', 'data_rna.npy'))
        print("[check_sim_to_data] Successfully loaded all parameters")
    except FileNotFoundError as e:
        print(f"[check_sim_to_data] Error: Missing parameter file: {e}")
        print("[check_sim_to_data] Please ensure network inference and simulation have been completed")
        sys.exit(1)
   
    names = adata.var_names
    t_data = list(set(times_data))
    t_data.sort()
    t_simul = list(set(times_simulation))
    t_simul.sort()
    print(f"[check_sim_to_data] Real data timepoints: {t_data}")
    print(f"[check_sim_to_data] Simulation timepoints: {t_simul}")

    # Generate synthetic data from mixture and simulation models
    print("[check_sim_to_data] Generating synthetic data for distribution comparison...")
    G = np.size(data_real, 0)-1
    data_ref = np.zeros((G+1, np.size(vect_kon_beta, 0)))
    data_ref[0, :] = times_data[:]
    data_ref[1:, :] = rna_ref[:, 1:].T
    
    data_beta = np.zeros((G+1, np.size(vect_kon_beta, 0)))
    data_netw_theta = np.zeros((G+1, np.size(vect_kon_theta, 0)))
    data_sim = np.zeros((G+1, np.size(vect_kon_sim, 0)))
    data_beta[0, :] = times_data[:]
    data_netw_theta[0, :] = times_data[:]
    data_sim[0, :] = times_simulation[:]

    # Generate zero-inflated samples from bursting parameters
    zero_mask = (np.random.uniform(0, 1, (data_sim[1:, :].shape)) < pi_zinb.reshape((G, 1)))
    zero_ratio_sim = np.sum(zero_mask == 1)/np.size(data_sim[1:, :])
    print(f"[check_sim_to_data] Simulation zero-inflation ratio: {zero_ratio_sim:.4f}")
    data_sim[1:, :] = np.random.negative_binomial((np.max(kz, 0)*vect_kon_sim)[:, 1:].T, (c / (c+1))[1:].reshape(G, 1))
    data_sim[1:, :] = np.where(zero_mask, 0, data_sim[1:, :])

    zero_mask = (np.random.uniform(0, 1, (data_beta[1:, :].shape)) < pi_zinb.reshape((G, 1)))
    zero_ratio_beta = np.sum(zero_mask == 1)/np.size(data_beta[1:, :])
    print(f"[check_sim_to_data] Beta (mixture) zero-inflation ratio: {zero_ratio_beta:.4f}")
    data_beta[1:, :] = np.random.negative_binomial((np.max(kz, 0)*vect_kon_beta)[:, 1:].T, (c / (c+1))[1:].reshape(G, 1))
    data_beta[1:, :] = np.where(zero_mask, 0, data_beta[1:, :])

    zero_mask = (np.random.uniform(0, 1, (data_netw_theta[1:, :].shape)) < pi_zinb.reshape((G, 1)))
    zero_ratio_theta = np.sum(zero_mask == 1)/np.size(data_netw_theta[1:, :])
    print(f"[check_sim_to_data] Theta (network) zero-inflation ratio: {zero_ratio_theta:.4f}")
    data_netw_theta[1:, :] = np.random.negative_binomial((np.max(kz, 0)*vect_kon_theta)[:, 1:].T, (c / (c+1))[1:].reshape(G, 1))
    data_netw_theta[1:, :] = np.where(zero_mask, 0, data_netw_theta[1:, :])

    model = NetworkModel(G)
    cardamom_dir = os.path.join(p, 'cardamomOT')

    # Save comparison datasets
    print("[check_sim_to_data] Saving comparison datasets...")
    try:
        adata_beta = ad.AnnData(X=data_beta[1:, ].T)
        adata_beta.var = adata.var.copy()
        adata_beta.obs['time'] = times_data
        adata_beta.write(os.path.join(cardamom_dir, f'adata_beta_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad'))

        adata_theta = ad.AnnData(X=data_netw_theta[1:, ].T)
        adata_theta.var = adata.var.copy()
        adata_theta.obs['time'] = times_data
        adata_theta.write(os.path.join(cardamom_dir, f'adata_theta_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad'))

        adata_sim = ad.AnnData(X=data_sim[1:, ].T)
        adata_sim.var = adata.var.copy()
        adata_sim.obs['time'] = times_simulation
        adata_sim.write(os.path.join(cardamom_dir, f'adata_sim_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad'))

        adata_rna_traj = ad.AnnData(X=data_ref[1:, :].T)
        adata_rna_traj.var = adata.var.copy()
        adata_rna_traj.obs['time'] = times_data
        adata_rna_traj.write(os.path.join(cardamom_dir, f'adata_rna_traj_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad'))

        data_prot_traj = np.load(os.path.join(cardamom_dir, 'data_prot_unitary.npy'))
        adata_prot_traj = ad.AnnData(X=data_prot_traj[:, 1:])
        adata_prot_traj.var = adata.var.copy()
        adata_prot_traj.obs['time'] = times_data
        adata_prot_traj.write(os.path.join(cardamom_dir, f'adata_prot_traj_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad'))
        
        data_prot_simul = np.load(os.path.join(cardamom_dir, 'data_prot_simul.npy'))
        adata_prot_simul = ad.AnnData(X=data_prot_simul[:, 1:])
        adata_prot_simul.var = adata.var.copy()
        adata_prot_simul.obs['time'] = times_simulation
        adata_prot_simul.write(os.path.join(cardamom_dir, f'adata_prot_simul_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad'))
        
        print(f"[check_sim_to_data] Successfully saved comparison datasets to {cardamom_dir}")
    except Exception as e:
        print(f"[check_sim_to_data] Error saving datasets: {e}")

    if plot_in_script:
        print("[check_sim_to_data] Generating distribution comparison plots...")
        try:
            plot_data_distrib(data_real, data_sim, t_data, t_simul, names, inputfile, outputfile, complement1)
            print("[check_sim_to_data] Plots successfully generated")
        except Exception as e:
            print(f"[check_sim_to_data] Warning: Could not generate plots: {e}")

if __name__ == "__main__":
   main(sys.argv[1:])
