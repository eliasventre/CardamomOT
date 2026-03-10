"""
check_test_to_train.py
----------------------
Validate network inference by comparing test set predictions to observations.

Compares inferred network dynamics on test set with observed test data.
Generates plots and AnnData objects for visualization of inference quality.

Usage:
    python check_test_to_train.py -i <project_path> -s <split> [-t <stim>] [-p <prior>]

Required input files:
    - Data/data_<split>.h5ad: observed test data
    - cardamom/data_kon_beta_test.npy, data_kon_theta_test.npy: inferred dynamics
    - cardamom/data_kon_simul_test.npy, data_rna_test.npy: simulation data
    - cardamom/mixture_parameters.npy, pi_zinb.npy: model parameters

Output files:
    - cardamom/adata_beta_test.h5ad: beta parameter trajectory
    - cardamom/adata_theta_test.h5ad: theta parameter trajectory
    - cardamom/adata_sim_test.h5ad: simulated trajectory
    - results_article/*_test_to_train.png: comparison plots
"""
import numpy as np
import sys
import getopt
import anndata as ad
from CardamomOT import NetworkModel, plot_data_distrib, plot_data_umap_altogether
import scipy.sparse
import os


def main(argv):
    """
    Validate network inference on test set.

    Loads test data and inferred dynamics, generates predictions,
    and compares with observed data through plots and saved AnnData objects.

    Args:
        argv: Command-line arguments (--input, --split, --stim, --prior).
    
    Returns:
        None. Saves validation plots and AnnData objects.
    """
    inputfile = ''
    split = ''
    stim = 1
    prior = 1
    try:
        opts, args = getopt.getopt(argv, "hi:s:t:p:", ["input=", "split=", "stim=", "prior="])
    except getopt.GetoptError:
        print("[check_test_to_train] Error: Invalid command-line arguments")
        print("[check_test_to_train] Usage: python check_test_to_train.py -i <project_path> -s <split> [-t <stim>] [-p <prior>]")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-s", "--split"):
            split = '{}'.format(arg)
        elif opt in ("-t", "--stim"):
            stim = float(arg)
        elif opt in ("-p", "--prior"):
            prior = float(arg)
        elif opt == "-h":
            print(__doc__)
            sys.exit(0)

    if not inputfile or not split:
        print("[check_test_to_train] Error: Missing required arguments --input and --split")
        sys.exit(1)

    p = '{}/'.format(inputfile)

    # Load observed test data
    data_path = os.path.join(p, 'Data', 'data_{}.h5ad'.format(split))
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Test data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[check_test_to_train] Loaded test data from {data_path}")
        print(f"[check_test_to_train] Dataset contains {adata.shape[0]} cells and {adata.shape[1]} genes")
    except FileNotFoundError as e:
        print(f"[check_test_to_train] Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[check_test_to_train] Error loading test data: {e}")
        sys.exit(1)
    
    # Extract RNA data matrix
    if scipy.sparse.issparse(adata.X):
        data_rna_extracted = adata.X.T.toarray()
    else:
        data_rna_extracted = adata.X.T
     
    # Validate temporal information
    try:
        times = adata.obs['time'].values 
        if len(np.unique(times)) <= 1:
            raise ValueError("Dataset must contain temporal information with multiple timepoints")
        print(f"[check_test_to_train] Found {len(np.unique(times))} unique timepoints")
    except (KeyError, ValueError) as e:
        print(f"[check_test_to_train] Error: {e}")
        sys.exit(1)
    
    data_real = np.vstack([times, data_rna_extracted]).astype(float)

    # Load model parameters
    try:
        mixture_parameters = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        c = mixture_parameters[-1, :]
        kz = mixture_parameters[:-1, :] + 1e-6
        pi_zinb = np.load(os.path.join(p, 'cardamomOT', 'pi_zinb.npy'))
        print(f"[check_test_to_train] Loaded model parameters")
    except FileNotFoundError as e:
        print(f"[check_test_to_train] Error: Missing parameter file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[check_test_to_train] Error loading parameters: {e}")
        sys.exit(1)

    # Load inference results
    try:
        vect_kon_beta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_beta_test.npy')) + 1e-6
        vect_kon_theta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_theta_test.npy')) + 1e-6
        vect_kon_sim = np.load(os.path.join(p, 'cardamomOT', 'data_kon_simul_test.npy')) + 1e-6
        rna_ref = np.load(os.path.join(p, 'cardamomOT', 'data_rna_test.npy'))
        times_data = np.load(os.path.join(p, 'cardamomOT', 'data_times_test.npy'))
        times_simulation = np.load(os.path.join(p, 'cardamomOT', 'simulation_times_test.npy'))
        print(f"[check_test_to_train] Loaded test set inference and simulation results")
    except FileNotFoundError as e:
        print(f"[check_test_to_train] Error: Missing forecast file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[check_test_to_train] Error loading forecast data: {e}")
        sys.exit(1)
   
    # Extract gene names and timepoints
    names = adata.var_names
    t_data = list(set(times_data))
    t_data.sort()
    t_simul = list(set(times_simulation))
    t_simul.sort()

    print(f"[check_test_to_train] Data timepoints: {t_data}, Simulation timepoints: {t_simul}")

    # Generate simulated data with noise
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

    # Add negative binomial noise and sparsity for simulations
    zero_mask = (np.random.uniform(0, 1, (data_sim[1:, :].shape)) < pi_zinb.reshape((G, 1)))
    print(f'[check_test_to_train] Sparsity ratio (simulation): {np.sum(zero_mask == 1)/np.size(data_sim[1:, :]):.3f}')
    data_sim[1:, :] = np.random.negative_binomial((np.max(kz, 0)*vect_kon_sim)[:, 1:].T, (c / (c+1))[1:].reshape(G, 1))
    data_sim[1:, :] = np.where(zero_mask, 0, data_sim[1:, :])

    zero_mask = (np.random.uniform(0, 1, (data_beta[1:, :].shape)) < pi_zinb.reshape((G, 1)))
    print(f'[check_test_to_train] Sparsity ratio (beta): {np.sum(zero_mask == 1)/np.size(data_beta[1:, :]):.3f}')
    data_beta[1:, :] = np.random.negative_binomial((np.max(kz, 0)*vect_kon_beta)[:, 1:].T, (c / (c+1))[1:].reshape(G, 1))
    data_beta[1:, :] = np.where(zero_mask, 0, data_beta[1:, :])

    zero_mask = (np.random.uniform(0, 1, (data_netw_theta[1:, :].shape)) < pi_zinb.reshape((G, 1)))
    print(f'[check_test_to_train] Sparsity ratio (theta): {np.sum(zero_mask == 1)/np.size(data_netw_theta[1:, :]):.3f}')
    data_netw_theta[1:, :] = np.random.negative_binomial((np.max(kz, 0)*vect_kon_theta)[:, 1:].T, (c / (c+1))[1:].reshape(G, 1))
    data_netw_theta[1:, :] = np.where(zero_mask, 0, data_netw_theta[1:, :])

    # Initialize model for plotting
    model = NetworkModel(G)
    model.stimulus = stim
    model.prior_network_pen = prior

    print(f"[check_test_to_train] Generating validation plots...")
    try:
        # Generate plots comparing predictions to observations
        order = np.arange(len(names[1:]))
        names = [names[0]] + list(names[order+1])

        plot_data_distrib(data_real, data_sim, t_data, t_simul, names, inputfile, 'Check', 'test_to_train')
        plot_data_umap_altogether(data_real, data_ref, data_beta, data_netw_theta, data_sim, t_data, t_simul, inputfile, 'Check', 'altogether_test')
        print(f"[check_test_to_train] Generated comparison plots")
    except Exception as e:
        print(f"[check_test_to_train] Warning: Error generating plots: {e}")

    # Save AnnData objects
    try:
        adata_beta = ad.AnnData(X=data_beta[1:, ].T)
        adata_beta.var = adata.var.copy()
        adata_beta.obs['time'] = times_simulation
        adata_beta.write(os.path.join(p, f'cardamom/adata_beta_test_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad'))
        print(f"[check_test_to_train] Saved beta parameter trajectory")

        adata_theta = ad.AnnData(X=data_netw_theta[1:, ].T)
        adata_theta.var = adata.var.copy()
        adata_theta.obs['time'] = times_simulation
        adata_theta.write(os.path.join(p, f'cardamom/adata_theta_test_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad'))
        print(f"[check_test_to_train] Saved theta parameter trajectory")

        adata_sim = ad.AnnData(X=data_sim[1:, ].T)
        adata_sim.var = adata.var.copy()
        adata_sim.obs['time'] = times_simulation
        adata_sim.write(os.path.join(p, f'cardamom/adata_sim_test_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad'))
        print(f"[check_test_to_train] Saved simulated trajectory")
        print("[check_test_to_train] Test set validation completed successfully")
    except Exception as e:
        print(f"[check_test_to_train] Error saving AnnData objects: {e}")
        sys.exit(1)

if __name__ == "__main__":
   main(sys.argv[1:])
