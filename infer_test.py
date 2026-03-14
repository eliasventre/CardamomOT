"""
infer_test.py
-------------
Infer regulatory network and simulate on test set.

Loads test data and network parameters, performs network inference,
and generates simulations on the test dataset for validation.

Usage:
    python infer_test.py -i <project_path>

Required input files:
    - Data/data_test.h5ad: test count matrix
    - cardamom/basal.npy, inter.npy: network parameters
    - cardamom/mixture_parameters.npy, degradations.npy: kinetic parameters
    - cardamom/basal_simul.npy, inter_simul.npy: simulation parameters

Output files:
    - cardamom/data_prot_test.npy, data_rna_test.npy: inferred test dynamics
    - cardamom/data_prot_simul_test.npy, data_kon_simul_test.npy: simulations
    - cardamom/simulation_times_test.npy: simulation timepoints
"""
import sys; sys.path += ['../']
import os
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta
import getopt
import anndata as ad
import scipy.sparse

def main(argv):
    """
    Infer network dynamics on test set and generate simulations.

    Loads learned network parameters and performs network inference
    on test data. Then simulates expression trajectories using
    learned dynamics for validation and comparison.

    Args:
        argv: Command-line arguments (--input).
    
    Returns:
        None. Saves inference and simulation results.
    """
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:", ["input="])
    except getopt.GetoptError:
        print("[infer_test] Error: Invalid command-line arguments")
        print("[infer_test] Usage: python infer_test.py -i <project_path>")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        elif opt == "-h":
            print(__doc__)
            sys.exit(0)

    if not inputfile:
        print("[infer_test] Error: Missing required argument --input")
        sys.exit(1)

    p = '{}/'.format(inputfile)

    # Load test data
    data_path = os.path.join(p, 'Data', 'data_test.h5ad')
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Test data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[infer_test] Loaded test data from {data_path}")
        print(f"[infer_test] Dataset contains {adata.shape[0]} cells and {adata.shape[1]} genes")
    except FileNotFoundError as e:
        print(f"[infer_test] Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[infer_test] Error loading test data: {e}")
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
        print(f"[infer_test] Found {len(np.unique(times))} unique timepoints: {sorted(np.unique(times))}")
    except (KeyError, ValueError) as e:
        print(f"[infer_test] Error: {e}")
        sys.exit(1)
    
    data_rna = np.vstack([times, data_rna_extracted]).T
    vect_samples_id = adata.obs['dataset_id'].values if 'dataset_id' in adata.obs else np.zeros(adata.n_obs)
    G = np.size(data_rna, 1)

    print(f"[infer_test] Data shape: {data_rna.shape} ({adata.shape[0]} cells, {G-1} genes)")

    # Initialize model
    try:
        model = NetworkModel_beta(G-1)
        print(f"[infer_test] Initialized network model with {G-1} genes")
    except Exception as e:
        print(f"[infer_test] Error initializing model: {e}")
        sys.exit(1)

    # Load model parameters
    try:
        model.basal = np.load(os.path.join(p, 'cardamomOT', 'basal.npy'))
        model.inter = np.load(os.path.join(p, 'cardamomOT', 'inter.npy'))
        model.a = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        model.times_data = np.load(os.path.join(p, 'cardamomOT', 'data_times.npy'))
        model.ratios = np.load(os.path.join(p, 'cardamomOT', 'ratios.npy'))
        model.n_networks = np.load(os.path.join(p, 'cardamomOT', 'n_networks.npy'))
        model.d = np.load(os.path.join(p, 'cardamomOT', 'degradations.npy'))
        print(f"[infer_test] Loaded network parameters")
    except FileNotFoundError as e:
        print(f"[infer_test] Error: Missing parameter file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[infer_test] Error loading parameters: {e}")
        sys.exit(1)

    # Perform network inference on test set
    print(f"[infer_test] Performing network inference on test set...")
    try:
        model.infer_test(data_rna, vect_samples_id=vect_samples_id, verb=1)
        print(f"[infer_test] Network inference completed")
    except Exception as e:
        print(f"[infer_test] Error during network inference: {e}")
        sys.exit(1)
    
    # Save inference results
    try:
        np.save(os.path.join(p, 'cardamomOT', 'data_prot_test'), model.prot)
        np.save(os.path.join(p, 'cardamomOT', 'data_rna_test'), model.rna)
        np.save(os.path.join(p, 'cardamomOT', 'data_times_test'), model.times_data)
        np.save(os.path.join(p, 'cardamomOT', 'data_kon_beta_test'), model.kon_beta)
        np.save(os.path.join(p, 'cardamomOT', 'data_kon_theta_test'), model.kon_theta)
        print(f"[infer_test] Saved inference results for test set")
    except Exception as e:
        print(f"[infer_test] Error saving inference results: {e}")
        sys.exit(1)

    # Load simulation parameters
    try:
        model.basal = np.load(os.path.join(p, 'cardamomOT', 'basal_simul.npy'))
        model.inter = np.load(os.path.join(p, 'cardamomOT', 'inter_simul.npy'))
        print(f"[infer_test] Loaded simulation parameters")
    except FileNotFoundError as e:
        print(f"[infer_test] Error: Missing simulation parameter file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[infer_test] Error loading simulation parameters: {e}")
        sys.exit(1)

    # Load simulation times if specified
    times_file = os.path.join(p, 'Data', 'times_to_simulate.txt')
    try:
        if os.path.exists(times_file):
            with open(times_file, "r") as f:
                times = [float(line.strip()) for line in f if line.strip()]
            if times[0] != 0:
                times = [0] + times
            print(f"[infer_test] Loaded simulation times from file: {times}")
        else:
            times = list(set(model.times_data))
            print(f"[infer_test] Using inferred timepoints for simulation")
    except Exception as e:
        print(f"[infer_test] Error loading simulation times: {e}")
        sys.exit(1)

    times.sort()
    N = int(model.prot.shape[0]/len(np.unique(model.times_data)))
    times_simulation = np.zeros(len(times)*N)
    for t in range(len(times)):
        times_simulation[t*N:(t+1)*N] = times[t]
    
    print(f"[infer_test] Simulation times: {np.unique(times_simulation)}, {len(times_simulation)} total timepoints")

    # Simulate network dynamics
    print(f"[infer_test] Simulating network dynamics on test set...")
    try:
        model.simulate_network(times)
        print(f"[infer_test] Network simulation completed")
    except Exception as e:
        print(f"[infer_test] Error during simulation: {e}")
        sys.exit(1)

    # Save simulation results
    try:
        np.save(os.path.join(p, 'cardamomOT', 'data_prot_simul_test'), model.prot)
        np.save(os.path.join(p, 'cardamomOT', 'data_kon_simul_test'), model.kon_theta)
        np.save(os.path.join(p, 'cardamomOT', 'simulation_times_test'), times_simulation)
        print(f"[infer_test] Saved test set simulation results")
        print("[infer_test] Test set inference and simulation completed successfully")
    except Exception as e:
        print(f"[infer_test] Error saving simulation results: {e}")
        sys.exit(1)

if __name__ == "__main__":
   main(sys.argv[1:])

