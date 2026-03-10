"""
simulate_network.py
-------------------
Simulate gene expression dynamics using the inferred network model.

Loads the inferred regulatory network structure and simulates stochastic
expression dynamics across specified timepoints for validation against
observed data.

Usage:
    python simulate_network.py -i <project_path> -s <split>

Required input files:
    - Data/data_<split>.h5ad: count matrix with temporal information
    - cardamom/basal_simul.npy, inter_simul.npy: inferred parameters
    - cardamom/mixture_parameters.npy, data_prot_unitary.npy: kinetic rates

Output files:
    - cardamom/data_prot_simul.npy: simulated protein abundance
    - cardamom/data_kon_simul.npy: simulated bursting events
    - cardamom/simulation_times.npy: timepoints used for simulation
"""
import sys; sys.path += ['../']
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta
import getopt
import anndata as ad
import scipy.sparse
import os

def main(argv):
    """
    Simulate gene expression dynamics using the inferred network model.

    Loads the previously inferred gene regulatory network and parameters,
    then simulates stochastic expression dynamics to generate synthetic
    data for comparison with experimental observations.

    Args:
        argv: Command-line arguments (--input, --split).
    
    Returns:
        None. Saves simulation results to cardamom/ directory.
    """
    inputfile = ''
    split = ''
    try:
        opts, args = getopt.getopt(argv, "hi:s:", ["input=", "split="])
    except getopt.GetoptError:
        print("[simulate_network] Error: Invalid command-line arguments")
        print("[simulate_network] Usage: python simulate_network.py -i <project_path> -s <split>")
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
        print("[simulate_network] Error: Missing required arguments --input and --split")
        sys.exit(1)

    p = '{}/'.format(inputfile)

    # Load gene expression data
    data_path = os.path.join(p, 'Data', 'data_{}.h5ad'.format(split))
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[simulate_network] Loaded data from {data_path}")
    except FileNotFoundError as e:
        print(f"[simulate_network] Error: {e}")
        print(f"[simulate_network] Please ensure Data/data_{split}.h5ad exists in {p}")
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
        print(f"[simulate_network] Detected {len(np.unique(times))} timepoints: {sorted(np.unique(times))}")
    except KeyError:
        print("[simulate_network] Error: data.obs['time'] not found")
        print("[simulate_network] Please ensure temporal information is in the 'time' column")
        sys.exit(1)
    except ValueError as e:
        print(f"[simulate_network] Error: {e}")
        sys.exit(1)
    
    data_rna = np.vstack([times, data_rna_extracted]).T
    G = np.size(data_rna, 1)
    print(f"[simulate_network] Data shape: {G} genes, {np.size(data_rna, 0)} cells")

    model = NetworkModel_beta(G-1)

    # Load inferred network parameters
    print("[simulate_network] Loading inferred network parameters...")
    try:
        model.d_t = np.load(os.path.join(p, 'cardamomOT', 'degradations_temporal.npy'))
        model.basal = np.load(os.path.join(p, 'cardamomOT', 'basal_simul.npy'))
        model.inter = np.load(os.path.join(p, 'cardamomOT', 'inter_simul.npy'))
        model.basal_t = np.load(os.path.join(p, 'cardamomOT', 'basal_t_simul.npy'))
        model.inter_t = np.load(os.path.join(p, 'cardamomOT', 'inter_t_simul.npy'))
        model.a = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        model.prot = np.load(os.path.join(p, 'cardamomOT', 'data_prot_unitary.npy'))
        model.times_data = np.load(os.path.join(p, 'cardamomOT', 'data_times.npy'))
        model.kon_beta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_beta.npy'))
        model.proba_traj = np.load(os.path.join(p, 'cardamomOT', 'proba_traj.npy'))
        model.ratios = np.load(os.path.join(p, 'cardamomOT', 'ratios.npy'))
        model.n_networks = np.load(os.path.join(p, 'cardamomOT', 'n_networks.npy'))
        print("[simulate_network] Successfully loaded all network parameters")
    except FileNotFoundError as e:
        print(f"[simulate_network] Error: Missing parameter file: {e}")
        print("[simulate_network] Please ensure network inference has been completed")
        sys.exit(1)
    except Exception as e:
        print(f"[simulate_network] Error loading parameters: {e}")
        sys.exit(1)

    # Determine simulation timepoints
    times_file = os.path.join(p, 'Data', 'times_to_simulate.txt')    
    if os.path.exists(times_file):
        print(f"[simulate_network] Custom timepoints found in {times_file}")
        try:
            with open(times_file, "r") as f:
                times = [float(line.strip()) for line in f if line.strip()]
            if not times:
                raise ValueError("times_to_simulate.txt is empty")
            if times[0] != 0:
                times = [0] + times
            print(f"[simulate_network] Using custom timepoints: {times}")
        except (ValueError, IOError) as e:
            print(f"[simulate_network] Error reading times_to_simulate.txt: {e}")
            print("[simulate_network] Falling back to unique timepoints from data")
            times = list(set(model.times_data))
    else:
        print("[simulate_network] Using timepoints from loaded data")
        times = list(set(model.times_data))
    
    times.sort()
    print(f"[simulate_network] Will simulate {len(times)} timepoints: {times}")

    # Simulate network dynamics
    print("[simulate_network] Starting network simulation...")
    model.simulate_network(times)
    print("[simulate_network] Simulation completed")
    
    # Save simulation results
    cardamom_dir = os.path.join(p, 'cardamomOT')
    try:
        np.save(os.path.join(cardamom_dir, 'data_prot_simul'), model.prot)
        np.save(os.path.join(cardamom_dir, 'data_kon_simul'), model.kon_theta)
        np.save(os.path.join(cardamom_dir, 'simulation_times'), model.times_simul)
        print(f"[simulate_network] Successfully saved simulation results to {cardamom_dir}")
    except Exception as e:
        print(f"[simulate_network] Error saving simulation results: {e}")
        sys.exit(1)

if __name__ == "__main__":
   main(sys.argv[1:])

