"""
simulate_network.py
-------------------
Simulate gene expression dynamics using the inferred network model.

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
from CardamomOT.inference.degradations import KonCorrectionMLP
import getopt
import anndata as ad
import os
import torch

def main(argv):
    """
    Simulate gene expression dynamics using the inferred network model.

    Args:
        argv: Command-line arguments (--input, --split).
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

    # Load gene expression data (for gene count and temporal validation)
    data_path = os.path.join(p, 'Data', 'data_{}.h5ad'.format(split))
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[simulate_network] Loaded data from {data_path}")
    except FileNotFoundError as e:
        print(f"[simulate_network] Error: {e}")
        sys.exit(1)

    # ─── LOAD STIMULUS SCHEDULE ──────────────────────────────────────────
    # Prefer simulation-specific schedule; fall back to inference schedule
    stim_sched = None
    for sched_name in ('stimulus_schedule_simul.txt', 'stimulus_schedule.txt'):
        sched_path = os.path.join(p, 'Data', sched_name)
        if os.path.exists(sched_path):
            stim_sched = np.loadtxt(sched_path)
            print(f"[simulate_network] Loaded stimulus schedule from {sched_path}")
            break

    # ─── DETECT n_stimuli FROM SCHEDULE ─────────────────────────────────
    _stim_arr = np.asarray(stim_sched) if stim_sched is not None else None
    n_stimuli = int(_stim_arr.shape[1]) if (_stim_arr is not None and _stim_arr.ndim == 2) else 1

    model = NetworkModel_beta(adata.shape[1], n_stimuli=n_stimuli)
    print(f"[simulate_network] Data: {adata.shape[1]} genes, {adata.shape[0]} cells, n_stimuli={n_stimuli}")

    # Load inferred network parameters
    print("[simulate_network] Loading inferred network parameters...")
    try:
        model.d_t = np.load(os.path.join(p, 'cardamomOT', 'degradations_temporal.npy'))
        model.basal = np.load(os.path.join(p, 'cardamomOT', 'basal_simul.npy'))
        model.inter = np.load(os.path.join(p, 'cardamomOT', 'inter_simul.npy'))
        # Validate n_stimuli against loaded inter (authoritative for simulation)
        n_stimuli_inter = model.inter.shape[0] - adata.shape[1]
        if n_stimuli_inter != model.n_stimuli:
            print(f"[simulate_network] Warning: correcting n_stimuli from {model.n_stimuli} "
                  f"to {n_stimuli_inter} based on loaded inter_simul.npy")
            model.n_stimuli = n_stimuli_inter
        model.basal_t = np.load(os.path.join(p, 'cardamomOT', 'basal_t_simul.npy'))
        model.inter_t = np.load(os.path.join(p, 'cardamomOT', 'inter_t_simul.npy'))
        model.a = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        model.prot = np.load(os.path.join(p, 'cardamomOT', 'data_prot_forsimul.npy'))
        model.rna = np.load(os.path.join(p, 'cardamomOT', 'data_rna.npy'))
        model.times_data = np.load(os.path.join(p, 'cardamomOT', 'data_times.npy'))
        model.kon_beta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_beta.npy'))
        model.proba_traj = np.load(os.path.join(p, 'cardamomOT', 'proba_traj.npy'))
        model.ratios = np.load(os.path.join(p, 'cardamomOT', 'ratios.npy'))
        model.n_networks = np.load(os.path.join(p, 'cardamomOT', 'n_networks.npy'))
        samples_path = os.path.join(p, 'cardamomOT', 'data_samples.npy')
        if os.path.exists(samples_path):
            model.samples_data = np.load(samples_path)
            print("[simulate_network] Loaded per-cell sample IDs")
        mlp_path = os.path.join(p, 'cardamomOT', 'kon_mlp.pt')
        cfg_path = os.path.join(p, 'cardamomOT', 'kon_mlp_config.npy')
        if os.path.exists(mlp_path) and os.path.exists(cfg_path):
            G_genes = int(np.load(cfg_path)[0])
            mlp = KonCorrectionMLP(G_genes)
            mlp.load_state_dict(torch.load(mlp_path, map_location="cpu"))
            mlp.eval()
            model.kon_mlp = mlp
            print("[simulate_network] Loaded kon correction MLP")
        print("[simulate_network] Successfully loaded all network parameters")
    except FileNotFoundError as e:
        print(f"[simulate_network] Error: Missing parameter file: {e}")
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
            times = list(set(model.times_data))
    else:
        print("[simulate_network] Using timepoints from loaded data")
        times = list(set(model.times_data))

    times.sort()
    print(f"[simulate_network] Will simulate {len(times)} timepoints: {times}")

    # Simulate network dynamics
    print("[simulate_network] Starting network simulation...")
    model.simulate_network(times, stimulus_schedule=stim_sched)
    print("[simulate_network] Simulation completed")

    # Save simulation results
    cardamom_dir = os.path.join(p, 'cardamomOT')
    try:
        np.save(os.path.join(cardamom_dir, 'data_prot_simul'), model.prot)
        np.save(os.path.join(cardamom_dir, 'data_kon_simul'), model.kon_theta)
        np.save(os.path.join(cardamom_dir, 'simulation_times'), model.times_simul)
        if model.simulate_full_with_harissa and hasattr(model, 'mrna_simul') and model.mrna_simul is not None:
            np.save(os.path.join(cardamom_dir, 'data_mrna_simul'), model.mrna_simul)
            print(f"[simulate_network] Saved mRNA simulation to data_mrna_simul.npy")
        print(f"[simulate_network] Successfully saved simulation results to {cardamom_dir}")
    except Exception as e:
        print(f"[simulate_network] Error saving simulation results: {e}")
        sys.exit(1)

if __name__ == "__main__":
   main(sys.argv[1:])
