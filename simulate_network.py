"""
simulate_network.py
-------------------
Simulate gene expression dynamics using the inferred network model.

Usage:
    python simulate_network.py -i <project_path>   (simulate_with_proliferation: model_parameters sheet)

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
from CardamomOT.inputs import input_dir
from CardamomOT.run_options import parse_step_options, settings, configure, configure_simulation
from CardamomOT.config import n_inference_stimuli, simulation_schedule
import anndata as ad
import os
import torch

def load_simulation_model(p, opts, adata, tag='[simulate_network]'):
    """NetworkModel with the inferred simulation parameters of the project (and the proliferation MLP if
    simulate_with_proliferation), and the simulated timepoints."""
    n_stimuli = n_inference_stimuli(input_dir(p))
    model = NetworkModel_beta(adata.shape[1], n_stimuli=n_stimuli)
    configure(model, opts)  # workbook, then the command-line options
    simulate_with_proliferation = bool(model.simulate_with_proliferation)
    model.simulate_with_proliferation = False  # enabled below once the proliferation network is loaded
    print(f"{tag} Data: {adata.shape[1]} genes, {adata.shape[0]} cells, n_stimuli={n_stimuli}")

    # Load inferred network parameters
    print(f"{tag} Loading inferred network parameters...")
    try:
        model.d_t = np.load(os.path.join(p, 'cardamomOT', 'degradations_temporal.npy'))
        model.basal = np.load(os.path.join(p, 'cardamomOT', 'basal_simul.npy'))
        model.inter = np.load(os.path.join(p, 'cardamomOT', 'inter_simul.npy'))
        # Validate n_stimuli against loaded inter (authoritative for simulation)
        n_stimuli_inter = model.inter.shape[0] - adata.shape[1]
        if n_stimuli_inter != model.n_stimuli:
            print(f"{tag} Warning: correcting n_stimuli from {model.n_stimuli} "
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
            print(f"{tag} Loaded per-cell sample IDs")
        print(f"{tag} Successfully loaded all network parameters")
    except FileNotFoundError as e:
        print(f"{tag} Error: Missing parameter file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"{tag} Error loading parameters: {e}")
        sys.exit(1)

    # Determine simulation timepoints
    times_file = os.path.join(input_dir(p), 'times_simulation.txt')
    if os.path.exists(times_file):
        print(f"{tag} Custom timepoints found in {times_file}")
        try:
            with open(times_file, "r") as f:
                times = [float(line.strip()) for line in f if line.strip()]
            if not times:
                raise ValueError("times_simulation.txt is empty")
            if times[0] != 0:
                times = [0] + times
            print(f"{tag} Using custom timepoints: {times}")
        except (ValueError, IOError) as e:
            print(f"{tag} Error reading times_simulation.txt: {e}")
            times = list(set(model.times_data))
    else:
        print(f"{tag} Using timepoints from loaded data")
        times = list(set(model.times_data))

    times.sort()
    print(f"{tag} Will simulate {len(times)} timepoints: {times}")

    # Load proliferation network if requested
    if simulate_with_proliferation:
        prolif_path = os.path.join(p, 'cardamomOT', 'prolif_network.pt')
        n_prot_path = os.path.join(p, 'cardamomOT', 'prolif_network_n_proteins.npy')
        if os.path.exists(prolif_path) and os.path.exists(n_prot_path):
            from CardamomOT.inference.proliferations import ProliferationMLP
            n_proteins = int(np.load(n_prot_path)[0])
            prolif_net = ProliferationMLP(n_proteins)
            # strict=False: networks saved before input standardisation keep identity scaling
            prolif_net.load_state_dict(torch.load(prolif_path, map_location='cpu', weights_only=True), strict=False)
            prolif_net.eval()
            model.prolif_network = prolif_net
            model.simulate_with_proliferation = True
            stim_pkl = os.path.join(p, 'cardamomOT', 'stimulus_rates.pkl')
            if os.path.exists(stim_pkl):
                import pickle
                model.stimulus_rate_model = pickle.load(open(stim_pkl, 'rb'))
                print(f"{tag} Effects of the inference stimuli on the net rate (perturbation_inference) "
                      "applied with the simulated schedule")
            print(f"{tag} Loaded proliferation network — branching simulation enabled")
        else:
            print(f"{tag} Warning: simulate_with_proliferation = True but prolif_network.pt not found; "
                  "run infer_network_simul.py with simulate_with_proliferation = True first")

    return model, times


def main(argv):
    """
    Simulate gene expression dynamics using the inferred network model.

    Args:
        argv: Command-line arguments (-i <project> and the options of run_options.STEP_OPTIONS).
    """
    opts = parse_step_options(argv, 'simulate_network', __doc__)
    p = opts.p
    split = settings(opts).split

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

    # Inference stimuli in simulation: first columns of stimulus_schedule_simulate.txt, else inference schedule
    stim_sched, _ = simulation_schedule(input_dir(p), n_inference_stimuli(input_dir(p)))
    model, times = load_simulation_model(p, opts, adata)
    configure_simulation(model, opts, adata)  # per-sample schedules of the simulation

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
        # Whether the cells were resampled by the proliferation MLP (reference used by check_sim_to_data)
        np.save(os.path.join(cardamom_dir, 'simulation_with_proliferation'),
                np.array([bool(model.simulate_with_proliferation and model.prolif_network is not None)]))
        if model.log_population is not None:
            np.save(os.path.join(cardamom_dir, 'data_log_population'), model.log_population)
        if model.simulate_full_with_harissa and hasattr(model, 'mrna_simul') and model.mrna_simul is not None:
            np.save(os.path.join(cardamom_dir, 'data_mrna_simul'), model.mrna_simul)
            print(f"[simulate_network] Saved mRNA simulation to data_mrna_simul.npy")
        print(f"[simulate_network] Successfully saved simulation results to {cardamom_dir}")
    except Exception as e:
        print(f"[simulate_network] Error saving simulation results: {e}")
        sys.exit(1)

if __name__ == "__main__":
   main(sys.argv[1:])
