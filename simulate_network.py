"""
simulate_network.py
-------------------
Simulate gene expression dynamics using the inferred network model.

Usage:
    python simulate_network.py -i <project_path>   (simulate_with_proliferation: model_parameters sheet)

Required input files:
    - Data/data_<split>.h5ad: count matrix with temporal information
    - cardamom/basal_simul.npy, inter_simul.npy: inferred parameters
    - cardamom/mixture_parameters.npy, data_prot.npy: kinetic rates and starting protein states (fit_network)

Output files:
    - cardamom/data_prot_simul.npy: simulated protein abundance
    - cardamom/data_kon_simul.npy: simulated bursting events
    - cardamom/simulation_times.npy: timepoints used for simulation
"""
import sys; sys.path += ['../']
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta, load_simul_parameters
from CardamomOT.inputs import input_dir
from CardamomOT.run_options import parse_step_options, settings, configure, configure_simulation
from CardamomOT.config import n_inference_stimuli, simulation_schedule, schedule_reference_times
import anndata as ad
import os
import torch

def attach_proliferation(model, p, tag='[simulate_network]', branching=True):
    """
    Proliferation of the simulations, as in the refine SDE. proliferation_mlp: the MLP (prolif_network.pt, one or two
    heads) gives the birth rate of the dilution (protein_dilution) and, with branching, the net rate of the branching
    (stimulus as input). Without it, each simulated cell takes at the start of each interval the rates of the nearest
    trajectory state of its sample: birth (data_state_rates.npz) and net growth over the interval (data_R_opt.npy, or
    data_R_prior.npy with use_recomputed_proliferation = False). Returns whether the simulation branches.
    """
    from CardamomOT import load_state_growth
    from CardamomOT.inference.proliferations import load_proliferation
    cdir = os.path.join(p, 'cardamomOT')
    model.simulate_with_proliferation = False
    if not model.proliferation_mlp:
        model.prolif_network = None
        if not branching:
            print(f"{tag} proliferation_mlp = False: dilution at the birth rates of the nearest trajectory states")
            return False
        # Net growth of the trajectory states (those of the model when it already holds them, e.g. test states)
        R, prot = getattr(model, 'R_opt', None), getattr(model, 'prot', None)
        if R is None or (prot is not None and len(R) != len(prot)):
            model.R_opt = load_state_growth(cdir, model.use_recomputed_proliferation)
        if model.R_opt is None:
            print(f"{tag} Warning: simulate_with_proliferation = True but no net growth of the trajectory states "
                  "(data_R_opt.npy / data_R_prior.npy, infer_network_structure): no branching")
            return False
        model.simulate_with_proliferation = True
        print(f"{tag} proliferation_mlp = False: branching and dilution at the rates of the nearest trajectory states "
              f"({'R_opt' if model.use_recomputed_proliferation else 'prior net rates'})")
        return True
    # One MLP (one or two heads), or one per network condition (ProliferationMLP_per_condition)
    prolif = load_proliferation(cdir)
    if prolif is None:
        if branching:
            print(f"{tag} Warning: simulate_with_proliferation = True but prolif_network.pt not found; "
                  "run infer_network_simul.py first")
        else:
            print(f"{tag} No prolif_network.pt: dilution at the birth rates of the trajectory slots")
        return False
    model.prolif_network = prolif
    model.simulate_with_proliferation = bool(branching)
    if not branching:
        print(f"{tag} Loaded proliferation network — birth rate of the dilution (no branching)")
        return True
    print(f"{tag} Loaded proliferation network — branching simulation enabled")
    return True


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
        prm = load_simul_parameters(os.path.join(p, 'cardamomOT'))  # outputs of infer_network_simul
        model.d_t, model.basal, model.inter = prm['d_t'], prm['basal'], prm['inter']
        # Validate n_stimuli against loaded inter (authoritative for simulation)
        n_stimuli_inter = model.inter.shape[-1] - adata.shape[1]  # inter: ([n_samples,] G_tot, G_tot)
        if n_stimuli_inter != model.n_stimuli:
            print(f"{tag} Warning: correcting n_stimuli from {model.n_stimuli} "
                  f"to {n_stimuli_inter} based on the loaded network")
            model.n_stimuli = n_stimuli_inter
        model.basal_t, model.inter_t = prm['basal_t'], prm['inter_t']
        model.a = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        model.prot = prm['prot']
        model.rna = np.load(os.path.join(p, 'cardamomOT', 'data_rna.npy'))
        model.times_data = np.load(os.path.join(p, 'cardamomOT', 'data_times.npy'))
        model.kon_beta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_beta.npy'))
        model.ratios = prm['ratios']
        model.load_state_rates(os.path.join(p, 'cardamomOT'))  # birth / death of the states (dilution)
        model.load_chromatin(os.path.join(p, 'cardamomOT'))  # chromatin regions, if any
        samples_path = os.path.join(p, 'cardamomOT', 'data_samples.npy')
        if os.path.exists(samples_path):
            model.samples_data = np.load(samples_path)
            print(f"{tag} Loaded per-cell sample IDs")
        model.set_gene_mask(adata)  # genes of each sample's condition (select_genes_per_condition), if any
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

    # Proliferation network: branching (simulate_with_proliferation) and birth rate of the dilution
    if simulate_with_proliferation or model.protein_dilution:
        attach_proliferation(model, p, tag, branching=simulate_with_proliferation)

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
    stim_sched, _, rows_t = simulation_schedule(input_dir(p), n_inference_stimuli(input_dir(p)), with_times=True)
    model, times = load_simulation_model(p, opts, adata)
    configure_simulation(model, opts, adata)  # per-sample schedules of the simulation

    # Simulate network dynamics (default schedule; the scenarios are simulated by simulate_network_KOV)
    print("[simulate_network] Starting network simulation...")
    model.simulate_network(times, stimulus_schedule=stim_sched,
                           schedule_times=None if rows_t is None else schedule_reference_times(rows_t, times))
    print("[simulate_network] Simulation completed")

    # Save simulation results
    cardamom_dir = os.path.join(p, 'cardamomOT')
    try:
        np.save(os.path.join(cardamom_dir, 'data_prot_simul'), model.prot)
        np.save(os.path.join(cardamom_dir, 'data_kon_simul'), model.kon_theta)
        np.save(os.path.join(cardamom_dir, 'simulation_times'), model.times_simul)
        # Whether the cells were resampled by the proliferation MLP (reference used by check_sim_to_data)
        np.save(os.path.join(cardamom_dir, 'simulation_with_proliferation'),
                np.array([bool(model.simulate_with_proliferation)]))
        if model.log_population is not None:
            np.save(os.path.join(cardamom_dir, 'data_log_population'), model.log_population)
        # Explicit mRNA counts (simulation_mrnas_direct) and chromatin accessibility (model.chromatin)
        for name, arr in [('data_mrna_simul', model.mrna_simul), ('data_chrom_simul', model.chrom_simul)]:
            path = os.path.join(cardamom_dir, name + '.npy')
            if arr is not None:
                np.save(path, arr)
                print(f"[simulate_network] Saved {name}.npy")
            elif os.path.exists(path):
                os.remove(path)  # output of an earlier run with other options
        print(f"[simulate_network] Successfully saved simulation results to {cardamom_dir}")
    except Exception as e:
        print(f"[simulate_network] Error saving simulation results: {e}")
        sys.exit(1)

if __name__ == "__main__":
   main(sys.argv[1:])
