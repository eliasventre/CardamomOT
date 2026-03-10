"""
simulate_network_KOV.py
----------------------
Simulate gene expression under in-silico knock-out (KO) and over-expression (OV).

Loads the inferred network model and simulates stochastic expression dynamics
under various genetic perturbations (knockouts and overexpressions) to predict
the effects of gene perturbations on network dynamics.

Usage:
    python simulate_network_KOV.py -i <project_path> -s <split>

Required input files:
    - Data/data_<split>.h5ad: count matrix with temporal information
    - Data/KO_OV_list.txt: table defining KO/OV combinations (tab-separated)
    - cardamom/inter_t_simul.npy, basal_simul.npy: inferred parameters

Output files:
    - cardamom/data_prot_simul_KO_*.npy: simulated protein for each perturbation
    - cardamom/data_kon_simul_KO_*.npy: simulated bursting for each perturbation

KO_OV_list.txt format:
    KO          OV
    gene1       gene2,gene3
    gene4
"""
import sys; sys.path += ['../']
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta
import getopt
import anndata as ad
import scipy.sparse
import os
import pandas as pd

def parse_gene_list(cell):
    """
    Parse a KO/OV cell that can contain multiple genes separated by commas.
    
    Args:
        cell: String or numeric value from DataFrame cell
    
    Returns:
        list: Parsed gene names, empty if null/zero
    """
    if pd.isna(cell):
        return []
    cell = str(cell).strip()
    if cell in ['0', '', 'nan']:
        return []
    return [g.strip() for g in cell.split(',') if g.strip() != '']


def load_ko_ov_combinations(file_path):
    """
    Load KO/OV combinations from tab-separated file.
    
    Expects columns 'KO' and/or 'OV' with gene names (comma-separated for multiple).
    
    Args:
        file_path: Path to KO_OV_list.txt
    
    Returns:
        list: List of dicts with 'KO' and 'OV' keys containing gene lists
    
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If no KO or OV column found
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"KO/OV list file not found: {file_path}")
    
    combos = []
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Parse header
    header = lines[0].strip().split('\t')
    header = [h.strip().upper() for h in header]

    ko_idx = header.index("KO") if "KO" in header else None
    ov_idx = header.index("OV") if "OV" in header else None

    if ko_idx is None and ov_idx is None:
        raise ValueError("KO_OV_list.txt must contain 'KO' or 'OV' column")

    # Parse combinations
    for line_num, line in enumerate(lines[1:], start=2):
        parts = line.rstrip().split('\t')

        def parse(idx):
            if idx is None or idx >= len(parts):
                return []
            cell = parts[idx].strip()
            if cell in ['', '0']:
                return []
            return [g.strip() for g in cell.split(',') if g.strip()]

        kos = parse(ko_idx)
        ovs = parse(ov_idx)

        if kos or ovs:  # Only add non-empty combinations
            combos.append({'KO': kos, 'OV': ovs})

    return combos


def main(argv):
    """
    Simulate knockout/overexpression perturbations of the inferred network.

    Loads the inferred network model and applies specified perturbations,
    then simulates dynamics for each condition to predict network responses
    to genetic manipulations.

    Args:
        argv: Command-line arguments (--input, --split).
    
    Returns:
        None. Saves simulation results for each perturbation.
    """
    inputfile = ''
    split = ''
    try:
        opts, args = getopt.getopt(argv, "hi:s:", ["input=", "split="])
    except getopt.GetoptError:
        print("[simulate_network_KOV] Error: Invalid command-line arguments")
        print("[simulate_network_KOV] Usage: python simulate_network_KOV.py -i <project_path> -s <split>")
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
        print("[simulate_network_KOV] Error: Missing required arguments --input and --split")
        sys.exit(1)

    p = f'{inputfile}/'

    # Load KO/OV combinations
    ko_ov_file = os.path.join(p, "Data", "KO_OV_list.txt")
    try:
        combos = load_ko_ov_combinations(ko_ov_file)
    except FileNotFoundError:
        print(f"[simulate_network_KOV] Error: KO/OV list file not found at {ko_ov_file}")
        sys.exit(1)
    except ValueError as e:
        print(f"[simulate_network_KOV] Error: {e}")
        sys.exit(1)

    if len(combos) == 0:
        print("[simulate_network_KOV] No KO/OV combinations found in KO_OV_list.txt")
        sys.exit(0)

    print(f"[simulate_network_KOV] Loaded {len(combos)} KO/OV combinations")

    # Load gene expression data
    data_path = os.path.join(p, 'Data', f'data_{split}.h5ad')
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[simulate_network_KOV] Loaded data from {data_path}")
    except FileNotFoundError as e:
        print(f"[simulate_network_KOV] Error: {e}")
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
        print(f"[simulate_network_KOV] Detected {len(np.unique(times))} timepoints")
    except KeyError:
        print("[simulate_network_KOV] Error: data.obs['time'] not found")
        sys.exit(1)
    except ValueError as e:
        print(f"[simulate_network_KOV] Error: {e}")
        sys.exit(1)

    data_rna = np.vstack([times, data_rna_extracted]).T
    G = np.size(data_rna, 1)
    print(f"[simulate_network_KOV] Data shape: {G} genes, {np.size(data_rna, 0)} cells")

    model = NetworkModel_beta(G-1)

    # Load network model parameters
    print("[simulate_network_KOV] Loading inferred network parameters...")
    try:
        model.d_t = np.load(os.path.join(p, 'cardamom', 'degradations_temporal.npy'))
        model.inter_t = np.load(os.path.join(p, 'cardamom', 'inter_t_simul.npy'))
        model.inter = np.load(os.path.join(p, 'cardamom', 'inter_simul.npy'))
        model.a = np.load(os.path.join(p, 'cardamom', 'mixture_parameters.npy'))
        model.times_data = np.load(os.path.join(p, 'cardamom', 'data_times.npy'))
        model.kon_beta = np.load(os.path.join(p, 'cardamom', 'data_kon_beta.npy'))
        model.proba_traj = np.load(os.path.join(p, 'cardamom', 'proba_traj.npy'))
        model.ratios = np.load(os.path.join(p, 'cardamom', 'ratios.npy'))
        model.n_networks = np.load(os.path.join(p, 'cardamom', 'n_networks.npy'))
        print("[simulate_network_KOV] Successfully loaded all parameters")
    except FileNotFoundError as e:
        print(f"[simulate_network_KOV] Error: Missing parameter file: {e}")
        sys.exit(1)

    # Determine simulation timepoints
    filepath = os.path.join(p, 'Data', 'times_to_simulate.txt')
    if os.path.exists(filepath):
        print("[simulate_network_KOV] Using custom timepoints from times_to_simulate.txt")
        try:
            with open(filepath, "r") as f:
                times = [float(line.strip()) for line in f if line.strip()]
            if times[0] != 0:
                times = [0] + times
        except (ValueError, IOError) as e:
            print(f"[simulate_network_KOV] Error reading times_to_simulate.txt: {e}")
            times = list(set(model.times_data))
    else:
        times = list(set(model.times_data))

    times.sort()
    print(f"[simulate_network_KOV] Will simulate {len(times)} timepoints")
    
    N = np.sum(model.times_data == 0)
    times_simulation = np.zeros(len(times)*N)
    for t in range(0, len(times)):
        times_simulation[t*N:(t+1)*N] = times[t]

    # Simulate perturbations
    print(f"[simulate_network_KOV] Starting simulation of {len(combos)} perturbations...")
    for idx, combo in enumerate(combos, start=1):
        kos = combo['KO']
        ovs = combo['OV']
        label = f"KO_{'-'.join(kos) if kos else 'none'}_OV_{'-'.join(ovs) if ovs else 'none'}"
        print(f"\n[simulate_network_KOV] Simulating condition {idx}/{len(combos)}: {label}")

        # Reset model to baseline
        try:
            model.basal = np.load(os.path.join(p, 'cardamom', 'basal_simul.npy'))
            model.basal_t = np.load(os.path.join(p, 'cardamom', 'basal_t_simul.npy'))
            model.prot = np.load(os.path.join(p, 'cardamom', 'data_prot_unitary.npy'))
            model.kon_theta = np.load(os.path.join(p, 'cardamom', 'data_kon_theta.npy'))
        except FileNotFoundError as e:
            print(f"[simulate_network_KOV] Error resetting model: {e}")
            continue

        # Apply knockouts
        for gene in kos:
            if gene in adata.var_names:
                ind = 1 + adata.var_names.get_loc(gene)
                model.basal_t[:, ind] = -100 - np.sum(model.inter_t[-1, :, ind])
                model.prot[:, ind] = 0
                print(f"[simulate_network_KOV]   KO: {gene} (index {ind})")
            else:
                print(f"[simulate_network_KOV]   Warning: Gene '{gene}' not found in data")

        # Apply overexpressions
        for gene in ovs:
            if gene in adata.var_names:
                ind = 1 + adata.var_names.get_loc(gene)
                model.basal_t[:, ind] = 100 + np.sum(model.inter_t[-1, :, ind])
                model.prot[:, ind] = 1
                print(f"[simulate_network_KOV]   OV: {gene} (index {ind})")
            else:
                print(f"[simulate_network_KOV]   Warning: Gene '{gene}' not found in data")

        # Simulate dynamics
        try:
            model.simulate_network(times)
            cardamom_dir = os.path.join(p, 'cardamom')
            np.save(os.path.join(cardamom_dir, f'data_prot_simul_{label}'), model.prot)
            np.save(os.path.join(cardamom_dir, f'data_kon_simul_{label}'), model.kon_theta)
            print(f"[simulate_network_KOV]   Results saved for condition: {label}")
        except Exception as e:
            print(f"[simulate_network_KOV]   Error simulating condition {idx}: {e}")
            continue

    print(f"\n[simulate_network_KOV] Completed simulation of all {len(combos)} conditions")

if __name__ == "__main__":
   main(sys.argv[1:])
