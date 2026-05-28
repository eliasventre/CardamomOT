"""
check_KOV_to_sim.py
-------------------
Validate knockouts/overexpressions by comparing simulations to observations.

Compares simulated gene expression under various perturbations (KO/OV)
with observed wildtype data. Generates AnnData objects for each perturbation
condition for downstream analysis and visualization.

Usage:
    python check_KOV_to_sim.py -i <project_path> -s <split>

Required input files:
    - Data/data_<split>.h5ad: observed count matrix (wildtype)
    - Data/KO_OV_simulate.txt: perturbation combinations
    - cardamomOT/data_prot_simul_KO_*.npy: simulated proteins for each perturbation
    - cardamomOT/data_kon_simul_KO_*.npy: simulated bursting for each perturbation

Output files:
    - cardamomOT/adata_sim_KO_*.h5ad: AnnData objects for each perturbation
    - cardamomOT/adata_prot_simul_KO_*.h5ad: Protein trajectories for each perturbation
"""
import re
import numpy as np
import sys, getopt
import anndata as ad
from CardamomOT import NetworkModel
import scipy.sparse
import os


def _parse_gene_with_pct(token):
    """Parse 'GENE-X' → (gene, X) or 'GENE' → (gene, None). X is a float 0–100."""
    token = token.strip()
    m = re.match(r'^(.+)-(\d+(?:\.\d+)?)$', token)
    if m:
        pct = float(m.group(2))
        if 0.0 < pct < 100.0:
            return m.group(1).strip(), pct
    return token, None


def _gene_label(gene, pct):
    return f"{gene}pct{int(pct)}" if pct is not None else gene


def load_ko_ov_combinations(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"KO/OV list file not found: {file_path}")

    combos = []
    with open(file_path, "r") as f:
        lines = f.readlines()

    header = lines[0].strip().split('\t')
    header = [h.strip().upper() for h in header]

    ko_idx = header.index("KO") if "KO" in header else None
    ov_idx = header.index("OV") if "OV" in header else None

    if ko_idx is None and ov_idx is None:
        raise ValueError("KO_OV_simulate.txt must contain 'KO' or 'OV' column")

    for line in lines[1:]:
        parts = line.rstrip().split('\t')

        def parse(idx):
            if idx is None or idx >= len(parts):
                return []
            cell = parts[idx].strip()
            if cell.lower() in ('', '0', 'none', 'nan'):
                return []
            return [_parse_gene_with_pct(g) for g in cell.split(',') if g.strip()]

        kos = parse(ko_idx)
        ovs = parse(ov_idx)

        if kos or ovs:
            combos.append({'KO': kos, 'OV': ovs})

    return combos


def main(argv):
    """
    Validate simulated perturbations against observed data.

    Loads simulated expression data for each KO/OV combination and
    creates AnnData objects for visualization and comparison with
    wildtype observations.

    Args:
        argv: Command-line arguments (--input, --split).
    
    Returns:
        None. Saves validation datasets to cardamomOT/ directory.
    """
    inputfile = ''
    split = ''
    stimulus = -1.0
    prior = -1.0
    try:
        opts, args = getopt.getopt(argv, "hi:s:t:p:",
                                   ["input=", "split=", "stimulus=", "prior="])
    except getopt.GetoptError:
        print("[check_KOV_to_sim] Error: Invalid command-line arguments")
        print("[check_KOV_to_sim] Usage: python check_KOV_to_sim.py -i <project_path> -s <split> "
              "[--stimulus <float>] [--prior <float>]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-s", "--split"):
            split = '{}'.format(arg)
        elif opt in ("-t", "--stimulus"):
            stimulus = float(arg)
        elif opt in ("-p", "--prior"):
            prior = float(arg)
        elif opt == "-h":
            print(__doc__)
            sys.exit(0)

    if not inputfile or not split:
        print("[check_KOV_to_sim] Error: Missing required arguments --input and --split")
        sys.exit(1)

    p = '{}/'.format(inputfile)

    # Load KO/OV combinations
    ko_ov_file = os.path.join(p, "Data", "KO_OV_simulate.txt")
    try:
        combos = load_ko_ov_combinations(ko_ov_file)
        if len(combos) == 0:
            print("[check_KOV_to_sim] No KO/OV combinations found in KO_OV_simulate.txt")
            sys.exit(0)
        print(f"[check_KOV_to_sim] Loaded {len(combos)} KO/OV combinations")
    except FileNotFoundError as e:
        print(f"[check_KOV_to_sim] Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[check_KOV_to_sim] Error loading KO/OV combinations: {e}")
        sys.exit(1)

    # Load observed data
    data_path = os.path.join(p, 'Data', f'data_{split}.h5ad')
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Observed data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[check_KOV_to_sim] Loaded observed data from {data_path}")
        print(f"[check_KOV_to_sim] Dataset contains {adata.shape[0]} cells and {adata.shape[1]} genes")
    except FileNotFoundError as e:
        print(f"[check_KOV_to_sim] Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[check_KOV_to_sim] Error loading observed data: {e}")
        sys.exit(1)

    if scipy.sparse.issparse(adata.X):
        data_rna_extracted = adata.X.T.toarray()
    else:
        data_rna_extracted = adata.X.T

    # Validate temporal information
    try:
        times = adata.obs['time'].values 
        if len(np.unique(times)) <= 1:
            raise ValueError("Dataset must contain temporal information with multiple timepoints")
        print(f"[check_KOV_to_sim] Found {len(np.unique(times))} unique timepoints: {sorted(np.unique(times))}")
    except (KeyError, ValueError) as e:
        print(f"[check_KOV_to_sim] Error: {e}")
        sys.exit(1)

    data_real = np.vstack([times, data_rna_extracted]).astype(float)

    # Load model parameters
    try:
        mixture_parameters = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        c = mixture_parameters[-1, :]
        kz = mixture_parameters[:-1, :]
        pi_zinb = np.load(os.path.join(p, 'cardamomOT', 'pi_zinb.npy'))
        times_simulation = np.load(os.path.join(p, 'cardamomOT', 'simulation_times.npy'))
        t_simul = list(set(times_simulation))
        t_simul.sort()
        print(f"[check_KOV_to_sim] Loaded model parameters and simulation times")
    except FileNotFoundError as e:
        print(f"[check_KOV_to_sim] Error: Missing parameter file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[check_KOV_to_sim] Error loading model parameters: {e}")
        sys.exit(1)

    G = np.size(data_real, 0) - 1
    # n_stimuli inferred from mixture_parameters: columns 0..ns-1 are stimulus slots
    ns = mixture_parameters.shape[1] - G

    model = NetworkModel(G)
    if stimulus >= 0:
        model.stimulus = stimulus
    if prior >= 0:
        model.prior_network_pen = prior
    print(f"[check_KOV_to_sim] stimulus={model.stimulus}, prior_network_pen={model.prior_network_pen}")

    # Create AnnData objects for each perturbation combination
    print(f"[check_KOV_to_sim] Creating AnnData objects for {len(combos)} KO/OV combinations")

    for idx, combo in enumerate(combos, start=1):
        kos = combo["KO"]
        ovs = combo["OV"]

        ko_label = '-'.join(_gene_label(g, p) for g, p in kos) if kos else 'none'
        ov_label = '-'.join(_gene_label(g, p) for g, p in ovs) if ovs else 'none'
        label = f"KO_{ko_label}_OV_{ov_label}"
        print(f"[check_KOV_to_sim] Processing combination {idx}/{len(combos)}: {label}")
        
        file_prefix = os.path.join(p, f"cardamomOT/data_kon_simul_{label}.npy")
        prot_prefix = os.path.join(p, f"cardamomOT/data_prot_simul_{label}.npy")

        if not os.path.exists(file_prefix):
            print(f"[check_KOV_to_sim] Warning: Simulation data missing for {label}, file {file_prefix} not found. Skipping.")
            continue

        try:
            vect_kon_sim = np.load(file_prefix)
            print(f"[check_KOV_to_sim] Loaded simulation data for {label}")
        except Exception as e:
            print(f"[check_KOV_to_sim] Error loading simulation data for {label}: {e}")
            continue
        # Generate simulated expression data
        data_sim = np.zeros((G+1, np.size(vect_kon_sim, 0)))
        data_sim[0, :] = times_simulation[:]

        # Generate negative binomial noise + sparsity
        zero_mask = (np.random.uniform(0, 1, data_sim[1:, :].shape) < pi_zinb.reshape((G, 1)))
        print(f'[check_KOV_to_sim] New zeros ratio for {label}: {np.sum(zero_mask == 1)/np.size(data_sim[1:, :]):.3f}')

        data_sim[1:, :] = np.random.negative_binomial(
            (np.max(kz, 0) * vect_kon_sim)[:, ns:].T,
            (c / (c + 1))[ns:].reshape(G, 1)
        )
        data_sim[1:, :] = np.where(zero_mask, 0, data_sim[1:, :])

        # Create AnnData object for simulated RNA
        adata_sim = ad.AnnData(X=data_sim[1:, ].T)
        adata_sim.var = adata.var.copy()
        adata_sim.obs["combo_label"] = label
        adata_sim.obs['time'] = times_simulation

        # Save simulated RNA data
        sim_rna_path = os.path.join(p, f'cardamomOT/adata_sim_{label}_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad')
        try:
            adata_sim.write(sim_rna_path)
            print(f"[check_KOV_to_sim] Saved simulated RNA data: {os.path.basename(sim_rna_path)}")
        except Exception as e:
            print(f"[check_KOV_to_sim] Error saving simulated RNA data: {e}")
            continue

        # Load and save simulated protein data
        if os.path.exists(prot_prefix):
            try:
                data_prot_simul = np.load(prot_prefix)
                adata_prot_simul = ad.AnnData(X=data_prot_simul[:, ns:])
                adata_prot_simul.var = adata.var.copy()
                adata_prot_simul.obs['time'] = times_simulation
                
                prot_path = os.path.join(p, f'cardamomOT/adata_prot_simul_{label}_stim{model.stimulus}_prior{model.prior_network_pen}.h5ad')
                adata_prot_simul.write(prot_path)
                print(f"[check_KOV_to_sim] Saved simulated protein data: {os.path.basename(prot_path)}")
            except Exception as e:
                print(f"[check_KOV_to_sim] Error processing protein data for {label}: {e}")
        else:
            print(f"[check_KOV_to_sim] Warning: Protein simulation data not found for {label}")

    print("[check_KOV_to_sim] KO/OV validation completed successfully")


if __name__ == "__main__":
    main(sys.argv[1:])
