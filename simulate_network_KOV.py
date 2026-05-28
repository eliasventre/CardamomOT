"""
simulate_network_KOV.py
----------------------
Simulate gene expression under in-silico knock-out (KO) and over-expression (OV).

Usage:
    python simulate_network_KOV.py -i <project_path> -s <split>

Required input files:
    - Data/data_<split>.h5ad: count matrix with temporal information
    - Data/KO_OV_simulate.txt: table defining KO/OV combinations (tab-separated)
    - cardamom/inter_t_simul.npy, basal_simul.npy: inferred parameters

Output files:
    - cardamom/data_prot_simul_KO_*.npy: simulated protein for each perturbation
    - cardamom/data_kon_simul_KO_*.npy: simulated bursting for each perturbation

KO_OV_simulate.txt format:
    KO          OV
    gene1       gene2,gene3
    gene4
"""
import sys; sys.path += ['../']
import re
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta
import getopt
import anndata as ad
import os
import copy


def _parse_gene_with_pct(token):
    """Parse 'GENE-X' → (gene, X) or 'GENE' → (gene, None). X is a float 0–100.

    A trailing '-<number>' suffix is interpreted as a percentage only when
    the number is strictly between 0 and 100 (exclusive) so that gene names
    like 'HIF-1A' are not accidentally matched.
    """
    token = token.strip()
    m = re.match(r'^(.+)-(\d+(?:\.\d+)?)$', token)
    if m:
        pct = float(m.group(2))
        if 0.0 < pct < 100.0:
            return m.group(1).strip(), pct
    return token, None


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

    for line_num, line in enumerate(lines[1:], start=2):
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
    Simulate knockout/overexpression perturbations of the inferred network.

    Args:
        argv: Command-line arguments (--input, --split).
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
    ko_ov_file = os.path.join(p, "Data", "KO_OV_simulate.txt")
    try:
        combos = load_ko_ov_combinations(ko_ov_file)
    except FileNotFoundError:
        print(f"[simulate_network_KOV] Error: KO/OV list file not found at {ko_ov_file}")
        sys.exit(1)
    except ValueError as e:
        print(f"[simulate_network_KOV] Error: {e}")
        sys.exit(1)

    if len(combos) == 0:
        print("[simulate_network_KOV] No KO/OV combinations found in KO_OV_simulate.txt")
        sys.exit(0)

    print(f"[simulate_network_KOV] Loaded {len(combos)} KO/OV combinations")

    # Load gene expression data (for gene count and var_names)
    data_path = os.path.join(p, 'Data', f'data_{split}.h5ad')
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[simulate_network_KOV] Loaded data from {data_path}")
    except FileNotFoundError as e:
        print(f"[simulate_network_KOV] Error: {e}")
        sys.exit(1)

    # ─── LOAD STIMULUS SCHEDULE ──────────────────────────────────────────
    stim_sched = None
    for sched_name in ('stimulus_schedule_simul.txt', 'stimulus_schedule.txt'):
        sched_path = os.path.join(p, 'Data', sched_name)
        if os.path.exists(sched_path):
            stim_sched = np.loadtxt(sched_path)
            print(f"[simulate_network_KOV] Loaded stimulus schedule from {sched_path}")
            break

    # ─── DETECT n_stimuli FROM SCHEDULE ─────────────────────────────────
    _stim_arr = np.asarray(stim_sched) if stim_sched is not None else None
    n_stimuli = int(_stim_arr.shape[1]) if (_stim_arr is not None and _stim_arr.ndim == 2) else 1

    print(f"[simulate_network_KOV] Data: {adata.shape[1]} genes, n_stimuli={n_stimuli}")

    model = NetworkModel_beta(adata.shape[1], n_stimuli=n_stimuli)

    # Load network model parameters
    print("[simulate_network_KOV] Loading inferred network parameters...")
    try:
        model.d_t = np.load(os.path.join(p, 'cardamomOT', 'degradations_temporal.npy'))
        model.inter_t = np.load(os.path.join(p, 'cardamomOT', 'inter_t_simul.npy'))
        model.inter = np.load(os.path.join(p, 'cardamomOT', 'inter_simul.npy'))
        # Validate n_stimuli against loaded inter (authoritative for simulation)
        n_stimuli_inter = model.inter.shape[0] - adata.shape[1]
        if n_stimuli_inter != model.n_stimuli:
            print(f"[simulate_network_KOV] Warning: correcting n_stimuli from {model.n_stimuli} "
                  f"to {n_stimuli_inter} based on loaded inter_simul.npy")
            model.n_stimuli = n_stimuli_inter
        model.a = np.load(os.path.join(p, 'cardamomOT', 'mixture_parameters.npy'))
        model.times_data = np.load(os.path.join(p, 'cardamomOT', 'data_times.npy'))
        model.kon_beta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_beta.npy'))
        model.rna = np.load(os.path.join(p, 'cardamomOT', 'data_rna.npy'))
        model.proba_traj = np.load(os.path.join(p, 'cardamomOT', 'proba_traj.npy'))
        model.ratios = np.load(os.path.join(p, 'cardamomOT', 'ratios.npy'))
        model.n_networks = np.load(os.path.join(p, 'cardamomOT', 'n_networks.npy'))
        samples_path = os.path.join(p, 'cardamomOT', 'data_samples.npy')
        if os.path.exists(samples_path):
            model.samples_data = np.load(samples_path)
            print("[simulate_network_KOV] Loaded per-cell sample IDs")
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

    # ─── COMPUTE CLEAN BASELINE (unperturbed by training KOVs) ───────────────
    basal_simul_raw = np.load(os.path.join(p, 'cardamomOT', 'basal_simul.npy'))
    basal_t_simul_raw = np.load(os.path.join(p, 'cardamomOT', 'basal_t_simul.npy'))
    EPS = 1e-16

    if basal_simul_raw.ndim == 3:
        mask_path = os.path.join(p, 'cardamomOT', 'basal_ref_mask.npy')
        n_samp, G_tot_b, _ = basal_simul_raw.shape

        # ── Build per-sample clean basal (3-D) ────────────────────────────────
        # Start from the per-sample inferred basal and replace, for each gene
        # that was forced during training (basal_ref_mask), the perturbed
        # samples with the mean of the *free* (unforced) samples.
        basal_clean_3d = basal_simul_raw.copy()   # (n_samp, G_tot, n_nw)
        if os.path.exists(mask_path):
            basal_ref_mask = np.load(mask_path)   # (n_samples, G_tot)
            for g in range(G_tot_b):
                perturbed = np.where(basal_ref_mask[:, g])[0]
                free      = np.where(~basal_ref_mask[:, g])[0]
                if len(perturbed) > 0:
                    clean_g = (basal_simul_raw[free, g, :].mean(axis=0)
                               if len(free) > 0
                               else basal_simul_raw[:, g, :].mean(axis=0))
                    basal_clean_3d[perturbed, g, :] = clean_g

        # 2-D average kept for ops that still need it (e.g. non-per-sample path)
        basal_clean = basal_clean_3d.mean(axis=0)   # (G_tot, n_nw)

        # ── Build temporal clean basal, preserving per-sample structure ────────
        if basal_t_simul_raw.ndim == 4:
            # (T-1, n_samp, G_tot, n_nw): normalise by *per-sample* static basal
            # so the temporal modulation is correct for each sample.
            temporal_factor = basal_t_simul_raw / (basal_simul_raw[np.newaxis] + EPS)
            basal_t_clean = basal_clean_3d[np.newaxis] * temporal_factor  # (T-1, n_samp, G_tot, n_nw)
            print(f"[simulate_network_KOV] basal_t_simul is 4-D (per-sample); "
                  f"keeping {n_samp} samples for per-sample routing")
        else:
            # (T-1, G_tot, n_nw): normalise by sample-mean static basal
            basal_mean_orig = basal_simul_raw.mean(axis=0)   # (G_tot, n_nw)
            temporal_factor = basal_t_simul_raw / (basal_mean_orig[np.newaxis] + EPS)
            basal_t_clean = basal_clean[np.newaxis] * temporal_factor     # (T-1, G_tot, n_nw)

        print(f"[simulate_network_KOV] Clean baseline from {n_samp}-sample basal "
              f"(basal_t shape: {basal_t_clean.shape})")
    else:
        # 2-D basal (no per-sample structure)
        basal_clean    = basal_simul_raw
        basal_clean_3d = None
        basal_t_clean  = basal_t_simul_raw

    N = np.sum(model.times_data == 0)
    times_simulation = np.zeros(len(times)*N)
    for t in range(0, len(times)):
        times_simulation[t*N:(t+1)*N] = times[t]

    ns = model.n_stimuli

    def _gene_label(gene, pct):
        return f"{gene}pct{int(pct)}" if pct is not None else gene

    # Simulate perturbations
    print(f"[simulate_network_KOV] Starting simulation of {len(combos)} perturbations...")
    for idx, combo in enumerate(combos, start=1):
        model_combo = copy.deepcopy(model)
        kos = combo['KO']   # list of (gene, pct_or_None)
        ovs = combo['OV']
        ko_label = '-'.join(_gene_label(g, p) for g, p in kos) if kos else 'none'
        ov_label = '-'.join(_gene_label(g, p) for g, p in ovs) if ovs else 'none'
        label = f"KO_{ko_label}_OV_{ov_label}"
        print(f"\n[simulate_network_KOV] Simulating condition {idx}/{len(combos)}: {label}")

        # Reset model to clean (unperturbed) baseline.
        # Use the per-sample 3-D basal when available so that simulate_trajectories_unitary
        # can route each cell to its own sample's basal (same as simulate_network.py).
        try:
            model_combo.basal = basal_clean_3d.copy() if basal_clean_3d is not None else basal_clean.copy()
            model_combo.basal_t = basal_t_clean.copy()
            model_combo.production_factor = None   # reset per-gene creation rate scaling
            model_combo.prot = np.load(os.path.join(p, 'cardamomOT', 'data_prot_forsimul.npy'))
            model_combo.kon_theta = np.load(os.path.join(p, 'cardamomOT', 'data_kon_theta.npy'))
        except FileNotFoundError as e:
            print(f"[simulate_network_KOV] Error resetting model: {e}")
            continue

        # Apply knockouts
        # - No percentage: silence via basal_t (complete KO, existing behaviour)
        # - With percentage X: scale the per-gene creation rate by (1 - X/100),
        #   which correctly reduces steady-state expression in both ODE and PDMP modes
        #   without altering degradation rates or dynamics speed.
        # model.prot[:N] stays WT so t=0 NB parameters are identical across conditions.
        for gene, pct in kos:
            if gene in adata.var_names:
                ind = ns + adata.var_names.get_loc(gene)
                if pct is None:
                    if model_combo.basal_t.ndim == 4:
                        model_combo.basal_t[:, :, ind] = -100 - np.sum(model_combo.inter_t[-1, :, ind])
                    else:
                        model_combo.basal_t[:, ind] = -100 - np.sum(model_combo.inter_t[-1, :, ind])
                    print(f"[simulate_network_KOV]   KO 100%: {gene} (index {ind})")
                else:
                    factor = max(1.0 - pct / 100.0, 1e-12)
                    if model_combo.production_factor is None:
                        model_combo.production_factor = np.ones(adata.shape[1] + ns)
                    model_combo.production_factor[ind] = factor
                    print(f"[simulate_network_KOV]   KO {pct:.0f}%: {gene} (index {ind}), creation ×{factor:.3g}")
            else:
                print(f"[simulate_network_KOV]   Warning: Gene '{gene}' not found in data")

        # Apply overexpressions
        # - No percentage: activate via basal_t (complete OV, existing behaviour)
        # - With percentage X: scale the per-gene creation rate by 1/(1 - X/100),
        #   which correctly increases steady-state expression in both ODE and PDMP modes.
        for gene, pct in ovs:
            if gene in adata.var_names:
                ind = ns + adata.var_names.get_loc(gene)
                if pct is None:
                    if model_combo.basal_t.ndim == 4:
                        model_combo.basal_t[:, :, ind] = 100 + np.sum(model_combo.inter_t[-1, :, ind])
                    else:
                        model_combo.basal_t[:, ind] = 100 + np.sum(model_combo.inter_t[-1, :, ind])
                    print(f"[simulate_network_KOV]   OV 100%: {gene} (index {ind})")
                else:
                    factor = 1.0 / max(1.0 - pct / 100.0, 1e-12)
                    if model_combo.production_factor is None:
                        model_combo.production_factor = np.ones(adata.shape[1] + ns)
                    model_combo.production_factor[ind] = factor
                    print(f"[simulate_network_KOV]   OV {pct:.0f}%: {gene} (index {ind}), creation ×{factor:.3g}")
            else:
                print(f"[simulate_network_KOV]   Warning: Gene '{gene}' not found in data")

        # Simulate dynamics
        try:
            model_combo.simulate_network(times, stimulus_schedule=stim_sched)
            cardamom_dir = os.path.join(p, 'cardamomOT')
            np.save(os.path.join(cardamom_dir, f'data_prot_simul_{label}'), model_combo.prot)
            np.save(os.path.join(cardamom_dir, f'data_kon_simul_{label}'), model_combo.kon_theta)
            print(f"[simulate_network_KOV]   Results saved for condition: {label}")
        except Exception as e:
            print(f"[simulate_network_KOV]   Error simulating condition {idx}: {e}")
            continue

    print(f"\n[simulate_network_KOV] Completed simulation of all {len(combos)} conditions")

if __name__ == "__main__":
   main(sys.argv[1:])
