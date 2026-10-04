"""
simulate_network_KOV.py
----------------------
Simulate gene expression under in-silico knock-out (KO) and over-expression (OV).

Usage:
    python simulate_network_KOV.py -i <project_path>   (simulate_with_proliferation: model_parameters sheet)

Required input files:
    - Data/data_<split>.h5ad: count matrix with temporal information
    - Data/KO_OV_Stim_simulate.txt (old name KO_OV_simulate.txt): perturbations (tab-separated)
Optional:
    - Data/stimulus_schedule_simulate.txt: schedules of the simulation, one row per simulated
      time: the inference stimuli, then the perturbation stimuli STIM1, STIM2... (default 0 at
      the first time, 1 after)
    - cardamom/inter_t_simul.npy, basal_simul.npy: inferred parameters

Output files:
    - cardamom/data_prot_simul_KO_*.npy: simulated protein for each perturbation
    - cardamom/data_kon_simul_KO_*.npy: simulated bursting for each perturbation

KO_OV_Stim_simulate.txt format (see CardamomOT/tools/perturbations.py):
    KO          OV            STIM
    gene1       gene2,gene3   0
    gene4       0             gene5+gene6-
"""
import sys; sys.path += ['../']
import re
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta
from CardamomOT.inputs import input_dir
from CardamomOT.run_options import parse_step_options, settings, configure
import anndata as ad
import os
import copy
import torch
from CardamomOT.tools.perturbations import (find_perturbation_file, load_perturbations, combo_label,
                                            perturbation_schedule, rate_target_genes)
from CardamomOT.config import n_inference_stimuli, simulation_schedule


def main(argv):
    """
    Simulate knockout/overexpression perturbations of the inferred network.

    Args:
        argv: Command-line arguments (-i <project> and the options of run_options.STEP_OPTIONS).
    """
    opts = parse_step_options(argv, 'simulate_network_KOV', __doc__)
    p = opts.p
    split = settings(opts).split

    ko_ov_file = find_perturbation_file(input_dir(p))
    if ko_ov_file is None:
        print("[simulate_network_KOV] Error: no Data/KO_OV_Stim_simulate.txt")
        sys.exit(1)

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

    # Load perturbations (STIM targets resolved with the gene names)
    try:
        combos = load_perturbations(ko_ov_file, genes=list(adata.var_names))
    except ValueError as e:
        print(f"[simulate_network_KOV] Error: {e}")
        sys.exit(1)
    if len(combos) == 0:
        print(f"[simulate_network_KOV] No perturbation found in {ko_ov_file}")
        sys.exit(0)
    print(f"[simulate_network_KOV] Loaded {len(combos)} perturbations from {ko_ov_file}")

    # ─── STIMULUS SCHEDULES (inference stimuli, then perturbation stimuli) ──
    n_stimuli = n_inference_stimuli(input_dir(p))
    try:
        stim_sched, pert_sched = simulation_schedule(input_dir(p), n_stimuli)
    except ValueError as e:
        print(f"[simulate_network_KOV] Error: {e}")
        sys.exit(1)

    print(f"[simulate_network_KOV] Data: {adata.shape[1]} genes, n_stimuli={n_stimuli}")

    model = NetworkModel_beta(adata.shape[1], n_stimuli=n_stimuli)
    configure(model, opts)  # workbook, then the command-line options
    simulate_with_proliferation = bool(model.simulate_with_proliferation)
    model.simulate_with_proliferation = False  # enabled below once the proliferation network is loaded

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
    filepath = os.path.join(input_dir(p), 'times_to_simulate.txt')
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
                print("[simulate_network_KOV] Effects of the inference stimuli on the net rate (perturbation_inference) "
                      "applied with the simulated schedule")
            print("[simulate_network_KOV] Loaded proliferation network — branching simulation enabled")
        else:
            print("[simulate_network_KOV] Warning: simulate_with_proliferation = True but prolif_network.pt not found (run infer_network_simul first)")

    # Simulate perturbations
    print(f"[simulate_network_KOV] Starting simulation of {len(combos)} perturbations...")
    for idx, combo in enumerate(combos, start=1):
        model_combo = copy.deepcopy(model)
        kos = combo['KO']   # list of (gene, pct_or_None)
        ovs = combo['OV']
        label = combo_label(combo)
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

        # Perturbation stimuli: targets with their sign, own schedules (applied in the simulation)
        model_combo.perturbation_stimulus = []
        try:
            for k, targets in sorted(combo['STIM'].items()):
                signs = np.zeros(adata.shape[1] + ns)
                for gene, sign in targets:
                    if gene in adata.var_names:
                        signs[ns + adata.var_names.get_loc(gene)] = sign
                        print(f"[simulate_network_KOV]   Stimulus {k} {'activates' if sign > 0 else 'inhibits'}: {gene}")
                    else:
                        print(f"[simulate_network_KOV]   Warning: stimulus {k} target '{gene}' not found in data")
                model_combo.perturbation_stimulus.append(
                    dict(signs=signs, schedule=perturbation_schedule(pert_sched, times, k)))
        except ValueError as e:
            print(f"[simulate_network_KOV]   Error: {e}")
            continue

        # RATE effects on the net proliferation rate: delta x signature score (protein / 99th percentile of
        # the trajectories, clipped to [0, 1], mean over the genes), with the schedule of stimulus k
        model_combo.rate_perturbation = []
        try:
            q99 = np.percentile(model_combo.prot[:, ns:], 99, axis=0)
            scale = 1.0 / np.maximum(q99, 1e-12)
            for k, effects in sorted(combo.get('RATE', {}).items()):
                for target, delta in effects:
                    genes_t = rate_target_genes(target, list(adata.var_names), input_dir(p))
                    weights = None
                    if genes_t is not None:
                        weights = np.zeros(adata.shape[1])
                        weights[[adata.var_names.get_loc(g) for g in genes_t]] = 1.0 / len(genes_t)
                    model_combo.rate_perturbation.append(
                        dict(weights=weights, scale=scale, delta=float(delta),
                             schedule=perturbation_schedule(pert_sched, times, k)))
                    print(f"[simulate_network_KOV]   Rate {k}: {target} ({'all cells' if genes_t is None else len(genes_t)}"
                          f"{'' if genes_t is None else ' genes'}) {delta:+g} per time unit at maximal score")
        except ValueError as e:
            print(f"[simulate_network_KOV]   Error: {e}")
            continue

        # Simulate dynamics
        try:
            model_combo.simulate_network(times, stimulus_schedule=stim_sched)
            cardamom_dir = os.path.join(p, 'cardamomOT')
            np.save(os.path.join(cardamom_dir, f'data_prot_simul_{label}'), model_combo.prot)
            np.save(os.path.join(cardamom_dir, f'data_kon_simul_{label}'), model_combo.kon_theta)
            if model_combo.log_population is not None:
                np.save(os.path.join(cardamom_dir, f'data_log_population_{label}'), model_combo.log_population)
            print(f"[simulate_network_KOV]   Results saved for condition: {label}")
        except Exception as e:
            print(f"[simulate_network_KOV]   Error simulating condition {idx}: {e}")
            continue

    print(f"\n[simulate_network_KOV] Completed simulation of all {len(combos)} conditions")

if __name__ == "__main__":
   main(sys.argv[1:])
