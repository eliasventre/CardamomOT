"""
report_results.py
-----------------
Build the final PDF report of a CardamomOT run (generative model, GRN top regulators,
KO/OV predictions) and write it in the project directory.

Usage:
    python report_results.py -i <project_path> -s <split> [--stimulus <float>] [--prior <float>]
                             [--net-index <int>] [--norm] [--no-log] [--n-umap <int>] [-o <out.pdf>]

If --stimulus/--prior are omitted (or negative), they are read from the most recent
cardamomOT/adata_sim_stim*_prior*.h5ad (fallback: model defaults).

Required input files:
    - Data/data_<split>.h5ad: observed data (obs['time'], optionally obs['cell_type'])
    - cardamomOT/adata_{rna_traj,beta,theta,sim}_stim*_prior*.h5ad: from check_sim_to_data.py
    - cardamomOT/inter_simul.npy: inferred network
Optional:
    - Data/KO_OV_simulate.txt + cardamomOT/adata_sim_KO_*_stim*_prior*.h5ad: from check_KOV_to_sim.py
    - cardamomOT/adata_prot_{traj,simul}_stim*_prior*.h5ad

Output:
    - <project_path>/CardamomOT_report_stim<s>_prior<q>.pdf
"""
import os
import re
import sys
import glob
import getopt
import logging
import matplotlib
matplotlib.use('Agg')
logging.getLogger('fontTools').setLevel(logging.WARNING)  # silence PDF font-subsetting logs

from CardamomOT import NetworkModel
from CardamomOT.inputs import input_dir
from CardamomOT.tools.report import generate_report
from CardamomOT.tools.perturbations import (find_perturbation_file, load_perturbations, combo_label,
                                            combo_description, combo_genes)


def _latest_tag(p):
    """(stim, prior) of the most recent WT check_sim_to_data output, or None."""
    pat = re.compile(r'^adata_sim_stim([-\d.eE]+)_prior([-\d.eE]+)\.h5ad$')
    found = [(os.path.getmtime(f), m) for f in glob.glob(os.path.join(p, 'cardamomOT', 'adata_sim_stim*_prior*.h5ad'))
             if (m := pat.match(os.path.basename(f)))]
    if not found:
        return None
    m = max(found, key=lambda x: x[0])[1]
    return float(m.group(1)), float(m.group(2))


def main(argv):
    inputfile, split, out_path = '', '', None
    stimulus, prior = -1.0, -1.0
    net_index, norm, log, n_umap = 0, False, True, 4000
    try:
        opts, _ = getopt.getopt(argv, "hi:s:t:p:o:", ["input=", "split=", "stimulus=", "prior=", "output=",
                                                       "net-index=", "norm", "no-log", "n-umap="])
    except getopt.GetoptError:
        print(__doc__)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(__doc__); sys.exit(0)
        elif opt in ('-i', '--input'):
            inputfile = arg
        elif opt in ('-s', '--split'):
            split = arg
        elif opt in ('-t', '--stimulus'):
            stimulus = float(arg)
        elif opt in ('-p', '--prior'):
            prior = float(arg)
        elif opt in ('-o', '--output'):
            out_path = arg
        elif opt == '--net-index':
            net_index = int(arg)
        elif opt == '--norm':
            norm = True
        elif opt == '--no-log':
            log = False
        elif opt == '--n-umap':
            n_umap = int(arg) if int(arg) > 0 else None

    if not inputfile or not split:
        print("[report_results] Error: missing required arguments --input and --split")
        sys.exit(1)
    p = os.path.join(inputfile, '')

    # Resolve the stim/prior tag of the run
    model = NetworkModel(1)
    model.apply_project_parameters(p)  # Data/CardamomOT_inputs.xlsx dominates the options
    if model.overridden('stimulus'):
        stimulus = model.stimulus
    if model.overridden('prior_network_pen'):
        prior = model.prior_network_pen
    if stimulus < 0 or prior < 0:
        latest = _latest_tag(p)
        default = latest if latest is not None else (model.stimulus, model.prior_network_pen)
        stimulus = stimulus if stimulus >= 0 else default[0]
        prior = prior if prior >= 0 else default[1]
    print(f"[report_results] stimulus={stimulus}, prior={prior}")

    # KO/OV conditions listed in the data folder
    perturbations = []
    ko_ov_file = find_perturbation_file(input_dir(p))
    if ko_ov_file is not None:
        import anndata as ad
        genes = list(ad.read_h5ad(os.path.join(p, 'Data', f'data_{split}.h5ad'), backed='r').var_names)
        for combo in load_perturbations(ko_ov_file, genes):
            perturbations.append((combo_label(combo), combo_description(combo), combo_genes(combo)))
        print(f"[report_results] {len(perturbations)} perturbations in {ko_ov_file}")
    else:
        print("[report_results] No KO_OV_Stim_simulate.txt: perturbation section will be empty")

    try:
        out = generate_report(p, split, stimulus, prior, perturbations, out_path=out_path,
                              net_index=net_index, normtransform=norm, logtransform=log, n_umap=n_umap)
    except FileNotFoundError as e:
        print(f"[report_results] Error: {e}")
        sys.exit(1)
    print(f"[report_results] Report written to {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
