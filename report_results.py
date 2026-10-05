"""
report_results.py
-----------------
Build the final PDF report of a CardamomOT run (generative model, GRN top regulators,
KO/OV predictions) and write it in the project directory.

Usage:
    python report_results.py -i <project_path> [--stimulus <float>] [--prior <float>] [-o <out.pdf>]
(split, report_net_index, report_normalize, report_log1p, report_n_umap: model_parameters sheet)

If --stimulus/--prior are given neither on the command line nor in the workbook, they are read from
the most recent cardamomOT/adata_sim_stim*_prior*.h5ad (fallback: model defaults).

Required input files:
    - Data/data_<split>.h5ad: observed data (obs['time'], optionally obs['cell_type'])
    - cardamomOT/adata_{rna_traj,beta,theta,sim}_stim*_prior*.h5ad: from check_sim_to_data.py
    - cardamomOT/inter_simul.npy: inferred network
Optional:
    - perturbation_simulation sheet + cardamomOT/adata_sim_KO_*_stim*_prior*.h5ad: from check_KOV_to_sim.py
    - cardamomOT/adata_prot_{traj,simul}_stim*_prior*.h5ad

Output:
    - <project_path>/CardamomOT_report_stim<s>_prior<q>.pdf
"""
import os
import re
import sys
import glob
from CardamomOT.run_options import parse_step_options, configure
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
    opts = parse_step_options(argv, 'report_results', __doc__, output=True)
    p, out_path = opts.p, opts.output
    model = configure(NetworkModel(1), opts)
    split = model.split
    net_index, norm, log = int(model.report_net_index), bool(model.report_normalize), bool(model.report_log1p)
    n_umap = int(model.report_n_umap) if model.report_n_umap and int(model.report_n_umap) > 0 else None

    # Tag of the run: given (option or workbook), else that of the latest check_sim_to_data output
    stimulus, prior = model.stimulus, model.prior_network_pen
    if not (model.overridden('stimulus') and model.overridden('prior_network_pen')):
        latest = _latest_tag(p)
        if latest is not None:
            stimulus = stimulus if model.overridden('stimulus') else latest[0]
            prior = prior if model.overridden('prior_network_pen') else latest[1]
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
