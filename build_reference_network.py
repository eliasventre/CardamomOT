"""
Literature prior network (cardamomOT/ref_network.csv) of the genes of Data/data_full.h5ad.

Weight of an edge A -> B: 1 when OmniPath holds a path A -> ... -> B of at most `depth` edges whose
last edge is transcriptional (TF -> target) and whose intermediates are not among the genes, or when
the literature does not cover the pair; 1 / (k + 1) when every such path goes through k observed
genes; 0 without any path (raised to prior_network_pen during inference). Same computation as the
prior written by the gene selection (select_genes_and_split -c 1, see CardamomOT/inference/
literature.py): this script is only needed for a gene list chosen without it.

Usage:
    python build_reference_network.py -i <project_path> [-d <depth>] [--species auto|human|mouse] [--resources extended|core]
    (defaults: model.literature_depth = 3, model.literature_resources = 'extended')
"""
import sys; sys.path += ['../']
import os
import getopt
import numpy as np
import pandas as pd
import anndata as ad

from CardamomOT.inference.literature import literature_graph
from CardamomOT.inference.halflife_db import detect_species


def main(argv):
    # Defaults: those of the gene selection (NetworkModel), so that both priors are identical
    from CardamomOT import NetworkModel
    m = NetworkModel(1)
    inputfile, depth, species, resources = '', m.literature_depth, 'auto', m.literature_resources
    try:
        opts, _ = getopt.getopt(argv, "hi:d:", ["input=", "depth=", "species=", "resources="])
    except getopt.GetoptError:
        print(__doc__)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(__doc__)
            sys.exit(0)
        if opt in ("-i", "--input"):
            inputfile = arg
        if opt in ("-d", "--depth"):
            depth = int(arg)
        if opt == "--species":
            species = arg.lower()
        if opt == "--resources":
            resources = arg.lower()

    p = inputfile
    m.apply_project_parameters(os.path.join(p, ''))  # Data/CardamomOT_inputs.xlsx dominates the options
    depth = m.literature_depth if m.overridden('literature_depth') else depth
    resources = m.literature_resources if m.overridden('literature_resources') else resources
    for name in ('data_full.h5ad', 'data_train.h5ad'):
        data_path = os.path.join(p, 'Data', name)
        if os.path.exists(data_path):
            break
    else:
        raise FileNotFoundError("No Data/data_full.h5ad nor Data/data_train.h5ad: run select_genes_and_split first")
    genes = list(ad.read_h5ad(data_path, backed='r').var_names)
    out = os.path.join(p, 'cardamomOT', 'ref_network.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if depth > 0:
        if species == 'auto':
            species = detect_species(genes)[0]
        lg = literature_graph(species, resources)
        prior = lg.prior(genes, depth)
        reg, tgt = lg.coverage(genes)
        off = ~np.eye(len(genes), dtype=bool)
        print(f"[build_reference_network] {species} ({resources}), depth {depth}: {reg.mean() * 100:.0f}% of the genes "
              f"covered as regulators, {tgt.mean() * 100:.0f}% as targets; {np.mean(prior[off] > 0) * 100:.0f}% of the "
              f"edges allowed ({int((prior[off] > 0).sum())} free interactions)")
    elif os.path.exists(out):
        print(f"[build_reference_network] depth 0: existing {out} kept")
        return
    else:
        prior = np.ones((len(genes), len(genes)))
    up = [g.upper() for g in genes]  # infer_network_structure matches upper-case names
    pd.DataFrame(prior, index=up, columns=up).to_csv(out)
    print(f"[build_reference_network] Saved {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
