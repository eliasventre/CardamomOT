"""
Literature prior network (cardamomOT/ref_network.csv) of the genes of Data/data_full.h5ad.

Weight of an edge A -> B: 1 when OmniPath holds a path A -> ... -> B of at most `depth` edges whose
last edge is transcriptional (TF -> target) and whose intermediates are not among the genes, or when
the literature does not cover the pair; 1 / (k + 1) when every such path goes through k observed
genes; 0 without any path (raised to prior_network_pen during inference). Same computation as the
prior written by the gene selection (select_genes_and_split with select_genes, see CardamomOT/inference/
literature.py): this script is only needed for a gene list chosen without it.

Usage:
    python build_reference_network.py -i <project_path>
    (literature_depth, literature_resources, species: Model_parameters sheet; run by the pipeline when
    build_prior_network is True and select_genes is False)
"""
import sys; sys.path += ['../']
import os
from CardamomOT.run_options import parse_step_options, settings
import numpy as np
import pandas as pd
import anndata as ad

from CardamomOT.inference.literature import literature_graph
from CardamomOT.inference.halflife_db import detect_species


def main(argv):
    # Parameters of the gene selection (NetworkModel), so that both priors are identical
    opts = parse_step_options(argv, 'build_reference_network', __doc__)
    p = opts.p
    m = settings(opts)
    depth, species, resources = int(m.literature_depth), str(m.species).lower(), str(m.literature_resources).lower()
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
