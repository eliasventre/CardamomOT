"""
get_proliferation_rates.py
----------------------------
Estimate per-cell net proliferation rates (birth - death) from literature
proliferation/death/senescence gene signatures.

Runs BEFORE differential gene selection (select_DEgenes_and_split.py), on the
full unfiltered dataset — DE gene selection can otherwise discard many of the
literature marker genes used to score proliferation/death, so scoring happens
first while the complete gene set is still available. Because the estimate
is written to `adata.obs`, it is preserved automatically when
select_DEgenes_and_split.py later subsets genes and splits cells into
train/test (obs columns are untouched by that subsetting); select_DEgenes_and_split.py
only ever reads Data/data.h5ad, never Data/data_complete.h5ad.

If `Data/data.h5ad` was already prepared with a pre-filtered gene set, the
literature marker genes may be missing from it. In that case, place the
unfiltered dataset alongside it as `Data/data_complete.h5ad`: it is used ONLY
to score the proliferation/death/senescence signatures (and is never
modified or written back), and the resulting per-cell rates are mapped onto
the matching cells of `Data/data.h5ad` by cell name (every cell in
Data/data.h5ad is assumed to also be present in Data/data_complete.h5ad; the
converse need not hold). If `Data/data_complete.h5ad` is absent,
`Data/data.h5ad` is used directly for both scoring and output, as before.

Usage:
    python get_proliferation_rates.py -i <project_path> [--species human|mouse]

Required input files:
    - Data/data.h5ad: count matrix (all genes, pre-selection, unless
      Data/data_complete.h5ad is provided)

Optional input files:
    - Data/data_complete.h5ad: unfiltered count matrix used only to score the
      gene signatures when Data/data.h5ad has already been gene-filtered;
      never modified
    - Data/proliferation_signatures.csv|txt: proliferation marker genes,
      one per line or comma-separated (overrides the built-in default list
      for the chosen --species)
    - Data/death_signatures.csv|txt: death marker genes, one per line or
      comma-separated (overrides the built-in default list for the chosen
      --species)
    - Data/senescence_signatures.csv|txt: senescence/cell-cycle-arrest/
      quiescence marker genes, one per line or comma-separated (overrides
      the built-in default list for the chosen --species). Used to gate the
      birth rate to ~0 for cells that are arrested/senescent rather than
      actively dying, even if they still carry residual cell-cycle gene
      expression — see CardamomOT.tools.estimate_proliferation. Pass
      an empty file (or set `--no-senescence-gating`) to disable this and
      recover the plain birth - death estimate.
    - Data/proliferation_rates.csv|txt: two columns, no header
      (cell_type, net_rate) — anchors the literature estimate's per-cell-type
      mean to a trusted population-level rate. Grouping uses
      `adata.obs['cell_type_proliferation']` if present, else falls back to
      `adata.obs['cell_type']`; add `cell_type_proliferation` during
      preprocessing to use a coarser/finer grouping for this task without
      touching `cell_type` itself. `net_rate` must be in the same
      time unit as `adata.obs['time']` (hours, not moscot/WOT's day^-1
      convention — see CardamomOT.tools.estimate_proliferation), since it is
      blended directly with the (already hour^-1) literature estimate

Output files:
    - Data/data.h5ad: updated in place with obs['proliferation_net_rate']
      (Data/data_complete.h5ad, if used, is never written to)
"""
import sys; sys.path += ['../']
import os
import getopt
import anndata as ad
import pandas as pd

from CardamomOT import find_data_file, read_gene_list, resolve_cell_type_obs
from CardamomOT.tools.estimate_proliferation import (
    estimate_growth_rates, combine_growth_rates_with_reference,
)


def assign_proliferation_rates(adata, prolif_path, species='human', proliferation_genes=None,
                                death_genes=None, senescence_genes=None, senescence_gating=True):
    """
    Compute and assign adata.obs['proliferation_net_rate'] (birth − death) in place.

    Estimated per-cell from literature proliferation/death/senescence gene
    signatures (tools.estimate_proliferation). A senescence/arrest score
    gates the birth term toward 0 for cells that look arrested/senescent
    rather than actively proliferating, without touching the death term
    (see estimate_proliferation.estimate_growth_rates); set
    `senescence_gating=False` to disable this and recover the plain
    birth - death estimate.

    If `prolif_path` points to a per-cell-type reference table
    (Data/proliferation_rates.csv or .txt, two columns: cell_type, rate), the
    literature estimate is anchored so its mean matches the reference rate
    within each cell type, keeping per-cell heterogeneity from the
    signature. Grouping uses `adata.obs['cell_type_proliferation']` if
    present, else falls back to `adata.obs['cell_type']`.
    `rate` must be expressed per hour, matching `adata.obs['time']`
    and the (already hour^-1) literature estimate it is blended with — see
    estimate_proliferation.estimate_growth_rates for why that estimate is in
    hours rather than moscot/WOT's native day^-1.
    """
    prefix = "[get_proliferation_rates]"
    kwargs = {'species': species, 'senescence_gating': senescence_gating}
    if proliferation_genes is not None:
        kwargs['proliferation_genes'] = proliferation_genes
    if death_genes is not None:
        kwargs['death_genes'] = death_genes
    if senescence_genes is not None:
        kwargs['senescence_genes'] = senescence_genes

    net_lit = estimate_growth_rates(adata, **kwargs)
    print(f"{prefix} Estimated literature-based proliferation rates for "
          f"{len(net_lit)} cells (mean={net_lit.mean():.4f})")

    celltype_col = resolve_cell_type_obs(adata, 'cell_type_proliferation')

    if prolif_path is not None and celltype_col is not None:
        user_rates = pd.read_csv(prolif_path, sep=None, engine='python',
                                  header=None, index_col=0).iloc[:, 0]
        user_rates.index = user_rates.index.astype(str)
        adata.obs['proliferation_net_rate'] = combine_growth_rates_with_reference(
            net_lit, adata.obs[celltype_col].astype(str).values, user_rates.to_dict()
        )
        print(f"{prefix} Anchored literature proliferation rates to {prolif_path} "
              f"per '{celltype_col}' ({len(user_rates)} types)")
    else:
        if prolif_path is not None:
            print(f"{prefix} Found {prolif_path} but neither adata.obs['cell_type_proliferation'] "
                  "nor adata.obs['cell_type'] is present; using unanchored literature estimate")
        else:
            print(f"{prefix} No Data/proliferation_rates.{{csv,txt}} found; "
                  "using literature-only proliferation rate estimate")
        adata.obs['proliferation_net_rate'] = net_lit


def main(argv):
    """
    Estimate and assign per-cell net proliferation rates.

    Args:
        argv: Command-line arguments (--input, --species, --no-senescence-gating).

    Returns:
        None. Updates Data/data.h5ad in place with obs['proliferation_net_rate'].
    """
    inputfile = ''
    species = 'human'
    senescence_gating = True
    try:
        opts, args = getopt.getopt(
            argv, "hi:", ["input=", "species=", "no-senescence-gating"]
        )
    except getopt.GetoptError:
        print("[get_proliferation_rates] Error: Invalid command-line arguments")
        print("[get_proliferation_rates] Usage: python get_proliferation_rates.py "
              "-i <project_path> [--species human|mouse] [--no-senescence-gating]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        elif opt == "--species":
            species = arg
        elif opt == "--no-senescence-gating":
            senescence_gating = False
        elif opt == "-h":
            print(__doc__)
            sys.exit(0)

    if not inputfile:
        print("[get_proliferation_rates] Error: Missing required argument --input")
        sys.exit(1)

    p = '{}/'.format(inputfile)
    data_dir = os.path.join(p, 'Data')

    # Data/data.h5ad is always the file that gets updated.
    data_path = os.path.join(data_dir, 'data.h5ad')
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        adata_target = ad.read_h5ad(data_path)
        print(f"[get_proliferation_rates] Loaded dataset from {data_path}")
        print(f"[get_proliferation_rates] Dataset contains "
              f"{adata_target.shape[0]} cells and {adata_target.shape[1]} genes")
    except FileNotFoundError as e:
        print(f"[get_proliferation_rates] Error: {e}")
        print(f"[get_proliferation_rates] Please ensure Data/data.h5ad exists in {p}")
        sys.exit(1)

    if 'proliferation_net_rate' in adata_target.obs.columns:
        print("[get_proliferation_rates] adata.obs['proliferation_net_rate'] "
              "already present in Data/data.h5ad; recomputing and overwriting it")

    # Data/data_complete.h5ad, if present, is used only to score the gene
    # signatures (e.g. when Data/data.h5ad was already gene-filtered) and is
    # never written to.
    complete_path = os.path.join(data_dir, 'data_complete.h5ad')
    using_complete = os.path.exists(complete_path)
    if using_complete:
        adata_score = ad.read_h5ad(complete_path)
        print(f"[get_proliferation_rates] Found {complete_path}; scoring gene signatures on it "
              f"({adata_score.shape[0]} cells, {adata_score.shape[1]} genes) instead of Data/data.h5ad")
    else:
        adata_score = adata_target

    prolif_path = find_data_file(data_dir, 'proliferation_rates')

    species = species.strip().lower()
    print(f"[get_proliferation_rates] Using species='{species}'")
    print(f"[get_proliferation_rates] Senescence gating: "
          f"{'enabled' if senescence_gating else 'disabled (plain birth - death)'}")

    proliferation_genes = None
    proliferation_genes_path = find_data_file(data_dir, 'proliferation_signatures')
    if proliferation_genes_path is not None:
        proliferation_genes = read_gene_list(proliferation_genes_path)
        print(f"[get_proliferation_rates] Loaded {len(proliferation_genes)} proliferation marker "
              f"genes from {proliferation_genes_path}")

    death_genes = None
    death_genes_path = find_data_file(data_dir, 'death_signatures')
    if death_genes_path is not None:
        death_genes = read_gene_list(death_genes_path)
        print(f"[get_proliferation_rates] Loaded {len(death_genes)} death marker genes "
              f"from {death_genes_path}")

    senescence_genes = None
    senescence_genes_path = find_data_file(data_dir, 'senescence_signatures')
    if senescence_genes_path is not None:
        senescence_genes = read_gene_list(senescence_genes_path)
        print(f"[get_proliferation_rates] Loaded {len(senescence_genes)} senescence/arrest marker "
              f"genes from {senescence_genes_path}")

    assign_proliferation_rates(adata_score, prolif_path, species=species,
                                proliferation_genes=proliferation_genes, death_genes=death_genes,
                                senescence_genes=senescence_genes, senescence_gating=senescence_gating)

    if using_complete:
        missing = adata_target.obs_names.difference(adata_score.obs_names)
        if len(missing) > 0:
            print(f"[get_proliferation_rates] Error: {len(missing)} cell(s) in Data/data.h5ad "
                  f"were not found in Data/data_complete.h5ad; cannot map proliferation rates "
                  f"(first missing: {list(missing[:5])})")
            sys.exit(1)
        adata_target.obs['proliferation_net_rate'] = adata_score.obs.loc[
            adata_target.obs_names, 'proliferation_net_rate'
        ].to_numpy()
        print("[get_proliferation_rates] Mapped proliferation rates from Data/data_complete.h5ad "
              "onto Data/data.h5ad cells")

    try:
        adata_target.write(data_path)
        print(f"[get_proliferation_rates] Saved updated dataset to {data_path}")
    except Exception as e:
        print(f"[get_proliferation_rates] Error saving dataset: {e}")
        sys.exit(1)

    print("[get_proliferation_rates] Proliferation rate assignment completed successfully")


if __name__ == "__main__":
    main(sys.argv[1:])