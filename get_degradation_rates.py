"""
get_degradation_rates.py
-------------------------
Assign literature mRNA (d0) and protein (d1) degradation rates to genes, in hour⁻¹.

The species is detected from the gene nomenclature (mouse "Gata1" vs human
"GATA1") unless given, and half-lives (hours) are read from that species' own
reference, shipped with the package in CardamomOT/data/halflife:
    - mouse: Schwanhäusser et al. 2011 (corrigendum 2013), NIH3T3, mRNA and protein;
    - human: RNADecayCafe v1.1 (mRNA, 11 cell lines) and Mathieson et al. 2018
      (protein, primary cells).
Genes absent from their species' table get a value estimated on that species'
scale from their ortholog in the other species' table (recalibrated on common
orthologs) and from biologically related measured genes (paralogs, gene family,
Gene Ontology function); see CardamomOT/inference/halflife_db.py.

Usage:
    python get_degradation_rates.py -i <project_path> -s <split> [--species auto|human|mouse] [--overwrite]

Required input files:
    - Data/data_full.h5ad: full count matrix

Output files:
    - Data/data_full.h5ad: updated with d0 (mRNA) and d1 (protein) degradation rates (h⁻¹)
    - Data/data_train.h5ad, data_test.h5ad: updated with degradation rates (if split != "full")
    - Data/degradation_rates_report.csv: per-gene match, half-lives, source and neighbours used
"""
import sys; sys.path += ['../']
import os
import numpy as np
from CardamomOT import extract_degradation_rates
from CardamomOT.inference.halflife_db import REFERENCES
import anndata as ad
import getopt

verb = 1


def assign_rates(adata, details, species, overwrite=False):
    """
    Write d0/d1 (h⁻¹) and their provenance into ``adata.var``.

    Rates are clipped to [median / 10, 10 × median] of the dataset, as before,
    now centred on the median so that a few very unstable genes do not shift
    the window. Existing columns are kept unless ``overwrite``.
    """
    assert list(details["gene"]) == [str(g) for g in adata.var_names], "details must follow adata.var_names"
    for col, quantity in (("d0", "mrna"), ("d1", "protein")):
        if col in adata.var.columns and not overwrite:
            print(f"[get_degradation_rates] {col} already present, skipping (use --overwrite to replace it)")
            continue
        rates = details[col].to_numpy(dtype=float)
        med = np.median(rates)
        adata.var[col] = np.clip(rates, med / 10, 10 * med)
        adata.var[f"{col}_source"] = details[f"{quantity}_source"].to_numpy()
        print(f"[get_degradation_rates] Assigned {quantity} degradation rates ({col}), "
              f"median: {med:.4f} h^-1 (half-life {np.log(2) / med:.1f} h)")
    adata.uns["degradation_rates"] = {"units": "hour^-1", "species": species,
                                      "reference_mrna": REFERENCES[species]["mrna"],
                                      "reference_protein": REFERENCES[species]["protein"]}


def main(argv):
    """
    Assign literature degradation rates to the genes of a project.

    Args:
        argv: Command-line arguments (--input, --split, --species, --overwrite).

    Returns:
        None. Updates AnnData files with degradation rates.
    """
    inputfile = ''
    split = ''
    species = 'auto'
    overwrite = False
    try:
        opts, args = getopt.getopt(argv, "hi:s:", ["input=", "split=", "species=", "overwrite"])
    except getopt.GetoptError:
        print("[get_degradation_rates] Error: Invalid command-line arguments")
        print("[get_degradation_rates] Usage: python get_degradation_rates.py -i <project_path> -s <split> "
              "[--species auto|human|mouse] [--overwrite]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-s", "--split"):
            split = arg
        elif opt == "--species":
            species = arg.strip().lower()
        elif opt == "--overwrite":
            overwrite = True
        elif opt == "-h":
            print(__doc__)
            sys.exit(0)

    if not inputfile:
        print("[get_degradation_rates] Error: Missing required argument --input")
        sys.exit(1)
    if species not in ("auto", "human", "mouse"):
        print(f"[get_degradation_rates] Error: --species must be auto, human or mouse (got '{species}')")
        sys.exit(1)

    p = '{}/'.format(inputfile)

    # Load full dataset
    data_path = os.path.join(p, 'Data', 'data_full.h5ad')
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Full data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        print(f"[get_degradation_rates] Loaded full dataset from {data_path}")
        print(f"[get_degradation_rates] Dataset contains {adata.shape[0]} cells and {adata.shape[1]} genes")
    except FileNotFoundError as e:
        print(f"[get_degradation_rates] Error: {e}")
        print(f"[get_degradation_rates] Please ensure Data/data_full.h5ad exists in {p}")
        sys.exit(1)

    # Look up half-lives for genes in dataset
    try:
        _, details = extract_degradation_rates(adata.var_names, species=species, return_details=True)
    except Exception as e:
        print(f"[get_degradation_rates] Error extracting degradation rates: {e}")
        sys.exit(1)
    species = details["species"].iloc[0]
    print(f"[get_degradation_rates] Species: {species}")
    print(f"[get_degradation_rates]   mRNA reference: {REFERENCES[species]['mrna']}")
    print(f"[get_degradation_rates]   protein reference: {REFERENCES[species]['protein']}")
    print(f"[get_degradation_rates] Name matching: {details['match'].value_counts().to_dict()}")
    for quantity in ("mrna", "protein"):
        sources = details[f"{quantity}_source"].str.split(":").str[0].value_counts().to_dict()
        print(f"[get_degradation_rates] {quantity} half-life sources: {sources}")
    unresolved = details.loc[details["match"] == "unresolved", "gene"].tolist()
    if unresolved:
        print(f"[get_degradation_rates] {len(unresolved)} gene names not recognised as {species} genes "
              f"(global median used): {unresolved[:20]}{' ...' if len(unresolved) > 20 else ''}")

    assign_rates(adata, details, species, overwrite)

    report_path = os.path.join(p, 'Data', 'degradation_rates_report.csv')
    report = details.copy()
    report["d0_assigned"] = adata.var["d0"].to_numpy()
    report["d1_assigned"] = adata.var["d1"].to_numpy()
    report.to_csv(report_path, index=False)
    print(f"[get_degradation_rates] Per-gene report written to {report_path}")

    # Save updated full dataset
    try:
        adata.write(os.path.join(p, 'Data', 'data_full.h5ad'))
        print(f"[get_degradation_rates] Saved updated full dataset to {os.path.join(p, 'Data', 'data_full.h5ad')}")
    except Exception as e:
        print(f"[get_degradation_rates] Error saving full dataset: {e}")
        sys.exit(1)

    # Handle train/test splits if specified
    if split != "full":
        print(f"[get_degradation_rates] Processing train/test split: {split}")

        # Load train data
        train_path = os.path.join(p, 'Data', 'data_train.h5ad')
        test_path = os.path.join(p, 'Data', 'data_test.h5ad')

        try:
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Train data not found at {train_path}")
            adata_train = ad.read_h5ad(train_path)
            print(f"[get_degradation_rates] Loaded train dataset from {train_path}")
        except FileNotFoundError as e:
            print(f"[get_degradation_rates] Error: {e}")
            sys.exit(1)

        try:
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test data not found at {test_path}")
            adata_test = ad.read_h5ad(test_path)
            print(f"[get_degradation_rates] Loaded test dataset from {test_path}")
        except FileNotFoundError as e:
            print(f"[get_degradation_rates] Error: {e}")
            sys.exit(1)

        # Look up rates for train genes (same species as the full dataset)
        try:
            _, details_train = extract_degradation_rates(adata_train.var_names, species=species,
                                                         return_details=True)
        except Exception as e:
            print(f"[get_degradation_rates] Error extracting train degradation rates: {e}")
            sys.exit(1)
        assign_rates(adata_train, details_train, species, overwrite)

        # Copy rates to test data
        for col in ("d0", "d1", "d0_source", "d1_source"):
            if col in adata_train.var.columns:
                adata_test.var[col] = adata_train.var[col].values
        adata_test.uns["degradation_rates"] = adata_train.uns["degradation_rates"]

        # Save updated train/test datasets
        try:
            adata_train.write(os.path.join(p, 'Data', 'data_train.h5ad'))
            adata_test.write(os.path.join(p, 'Data', 'data_test.h5ad'))
            print(f"[get_degradation_rates] Saved updated train/test datasets")
        except Exception as e:
            print(f"[get_degradation_rates] Error saving train/test datasets: {e}")
            sys.exit(1)

    # Report final statistics
    mean_d1 = np.mean(adata.var['d1'].values)
    mean_d0 = np.mean(adata.var['d0'].values)
    print(f"[get_degradation_rates] Final statistics (h^-1):")
    print(f"[get_degradation_rates]   Mean protein degradation rate (d1): {mean_d1:.4f}")
    print(f"[get_degradation_rates]   Mean mRNA degradation rate (d0): {mean_d0:.4f}")
    print("[get_degradation_rates] Kinetic rates assignment completed successfully")

if __name__ == "__main__":
   main(sys.argv[1:])
