"""
infer_rd.py
-----------
Estimate cellular read depth factors (r_i) using the Chronocell method.

Implements the read depth inference method from Fang, Gorin & Pachter (Chronocell, 2024).
Estimates normalization factors for each cell based on Poisson gene selection.

The method iteratively selects genes with Poisson-like variance characteristics
and uses their mean expression to compute cell-specific read depth factors.
Factors are stored in adata.obs['rd'] for downstream normalization.

Usage:
    python infer_rd.py -i <project_path> [--var_threshold 1.2] [--min_mean 0.01] [--quiet]

Required input files:
    - Data/data.h5ad: count matrix with temporal information

Output files:
    - Data/data.h5ad: updated with rd (read depth) in obs and is_poissonian_rd in var

Reference:
    Fang M, Gorin G, Pachter L. "Trajectory inference from single-cell genomics
    data with a process time model." bioRxiv 2024.
    https://github.com/pachterlab/FGP_2024
"""

import sys
import os
import getopt
import numpy as np
import anndata as ad
import scipy.sparse
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Estimation du CV² et des gènes Poissoniens
# ─────────────────────────────────────────────────────────────────────────────

def estimate_cv2_read_depth(X, min_mean=0.01):
    """
    Estimate sequencing read depth coefficient of variation (CV²).

    Computes the normalized covariance matrix averaged over gene pairs.
    Implementation follows Chronocell (Fang, Gorin & Pachter 2024, Cell 21-22):

        X_rho = Cov(X) / mean(X)[:,None] / mean(X)[None,:]
        xi = (sum(X_rho) - sum(diag(X_rho))) / (p*(p-1))

    Represents the average normalized covariance of off-diagonal pairs.

    Parameters
    ----------
    X        : (N, G) array of raw counts
    min_mean : float, minimum gene mean threshold for inclusion

    Returns
    -------
    xi : float, estimated CV² of read depth
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)                          # (G,)
    valid = mu > min_mean
    X_v  = X[:, valid]
    mu_v = mu[valid]
    p    = X_v.shape[1]

    if p < 2:
        return 0.0

    # Matrice de covariance  (np.cov : rowvar=False -> genes en colonnes)
    X_cov = np.cov(X_v, rowvar=False)            # (p, p)

    # Covariance normalisee : rho[a,b] = Cov(a,b) / mean(a) / mean(b)
    X_rho = X_cov / mu_v[:, None] / mu_v[None, :]

    # Moyenne hors-diagonale (exactement comme Chronocell)
    xi = (np.sum(X_rho) - np.sum(np.diag(X_rho))) / (p * (p - 1))
    xi = max(float(xi), 0.0)
    return xi


def select_poissonian_genes(X, xi, var_threshold=1.2, min_mean=0.1):
    """
    Select Poisson-like genes and iteratively refine CV² estimate.

    Implements iterative Poisson gene selection from Chronocell (Cell 22):

        for i in range(5):
            Pgene_mask = (X_mean > 0.01) & (X_var / (a*X_mean + sp*X_mean²) < 1.2)
            sp = estimate_cv2_read_depth(X[:, Pgene_mask])

    where a=1 (intrinsic Poisson term).

    Parameters
    ----------
    X             : (N, G) array of raw counts
    xi            : float, initial CV² estimate
    var_threshold : float, variance threshold (default 1.2 from Chronocell)
    min_mean      : float, minimum gene mean threshold (default 0.01)

    Returns
    -------
    poisson_mask : (G,) bool array indicating Poisson genes
    xi_final     : float, converged CV² estimate on Poisson subset
    """
    X   = np.asarray(X, dtype=float)
    mu  = X.mean(axis=0)
    var = X.var(axis=0)
    a   = 1.0     # intrinsic Poisson term
    sp  = xi

    poisson_mask = mu > min_mean   # initialize broadly

    for _ in range(10):             # up to 5 iterations 
        # criterion: var / (a*mu + sp*mu^2) < var_threshold
        denom        = a * mu + sp * mu ** 2
        poisson_mask = (mu > min_mean) & (var / np.where(denom > 0, denom, np.inf) < var_threshold)

        if poisson_mask.sum() < 5:
            poisson_mask = mu > min_mean
            break

        sp = estimate_cv2_read_depth(X[:, poisson_mask], min_mean=min_mean)

    return poisson_mask, sp


def estimate_read_depth(X, var_threshold=1.2, min_mean=0.01, normalize=True, verb=True):
    """
    Complete pipeline for estimating cellular read depth factors.

    Implements Chronocell algorithm (Fang, Gorin & Pachter 2024, Cells 21-24):

        1. Estimate CV² on all genes
        2. Iteratively select Poisson-like genes (5 iterations)
        3. Compute read depth = cell mean / global mean on Poisson genes

    Parameters
    ----------
    X             : (N, G) array of raw counts
    var_threshold : float, variance threshold (default 1.2 from Chronocell)
    min_mean      : float, minimum gene mean threshold (default 0.01)
    normalize     : bool, if True divide by mean to ensure rd mean = 1
    verb          : bool, verbosity flag

    Returns
    -------
    r            : (N,) array of cell read depth factors (mean = 1)
    xi           : float, final CV²(r) estimate
    poisson_mask : (G,) bool array
    """
    X = np.asarray(X, dtype=float)
    N, G = X.shape

    # Etape 1 : CV² initial sur tous les genes
    xi_init = estimate_cv2_read_depth(X, min_mean=min_mean)
    if verb:
        print(f"  [rd] CV2(r) initial (tous genes) : {xi_init:.4f}")

    # Etape 2 : genes Poissoniens (5 iterations comme Chronocell)
    poisson_mask, xi_final = select_poissonian_genes(
        X, xi_init, var_threshold=var_threshold, min_mean=min_mean
    )
    n_poisson = poisson_mask.sum()
    if verb:
        print(f"  [rd] Genes Poissoniens selectionnes : {n_poisson}/{G}")
        print(f"  [rd] CV2(r) final (genes Poissoniens) : {xi_final:.4f}")

    if n_poisson < 100:
        if verb:
            print("  [rd] Aucun gene Poissonien -> fallback sur lib size")
        X_p = X   # fallback : tous les genes
    else:
        X_p = X[:, poisson_mask]

    # Etape 3 : rd = mean par cellule / mean globale  
    # Compute cell-specific read depth factors
    X_p = X[:, poisson_mask]
    global_mean = X_p.mean()
    if global_mean == 0:
        r = np.ones(N, dtype=float)
    else:
        r = X_p.mean(axis=1) / global_mean   # mean = 1 by construction

    if verb:
        print(f"  [infer_rd] Read depth statistics:")
        print(f"  [infer_rd]   min={r.min():.3f}  mean={r.mean():.3f}  "
              f"median={np.median(r):.3f}  max={r.max():.3f}  "
              f"CV²={np.var(r)/np.mean(r)**2:.4f}")

    return r, xi_final, poisson_mask


def main(argv):
    """
    Estimate and save cellular read depth factors.

    Loads count matrix, estimates Poisson genes, computes cell-specific
    read depth normalizing factors, and saves them to AnnData object.

    Args:
        argv: Command-line arguments (--input, --var_threshold, --min_mean, --quiet).
    
    Returns:
        None. Updates Data/data.h5ad with read depth annotations.
    """
    inputfile = ''
    var_threshold = 1.2
    min_mean = 0.1
    verb = True

    try:
        opts, args = getopt.getopt(
            argv, "hi:",
            ["input=", "var_threshold=", "min_mean=", "quiet"]
        )
    except getopt.GetoptError:
        print("[infer_rd] Error: Invalid command-line arguments")
        print("[infer_rd] Usage: python infer_rd.py -i <project_path> [--var_threshold 1.2] [--min_mean 0.01] [--quiet]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("--var_threshold",):
            var_threshold = float(arg)
        elif opt in ("--min_mean",):
            min_mean = float(arg)
        elif opt in ("--quiet",):
            verb = False
        elif opt == "-h":
            print(__doc__)
            sys.exit(0)

    if not inputfile:
        print("[infer_rd] Error: Missing required argument --input")
        sys.exit(1)

    p = f"{inputfile}/"
    data_path = os.path.join(p, "Data", "data.h5ad")

    # Load count matrix
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        adata = ad.read_h5ad(data_path)
        if verb:
            print(f"[infer_rd] Loaded data from {data_path}")
    except FileNotFoundError as e:
        print(f"[infer_rd] Error: {e}")
        print(f"[infer_rd] Please ensure Data/data.h5ad exists in {p}")
        sys.exit(1)
    except Exception as e:
        print(f"[infer_rd] Error loading data: {e}")
        sys.exit(1)

    # Extract count matrix
    try:
        if scipy.sparse.issparse(adata.X):
            X_full = adata.X.toarray().astype(float)
        else:
            X_full = np.asarray(adata.X, dtype=float)
    except Exception as e:
        print(f"[infer_rd] Error extracting count matrix: {e}")
        sys.exit(1)

    N_cells, G_total = X_full.shape
    if verb:
        print(f"[infer_rd] Dataset contains {N_cells} cells and {G_total} genes")

    # Validate temporal information
    try:
        if "time" not in adata.obs.columns:
            raise ValueError("Dataset must contain 'time' column in obs")
        times = adata.obs['time'].values
        if len(np.unique(times)) <= 1:
            raise ValueError("Dataset must contain multiple timepoints")
        if verb:
            print(f"[infer_rd] Found {len(np.unique(times))} unique timepoints")
    except (KeyError, ValueError) as e:
        print(f"[infer_rd] Error: {e}")
        sys.exit(1)

    # Estimate read depth factors
    try:
        print(f"[infer_rd] Estimating read depth with var_threshold={var_threshold}, min_mean={min_mean}")
        r, xi, poisson_mask = estimate_read_depth(
            X_full,
            var_threshold=var_threshold,
            min_mean=min_mean,
            normalize=True,
            verb=verb,
        )
        print(f"[infer_rd] Selected {poisson_mask.sum()} Poisson genes out of {G_total}")
    except Exception as e:
        print(f"[infer_rd] Error estimating read depth: {e}")
        sys.exit(1)

    # Clip and normalize
    r = np.clip(r, 0.2, 5)
    r /= r.mean()
    if verb:
        print(f"[infer_rd] Read depth factors clipped to [0.2, 5] and normalized to mean=1")

    # Store in AnnData object
    try:
        adata.obs["rd"] = r.astype(np.float32)
        adata.obs["rd_cv2"] = float(xi)
        
        # Store which genes were used
        poisson_col = np.zeros(G_total, dtype=bool)
        poisson_col[:len(poisson_mask)] = poisson_mask
        adata.var["is_poissonian_rd"] = poisson_col
        
        if verb:
            print(f"[infer_rd] Stored read depth factors in adata.obs['rd']")
            print(f"[infer_rd] Stored Poisson gene mask in adata.var['is_poissonian_rd']")
    except Exception as e:
        print(f"[infer_rd] Error storing annotations: {e}")
        sys.exit(1)

    # Save updated AnnData
    try:
        adata.write_h5ad(data_path)
        print(f"[infer_rd] Saved updated dataset to {data_path}")
        print("[infer_rd] Read depth estimation completed successfully")
    except Exception as e:
        print(f"[infer_rd] Error saving dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])