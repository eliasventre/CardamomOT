from _scripts.genie3 import genie3
from scipy import stats
import os
num_threads = "8"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
import time as timer
import numpy as np
import scanpy as sc
import anndata as ad
import scipy
import os
import sys, getopt
import torch
import time
sys.path.append("./_scripts/") 
import rf

# Code adapted from https://github.com/zsteve/referencefitting/tree/main

def future_pred(estimator, adata, T_start, n_timepoints):
    Xs_copy = estimator.Xs[adata].copy()[T_start]
    Xs_pred = [Xs_copy]
    A = estimator.A * estimator.Ms[adata]
    b = estimator.b * estimator.Ms[adata][0, :]
    t = 1 /estimator.T
    P = torch.linalg.matrix_exp(t*A)
    for n in range(n_timepoints):
        future_pred = torch.relu(((Xs_pred[n] / estimator.std) @ P + t * b) * estimator.std)
        Xs_pred.append(future_pred)
    return Xs_pred

def infer(adata, save_path, timepoints):
    #options
    t_key = 'timepoint'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    

    start_runtime = time.time()
    print("Starting reference fitting model training")

    options = {
        "lr" : 0.05, 
        "reg_sinkhorn" : 0.1,
        "reg_A" : 1, 
        "reg_A_elastic" : 0.5, 
        "iter" : 1000,
        "ot_coupling" : True,
        "optimizer" : torch.optim.Adam,
        "n_pca_components" : -1
    }

    estim_alt_wt = rf.Estimator([adata], kos = [None],
            lr = options["lr"],
            reg_sinkhorn = options["reg_sinkhorn"], 
            reg_A = options["reg_A"], 
            reg_A_elastic = options["reg_A_elastic"], 
            iter = options["iter"], 
            ot_coupling = options["ot_coupling"],
            optimizer = options["optimizer"],
            norm = False,
            t_key = t_key)
    
    estim_alt_wt.fit(print_iter=10, alg = "alternating", update_couplings_iter=250)

    t = 1/estim_alt_wt.T
    #P = torch.linalg.matrix_exp(t*estim_alt_wt.A)
    #grn = P.cpu().numpy()
    grn = estim_alt_wt.A.cpu().numpy()
    print(grn.mean())
    np.save(save_path+f'inter', grn)

### Inference for the three methods

datasets = ['Semrau', 'Kameneva', 'Schiebinger']
split = ['full', 'full', 'train']

for i, dataset in enumerate(datasets):

    p = f'./../../experimental_datasets/{dataset}'

    data_path = os.path.join(p, 'Data', 'data_{}.h5ad'.format(split[i]))
    if os.path.exists(data_path):
        adata = ad.read_h5ad(data_path)
    else:
        error_msg = (
            f"Error: Data file not found at {data_path}.\n"
            f"Ensure you have created a 'data_{split[i]}.h5ad' file in the Data/ directory."
        )
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    if scipy.sparse.issparse(adata.X):
        data_rna_extracted = adata.X.T.toarray()
    else:
        data_rna_extracted = adata.X.T

    # ─── CHECK TEMPORAL INFORMATION ──────────────────────────────────────
    try:
        times = adata.obs['time'].values 
        if len(np.unique(times)) <= 1:
            raise ValueError(
                "Data must contain temporal information with at least 2 distinct timepoints."
            )
    except KeyError as e:
        error_msg = f"Error: 'time' column not found in adata.obs. {e}"
        print(error_msg)
        raise SystemExit(error_msg)
    data_rna = np.vstack([times, data_rna_extracted]).T

    x = data_rna[:, :]
    x[times > 0, 0] = 100
    x[times == 0, 0] = 0

    for i in range(x.shape[0]):
        if np.sum(x[i, :]) < 1:
            x[i, np.random.choice(x.shape[1])] = 1
    for j in range(x.shape[1]):
        if np.sum(x[:, j]) < 1:
            x[np.random.choice(x.shape[0]), j] = 1
    
    print(x.shape)

    # adata_rna = ad.AnnData(X=x)
    # genes_list = ['Stimulus'] + list(adata.var_names.values)
    # print(len(genes_list))
    # adata_rna.var_names = genes_list[:]
    # adata_rna.obs['time'] = times.astype(int)
    # adata_rna.obs['timepoint'] = times.astype(int)
    # infer(adata_rna, p + '/RF/', np.unique(adata_rna.obs['timepoint']))

    # score = genie3(x)
    # np.save(p + '/GENIE3/inter', score)

    G = np.size(x, 1)
    score = np.zeros((G, G))
    for i in range(0, G):
        for j in range(0, G):
            score[i, j] = stats.pearsonr(x[:, i], x[:, j])[0]
    np.save(p + '/PEARSON/inter', score)
