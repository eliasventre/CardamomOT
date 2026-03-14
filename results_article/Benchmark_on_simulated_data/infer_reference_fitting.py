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

# Number of runs
N = 1
# Print information
verb = 1
# number of initial couplings
n_repet=4

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

def infer(adata, save_path, timepoints, replicate_number):
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
    print(grn.shape)
    np.save(save_path+f'score_{replicate_number}', grn)

    pred_data = future_pred(estim_alt_wt, 0, 0, len(timepoints))
    pred_array = np.array([pred_data[i].cpu().numpy() for i in range(len(pred_data))])
    n_cells_per_time = len(adata[adata.obs[t_key]==timepoints[-1]])
    pred_array = np.reshape(pred_array, newshape = (len(pred_data)*n_cells_per_time, adata.shape[1]))
    list_times_pred = []
    for i in range(len(pred_data)):
        list_times_pred+= [i] * n_cells_per_time ## To check ; need to correspond to timepoints
    pred_adata = ad.AnnData(pred_array)
    pred_adata.obs['time'] = list_times_pred
    pred_adata.obs[t_key] = list_times_pred
    pred_adata.var_names = adata.var_names
    runtime = time.time() - start_runtime
    pred_adata.uns['time'] = runtime
    np.savetxt(f'{save_path}runtime.txt', [runtime])
    pred_adata.write_h5ad(save_path+f'pred_adata_{replicate_number}.h5ad')
    

# Inferences
Networks = ['FN4', 'CN5', 'BN8', 'FN8', 'Trees5', 'Trees10', 'Trees20', 'Trees50', 'Trees100']
for net in Networks:
    for r in range(0, N):
        print(net)
        fname = f'{net}/Data/data_{r + 1}.txt'
        raw_matrix = np.loadtxt(fname, delimiter='\t').astype(np.int64)
        timepoints = raw_matrix[0, 1:]
        data_rna = raw_matrix[1:, 1:].T 
        genes_names = raw_matrix[:, 0]
        adata = ad.AnnData(X=data_rna)
        adata.var_names = genes_names[1:].astype(str)
        adata.obs['time'] = timepoints.astype(int)
        adata.obs['timepoint'] = timepoints.astype(int)
        t0 = timer.time()
        infer(adata, f'{net}/REFERENCE_FITTING/', np.unique(adata.obs['timepoint']), r+1)
        t1 = timer.time()
        print(t1 - t0)