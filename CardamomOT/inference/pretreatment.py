"""
Core functions for selecting the most variable genes according to the
Zero‑Inflated Negative Binomial (ZiNB) model.
"""
from typing import Any
import pandas as pd
import numpy as np
from difflib import get_close_matches
import anndata as ad
import logging

from CardamomOT.logging import get_logger

# module logger
logger = get_logger(__name__)

def ln2(x):
    return np.log(2) / x if x > 0 else np.nan

def extract_degradation_rates(df, gene_list, cell_line=None, similarity_threshold=np.linspace(.99, 0.01, 10)):
    df = df.dropna(subset=["gene_symbol"])   
    
    if cell_line:
        df = df[df["cell_line"].str.lower() == cell_line.lower()]
    
    known_genes = df["gene_symbol"].unique()
    
    deg = np.zeros((2, len(gene_list)))
    mean_ratio = df["prot_half_life"].astype(float).mean(skipna=True) / df["rna_half_life"].astype(float).mean(skipna=True)

    for cnt, gene in enumerate(gene_list):
        gene_len: int = len(gene)

        for pct in range(100, -1, -10):  # from 100% to 0% in steps of 10%
            min_len = int(gene_len * pct / 100)
            prefix = gene[:min_len]

            similar_genes = [g for g in known_genes if g.startswith(prefix)]

            if similar_genes:
                sim_matches = df[df["gene_symbol"].isin(similar_genes)]
                prot_half_life = sim_matches["prot_half_life"].astype(float).mean(skipna=True)
                rna_half_life = sim_matches["rna_half_life"].astype(float).mean(skipna=True)
                break

        if np.isnan(prot_half_life):
            prot_half_life = rna_half_life * mean_ratio
        
        deg[0, cnt] = np.log(2)/rna_half_life
        deg[1, cnt] = np.log(2)/prot_half_life

    return deg


def select_DEgenes(data_rna, vect_samples_id, vect_celltype_id, proba,  
                   list_genes, n_genes_tokeep_temporal=[1000], n_genes_tokeep_celltype=[1000], 
                   limit_min=.01, verb=0):

    G: int = len(list_genes)
    vect_t = data_rna[:, 0]
    times_full = np.sort(np.unique(vect_t))
    times = times_full[:-1]
    samples_id = np.unique(vect_samples_id)
    celltype_id = np.unique(vect_celltype_id)

    if len(n_genes_tokeep_temporal) < len(times):
        n_genes_tokeep_temporal = np.ones(len(times), dtype=int) * int(np.mean(n_genes_tokeep_temporal))
    if len(n_genes_tokeep_celltype) < len(celltype_id):
        n_genes_tokeep_celltype = np.ones(len(celltype_id), dtype=int) * int(np.mean(n_genes_tokeep_celltype))


    proba_class = np.argmax(proba[:, 1:], axis=-1)
    temporal_variations = np.zeros((G, len(times), len(samples_id)))
    celltype_variations = np.zeros((G, len(celltype_id), len(samples_id))) if len(celltype_id) > 1 else None

    ### ----- TEMPORAL VARIATIONS -----
    selection_info = {g: [] for g in range(G)}
    for s_i, s in enumerate(samples_id):
        for t_i, t in enumerate(times):
            idx_init = (vect_t == t) & (vect_samples_id == s)
            idx_end  = (vect_t == times_full[t_i+1]) & (vect_samples_id == s)
            if np.sum(idx_init) == 0 or np.sum(idx_end) == 0:
                continue
            for g in range(G):
                diff = np.sum([abs(
                    np.mean(proba_class[idx_end, g] == i) -
                    np.mean(proba_class[idx_init, g] == i)
                ) for i in range(proba.shape[-1])])
                temporal_variations[g, t_i, s_i] = diff

    list_genes_tokeep_temporal = []
    for t_i, t in enumerate(times):
        variations_max = temporal_variations[:, t_i, :].sum(axis=1)
        ranked_idx: np.ndarray[Any, np.dtype[np.signedinteger[Any]]] = np.argsort(variations_max)[::-1]
        top_genes: np.ndarray[Any, np.dtype[np.signedinteger[Any]]] = ranked_idx[:n_genes_tokeep_temporal[t_i]]
        for rank, g in enumerate(top_genes, start=1):
            if variations_max[g] >= limit_min:
                list_genes_tokeep_temporal.append(g)
                selection_info[g].append(f"temporal - {t} - {rank}")
        if verb:
            logger.info("[Temporal] t=%s → %s genes kept", t, len(top_genes))

    ### ----- CELLTYPE VARIATIONS -----
    list_genes_tokeep_celltype = []
    if len(celltype_id) > 1:
        for s_i, s in enumerate(samples_id):
            for c_i, c in enumerate(celltype_id):
                idx_init = (vect_celltype_id == c) & (vect_samples_id == s)
                idx_end  = (vect_celltype_id != c) & (vect_samples_id == s)
                if np.sum(idx_init) == 0 or np.sum(idx_end) == 0:
                    continue
                for g in range(G):
                    diff = np.sum([abs(
                            np.mean(proba_class[idx_end, g] == i) -
                            np.mean(proba_class[idx_init, g] == i)
                        ) for i in range(proba.shape[-1])])
                    celltype_variations[g, c_i, s_i] = diff

        for c_i, c in enumerate(celltype_id):
            variations_max = celltype_variations[:, c_i, :].sum(axis=1)
            ranked_idx: np.ndarray[Any, np.dtype[np.signedinteger[Any]]] = np.argsort(variations_max)[::-1]
            top_genes: np.ndarray[Any, np.dtype[np.signedinteger[Any]]] = ranked_idx[:n_genes_tokeep_celltype[c_i]]
            for rank, g in enumerate(top_genes, start=1):
                if variations_max[g] >= limit_min:
                    list_genes_tokeep_celltype.append(g)
                    selection_info[g].append(f"celltype - {c} - {rank}")
            if verb:
                logger.info("[Celltype] %s → %s genes kept", c, len(top_genes))

    ### ----- FINAL SELECTION -----
    indices_to_keep = sorted(set(list_genes_tokeep_temporal + list_genes_tokeep_celltype))
    genes_to_keep = [list_genes[i] for i in indices_to_keep]

    ### ----- REPORT BUILDING -----
    # mean of variations across samples
    mean_temp = np.mean(temporal_variations, axis=2)  # (G, len(times))
    mean_cell: Any | None = np.mean(celltype_variations, axis=2) if celltype_variations is not None else None

    report_rows = []
    for g in indices_to_keep:
        gene_name = list_genes[g]
        select_text: str = " / ".join(selection_info[g])

        row = {"gene": gene_name, "selection_summary": select_text}

        for s_i, s in enumerate(samples_id):
            # Add temporal columns
            for t_i, t in enumerate(sorted(times, key=str)):
                row[f"sample_{s}-temporal_{t}"] = temporal_variations[g, t_i, s_i]

            # Add celltype columns
            if celltype_variations is not None:
                for c_i, c in enumerate(sorted(celltype_id, key=str)):
                    row[f"sample_{s}-celltype_{c}"] = celltype_variations[g, c_i, s_i]

        report_rows.append(row)

    df_report = pd.DataFrame(report_rows)
    df_report: pd.DataFrame = df_report.sort_values("gene").reset_index(drop=True)

    # sums for output; celltype may be None when only one cell type present
    temporal_sum = temporal_variations.sum(axis=(1, 2))
    if celltype_variations is not None:
        celltype_sum = celltype_variations.sum(axis=(1, 2))
    else:
        celltype_sum = np.zeros(G)

    return genes_to_keep, temporal_sum, celltype_sum, df_report
