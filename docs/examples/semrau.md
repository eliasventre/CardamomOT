# Semrau et al. (2017) — Mouse ESC differentiation

> **Dataset:** Semrau S, Goldmann JE, Soumillon M, Mikkelsen TS, Jaenisch R, and van Oudenaarden A. *Dynamics of lineage commitment revealed by single-cell transcriptomics of differentiating embryonic stem cells.* Nat Commun 2017; 8:1–16.  
> **Organism / tissue:** Mouse embryonic stem cells (ESC) differentiating toward epiblast (Epi) and primitive endoderm (PE).  
> **Time points:** 5 consecutive days (days 0–4).  
> **Genes analysed:** Selected markers of the Epi/PE decision.

## Biological context

This dataset captures the earliest bifurcation in the mouse blastocyst inner cell mass: from naive ESCs at day 0, cells commit either to the epiblast or to the primitive endoderm fate by day 4. The small number of genes and cells, combined with clear cell-type labels, makes this dataset a well-controlled benchmark for GRN inference.

## CardamomOT configuration

```bash
cardamomot pipeline \
    -i experimental_datasets/Semrau \
    -s full \
    -r 0.7 \
    -c 0 \
    --mean-forcing 1.0 \
    --stimulus 1.0 \
    --prior 1.0 \
    --force-basins 1.0 \
    --temporal-basins 1

```

Key choices:
- `-m 1.0` sets the mean expression threshold to 1 (appropriate for this low-gene-count dataset).
- `--force-basins 1.0 --temporal-basins 1` enforces temporal consistency of NB modes across time points.
- `--prior 1.0` uses full prior-network weighting for the network constraint.
- KO/OV perturbation steps run by default (`--no-kov` would disable them).

## Results

### Inferred kinetic parameters

CardamomOT fits a negative-binomial burst model to the marginal gene-expression distributions at each time point, yielding per-gene burst frequencies (κ_on, κ_off) and degradation rates (δ).

### Inferred network

The inferred regulatory network recovers known interactions in the Epi/PE decision, including activation of *Gata6* and *Pdgfra* in the PE branch and maintenance of *Nanog* and *Sox2* in the Epi branch.

### Simulated trajectories

Stochastic simulations of the inferred network reproduce the bimodal distributions observed at intermediate time points and the two separated clusters at day 4.

### In-silico knock-outs

CardamomOT supports virtual knock-out (KO) experiments by setting one or more gene's basal activity to zero in silico and re-simulating the network. Results for single and double KOs of key Epi/PE regulators are stored in:

```
experimental_datasets/Semrau/cardamomOT/adata_prot_simul_KO_none_OV_<gene>_*.h5ad
```

Wild-type and KO outputs can be compared using the `utils/plot_data_to_sim_KOV.ipynb` notebook,
which calls `plot_results_rna_clean` for each perturbation condition.

## Post-analysis notebooks

The following notebooks in `utils/` apply to this dataset:

| Notebook | Content |
|---|---|
| `plot_networks.ipynb` | Inferred GRN with edge thresholding |
| `plot_data_to_sim.ipynb` | UMAP and marginal comparison between data and simulation |
| `plot_data_to_sim_KOV.ipynb` | Wild-type vs KO/OV simulations (`KO_none_OV_Zfp42`, `KO_none_OV_Dnmt3a`, …) |
| `compare_cell_types.ipynb` | Cell-type proportions in data vs simulation |
