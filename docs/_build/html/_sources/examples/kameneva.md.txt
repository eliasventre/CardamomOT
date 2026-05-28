# Kameneva et al. (2021) — Human sympathoadrenal development

> **Dataset:** Kameneva P. et al., *Single-cell transcriptomics of human embryos identifies multiple sympathoblast lineages with potential implications for neuroblastoma origin*, Nat. Genet. 2021.  
> **Organism / tissue:** Human embryonic adrenal gland (sympathoadrenal progenitors).  
> **Time points:** 4 developmental stages (Carnegie stages 15–21).  
> **Genes analysed:** ~30 selected markers of neural crest and sympathoadrenal commitment.

## Biological context

The sympathoadrenal lineage arises from neural crest cells that migrate to the adrenal gland and differentiate into sympathoblasts, chromaffin cells, and Schwann cell precursors. Aberrant differentiation in this lineage is implicated in neuroblastoma. This dataset challenges CardamomOT with multi-branch topology and a relatively small number of time points.

## CardamomOT configuration

```bash
cardamomot pipeline \
    -i experimental_datasets/Kameneva \
    -s full \
    -c 0 \
    -r 0.7 \
    --mean-forcing 0.5 \
    --stimulus 1.0 \
    --prior 1.0 \
    --force-basins 1.0 \
    --temporal-basins 1
```

Multiple `--stimulus` values (0.2, 0.3, 0.5, 1.0) were explored to assess sensitivity to the stimulus-edge penalisation, as reflected in the pre-computed outputs.

## Results

### Inferred network

The regulatory network captures the transition from multipotent progenitors (*SOX10*+, *TFAP2A*+) to committed sympathoblasts (*PHOX2B*+, *HAND2*+) and chromaffin precursors (*CHGA*+, *STMN2*+).

### In-silico perturbations

KO experiments targeting *CHGA* with over-expression of *STMN2* test the role of chromaffin-to-sympathoblast reversibility. Pre-computed results for several `--stimulus` values are stored in `experimental_datasets/Kameneva/cardamomOT/` and visualised with `utils/plot_data_to_sim_KOV.ipynb`.

## Sensitivity to the stimulus parameter

A sweep over `--stimulus` values 0.2, 0.3, 0.5 and 1.0 demonstrates that the core regulatory topology is robust while precise edge weights vary. The pre-computed outputs for each value are available in `experimental_datasets/Kameneva/cardamomOT/`.

## Post-analysis notebooks

| Notebook | Content |
|---|---|
| `plot_networks.ipynb` | Inferred GRN across stimulus values |
| `plot_data_to_sim.ipynb` | Data vs simulation comparison (UMAP, marginals) |
| `plot_data_to_sim_KOV.ipynb` | `KO_CHGA_OV_none`, `KO_none_OV_STMN2`, and combined perturbations |
| `compare_cell_types.ipynb` | Sympathoblast/chromaffin proportions in data vs simulation |
