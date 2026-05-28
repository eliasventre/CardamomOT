# Semrau et al. (2017) — Mouse ESC differentiation

> **Dataset:** Semrau S. et al., *Identifying cellular trajectories with SCORPIUS*, Nat. Methods 2017.  
> **Organism / tissue:** Mouse embryonic stem cells (ESC) differentiating toward epiblast (Epi) and primitive endoderm (PE).  
> **Time points:** 5 consecutive days (days 0–4).  
> **Genes analysed:** ~25 selected markers of the Epi/PE decision.

## Biological context

This dataset captures the earliest bifurcation in the mouse blastocyst inner cell mass: from naive ESCs at day 0, cells commit either to the epiblast or to the primitive endoderm fate by day 4. The small number of genes and cells, combined with clear cell-type labels, makes this dataset a well-controlled benchmark for GRN inference.

## CardamomOT configuration

```bash
cardamomot pipeline \
    -i experimental_datasets/Semrau \
    -s full \
    -c 1 \
    -r 0.6 \
    -m 0.5 \
    --stimulus 1.0 \
    --prior 1.0
```

Key choices:
- `-c 1` activates differential gene selection, keeping genes that change significantly across time.
- `--prior 1.0` uses full prior-knowledge weighting for the network structure.

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

Load and compare with wild-type simulation:

```python
import anndata as ad
import matplotlib.pyplot as plt

wt  = ad.read_h5ad("cardamomOT/adata_prot_simul_stim1.0_prior1.0.h5ad")
ko  = ad.read_h5ad("cardamomOT/adata_prot_simul_KO_none_OV_Zfp42_stim1.0_prior1.0.h5ad")

from CardamomOT import compare_marginals
compare_marginals(wt, ko, genes=["Zfp42", "Nanog", "Gata6"])
plt.show()
```

## Notebook

An interactive notebook walking through this analysis is available in the repository at `experimental_datasets/Semrau/pipeline.ipynb`.
