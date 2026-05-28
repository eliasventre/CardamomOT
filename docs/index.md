# CardamomOT

**CardamomOT** is a Python package for joint gene regulatory network (GRN) and cell trajectory inference from time-course single-cell RNA-seq data. It calibrates the parameters of a mechanistic model of gene expression — the calibrated network can then be simulated to reproduce and extrapolate the observed data.

::::{grid} 2
:::{grid-item-card} 🚀 Installation
:link: installation
:link-type: doc
Get up and running in minutes.
:::
:::{grid-item-card} ⚡ Quick Start
:link: quickstart
:link-type: doc
Run your first CardamomOT analysis.
:::
:::{grid-item-card} 📚 API Reference
:link: api
:link-type: doc
Explore the full Python API.
:::
:::{grid-item-card} 🔬 Worked Examples
:link: examples/index
:link-type: doc
Three real-data applications.
:::
::::

## Overview

CardamomOT combines optimal transport and mechanistic modelling to jointly infer:

- **Kinetic parameters** (burst frequencies, degradation rates) from the marginal distributions at each time point
- **Gene regulatory interactions** using an optimal-transport trajectory that aligns consecutive snapshots
- **Stochastic simulations** of the inferred network for in-silico perturbation experiments

The package is described in:

> Maugé Y. and Ventre E. (2026). *CardamomOT: a mechanistic optimal transport-based framework for gene regulatory network inference, trajectory reconstruction and generative modeling*. doi: [10.64898/2026.03.31.715390](https://doi.org/10.64898/2026.03.31.715390)

```{toctree}
:maxdepth: 2
:hidden:
:caption: User guide

installation
quickstart
advanced
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Worked examples

examples/index
examples/semrau
examples/kameneva
examples/schiebinger
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Reference

api
autoapi/CardamomOT/index
```
