# CARDAMOM

[![Documentation](https://readthedocs.org/projects/cardamomot/badge/?version=latest)](https://cardamomot.readthedocs.io/en/latest/)

**Documentation:** [https://cardamomot.readthedocs.io/en/latest/](https://cardamomot.readthedocs.io/en/latest/)

> **Version of the article (PLOS Computational Biology).** The code of the published article ([[3](#Mauge2026)])
> is the tag [`plos-cb-2026`](https://github.com/eliasventre/CardamomOT/tree/plos-cb-2026); the inference may have
> changed slightly in later versions. To use it:
>
> ```bash
> git clone https://github.com/eliasventre/CardamomOT.git
> cd CardamomOT
> git checkout plos-cb-2026
> ```
>
> or download it as a zip from its [GitHub page](https://github.com/eliasventre/CardamomOT/tree/plos-cb-2026) (Code → Download ZIP).

CARDAMOM is an executable gene regulatory network (GRN) inference method, adapted to time-course scRNA-seq datasets. The algorithm consists in calibrating the parameters of a mechanistic model of gene expression: the calibrated model can then be simulated, which allows to reproduce the dataset used for inference. The first inference method has been introduced in [[1](#Ventre2021)]. It has been benchmarked along with other GRN inference tools and applied to a real dataset in [[2](#Ventre2023)]. The second version is presented in [[3](#Mauge2026)] and combines GRN and trajectory inference method and shows a strong improvement over the first version.  The simulation part is based on the [Harissa](https://github.com/ulysseherbach/harissa) package.

## 🚀 Quick Start

### 1. Installation

#### Create a virtual environment (recommended)

```bash
# Create a new conda environment
conda create -n cardamom_env python=3.12 -y
conda activate cardamom_env

# IMPORTANT on macOS Apple Silicon (arm64): install numba + threading runtimes from conda-forge
# This avoids pip wheels without TBB backend support.
conda install -c conda-forge numba llvmlite llvm-openmp tbb tbb-devel -y

# OR with venv (standard Python)
python -m venv cardamom_env
source cardamom_env/bin/activate  # Linux/Mac
# cardamom_env\Scripts\activate   # Windows
```

#### Install CARDAMOM

```bash
# Clone the repository
git clone https://github.com/eliasventre/CardamomOT.git
cd CardamomOT

# Install the package in development mode
pip install -e .

# For development (tests, linting):
pip install -e ".[dev]"

# For Jupyter notebooks:
pip install -e ".[notebooks]"

# Note: omnipath (literature prior of the gene selection and build_reference_network) is included in the default install.
```

### 2. Prepare your data

CARDAMOM requires a project directory containing your scRNA-seq data. Create a folder `my_project/` with:

```
my_project/
└── Data/
    └── data.h5ad          # Your count matrix (required)
    └── gene_list.txt      # Gene list (optional)
```

**Required format for `data.h5ad`:**
- Gene counts (rows = genes, columns = cells)
- `data.obs['time']`: measurement time for each cell (if absent or unique, the data are treated as stationary: gene selection and mixture run on cell types only, then the pipeline stops — network inference will be handled by CardamomOT-stat, in prep.)
- `data.obs['cell_type']`: cell types (optional)
- `data.obs['cell_type_selection']`: optional override of `cell_type` for DE gene selection
- `data.obs['cell_type_proliferation']` / `data.obs['cell_type_transition']`: groupings matching the rows of `Data/proliferation_rates` (optional anchor of the literature estimate) / `Data/transition_rates` (structures the OT). They may differ; if only one of the two is defined it is used for both, and if neither is defined `cell_type` is used


**Optional run inputs: `Data/CardamomOT_inputs.xlsx`.** All optional information steering a run (genes of interest, stimulus schedules and targets, in-silico perturbations, timepoints, proliferation anchors, transition rates…) lives in one Excel workbook, created with its documented structure at the first run (README sheet + one sheet per input; hover the header cells for their meaning). Empty cells mean "not given": the defaults of CardamomOT apply. Text files of older projects (`genes_queries.txt`, `KO_OV_simulate.txt`, `stimulus_schedule.txt`, …) still present in `Data/` are imported into the workbook at each run, overwriting the corresponding cells (with a warning): older projects run unchanged and get a filled workbook; delete the text files to edit the workbook instead. Large numeric arrays (`reference_network.csv`, `basal_init`/`basal_ref`, `inter_init`/`inter_ref`, `inter_simul_ref`) stay files in `Data/`. The text files named in this documentation correspond to the sheets: genes_queries / signatures → `Gene_lists`; stimulus_schedule_inference → `Stimulus_inference`; stimulus_targets → `Stimulus_targets`; stimulus_schedule_simulate → `Simulation_schedule`; KO_OV_Stim_simulate → `Perturbations`; KO_OV_inference → `KO_OV_inference`; times_to_inference / times_to_simulate → `Times`; proliferation_rates, population_sizes, transition_rates → sheets of the same name. The sheet `Model_parameters` fixes model parameters for the project (`parameter` / `value` / `default` / `description`, the description being the comment of `CardamomOT/model/base.py`): a filled value replaces the default of `base.py`, and is itself overridden by the command-line options of the pipeline (`--stimulus`, `--prior`, `--mean-forcing`, `--force-basins`, `--temporal-basins`); an empty value keeps the default.
### 3. Run the full analysis

#### Full pipeline (recommended for beginners)

```bash
# Activate the environment
conda activate cardamom_env

# Run the full pipeline (only -i is required)
python -m CardamomOT.cli pipeline -i my_project --stimulus 1 --prior 1 --mean-forcing 0.5
# or, equivalently (positional, in this order; empty or -1 = workbook value / default)
./run.sh my_project 1 1 0.5
```

**Command-line parameters: only the hard-to-calibrate ones.**

| Argument | `run.sh` position | Model parameter | Description |
|---|---|---|---|
| `-i my_project` | 1 (required) | — | path to your project directory |
| `--stimulus` | 2 | `stimulus` (`1.0`) | stimulus edge penalization (`0`–`1`) |
| `--prior` | 3 | `prior_network_pen` (`1.0`) | weight of the edges absent from the literature prior (`0` = hard constraint, `1` = prior ignored) |
| `--mean-forcing` | 4 | `mean_forcing_em` (`0.5`) | mean-forcing intensity of the NB mixture |
| `--force-basins` | 5 | `force_basins` (`1.0`) | preservation of the basin weights in the network inference (`0`–`1`) |
| `--temporal-basins` | 6 | `temporal_basins` (`1`) | temporal consistency of the basins (`0` or `1`) |

**Everything else is a parameter of `NetworkModel`** (`CardamomOT/model/base.py`), fixed per project in the
`Model_parameters` sheet of `Data/CardamomOT_inputs.xlsx` (empty value = default of `base.py`). Precedence:
**default < workbook < command line**. The pipeline settings are:

| Parameter | Default | Effect |
|---|---|---|
| `split` | `'train'` | `'train'`: train/test split of the cells; `'full'`: all cells |
| `train_rate` | `0.7` | share of the cells of each (sample, time) in the train split |
| `select_genes` | `False` | gene selection (queries, entropy genes, global network, Steiner tree); otherwise all genes kept |
| `build_prior_network` | `False` | literature prior `cardamomOT/ref_network.csv` (by the selection if `select_genes`, else `build_reference_network`) |
| `estimate_proliferation_rates` | `False` | `get_proliferation_rates` (net proliferation rate per cell from gene signatures) |
| `run_test` | `False` | `infer_test` + `check_test_to_train` (needs `split = 'train'`) |
| `simulate_perturbations` | `True` | `simulate_network_KOV` + `check_KOV_to_sim` |
| `simulate_with_proliferation` | `False` | train the proliferation MLP and simulate with branching |
| `species` | `'auto'` | `human` / `mouse` for degradation rates, proliferation signatures, literature prior (auto = from gene names) |

Step-specific settings are parameters too (`allow_depth_correction`, `depth_method`, `use_depth_factor`,
`literature_depth`, `literature_resources`, `integrate_samples`, `senescence_gating`,
`overwrite_degradation_rates`, `report_n_umap`...): see the `Model_parameters` sheet and `base.py`.

**The pipeline starts with a per-cell depth diagnostic** (`estimate_cell_depth`, on `Data/data_complete.h5ad` or on `Data/data.h5ad` if it has at least 10,000 genes): when depth differences between cells of the same sample and time dominate the co-variation of genes, a depth factor per cell is stored in `adata.obs['depth_factor']` (unless `model.allow_depth_correction = False`). It is used by the later steps (counts modelled as NB(k, c / s_i)) only with `model.use_depth_factor = True` (default `False`: the factor is kept but ignored); `model.compute_depth_factor = False` makes the step only read an existing factor (never removed). See [Per-cell sequencing depth](docs/advanced.md#per-cell-sequencing-depth-estimate_cell_depth-first-step).

**Net proliferation rates** (`estimate_proliferation_rates = True`): the `get_proliferation_rates` step (run right after the depth step, on the full gene set) scores each cell against built-in proliferation/death marker gene sets of the species and writes `adata.obs['proliferation_net_rate']`, used to correct the optimal-transport marginals during network inference (it overwrites an existing column; keep the parameter `False` to use your own values). See [Population dynamics](#population-dynamics-proliferation-death-and-cell-type-transition-rates) to force the species, supply your own marker genes, or anchor the estimate to known growth rates.

Separately, `simulate_with_proliferation = True` turns on learning an `R(P)` MLP from the inferred OT couplings and simulating branching PDMP trajectories. See [Proliferation-aware simulation](docs/advanced.md#proliferation-aware-simulation---simulate-proliferation).

#### Results

The pipeline automatically creates these directories:
- `cardamomOT/`: calibrated model parameters (and `cardamomOT/inputs/`, the run inputs exported from the workbook)
- `Check/`: inference vs data comparisons
- `CardamomOT_report_stim<s>_prior<p>.pdf`: final report

## 📋 Detailed Usage

### Step-by-step (for advanced users)

Each step takes `-i` and only the hard-to-calibrate options it uses; the rest comes from the workbook:

```bash
python -m CardamomOT.cli step estimate_cell_depth     -i my_project
python -m CardamomOT.cli step get_proliferation_rates -i my_project   # if estimate_proliferation_rates
python -m CardamomOT.cli step select_genes_and_split  -i my_project --prior 1
python -m CardamomOT.cli step build_reference_network -i my_project   # if build_prior_network and not select_genes
python -m CardamomOT.cli step get_degradation_rates   -i my_project
python -m CardamomOT.cli step infer_mixture           -i my_project --mean-forcing 0.5
python -m CardamomOT.cli step check_mixture_to_data   -i my_project
python -m CardamomOT.cli step infer_network_structure -i my_project --stimulus 1 --prior 1 --force-basins 1 --temporal-basins 1
python -m CardamomOT.cli step infer_network_simul     -i my_project --stimulus 1 --prior 1
python -m CardamomOT.cli step simulate_network        -i my_project
python -m CardamomOT.cli step check_sim_to_data       -i my_project --stimulus 1 --prior 1
python -m CardamomOT.cli step infer_test              -i my_project --stimulus 1 --prior 1 --force-basins 1 --temporal-basins 1   # if run_test
python -m CardamomOT.cli step check_test_to_train     -i my_project --stimulus 1 --prior 1                                        # if run_test
python -m CardamomOT.cli step simulate_network_KOV    -i my_project
python -m CardamomOT.cli step check_KOV_to_sim        -i my_project --stimulus 1 --prior 1
python -m CardamomOT.cli step report_results          -i my_project --stimulus 1 --prior 1
```

Degradation rates (`get_degradation_rates`) come from per-species literature tables (mouse: Schwanhäusser et al. 2011; human: RNADecayCafe for mRNA, Mathieson et al. 2018 for protein), with ortholog recalibration for missing genes; provenance in `Data/degradation_rates_report.csv`. See docs/advanced.md (Literature degradation rates).

### Individual scripts (expert mode)

The scripts can be run directly with the same options, e.g.:

```bash
python infer_network_structure.py -i my_project --stimulus 0.0 --prior 0.5
```

A removed option (`-s`, `-c`, `-r`, `--ref`, `--species`, `--simulate-proliferation`...) stops the script with the name of the parameter to set in the workbook instead.

## 🧬 Advanced Features

This section describes optional input files that activate advanced modes of the algorithm. All optional files are placed either in `my_project/Data/` or directly in `adata.obs`.

---

### Multiple experimental samples (`dataset_id`)

If your experiment contains several biological conditions that should share a common gene regulatory network but may have **different basal transcription rates** (e.g. different cell lines, donors, or perturbation backgrounds), annotate each cell with a sample label:

```python
adata.obs['dataset_id'] = ...   # string or integer label per cell
```

When `dataset_id` is present, CARDAMOM automatically:
- Identifies unique samples and builds one set of **per-sample basal parameters** `θ_basal(s)` for each sample `s`.
- Solves a **single joint optimisation** over shared interaction weights `inter` and all per-sample basals simultaneously — so perturbation information is exploited during network inference.
- Saves `basal.npy` with shape `(n_samples, G, n_networks)` instead of the standard `(1, G, n_networks)`.

> Without `dataset_id`, all cells are treated as a single sample (`n_samples = 1`).

**Keeping per-sample basals close to each other (`constrain_basal_uniform`):**
By default the per-sample basals are free to diverge, which gives maximum flexibility but may overfit when samples differ only by targeted perturbations. Setting `model.constrain_basal_uniform = λ` (e.g. `λ = 100–1000`) adds an L2 penalty that pushes each gene's free-sample basals toward their common mean. The penalty is applied **per (sample, gene) pair**: a sample's basal for a given gene is excluded from the penalty if and only if that specific gene has a non-zero `basal_ref` for that sample (i.e. a KO or OV prior — see below). Concretely, for a `KO_CHGA` sample, only the CHGA basal is excluded; all other genes in that sample are still constrained to stay close to the wild-type values.

---

### Stimulus / exogenous signal (`n_stimuli`, `stimulus_schedule_inference.txt`)

CARDAMOM supports one or several **exogenous inputs** (stimuli) that are not inferred but act as known regulators of the network. Stimuli occupy the first `n_stimuli` columns of the full gene-plus-stimulus state vector.

**Default behaviour (no file needed):** one stimulus that is `0` at the first timepoint and `1` at all subsequent timepoints.

**Custom schedule:** place `Data/stimulus_schedule_inference.txt` in the project folder. Each row corresponds to a timepoint (in chronological order); each column to one stimulus:

```
# stimulus_schedule_inference.txt  (tab or space separated, no header)
# rows = timepoints, cols = stimulus channels
0.0    0.0
1.0    0.0
1.0    1.0
```

For a single stimulus channel, a single-column file suffices. Values between 0 and 1 are allowed and interpreted as partial stimulus strength.

**Fewer rows than timepoints:** if the file contains fewer rows than the number of unique timepoints in the data, the missing timepoints automatically inherit the value of the **last row**. This is useful when a stimulus reaches a plateau and you only want to specify the transition rows explicitly. Providing *more* rows than timepoints raises an error.

**Simulation schedule:** `Data/stimulus_schedule_simulate.txt` (one row per simulated time) gives the schedules of the simulations: first one column per inference stimulus (e.g. to test a new protocol after training), then one column per perturbation stimulus STIM1, STIM2… of `KO_OV_Stim_simulate.txt`. Without it, the inference schedule is used and the perturbation stimuli are 0 at the first time and 1 after (the old `stimulus_schedule_simul.txt` is still read).

---

### Stimulus and prior-network penalization (`--stimulus`, `--prior`)

Two scalar parameters let you **tune the influence of the stimulus and of a prior interaction graph** on network inference and simulation. They are given on the command line (`cardamomot pipeline --stimulus/--prior`, positions 2 and 3 of `run.sh`), which forwards them to every step that uses them, or fixed in the `Model_parameters` sheet.

**Default values** are defined in `NetworkModel` (`base.py`) as `model.stimulus = 1.0` and `model.prior_network_pen = 1.0`. Omitting these arguments (or passing `-1`) keeps the workbook value, else the default.

#### `--stimulus` (model default `1.0`)

Controls how strongly the **stimulus** regulates genes in the reference network matrix:

| Value | Effect |
|---|---|
| `1.0` | Model default — stimulus rows are set to 1 (full influence) |
| `0.0` | Stimulus rows are zeroed out — the stimulus has no regulatory influence |
| `0.5` | Intermediate penalization |

```bash
./run.sh experimental_datasets/Kameneva 0.0        # disable stimulus (run.sh position 2)
cardamomot pipeline -i my_project --stimulus 0.0   # same
```

#### `--prior` (model default `1.0`)

Controls how strongly the **prior interaction graph** (`ref_network.csv`) penalizes edges that are absent from the prior:

| Value | Effect |
|---|---|
| `1.0` | Model default — no penalization; all edges are equally possible |
| `0.0` | Edges absent from the prior are forbidden (`ref_network` acts as a hard sparsity mask) |
| `0.5` | Soft penalization — absent edges are allowed but discouraged |

> **Important:** the same `--prior` must reach `infer_network_structure.py` (edges *learned* during OT inference), `infer_network_simul.py` (simulation reference network) and the checks/report (file tags). `cardamomot pipeline` / `run.sh` forward it to all of them; fixing it in the workbook does the same for steps run by hand.

```bash
./run.sh experimental_datasets/Kameneva 1.0 0      # literature prior as a hard mask (sparse)
./run.sh experimental_datasets/Kameneva 1.0 0.5    # soft prior
# the prior itself: select_genes = True (written by the selection) or build_prior_network = True (workbook)
```

#### `--force-basins` (model default `1.0`) and `--temporal-basins` (model default `1`)

`force_basins` (float in `[0, 1]`) controls, during the network inference, how strongly the basin weights of each gene stay those of the NB mixture (`1.0`) rather than the posteriors of the current trajectories (`0.0`); `temporal_basins` (0 or 1) applies this per timepoint. They are used by `infer_network_structure.py` and `infer_test.py`; `--mean-forcing` (`mean_forcing_em`) is used by `infer_mixture.py`.

```bash
./run.sh experimental_datasets/Kameneva 1.0 1.0 0.5 0.5 0   # relaxed basin weights, not per timepoint
```

**Scripts using `--stimulus` / `--prior`:** `infer_network_structure.py`, `infer_network_simul.py`, `check_sim_to_data.py`, `infer_test.py`, `check_test_to_train.py`, `check_KOV_to_sim.py`, `report_results.py` (and `--prior` alone: `select_genes_and_split.py`, gene budget with a hard prior). Output file names embed `stimulus` and `prior` values (e.g. `adata_sim_stim1.0_prior0.5.h5ad`) so runs with different settings are kept separate.

---

### Initialisation and reference arrays (`Data/`)

The network inference step accepts optional arrays to **warm-start** the optimiser or to **anchor** the solution towards a prior network. All files go in `my_project/Data/` and can be provided as `.npy` or gene-indexed `.csv`.

| File | Shape | Role |
|---|---|---|
| `basal_init.npy` / `.csv` | `(G,)` or `(n_samples, G, n_networks)` | Initial values for basal parameters |
| `inter_init.npy` / `.csv` | `(G, G)` or `(G, G, n_networks)` | Initial values for interaction matrix |
| `basal_ref.npy` / `.csv` | same as `basal_init` | Regularisation target for basal (penalises deviations; entries ≠ 0 also exclude that sample/gene from the `constrain_basal_uniform` penalty) |
| `inter_ref.npy` / `.csv` | same as `inter_init` | Regularisation target for interactions |
| `ref_network.csv` | `(G, G)` gene-indexed CSV | Prior interaction graph: `0` = absent edge, `0 < |v| ≤ 1` = sign free, the edge costs `1/|v|` more than a sure edge; `|v| > 1` = sure edge with forced sign (e.g. `2` = activation, `-2` = inhibition) |

**CSV format for `basal_ref` / `inter_ref`:** rows and columns must be gene names matching `adata.var_names` (upper-cased). Stimulus rows/columns (`Stimulus`, `Stimulus_0`, …) are handled automatically.

A 3-D `basal_init` of shape `(n_samples, G, n_networks)` warm-starts each sample's basal independently. If a 2-D array `(G, n_networks)` is provided it is broadcast to all samples.

---

### Per-sample KO / OV prior (`Data/KO_OV_inference.txt`)

To encode prior knowledge about **which genes are knocked out (KO) or overexpressed (OV)** in specific samples, create a tab-separated file:

```
# KO_OV_inference.txt
sample_id   KO  OV
wt  0   0	
ko_CHGA CHGA	0
ov_STMN2    0   STMN2
ko_CHGA_ov_STMN2    CHGA    STMN2
```

- `sample_id` values must match `adata.obs['dataset_id']`.
- `KO` column: comma-separated gene names forced to **basal = −100** (silent during inference).
- `OV` column: comma-separated gene names forced to **basal = +100** (always active during inference).
- Multiple genes per cell: `CHGA,POSTN`.
- Missing or `0` entries mean no constraint for that sample.

When this file is present, `infer_network_structure.py` replaces `basal_ref` for the affected genes and samples, and saves `cardamomOT/basal_ref_mask.npy` (a boolean `(n_samples, G)` array) so that `simulate_network_KOV.py` can compute a clean wild-type baseline when running new perturbations.

---

### In-silico perturbation simulation (`Data/KO_OV_Stim_simulate.txt`)

After training, you can simulate arbitrary knock-out / over-expression combinations with `simulate_network_KOV.py`. Define the combinations in:

```
# KO_OV_Stim_simulate.txt  (tab-separated, header required)
KO	        OV
CHGA	STMN2           # wild-type (no perturbation)
POSTN	S100B,STMN2
```

Each row produces one independent simulation. Results are saved as `cardamomOT/adata_prot_simul_KO_*_OV_*_stim*.h5ad`.

If `KO_OV_inference.txt` was used during training and `basal.npy` is therefore 3-D (per-sample), `simulate_network_KOV.py` automatically reconstructs a clean wild-type basal by averaging the unconstrained samples before applying each new perturbation.

#### Partial KO / OV via degradation rates (`GENE-X` syntax)

By default a KO silences a gene by setting its basal transcription to −∞ (complete silencing). For a **partial** perturbation of strength X% (0 < X < 100) append `-X` to the gene name:

```
# KO_OV_Stim_simulate.txt
KO	        OV
CHGA-80	    STMN2-60     # 80 % KO of CHGA # 60 % OV of STMN2
POSTN	S100B            # full KO / full OV (no suffix = 100 %, existing behaviour)
```

**Mechanism:** instead of modifying the basal transcription, partial perturbations scale the **per-gene creation rate** (burst rate in the stochastic PDMP, effective `ks` in the ODE) without touching the degradation rates:

| Mode | Creation rate factor | Effect on steady-state |
|---|---|---|
| KO `X`% | `× (1 − X/100)` | production reduced → lower steady-state |
| OV `X`% | `× 1 / (1 − X/100)` | production increased → higher steady-state |

At X = 0 the factor is 1 (no effect). As X → 100 the KO factor → 0 (full silencing) and the OV factor → ∞. The output file is labelled, for example, `KO_CHGApct80_OV_STMN2pct60` to distinguish it from a complete perturbation.

> **Note on gene names with hyphens:** the `-X` suffix is recognised as a percentage only when `X` is a number strictly between 0 and 100. Gene names such as `HIF-1A` are therefore parsed correctly (the suffix `1` is outside the 0–100 exclusive range).

---

### Population dynamics: proliferation, death and cell-type transition rates

CARDAMOM's optimal transport step corrects for cell proliferation/death **by default**, but cell-type transitions are assumed equally likely unless you opt in with a transition-rate matrix (see below).

#### Net proliferation rate — default behaviour

Every run of `get_proliferation_rates.py` — run right after `estimate_cell_depth` in the standard pipeline, on the full, unfiltered dataset before any gene selection — estimates a per-cell **net** growth rate (birth − death; CardamomOT only ever uses the difference, never the two terms separately) and writes it to:

```python
adata.obs['proliferation_net_rate']   # float, net proliferation rate per cell (birth − death)
```

It runs directly on `Data/data.h5ad` (all genes) rather than after gene selection, because differential-expression filtering could otherwise discard many of the literature marker genes needed to score the signature. Since the rate is stored in `adata.obs` (per-cell, not per-gene), it survives the later gene-subsetting and train/test splitting done by `select_genes_and_split.py` unchanged — no need to re-estimate it per split.

By default this uses built-in **human** proliferation/death marker gene signatures (moscot/Waddington-OT style — see `CardamomOT/tools/estimate_proliferation.py`), scored with `scanpy.tl.score_genes` and mapped to a rate with the same shifted-logistic curve as moscot. `get_proliferation_rates.py` always (re)computes and overwrites `adata.obs['proliferation_net_rate']`, even if that column is already present. If you set it yourself from an external measurement (e.g. EdU staining) and want to keep it, skip the step entirely instead: `--no-use-proliferation` on `cardamomot pipeline`, or `use_proliferation=0` on `run.sh` (both default to running the step).

moscot/WOT calibrate this curve for a **per-day** rate (their `TemporalProblem` computes elapsed time from a `day` obs field and raises the growth score to that many days). CardamomOT expresses `adata.obs['time']` and every internal kinetic rate in **hours** instead, so the estimate is divided by 24 before being written to `obs['proliferation_net_rate']` — see [Advanced Features](docs/advanced.md#net-proliferation-rate--default-behaviour) for the exact conversion (overridable via `hours_per_day=` on `estimate_growth_rates` if your own `adata.obs['time']` is in days).

Once populated, the OT marginals between consecutive timepoints t₁ and t₂ are modified:
- **Source marginal** µᵢ ∝ exp(+netᵢ · Δt/2) — cells with higher net growth carry more weight as trajectory sources
- **Target marginal** νⱼ ∝ exp(−netⱼ · Δt/2) — fast-growing cells at t₂ are down-weighted (they represent fewer distinct lineages)

This is equivalent to computing a **demographically corrected** optimal transport (as in Waddington OT, Schiebinger et al. 2019): the resulting coupling captures intrinsic lineage transitions independently of population-level growth effects.

#### Refining the proliferation-rate estimate

Five levers, from least to most involved, all optional:

| Refinement | How | Why |
|---|---|---|
| Species | parameter `species = 'human'\|'mouse'` (Model_parameters sheet; default `'auto'`: detected from gene names) | Forces the built-in human or mouse marker gene lists (moscot uses different death markers per species — see `docs/advanced.md` for details) |
| Score on the unfiltered gene set | Place `Data/data_complete.h5ad` (all genes) alongside an already gene-filtered `Data/data.h5ad` | If `Data/data.h5ad` was prepared with genes already filtered, the literature marker genes may be missing from it; `data_complete.h5ad` is used only to score the signature (never modified), and the result is mapped back onto `Data/data.h5ad` by cell name — every cell in `data.h5ad` must also be present in `data_complete.h5ad` |
| Custom marker genes | `Data/proliferation_signatures.csv`/`.txt`, `Data/death_signatures.csv`/`.txt` (one gene per line or comma-separated) | Override the built-in lists with signatures specific to your system (e.g. a disease- or lineage-specific gene set) |
| Anchor to a known rate | `Data/proliferation_rates.csv`/`.txt` (two columns, no header: `cell_type, rate`) — **`rate` in hour⁻¹**, matching `adata.obs['time']` (growth curves are often reported per day — divide by 24 first) | If you have a trusted population-level growth rate per cell type (e.g. from a growth curve), the literature-based per-cell estimate is recentred so its mean matches your value within each cell type, while keeping the per-cell heterogeneity from the signature. Grouping uses `adata.obs['cell_type_proliferation']` if present, else `cell_type_transition`, else `cell_type`. All-or-nothing: if any cell type is missing from the table (or no grouping is found), the unanchored estimate is kept for all cells |
| Full manual override | Set `adata.obs['proliferation_net_rate']` yourself **and** keep `estimate_proliferation_rates = False` (default) | The step no longer preserves pre-existing values on its own — it always overwrites them when run |

See [Advanced Features → Refining proliferation rates](docs/advanced.md#refining-proliferation-rates) for the exact formulas and defaults.

#### Cell-type transition rates (`Data/transition_rates.csv`) — opt-in

To bias the OT cost toward biologically plausible cell-type transitions, place a square CSV of **transition rates** (same units as proliferation/death rates) in the project's `Data/` folder:

```
# transition_rates.csv — rows = source type at t1, cols = target type at t2
# values are instantaneous rates (≥ 0); higher rate = more likely transition
         ,TypeA,TypeB,TypeC
TypeA    ,  0.3,  0.1,  0.01
TypeB    ,  0.05, 0.2,  0.05
TypeC    ,  0.01, 0.05, 0.3
```

Row/column names must match the values of `adata.obs['cell_type_transition']` if present, else `cell_type_proliferation`, else `cell_type`. All-or-nothing: if any cell type is missing from the matrix (or no grouping is found), the OT runs without transition constraint.

At each pair of consecutive timepoints separated by Δt, transition probabilities are computed as `exp(rate × Δt)` and each row is rescaled to sum to `n_types` (number of cell types), so the mean weight per row equals 1 and the overall cost scale is preserved on average.

The OT pairwise distance is then divided element-wise by these weights: a transition with weight > 1 becomes cheaper (preferred), and a transition with weight < 1 becomes more expensive (penalised). The weights therefore adapt automatically to the interval Δt — short intervals produce weights close to 1 for all transitions, while long intervals amplify the contrast between fast and slow transitions. Missing cell types default to index 0.

Both corrections are active simultaneously when the corresponding files are present. They apply during training (`infer_network_structure.py`) and on the test set (`infer_test.py`).

---

### Summary: project directory structure

```
my_project/
├── Data/
│   ├── data.h5ad                  # required — obs['time'], obs['dataset_id'] (opt.)
│   │                              #            obs['proliferation_net_rate'] (opt.)
│   │                              #            obs['cell_type'] (opt.)
│   ├── data_complete.h5ad         # optional — unfiltered gene set, used only to score
│   │                              #            proliferation/death signatures if data.h5ad
│   │                              #            was already gene-filtered; never modified
│   ├── gene_list.txt              # optional — subset of genes to use
│   ├── stimulus_schedule_inference.txt      # optional — stimulus values per timepoint
│   ├── stimulus_schedule_simulate.txt # optional — schedules of the simulations (inference stimuli, then STIM1, STIM2...)
│   ├── ref_network.csv            # optional — prior interaction graph (sparsity mask)
│   ├── basal_init.npy / .csv      # optional — warm-start for basal parameters
│   ├── inter_init.npy / .csv      # optional — warm-start for interactions
│   ├── basal_ref.npy / .csv       # optional — regularisation target for basal
│   ├── inter_ref.npy / .csv       # optional — regularisation target for interactions
│   ├── KO_OV_inference.txt          # optional — per-sample KO/OV prior (requires dataset_id)
│   ├── KO_OV_Stim_simulate.txt             # optional — in-silico perturbations to simulate
│   ├── transition_rates.csv|txt   # optional — cell-type transition cost matrix for OT
│   ├── proliferation_signatures.csv|txt # optional — custom proliferation marker genes
│   ├── death_signatures.csv|txt   # optional — custom death marker genes
│   └── proliferation_rates.csv|txt# optional — per-cell-type rate (hour⁻¹) to anchor the estimate to
└── cardamomOT/                    # generated by the pipeline
    ├── basal.npy                  # (n_samples, G, n_networks)
    ├── inter.npy                  # (G, G, n_networks)
    ├── basal_ref_mask.npy         # (n_samples, G) bool — generated from KO_OV_inference.txt
    └── ...
```

---

## 📊 Understanding Results

### Main generated files

**In `my_project/cardamomOT/`:**
- `inter.npy`: gene regulatory interaction matrix `(G, G, n_networks)`
- `basal.npy`: basal transcription parameters `(n_samples, G, n_networks)`
- `mixture_parameters.npy`: burst kinetics parameters `(G+1, G)`

**In `my_project/Check/`:**
- Visual comparisons between real data and simulations

### Parameter Interpretation

- **Interactions (`inter[i, j, k]`)**: weight of gene `i` on gene `j` in network state `k`; positive = activation, negative = repression
- **Basal (`basal[s, g, k]`)**: constitutive transcription of gene `g` in sample `s` under network state `k`
- **Mixture parameters**: frequency and size of transcriptional bursts per gene

## 🔧 Troubleshooting

### Common Issues

**Import error (numpy/scipy):**
```bash
# Verify active environment
conda activate cardamom_env
python -c "import numpy, scipy; print('OK')"
```

**Non-conforming data:**
- Verify that `data.obs['time']` exists
- Ensure counts are positive integers

**Insufficient memory:**
- Reduce the number of genes in `gene_list.txt`
- Use subsampling of cells

### Debug Commands

```bash
# Verify installation
python -c "import CardamomOT; print('CARDAMOM imported')"

# Test a specific module
python -c "from CardamomOT.inference import mixture; print('Module OK')"

# Check dependencies
python -m CardamomOT.cli --help
```

## 📚 Advanced Tutorials

### Converting data from CARDAMOM v1

If you have data in CARDAMOM v1 format:

```bash
# Convert old .txt format to .h5ad
python ./utils/old_to_new/convert_old_data_to_ad.py -i my_project
python ./utils/old_to_new/add_degradations_to_ad.py -i my_project
```

### Customizing Parameters

See source files to modify:
- `select_genes_and_split.py`: gene selection criteria
- `infer_mixture.py`: burst kinetics parameters
- `infer_network_*.py`: network inference algorithms

## 📖 Références

[3] Y. Maugé, E. Ventre. [CardamomOT: a mechanistic optimal transport-based framework for gene regulatory network inference, trajectory reconstruction and generative modeling](https://doi.org/10.64898/2026.03.31.715390). *bioRxiv*, 2026.

[2] E. Ventre, U. Herbach et al. [One model fits all: Combining inference and simulation of gene regulatory networks](https://doi.org/10.1371/journal.pcbi.1010962). *PLOS Computational Biology*, 2023.

[1] E. Ventre. [Reverse engineering of a mechanistic model of gene expression using metastability and temporal dynamics](https://content.iospress.com/articles/in-silico-biology/isb210226). *In Silico Biology*, 2021.

## Article data
Datas analyzed in this article are available at :
• Semrau et al. (2017): GEO accession GSE79578.
• Kameneva et al. (2021): GEO accession GSE147821.
• Schiebinger et al. (2019): GEO accession GSE106340 and
https://singlecell.broadinstitute.org/single_cell/study/SCP295/