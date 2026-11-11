# Pseudotime CardamomOT — implementation plan (Options 1, 2, 3)

Status: plan (no code yet beyond `simulate_pseudotime_data.py`)
Scope: implement three variations of the CardamomOT optimisation cycle for **static data without
time labels** (pseudotime problem of `plan.tex`), as a **new subpackage**; `CardamomOT/model/base.py`
and the rest of the base package stay **untouched**.

---

## 0. Implementation status

| increment | state | file |
|---|---|---|
| simulated datasets (CN5, BN8, FN8) | **done** | `simulate_pseudotime_data.py` |
| τ → bins (`BinScheme`, equal-width/count, merging) | **done** | `binning.py` |
| NB mixture on bins + the 3 circularity guards | **done** | `base_pseudotime.py` |
| validation of the mixture on bins | **done** | `check_mixture_bins.py`, `tests/test_pseudotime_binning.py` |
| mechanistic edge costs (bridge, network CE, geometric) | **done** | `costs.py` |
| **Option 3** (mechanistic Palantir) + `update_pseudotime` | **done** | `options/palantir_like.py`, `evaluate.py` |
| validation of the ordering | **done** | `check_pseudotime_order.py`, `tests/test_pseudotime_order.py` |
| **`loop_pseudotime`** (mixture -> ordering -> E / M / α / basins) | **done** | `base_pseudotime.py` |
| **growth pass** on the pseudotime bins (`R_opt`) | **done** | `growth_pass_pseudotime` |
| validation of the joint cycle (oracle and inferred τ) | **done** | `check_pseudotime_fit.py` |
| tuning (harmonised geometric term, kNN/entropic sweep) | **done** | `costs.py`, PLAN.md §0.7 |
| **Option 1** (waves of partial OT) + validation | **done** | `options/partial_ot.py`, `tests/test_pseudotime_partial_ot.py`, §0.8 |
| **Option 2** (path-dependent TSP) + validation | **done** | `options/tsp.py`, `tests/test_pseudotime_tsp.py`, §0.9 |
| `refine` (degradations/simulations) in pseudotime units | **done** | `check_pseudotime_simul.py`, `evaluate.marginal_agreement`, §0.10 |
| M6 benchmark, continuous **and** snapshot sampling designs | **done** | `check_benchmark.py`, `evaluate.snapshot_metrics`, §0.11–§0.12 |

### 0.4 Joint cycle: results (3 iterations, run 1, seed 0, AUPRC vs the true graph)

| dataset | genes | random | oracle τ (frozen true bins) | free τ (Option 3) |
|---|---|---|---|---|
| CN5 | 5 | 0.24 | **0.93** (AUROC 0.96) | **0.62** (AUROC 0.81), ρ 0.46 / 0.65 transient |
| CH20 (tree, 4 arms) | 20 | 0.06 | 0.52 (AUROC 0.86) | 0.36 (AUROC 0.82), **ρ 0.84 / 0.93 transient** |
| BN8 | 8 | 0.27 | 0.27 | 0.45 |
| FN8 | 8 | 0.27 | 0.18 | 0.19 |

What this says: with a **correct time unit** (see §4.3, implemented) and the true roots, the reused
CardamomOT machinery recovers the cyclic network nearly perfectly (AUPRC 0.93) and gives 0.52 on the
20-gene tree — the E/M/basin/growth code is validated. With the *inferred* pseudotime, CN5 keeps
0.62 and CH20 reaches ρ = 0.84 (0.93 on the transient) with 0.36, so the ordering + network loop
works on the larger networks. BN8/FN8 (8 genes, weak bimodality, one nearly dead gene) remain weak
and are kept as the hard cases.

### 0.5 Three findings of this increment

1. **The time unit was the main blocker.** With `Δτ` from the prior `1/d1` (and `d1` read from
   `var['d1']` as in the figure pipelines), CN5's oracle AUPRC went from 0.55 to 0.93 and CH20's
   from 0.27 to 0.52, at `Δτ = 10 h` (the true 96 h / 9). Implemented as
   `pseudotime_duration_mode ∈ {'prior', 'fixed', 'grid'}`: `'prior'` sets `T_tot = (K-1)·relaxation/d1`
   (relaxation = 1 by default), `'grid'` selects the bin width among `pseudotime_dt_grid × 1/d1`
   by the model error (one iteration each), `'fixed'` takes `pseudotime_total_duration`.
2. **`set_total_duration` must keep the roots** — it re-binned with the default root rule (the single
   earliest cell), which silently degraded every run; caught by
   `test_loop_pseudotime_frozen_tau_keeps_the_bins`.
3. **Inside the cycle, the geometric edge weight is much better than the mode-based ones** (CN5,
   3 iterations: AUPRC 0.62 geometric, 0.37 plausibility, 0.31 mechanistic): with a network fitted
   on one or two iterations, `σ(Θ P)` and the bridge are too noisy to define the graph, while the
   continuous log-expression distance tracks the trajectory. Default set to `'geometric'`;
   `'auto'` (plausibility → mechanistic once Θ is fitted) and the mechanistic weights stay available
   and are expected to win once Θ is converged.

### 0.6 Benchmark networks

`simulate_pseudotime_data.py` provides six networks (per-network cell/run defaults):
`CN5`/`BN8`/`FN8` (2000×5), `CH20` (tree: stimulus → 4 hubs → 4-gene chains with a delayed negative
feedback, 1000×3), `RG50` (sparse random DAG in topological order, mixed signs, 800×3) and `MD100`
(five modules of 20, switched on in cascade, 500×3). All are checked for a sane expression range and
temporal variation (`~80/100` genes informative for MD100).

### 0.7 Tuning of the free mode (harmonised geometric term + sweep)

**Harmonisation.** The geometric term was previously inconsistent: `plausibility_cost` compared the
*log counts*, `mechanistic_cost` compared the *basin modes*. All three costs now take the same term
through `CostContext.geom_space ∈ {expression, modes}` (`pseudotime_geom_space`, default
`expression`), and `edge_cost` passes the same `(rna_i, rna_j, modes_i, modes_j)` to every variant, so
`mechanistic = network CE + geometric` holds exactly (asserted by
`test_mechanistic_cost_is_the_network_term_plus_the_geometric_term` for both spaces).

**Sweep** (free τ, 3 iterations, growth off, seed 0, run 1; CN5 = cyclic, CH20 = tree):

| graph | entropic | CN5 AUPRC / ρ / ρ_trans | CH20 AUPRC / ρ / ρ_trans |
|---|---|---|---|
| k=15 | 0.3 | 0.667 / +0.44 / +0.52 | 0.324 / +0.86 / +0.91 |
| k=15 | 1.0 | 0.734 / +0.44 / +0.52 | 0.438 / +0.86 / +0.91 |
| k=15 | 3.0 | 0.513 / +0.44 / +0.52 | 0.452 / +0.86 / +0.91 |
| k=30 | 0.3 | 0.572 / +0.46 / +0.65 | 0.290 / +0.84 / +0.93 |
| k=30 | 1.0 | 0.618 / +0.46 / +0.65 | 0.363 / +0.84 / +0.93 |
| k=30 | 3.0 | 0.463 / +0.46 / +0.65 | 0.344 / +0.84 / +0.93 |
| k=60 | 0.3 | 0.839 / +0.43 / +0.61 | 0.439 / +0.85 / +0.93 |
| **k=60** | **1.0** | **0.810** / +0.43 / +0.61 | **0.485** / +0.85 / +0.93 |
| k=60 | 3.0 | 0.760 / +0.43 / +0.61 | 0.323 / +0.85 / +0.93 |

Findings:

1. **The kNN size drives the network, not τ.** ρ is essentially flat in `k` (CN5 +0.43…+0.46, CH20
   +0.84…+0.86) but AUPRC moves a lot (CH20 0.29 → 0.49, CN5 0.46 → 0.84). A denser graph gives a
   *smoother* τ, so cells that are close in pseudo-time also share a bin; the global rank correlation
   does not see this, the bins do. Default raised to `palantir_n_neighbors = 60`.
2. **`init_entropic_noise = 1.0` is the best or near-best everywhere** (k=60: 0.810/0.485 vs 0.760/0.323
   at 3.0); too much entropic blur destroys the basin structure the M-step needs. Kept as is.
3. **`intensity_prior` cannot be tuned in this regime.** `weight_init ∝ (n_iter ≤ intensity_prior)`
   only depends on `n_iter`, so with 3 iterations every value ≥ 3 gives the identical schedule:
   measured CN5 0.835 (ip=2) vs 0.810 (ip=100), CH20 0.395 vs 0.485 — the differences are the effect of
   switching the mRNA weight off at iteration 3, not a tuning knob. Left at the base default (100).
   Tuning it needs ≥ 12 iterations, i.e. after the runtime work of Option 1/2.
4. With k=60 the §0.4 table improves to **CN5 0.84 (AUROC 0.93)** and **CH20 0.49 (AUROC 0.89)** in the
   free mode at ρ = 0.85 (0.93 transient), i.e. the gap to the oracle τ narrows from 0.93/0.62 and
   0.52/0.36 to 0.93/0.84 and 0.52/0.49.

Defaults after this increment: `palantir_edge_weight='geometric'`, `palantir_n_neighbors=60`,
`init_entropic_noise=1.0`, `pseudotime_geom_space='expression'`, `pseudotime_duration_mode='prior'`.

### 0.8 Option 1 (partial OT) — results and gate

Implemented in `options/partial_ot.py` (`PartialOT`), selected with `pseudotime_method='partial_ot'`;
the feature space of the candidate search is shared with Option 3 (`options/features.py`, extracted
from `PalantirLike._features`).

**Free τ, 3 iterations, seed 0, run 1, growth off** (`palantir` at its tuned defaults,
`partial_ot_alpha_geom=1.0`):

| dataset | method | AUPRC | AUROC | ρ | ρ transient | waves | unreached | time |
|---|---|---|---|---|---|---|---|---|
| CN5 | palantir | **0.810** | **0.947** | **+0.425** | **+0.611** | – | 0 | 39 s |
| CN5 | partial_ot | 0.717 | 0.760 | +0.343 | +0.579 | 50 | 0 | 35 s |
| CH20 | palantir | 0.485 | 0.887 | **+0.847** | **+0.934** | – | 0 | 6 s |
| CH20 | partial_ot | **0.519** | **0.923** | +0.828 | +0.840 | 25 | 0 | 27 s |
| RG50 | palantir | 0.086 | 0.563 | **+0.727** | **+0.920** | – | 0 | 15 s |
| RG50 | partial_ot | **0.096** | **0.584** | +0.510 | +0.743 | 29 | 0 | 37 s |

**Gate of M2** (`PLAN.md` §7):

| criterion | result |
|---|---|
| no cell left behind | **pass** — `n_unreached = 0` on all three networks; the neighbour fill only fires on a disconnected component, and `n_fallback` counts the waves where the candidate pool had to be widened |
| mass accounting | **pass** — `sum(gamma) = β` exactly and the marginals respect `a` / `b` (unit-tested) |
| ρ ≥ Option 3 | **fail** — ρ is 0.02–0.09 *below* Option 3 everywhere (−0.22 transient on RG50) |
| edges ≥ Option 3 | **pass on 2/3** — +0.034 AUPRC on CH20, +0.010 on RG50, −0.093 on CN5 |

**Findings**

1. **The step cost needs a strong geometric anchor.** With the planned `alpha_geom=0.2` the fitted
   network overrides expression continuity and the ordering collapses on the CN5 oscillator:
   ρ +0.52 → +0.32 → **−0.01** over three iterations (AUPRC 0.21). Sweep at 3 iterations
   (AUPRC / ρ):

   | α_geom | CN5 mechanistic | CN5 plausibility | CH20 mechanistic | CH20 plausibility |
   |---|---|---|---|---|
   | 0.2 | 0.211 / −0.01 | – | 0.496 / +0.85 | – |
   | 0.5 | 0.391 / +0.35 | 0.295 / +0.27 | 0.565 / +0.84 | 0.597 / +0.84 |
   | **1.0** | **0.717 / +0.34** | 0.609 / +0.37 | 0.519 / +0.83 | 0.519 / +0.82 |
   | 2.0 | 0.442 / +0.50 | 0.648 / +0.38 | 0.563 / +0.83 | 0.494 / +0.83 |

   Default set to `partial_ot_alpha_geom = 1.0` (best mean AUPRC over the two datasets).
2. **Option 1 improves the edges, not the axis.** ρ is slightly worse on every network while AUPRC
   is better on the two non-cyclic ones: the waves give a *layered* τ (a mass-weighted step count)
   that is coarser than the graph distance, so the bins are less even — but the network that the
   bins produce is at least as good, and on the tree clearly better (CH20 AUROC 0.923 vs 0.887).
3. **CN5 (oscillator) remains Option 3's.** The wave has to return to a basin it has already left,
   and the partial-OT formulation forbids it: a cell is reached exactly once. Documented limitation,
   not a bug — the same reason a monotone τ is ill-defined for CN5 at all (§0.4).
4. Runtime is comparable to Option 3; the number of waves (25–50) is set by `mass_per_step` and the
   kNN pool, not by the network size.

**Decision**: `pseudotime_method='palantir'` stays the default (it wins the gate on ρ);
`'partial_ot'` is the option of choice for a **branching** dataset, i.e. exactly where Option 3 is
weakest.

### 0.9 Option 2 (path-dependent TSP) — results and gate

Implemented in `options/tsp.py` (`PathTSP`), selected with `pseudotime_method='tsp'`; `options/common.py`
now holds the root mask shared by the three options.

The tour is warm-started from the τ of the previous iteration (and from `tsp_init` only when there is
no τ yet), the 2-opt is **first improvement** on random windows (the plan said best-improvement; the
first-improvement variant costs one segmentation per move instead of a full scan and is what the
budget `n_moves` then controls), and τ is the **cumulative path length**.

**Free τ, 3 iterations, seed 0, run 1, growth off** (all three options measured in the same run, so
the times are comparable to each other; they varied by ~5× between runs because of machine load):

| dataset | method | AUPRC | AUROC | ρ | ρ transient | tour gain | time |
|---|---|---|---|---|---|---|---|
| CN5 (cycle) | palantir | **0.810** | **0.947** | **+0.425** | +0.611 | – | 211 s |
| CN5 | partial_ot | 0.717 | 0.760 | +0.343 | +0.579 | – | 465 s |
| CN5 | tsp | 0.663 | 0.680 | +0.417 | **+0.637** | 4.0 % | **36 s** |
| CH20 (tree) | palantir | 0.485 | 0.887 | **+0.847** | **+0.934** | – | 29 s |
| CH20 | partial_ot | **0.519** | **0.923** | +0.828 | +0.840 | – | 164 s |
| CH20 | tsp | 0.311 | 0.780 | +0.707 | +0.895 | 1.7 % | **16 s** |
| RG50 (DAG) | palantir | 0.086 | 0.563 | **+0.727** | **+0.920** | – | 36 s |
| RG50 | partial_ot | **0.096** | **0.584** | +0.510 | +0.743 | – | 208 s |
| RG50 | tsp | 0.081 | 0.544 | +0.679 | +0.913 | 1.0 % | 65 s |

**Gate of M3** (`PLAN.md` §7):

| criterion | result |
|---|---|
| the tour never degrades | **pass** — `tour_length_end <= tour_length_start` by construction, and the 2-opt accepts a move only on a strict decrease |
| ρ ≥ Option 3 | **fail** — Option 2 ends below Option 3 on AUPRC on all three networks and below its ρ on CH20 / RG50 |
| cheaper than an E-step | **pass** — 16–65 s for the whole 3-iteration fit, the cheapest of the three |

**Findings**

1. **The 2-opt barely moves.** The tour shortens by only 1–4 % of its length: warm-starting from
   Option 3's τ and then re-ordering by adjacent swaps is nearly a no-op. That is the expected
   outcome — the ordering is *already* the shortest path the graph allows — and it means Option 2's
   value here is as a diagnostic (does an unconstrained re-ordering improve anything? no) rather than
   as a solver.
2. **A tour is the wrong object for a tree and the right one for a cycle.** On CH20 (4 arms) it is
   the worst of the three by a wide margin (AUPRC 0.311 vs 0.485 / 0.519): one path must interleave
   the arms, so half of the adjacent steps are physically impossible. On CN5 (a single cycle) it has
   the **best transient correlation of all three** (+0.637 vs +0.611 / +0.579) — a tour is exactly a
   cycle, which confirms that its weakness on the trees is structural and not an implementation
   artefact.
3. **First improvement on 1000 random windows is enough** for the tour to converge: the extra
   accepted moves beyond a few hundred change τ by < 1 % (the tour gain stays at 1–4 % from 500 to
   2000 moves).

**Decision**: `pseudotime_method='palantir'` remains the default. Option 2 is kept as the third arm
of the benchmark and as the cheap refinement of a given ordering (`tsp_n_moves`, `tsp_window`), and
its CH20 result is the quantitative statement of the limitation already in `plan.tex`.

### 0.10 M5 — growth, degradations and simulations in pseudotime mode

The whole downstream chain runs **unchanged** on pseudotime bins:

```
mixture + ordering cycle -> growth pass (R_opt) -> fit_birth_death_to_growth
   -> refine_network_degradations (network refit, d1/d0, ProliferationMLP) -> simulate_network
```

Why it works without a rewrite: `loop_pseudotime` leaves `times_data = np.repeat(bin_times, n_full)`
with `prot` / `kon_beta` / `alpha` on the same `n_full`-slots-per-bin layout, which is exactly the
**balanced design** the base assumes (`N_tot * a + slots`). The imbalance of the bins was already
consumed by the growth pass through `real_cell_batches`, so nothing downstream needs to know it.

`fit_pseudotime(refine=True)` is the entry point; `check_pseudotime_simul.py` reports the gate
(3 iterations, run 1, seed 0):

| dataset | $R$ (net, per unit pseudotime) | $d_1$ recovered / true | within 2x | simulated vs observed marginals | within 2x |
|---|---|---|---|---|---|
| CN5 | $-5.6\times10^{-18}$ | 0.63 – 0.98 | 5/5 genes | 0.90 – 1.32 | 8/8 bins, 5/5 genes |
| BN8 | $-1.4\times10^{-18}$ | 0.58 – 0.94 | 8/8 genes | 0.56 – 3.08 | 94 % of the pairs |

**Gate of M5** (`PLAN.md` §8):

| criterion | result |
|---|---|
| simulated mRNA marginals within 2x of the data | **pass** — CN5 entirely inside 0.90–1.32; BN8 at 94 % (one nearly-dead gene at 3.08, the same gene that already fails the oracle ordering in §0.4) |
| $d_1$ recovered (the axis is a prior, this is the only place it is inferred) | **pass** — within a factor 2 on every gene, but **systematically low** (0.58–0.98) |
| $R$ with the right sign | **not testable here** — $R = 0$ to machine precision, which is the *correct* answer since the simulated networks do not proliferate. The gate as written in §8 ("$R > 0$") was wrong for these datasets; a proliferating simulation would be needed (risk #2) |

**Findings**

1. **$d_1$ is biased low, and that bias is the circularity of the gauge.** The axis was scaled by
   $\Delta\tau = \text{relaxation}/\text{median}(d_1)$ from the *prior*; the refinement then re-estimates
   $d_1$ at 0.58–0.98 of that prior. The two are not independent measurements of the same quantity:
   $T_{\text{tot}}$ and $d_1$ can still drift together (§10.1). What the result *does* establish is
   that the ODE fit along the inferred axis is consistent with the prior to within a factor 2 —
   i.e. that the inferred $\tau$ is a plausible time coordinate, not that its absolute scale is
   certified.
2. **$d_0$ is not re-estimated on its own.** The base writes only `d_t[:, 1, :]` ($d_1$); the ratio
   $d_0/d_1$ goes to `model.ratios` and $d_0^{\text{sim}} = d_1 \cdot \text{ratios}$. Reading
   `d_t[:, 0, :]` after the refinement therefore returns the untouched prior — a trap worth knowing
   before quoting $d_0$ anywhere.
3. **The expected count must use $kz/c$.** The simulated mean of a state is
   $\mathrm{kon}\cdot kz / c$ with $kz = \max_{\text{mode}}\mathrm{sigmoid\_amplitudes}(ks)\cdot k_1$,
   because `negative_binomial(n, p=c/(c+1))` has mean $n/c$. Using the mixture mode amplitude
   instead of $k_1$, or forgetting the $1/c$, gives a **30x** error that looks like a broken model.
   Both mistakes were made and corrected while building `marginal_agreement`; the metric is now
   unit-tested against the exact formula.
4. **$R$ is estimated on an all-zero signal.** `growth_pass_pseudotime` returns values of order
   $10^{-17}$, not merely "small": with a uniform cell density along $\tau$ the unbalanced OT has
   nothing to rebalance. The `ProliferationMLP` trained on that target is degenerate (R² $\approx
   -10^{15}$), which is harmless here but means the proliferation path is **untested** by these
   datasets.

**Status**: the pseudotime pipeline is end-to-end — fit, order, infer, simulate — and the two
numerical quantities that a user would quote from it ($d_1$ and the mRNA marginals) are validated
against ground truth.

### 0.11 A second sampling design: unequally spaced snapshots

Every dataset so far gave each cell its own time (``uniform`` in ``[0, 96]``). Real sequencing data
does not: cells come from a handful of libraries, so the truth is a **step function** of time. The
benchmark therefore runs on two designs, generated by the same simulator:

```bash
# design 1 (existing): one time per cell -> data/<NET>/Data/
conda run -n cardamom_env python -m CardamomOT.pseudotime.simulate_pseudotime_data \
    --networks CN5 BN8 FN8 --runs 3
# design 2: 10 unequally spaced snapshots -> data/<NET>/Data_grid/
conda run -n cardamom_env python -m CardamomOT.pseudotime.simulate_pseudotime_data \
    --networks CN5 BN8 FN8 --runs 3 --time-design grid --snapshots 10 --grid-ratio 1.3
```

**The grid.** The gaps follow a geometric progression of ratio 1.3, so the last gap is
$1.3^{8} \approx 8.2$ times the first: for a 96 h horizon the snapshots fall at
0, 3.0, 6.9, 12.0, 18.6, 27.3, 38.4, 52.9, 71.7, 96.0 h. The number of cells is **equal per
snapshot** (200 of 2000), so the irregularity is in the *spacing* only and does not leak into the
growth signal, which reads cell density. The snapshot times and the per-cell snapshot index are
stored in ``uns['snapshot_times']`` / ``obs['snapshot']``.

The model never sees the times, so the design changes two things only: the density of cells along
the axis, and how the result can be scored.

**Metrics that survive a discrete truth** (``evaluate.snapshot_metrics``). Spearman against a step
function is tie-attenuated, so three quantities replace it:

| metric | definition | what it tests |
|---|---|---|
| ``order_accuracy`` | fraction of snapshot pairs $(i,j)$, $t_i<t_j$, with median $\tau_j > \tau_i$ | are the libraries in the right order? |
| ``separation`` | fraction of pairs with $\min\tau_j > \max\tau_i$ (no overlap of the two $\tau$ ranges) | a strict, tie-free ordering |
| ``gap_ratio`` | $(m_{i+1}-m_i)/\sum_j(m_{j+1}-m_j)$ divided by $(t_{i+1}-t_i)/(t_K-t_1)$, with $m_i$ the median $\tau$ of snapshot $i$ | is the **spacing** recovered, not just the order? |

``gap_ratio_within_2x`` is the fraction of consecutive gaps recovered within a factor 2. This is the
metric that separates a purely geometric method — which can only reproduce the ranks, hence an
evenly spaced $\tau$ — from a dynamical one, for which the amount of change over a gap carries its
length.
### 0.12 M6 — results on the two designs

3 networks × 3 runs × 2 designs × 6 methods = 108 fits, no failure. Full data in
``data/_benchmark/results.json``, the aggregated tables in ``data/_benchmark/aggregate.md``
(``check_benchmark.py --append`` regenerates them without refitting). Mean over the runs; ``sep``
and ``gap2x`` only exist for the snapshot design; ``dur`` is the total duration of the axis and ``s``
the wall-clock time of one fit.

| CN5 / **continuous** | AUPRC (sd) | rho | dur | s |
|---|---|---|---|---|
| palantir | 0.657 (0.194) | +0.396 | 76.7 | 17.2 |
| partial_ot | 0.605 (0.147) | +0.299 | 86.7 | 29.9 |
| tsp | 0.637 (0.019) | +0.341 | 90.0 | 6.6 |
| oracle | **0.989** (0.016) | +1.000 | 90.0 | 4.0 |
| palantir_pca | 0.485 (0.096) | +0.351 | 66.7 | 4.6 |
| random | 0.517 (0.122) | −0.002 | 90.0 | 5.2 |

| CN5 / **grid** | AUPRC (sd) | rho | order | sep | gap2x | dur | s |
|---|---|---|---|---|---|---|---|
| palantir | 0.664 (0.013) | +0.479 | 0.778 | 0.200 | 0.000 | 73.3 | 3.3 |
| partial_ot | 0.642 (0.063) | **+0.625** | **0.948** | 0.200 | 0.000 | 80.0 | 35.0 |
| tsp | 0.470 (0.046) | +0.484 | 0.778 | 0.200 | 0.000 | 90.0 | 8.0 |
| oracle | **0.867** (0.059) | +1.000 | 1.000 | **1.000** | **1.000** | 70.0 | 2.5 |
| palantir_pca | 0.640 (0.025) | +0.433 | 0.770 | 0.200 | 0.000 | 66.7 | 3.4 |
| random | 0.540 (0.084) | +0.003 | 0.556 | 0.000 | 0.000 | 90.0 | 5.4 |

| BN8 / **continuous** | AUPRC (sd) | rho | dur | s |
|---|---|---|---|---|
| palantir | 0.494 (0.056) | +0.702 | 160.0 | 3.5 |
| partial_ot | 0.404 (0.058) | +0.519 | 180.0 | 28.8 |
| tsp | 0.338 (0.004) | +0.687 | 180.0 | 6.7 |
| oracle | 0.595 (0.066) | +1.000 | 180.0 | 3.6 |
| palantir_pca | 0.375 (0.057) | +0.693 | 146.7 | 3.4 |
| random | 0.386 (0.095) | +0.007 | 180.0 | 4.2 |

| BN8 / **grid** | AUPRC (sd) | rho | order | sep | gap2x | dur | s |
|---|---|---|---|---|---|---|---|
| palantir | 0.393 (0.084) | +0.854 | 0.993 | 0.200 | 0.222 | 160.0 | 3.2 |
| partial_ot | **0.504** (0.074) | +0.725 | 0.756 | 0.200 | 0.000 | 153.3 | 23.7 |
| tsp | 0.384 (0.071) | +0.846 | 0.993 | 0.200 | **0.296** | 180.0 | 6.1 |
| oracle | 0.437 (0.020) | +1.000 | 1.000 | **1.000** | **1.000** | 140.0 | 4.4 |
| palantir_pca | **0.577** (0.058) | +0.822 | 1.000 | 0.200 | 0.259 | 140.0 | 3.3 |
| random | 0.495 (0.058) | +0.006 | 0.489 | 0.000 | 0.000 | 180.0 | 3.3 |

| FN8 / **continuous** | AUPRC (sd) | rho | dur | s |
|---|---|---|---|---|
| palantir | 0.341 (0.075) | +0.636 | 100.0 | 4.7 |
| partial_ot | 0.320 (0.043) | +0.301 | 95.8 | 24.5 |
| tsp | 0.421 (0.044) | +0.501 | 112.5 | 8.6 |
| oracle | **0.549** (0.031) | +1.000 | 112.5 | 5.1 |
| palantir_pca | 0.333 (0.026) | +0.568 | 100.0 | 4.3 |
| random | 0.264 (0.016) | −0.003 | 112.5 | 3.5 |

| FN8 / **grid** | AUPRC (sd) | rho | order | sep | gap2x | dur | s |
|---|---|---|---|---|---|---|---|
| palantir | 0.357 (0.028) | +0.772 | 0.874 | 0.200 | 0.000 | 95.8 | 3.8 |
| partial_ot | 0.362 (0.028) | +0.725 | 0.881 | 0.200 | 0.000 | 100.0 | 24.7 |
| tsp | 0.367 (0.027) | +0.806 | **0.911** | 0.200 | 0.000 | 112.5 | 8.1 |
| oracle | **0.696** (0.045) | +1.000 | 1.000 | **1.000** | **1.000** | 87.5 | 3.8 |
| palantir_pca | 0.356 (0.028) | +0.779 | 0.881 | 0.200 | 0.000 | 91.7 | 4.5 |
| random | 0.297 (0.008) | +0.001 | 0.570 | 0.000 | 0.000 | 112.5 | 4.1 |

**Findings**

1. **The snapshot design breaks the *spacing*, not the *order*.** ``order_accuracy`` is 0.76–0.99
   for every real method (0.49–0.57 for ``random``) — the libraries are put in the right order. But
   ``gap_ratio_within_2x`` is **0.00 for every method on CN5 and FN8**, and at best 0.30 (``tsp`` on
   BN8). Only the oracle, which is given the true times, reaches 1.00. Conclusion: **the relative
   spacing of an irregular design is not recovered by any of the three options.** The 3.0 → 6.9 h gap
   and the 52.9 → 71.7 h gap are not distinguishable from the inferred $\\tau$.
2. **``separation`` is exactly 0.200 for every method on every network** — 9 of the 45 pairs, i.e.
   only the pairs involving the first snapshot, whose cells are pinned at $\\tau=0$ as roots. So the
   $\\tau$ ranges of *any two non-root snapshots overlap*: the inferred $\\tau$ is smooth while the
   truth is a step function, and consecutive libraries sit below the resolution of the ordering. This
   is a property of the problem, not of a particular option.
3. **$\\rho$ is not comparable across designs — it *increases* on the harder one** (CN5 0.40 → 0.48,
   BN8 0.70 → 0.85, FN8 0.64 → 0.77). With 10 distinct true values the ties absorb the errors, so
   Spearman rewards getting the coarse order right. The snapshot metrics are the valid comparator;
   reporting $\\rho$ alone on a snapshot design would have suggested the opposite of the truth.
4. **The network metric is only interpretable on CN5 and, weakly, FN8.** The oracle wins clearly on
   CN5 (0.99 vs 0.66 continuous, 0.87 vs 0.66 grid) and on FN8 continuous (0.55 vs 0.42); on **BN8 the
   oracle *loses* to ``palantir_pca`` (0.437 vs 0.577) and to ``random`` (0.495) in the grid
   design** — with 8 genes AUPRC is a handful of edges and is dominated by noise. BN8's AUPRC must
   not be read as evidence for or against any method.
5. **The run-to-run spread swallows most of the difference between the options.** On CN5 continuous
   the sd is 0.19 (``palantir``), 0.15 (``partial_ot``), 0.02 (``tsp``), against means of
   0.657 / 0.605 / 0.637: the earlier single-run orderings (§0.7–§0.9, run 1 only) were inside this
   spread. Over 3 runs **no option is significantly better than another on the network**, and
   ``palantir`` — the default — is not beaten by either alternative. On $\\tau$ the picture is more
   stable: ``partial_ot`` is the best on CN5/grid (+0.625 and order 0.948), ``tsp``/``palantir`` on
   BN8 (~0.85 and 0.99), ``tsp`` on FN8/grid (+0.806, order 0.911).
6. **Runtime.** ``tsp`` is the cheapest option (6–9 s), ``partial_ot`` the most expensive
   (24–35 s), ``palantir`` in between — and on the grid design it is much faster than on the
   continuous one (17.2 → 3.3 s on CN5) because the cycle stops as soon as $\\tau$ is stable, which
   happens immediately when the cells are clumped.
7. **The duration stays within a factor 2 but is a prior, not a measurement.** CN5 73–90,
   FN8 88–112 against a true 96 h; BN8 153–180 against 96 h (1.6–1.9x), because $d_1 = 0.05$ gives
   $\\Delta\\tau = 20$ with 9 bins. This is §0.10 finding 1 restated at the benchmark level.
8. **Missing baseline.** ``stat`` (no time at all) is not available: ``CardamomOT_method='stat'`` is
   only read by ``fit_population_anchors.py`` / ``get_proliferation_rates.py``;
   ``NetworkModel.fit_network`` always runs ``loop_trajectories``, which needs ≥ 2 distinct times
   (``self.alpha[0] = .01`` on an empty array). ``random`` plays that role here; a faithful ``stat``
   needs its own code path.

**Verdict for M6**: the three options are *indistinguishable* on the network once the run-to-run
spread is taken into account, all of them beat ``random`` on $\\tau$ and on the order, and all of them
recover the ordering of an irregular design without recovering its spacing. The default
(``palantir``) stays; the choice between options should be made on runtime and on the structure of
the dataset (branching → ``partial_ot``), not on AUPRC.
### 0.1 Validated effect of fitting the mixture on bins

Oracle pseudotime (= true simulation time), 10 equal-width bins, root = first 2% of the true time,
run 1, seed 0 (`check_mixture_bins.py`, figures in `data/_mixture_bins/`):

| dataset | bins | mean error per bin (median / p90) | genes bimodal pooled → binned | genes with \|ρ\| ≥ 0.8 | mode-range ratio | cells relabelled |
|---|---|---|---|---|---|---|
| CN5 | 10 | 0.010 / 0.079 | 5 → 5 | 1 / 5 | ×1.22 | 8 % |
| BN8 | 10 | 0.080 / 0.921 | 8 → 8 | **5 / 8** (ρ ≈ 0.95) | ×1.16 | 5 % |
| FN8 | 10 | 0.104 / 0.485 | 7 → **8** | 4 / 8 (ρ ≈ 1.0) | ×1.25 | **26 %** |

What this already shows: (i) the per-bin masses reproduce each bin's mean (median error 1–10 %), i.e.
the mean-preservation constraint is effectively local; (ii) the mass of the highest mode becomes a
function of the bin with a clean monotone trend on the feed-forward and bifurcating networks;
(iii) the binned fit recovers a *bimodal* gene that the pooled fit collapses (FN8: 7 → 8 genes,
26 % of the cells relabelled), and always gives a wider — never narrower — mode range, closer to the
range observed along the axis.

Two caveats to keep in mind when reading these numbers:

* the p90 of the mean error reaches ~0.9 because the mean constraint is only enforced with weight
  `mean_forcing_em = 0.5` (a compromise with the likelihood, as in the temporal model), not exactly;
* **CN5 is an oscillator**: the ON-fraction of its genes is periodic along the time axis, so a single
  monotone pseudotime coordinate is ill-defined (ρ ≈ 0 for most genes is *expected*, not a failure).
  CN5 is therefore a degenerate test for τ (keep it for the network/`force_basins` behaviour, use
  BN8/FN8 to judge the ordering).

### 0.2 Validated ordering (Option 3, milestone M1)

Oracle roots (first 2% of the true time), 30 neighbours, 50 waypoints, seed 0, run 1
(`check_pseudotime_order.py`, figures in `data/_pseudotime_order/`):

| dataset | Palantir (plausibility, iter 1) | DPT baseline (scanpy, expression space) |
|---|---|---|
| CN5 | ρ = 0.37 (transient 0.60) | ρ = 0.04 (transient 0.48) |
| BN8 | **ρ = 0.67 (transient 0.70)** | ρ = 0.46 (transient 0.54) |
| FN8 | ρ = 0.70 (transient 0.79) | ρ = 0.73 (transient 0.47) |

*transient* = Spearman on the cells of the first half of the true time. The two metrics matter
because the simulations saturate: once a network has reached its steady state the cells are
indistinguishable, so **no** pseudotime can order them and the global ρ is capped (~0.7 here). On
the transient our ordering is clearly better than the diffusion pseudotime (0.70/0.79 vs 0.54/0.47),
and on BN8 it is better on both. CN5 is an oscillator (τ ill-defined, §0.1).

The `mechanistic` edge weight is *worse than the geometric one at iteration 0* (0.38/0.47/0.50):
with :math:`\Theta=0` its network term degenerates to the target mode alone while the bridge adds
noise. It is meant to be used once :math:`\Theta` is fitted (from iteration 1 of the cycle on).

### 0.3 Four findings that changed the design

1. **The kNN graph must be built in a continuous space.** Hard burst modes are a discrete lattice
   (one value per basin): with 2 modes and 8 genes, most cells are exact duplicates, the kNN graph
   splits into one component *per basin pattern* and up to 1 600 of the 2 000 cells have no path
   from the roots. Building it on the **expected** mode :math:`\sum_m \pi_{c,g,m} k_{s,g,m}`
   (`palantir_feature='posterior'`, the default) gives a connected graph (0 unreached) and the best ρ.
2. **Palantir's global weighted mean is biased when the waypoints are sparse**: the many far
   waypoints dominate and the cells near the root get a large τ. Default `consensus='min'` (the
   tightest perspective :math:`\min_w (d(c_0,w)+d(w,c))/2`, which corrects the graph shortcuts and
   is jitter-free); `'mean'` is available, restricted to the nearest waypoints.
3. Unreached cells are now **relaxed** from their reached neighbours (``d_j = min_i d_i + w_ij``)
   instead of all receiving the same value, which used to destroy the ranking.
4. **The reported metric must be the transient one**: the global Spearman is capped by saturation,
   so `evaluate.pseudotime_metrics` also returns `spearman_transient`, and the validation script
   compares against a diffusion pseudotime (scanpy `dpt`) baseline.

---

## 1. Goal and non-goals

### Goal
Given one static snapshot of counts $x_{c,g}$ and a set of root cells $\mathcal{C}_0$, jointly infer

1. a per-cell pseudotime $\tau_c$ (and its binning),
2. mechanistic protein trajectories $P_c(\tau)$ under the bursty model,
3. the directed signed network $\Theta$ (basal + interactions), in the *same* parameterisation as
   CardamomOT (so that `refine_network_degradations`, simulations, reports and every downstream
   tool keep working),

by replacing only the **trajectory/ordering step** of the existing EM cycle.

### Non-goals (first version)
- No new mixture model: the NB mixture is fitted **pooled on all cells** (`fit_mixture` with a
  single time value), as in `CardamomOT_method='stat'`.
- No modification of the degradation/simulation stage: `refine_network_degradations`,
  `simulate_network`, `report` are reused as-is (with `times_data` = pseudotime bins).
- No multi-sample-specific work: pseudotime is per sample (`obs['dataset_id']`), but v1 is
  validated per sample; joint multi-sample ordering is out of scope.
- No new input format: an `AnnData` with counts in `X`, a root annotation, no `obs['time']`.

---

## 2. What the base implementation actually does (the hook points)

### 2.1 The cycle, as implemented

`NetworkModel.loop_trajectories` (`base.py:1973`) is a fixed-point iteration over a **time-bin
skeleton**:

- `times` ($T$ bins), `N_full[s]` trajectory slots per bin, `y_prot` of shape `(T·N_tot, G)`, with
  `vect_t_sim = np.repeat(times, N_tot)`;
- `observed` ($S\times T$ bool), `valid` (flat), `pairs = trajectory_pairs(valid, T)` (consecutive
  observed bins per sample), `real_cell_batches[s][t]` (real target cells of bin $t+1$).

One iteration:

| step | code | what it does |
|---|---|---|
| ordering | — | *does not exist*: bin membership is given by `obs['time']` |
| **E** | `estimate_trajectories_given_model` (`base.py:1371`) | per consecutive bin pair and per batch: cost `my_otdistance(...)` (`inference/trajectory.py:392`) → `_solve_ot` (entropic/UOT Sinkhorn) → one target sampled per source → end proteins by `find_next_prot` / `find_next_prot_dilution` / `prot_bridge_end` → writes `y_prot, y_kon, y_rna` (and `sim_real_idx`, `to_keep_for_update`) |
| **M** | `fit_theta` → `_fit_theta_averaged` → `inference_network_multi` (`inference/network.py:886`) | per-gene L-BFGS on the cross-entropy between $\sigma(\theta_0+\Theta P)$ and `sigmoid_target(kon)` |
| $\alpha$ | `inference_alpha` (`trajectory.py`) | switching fraction of each gene inside a step |
| basins | inline block, `base.py:2380-2440` | one EMD per gene between mixture posteriors and modes, network-weighted by `to_keep_for_update` |
| growth | `_growth_log_mass` (`base.py:1896`), called with `growth_only=True` | WOT-style net growth per bin pair → `R_opt` |

**Key observation** (this is what makes the three options cheap to implement): everything after
binning is *time-label agnostic*. `inference_network_multi` only sees `y_prot`, `y_kon`,
`y_samples`; the basin block only sees `vect_t == t_i` (i.e. *bin* membership). Therefore a
pseudotime method only has to produce, at each iteration, the arrays
`(times, N_full, real_cell_batches, observed, valid, vect_t)`. The E/M/$\alpha$/basin code is
reused verbatim.

### 2.2 Where the growth / proliferation part integrates (your question)

The growth machinery is a **5-stage pipeline**, and it is *entirely downstream of the OT
couplings*, so it transfers to pseudotime with one condition (see §6):

1. **Growth pass inside the loop** — `estimate_trajectories_given_model(..., growth_only=True)`
   re-solves, on the *same* cost matrix as the trajectory OT, an unbalanced Sinkhorn where the
   source marginal is relaxed (prior $\exp(R_{\text{prior}}\Delta t)$, weight
   `growth_reg_source`) and the target marginal is almost hard (`reg_m = [·, 1e3]`), iterating
   `n_growth_iter` times with $\mu \leftarrow$ row marginals (`_growth_log_mass`).
   It returns the log mass gain of each source state; dividing by $\Delta t$ gives
   `R_opt` (net proliferation rate per unit time) of every trajectory state. The mass that has to
   be *created* between bin $k$ and bin $k+1$ is exactly the observed **density change of cells
   along the bins**, transported through the coupling — this is the only place where population
   size enters.
2. **`fit_birth_death_to_growth`** splits $R$ into birth/death consistent with the cell-level
   priors (`split_net_change`), so that a non-dividing cell stays non-dividing.
3. **`refine_network_degradations`** retrains `prolif_network` (the `ProliferationMLP`, inputs =
   proteins + stimulus, two heads birth/death) on `(self.prot, R_target)` along the paths, and
   simultaneously refits $d_1$ (and the ratio $d_0/d_1$, `infer_ratio_d0_d1_unitary`).
4. **Simulation** (`simulate_trajectories_unitary`): $R$ is integrated along each simulated path
   (quadrature over `n_growth_nodes`), then a multinomial resampling inside each sample implements
   branching; `log_population` is recorded (relative to $t_0$).
5. Optionally, **back into the E-step**: if per-cell net rates exist (`obs['proliferation_net_rate']`),
   `_traj_marginals` reweights source/target marginals by $\exp(\pm R\,\Delta t/2)$, i.e. the OT
   itself accounts for growth (midpoint rule, documented in `docs/advanced.md`).

So the "growth" of the model is not an extra block bolted on the end: it is (i) a second,
unbalanced reading of the same couplings, (ii) an MLP trained on the states, (iii) a resampling
during simulation. In pseudotime mode, stages 1–5 are reused unchanged **provided the
pseudotime axis is an absolute coordinate** (§6.2) and the bins are **equal-width** (§6.3).

### 2.3 Reusable primitives (imported, never copied)

| need | primitive |
|---|---|
| mixture / modes / posteriors | `NetworkModel.fit_mixture`, `_mixture_terms()` |
| network forward | `kon_ref_vector`, `base_kon_vector`, `sigmoid_target`, `sigmoid_amplitudes`, `gene_weights`, `count_errors` |
| OT cost | `my_otdistance` (numba) |
| OT solve | `NetworkModel._solve_ot`, `_traj_marginals`, `_uot_ramp`, `_transition_log_penalty` |
| trajectories | `NetworkModel.estimate_trajectories_given_model`, `estimate_trajectories` |
| switching fractions | `inference_alpha` / `inference_alpha_1thread`, `birth_alpha` |
| M-step | `NetworkModel._fit_theta_averaged`, `_n_params_per_target`, `inference_network_multi` |
| bridges / states | `find_next_prot`, `find_next_prot_dilution`, `prot_bridge_end`, `adaptive_shrinkage_init` |
| growth & sim | `_growth_log_mass`, `fit_birth_death_to_growth`, `refine_network_degradations`, `simulate_network` |

Only **one** block must be duplicated (the per-gene basin/mode EMD, ~60 lines of
`base.py:2380-2440`), because it is inline in `loop_trajectories` and `base.py` must stay
untouched. A unit test on oracle bins asserts identical output (§8, M0).

---

## 3. Architecture

```
CardamomOT/pseudotime/
├── __init__.py                 # exports the 3 models + helpers
├── base_pseudotime.py          # PseudotimeNetworkModel(NetworkModel): params, prep, loop, growth, duration
├── binning.py                  # tau -> bins / masses / real_cell_batches / N_full
├── costs.py                    # mechanistic edge cost (bridge over dtau) shared by the 3 options
├── options/
│   ├── __init__.py             # OPTIONS = {'palantir': ..., 'partial_ot': ..., 'tsp': ...}
│   ├── palantir_like.py        # Option 3  (implemented first)
│   ├── partial_ot.py           # Option 1
│   └── tsp.py                  # Option 2
├── evaluate.py                 # GRN AUPRC, tau vs true time, runtime
├── infer_pseudotime.py         # CLI: data.h5ad + roots -> run -> results
├── simulate_pseudotime_data.py # (done, unchanged)
└── README.md                   # (done, extended with the pseudotime models)
```

### 3.1 Class design

```python
# base_pseudotime.py
class PseudotimeNetworkModel(NetworkModel):
    """CardamomOT with the time labels replaced by an inferred pseudotime."""

    # --- new parameters (defaults; all settable through Data/CardamomOT_inputs.xlsx) -------------
    pseudotime_method = 'palantir'          # 'palantir' | 'partial_ot' | 'tsp'
    pseudotime_bins = 10                    # T (number of pseudotime bins)
    pseudotime_bin_rule = 'equal_width'     # 'equal_width' (keeps the density -> growth) | 'equal_count'
    pseudotime_duration = None               # T_tot in model time units; None = estimated (gauge/grid)
    pseudotime_duration_mode = 'grid'        # 'grid' | 'gauge' | 'fixed'
    pseudotime_dt_grid = np.array([0.05, .1, .25, .5, 1., 2.]) / d1_prior   # candidate dtau
    pseudotime_update_every = 1              # recompute tau every n EM iterations
    pseudotime_root_key = 'is_root'          # obs bool
    pseudotime_root_group_key = 'root_group'  # optional obs categorical (several roots)
    pseudotime_fixed_tau = False             # DEBUG ONLY: freeze tau (oracle bins) to validate the wiring
    pseudotime_growth = True                 # run the growth pass between consecutive bins

    # --- state ----------------------------------------------------------------------------------
    tau: (N,) float          # pseudotime of each real cell, normalised to [0, 1]
    bin_of_cell: (N,) int    # 0 = roots, 1..T-1 = the rest
    bin_times: (T,) float    # k * dtau, model time units (self.times_data of the loop)
    n_bins_used: int

    # --- API ------------------------------------------------------------------------------------
    def fit_pseudotime(self, data, gene_names=None, verb=True): ...   # main entry point
    def prepare_pseudotime(self, data): ...        # counts, roots, pooled mixture
    def _initial_tau(self): ...                    # from the option with Theta = 0 (mixture only)
    def _bin_cells(self, tau): ...                 # binning.py -> the loop scaffolding arrays
    def update_pseudotime(self): ...               # dispatch to the option (Theta-dependent)
    def loop_pseudotime(self): ...                 # the modified cycle (§4.4)
    def _update_duration(self, stage): ...         # grid search / gauge fixing (§4.3)
```

Each option is a small class with **one public method** and no state of its own beyond
hyperparameters:

```python
class PalantirLike:
    def update(self, model: PseudotimeNetworkModel, M, P, dtau) -> np.ndarray:  # returns tau (N,)
```

so that options are unit-testable on toy inputs (a cost matrix and a monotone τ) without running
the whole model.

### 3.2 Entry point

```python
# infer_pseudotime.py
python -m CardamomOT.pseudotime.infer_pseudotime \
    --data CardamomOT/pseudotime/data/CN5/Data/data_1.h5ad \
    --root-quantile 0.02            # or --root-key is_root / --root-group-key root_group
    --method palantir --bins 10 --outdir results_pseudotime/CN5/r1
```
which writes `tau.npy`, `bin_of_cell.npy`, `inter.npy`, `basal.npy`, `dsim.npy` (the usual
`cardamomOT/` artefacts), plus `metrics.json` from `evaluate.py`.

---

## 4. Shared substrate (common to the three options)

### 4.1 Data preparation

1. `data_rna = self._parse_input(data)`; everything is one sample (`vect_t = 0`), since no time is
   known. If `obs['dataset_id']` has $\ge 2$ values, `fit_mixture_samples` is used unchanged; the
   pseudotime is then computed **per sample** (`vect_samples_id` kept).
2. Roots: `is_root` (bool) or `root_group` (categorical, in which case roots of group *r* all get
   $\tau = 0$; a multi-root problem is not distinguished in v1 but the groups are propagated and
   used by Option 1's frontier seeding).
3. Pooled mixture: `self.fit_mixture(data, ...)` with all cells at one time →
   `self.modes` ($k^{\text{mix}}_c$), `self.proba_init`, `self.basins`, and
   `ks, s1, k1 = self._mixture_terms()`.
   With `pseudotime_refit_mixture=True` (§10.6), the mixture is re-fitted on the current bins
   instead (a temporary `obs['time'] = bin_times[bin_of_cell]` is enough: `fit_mixture` is reused
   unchanged), every `pseudotime_refit_every` iterations, with `force_basins` ramped 0.3 → 1.0.
4. Initial proteins: `adaptive_shrinkage_init(x_c · s1, modes_c)` (the same equilibrium
   initialisation as the temporal loop, `base.py:1440`).

### 4.2 τ → bins (`binning.py`)

```python
def bin_from_tau(tau, n_bins, roots, rule='equal_width', min_cells=50, samples=None):
    """returns (bin_of_cell, bin_times, N_full, real_cell_batches, observed, valid)"""
```

Rules:
- **bin 0 = the root cells only** (τ = 0). Non-root cells with τ below the first edge are moved to
  bin 1: bin 0 is a *source* bin and is never a target, so leaving non-roots there would make them
  permanently "unreached" (`to_keep_for_update = 0`).
- `equal_width`: edges on $[\tau_{\min}, \tau_{\max}]$ → cells per bin ∝ density (**required for
  the growth pass**, §6.3).
- `equal_count`: quantile edges → equal cells per bin (growth then reads $R\approx0$: available as
  an ablation, `pseudotime_growth` is forced off).
- Merge consecutive bins with fewer than `min_cells` cells (avoid degenerate OT), and cap the
  effective $T$ by the number of distinct τ values.
- `real_cell_batches[s][k]`: the actual cells of bin $k$ (tiled/stratified exactly as
  `base.py:2185-2205`, reusing `stratified_order`), so that unequal bin sizes — and therefore
  density/growth — are handled by the *existing* machinery.
- `times = k * dtau`, `dtau = T_tot / (T - 1)`; `observed = ones((S, T))`, `valid = ones(T·N_tot)`.

### 4.3 The time unit: free total duration $T_{\text{tot}}$ (identifiability)

In the deterministic limit of the trajectory step, $\Delta\tau$ enters **only** through
$d_1\,\Delta\tau$ (the relaxation $P \leftarrow u + (P-u)e^{-d_1\Delta\tau}$), $\alpha$ being the
dimensionless switching fraction; $d_0$ is not used during the inference (only in
`refine_network_degradations`, through the ratio prior). Hence:

$$(d_1,\ \Delta\tau)\ \text{is a gauge}:\ \text{only } \lambda := d_1\,\Delta\tau \text{ matters.}$$

Fix the gauge by anchoring $d_1$ to its prior (`halflife_db`) and letting $T_{\text{tot}}$ absorb the
scale:

1. **Grid** (`pseudotime_duration_mode='grid'`, first 2–3 EM iterations): run the cycle for
   $\Delta\tau \in \{0.05,0.1,0.25,0.5,1,2\}/d_{1,\text{prior}}$ (≈ 3 h … 130 h with
   $d_1=\log 2/46$), keep the $\Delta\tau$ minimising the model error
   `_count_errors_per_sample`; warm-start the cycles from each other (τ of the previous candidate).
2. **Gauge** (`'gauge'`, afterwards, cheap): estimate the optimal single $\lambda$ from the current
   trajectories by 1-D least squares
   $$\lambda^\* = \arg\min_\lambda \sum_{c,k}\big\|P_{k+1}^c - [u_k^c + (P_k^c-u_k^c)e^{-\lambda}]\big\|^2,$$
   then $T_{\text{tot}} \leftarrow (T-1)\,\lambda^\*/d_{1,\text{prior}}$ (with damping,
   `|log(update)| ≤ 0.5` per iteration). Reported at every iteration so the user sees the
   inferred time scale in hours.
3. `'fixed'` + `pseudotime_duration` for tests.

### 4.4 The modified cycle (`loop_pseudotime`)

```mermaid
flowchart TD
    A[prepare: pooled mixture, roots, tau from option] --> B[bin cells -> scaffolding]
    B --> C[E-step: estimate_trajectories_given_model between consecutive bins]
    C --> D[M-step: inference_network_multi]
    D --> E[alpha update: inference_alpha]
    E --> F[basin / mode update: EMD per gene]
    F --> G[update tau: option.update(Theta, P, dtau)]
    G --> H[gauge / grid update of T_tot]
    H --> I{stopping: error, delta tau, max_iter}
    I -->|no| B
    I -->|yes| J[growth pass: R_opt, fit_birth_death_to_growth]
    J --> K[refine_network_degradations, simulate_network, report]
```

Differences with `loop_trajectories`, all of them at the *ordering* level:

1. `vect_t` (real cells) = `bin_times[bin_of_cell]` and is **recomputed at each iteration**
   (cells move between bins). Everything derived from it (`observed`, `pairs`, `vect_t_sim`,
   `strata`, `t0_real`, `real_cell_batches`) is rebuilt each iteration — this is why a dedicated
   loop is needed rather than a monkey-patch of the caller.
2. Batches of *real* cells per bin: as in base (tiled stratified shuffles), sizes free.
3. `y_prot/y_kon/y_rna/y_alpha` are **carried across iterations**, reindexed when a cell changes
   bin: the trajectory slot of a cell follows the cell (`sim_real_idx` → new slot). Warm start of
   the E-step is essential: without it each iteration restarts trajectories from the modes.
4. Stopping: the base criteria plus
   $\text{Spearman}(\tau^{n}, \tau^{n-1}) \ge$ `pseudotime_stable` (default 0.999) over
   `pseudotime_count_max` (default `count_max`) consecutive iterations.
5. The growth pass and `R_opt` reuse `_growth_log_mass` and `estimate_trajectories_given_model(growth_only=True)`
   with the *final* bins (§6).

`pseudotime_fixed_tau=True` replaces steps 1 and 7 by the true-time bins (test only).

---

## 5. The three options

Common notation: $k^{\text{mix}}_c \in \mathbb{R}^G$ = burst modes of cell $c$ (from the mixture),
$P_c$ = current protein state, $\Theta$ = current network, $d_1$ = protein degradation,
$\Delta\tau$ = bin width, $\sigma$ = logistic, $k_{on}(P) = k_{low} + (k_{high}-k_{low})\sigma(\theta_0+\Theta P)$,
$\text{loss}$ = the cross-entropy of `count_errors` between $k_{on}$ and the sigmoid target of the modes.

### 5.1 Option 3 — Mechanistic Palantir (`options/palantir_like.py`) — **implemented first**

No OT at all: τ is a shortest-path distance in a GRN-weighted kNN graph, refined by waypoints.

```python
class PalantirLike:
    n_neighbors = 20; n_waypoints = 50
    edge_weight = 'plausibility'   # 'plausibility' (graph building) | 'mechanistic' (Dijkstra)
    alpha_geom = 0.5               # weight of the geometric term
    n_consensus = 3                # perspective-matrix iterations per EM iteration
    n_pcs = 0                      # 0 = raw modes; >0 = PCA of the modes before the kNN (large G)
```

Algorithm (per EM iteration, given $\Theta$):

1. **Graph space.** $k^{\text{mix}}_c$ (optionally PCA-reduced), restricted to genes with $\ge2$
   modes (`ks`). kNN graph with `sklearn.neighbors.NearestNeighbors`, edges $c\to c'$ (directed,
   as in `plan.tex`).
2. **Edge weights** (two variants, both from `plan.tex` §`mech-palantir`):
   - *plausibility* (independent of $P$, so it can build the graph):
     $$d_\Theta(c,c') = \alpha\Big\|\log\tfrac{1+M_{c'}}{1+M_c}\Big\|_2
       + \sum_g \begin{cases}0 & k^{\text{mix}}_{c',g}=k^{\text{mix}}_{c,g}\\
       1-\sigma(\theta_{0,g}+\Theta_g k^{\text{mix}}_c) & k^{\text{mix}}_{c',g} > k^{\text{mix}}_{c,g}\\
       \sigma(\theta_{0,g}+\Theta_g k^{\text{mix}}_{c}) & k^{\text{mix}}_{c',g} < k^{\text{mix}}_{c,g}\end{cases}$$
   - *mechanistic* (full cost, needs the PDMP bridge from $P_c$ over $\Delta\tau$):
     $$w_\Theta(c,c') = \text{loss}\big(k_{on}(P_{c\to c'};\Theta),\,k^{\text{mix}}_{c'}\big)
       + \alpha\|k^{\text{mix}}_{c'}-k^{\text{mix}}_c\|_2 ,$$
     $P_{c\to c'}$ from `find_next_prot(_dilution)` / `prot_bridge_end` — **the same functions the
     E-step uses**, wrapped in `costs.py:bridge_cost(...)`, vectorised over the $k$ neighbours of a
     cell (numba-jitted, reusing the `my_otdistance` inner kernel).
3. **Dijkstra** from the root set (`scipy.sparse.csgraph.dijkstra`, `indices=roots`, or a virtual
   super-root with 0-cost edges) → $\tau^{(0)}_c$.
4. **Waypoints** by max-min sampling in modes space ($W$ = `n_waypoints`), distances $d(c,w)$ by
   Dijkstra; **consensus** (perspective matrix, iterated `n_consensus` times):
   $$\tau_c = \sum_{w}\hat w_c\,[d(c_0,w)+d(w,c)],\qquad \hat w_c \propto 1/d(c,w),$$
   with the same cleaning as Palantir (discard waypoints out of order, cap the maximum step).
5. Normalise: $\tau \leftarrow \tau/\max\tau$ (dividing by the max keeps the *relative* distances,
   hence the density, hence the growth signal — do **not** rank-normalise).
6. *(optional, for BN8)* terminal states + absorbing Markov chain → branch probabilities; used by
   `evaluate.py` (fate accuracy), not by the EM.

Complexity per EM iteration: $O(NkG)$ for the weights + $W+1$ Dijkstras on a sparse graph
($\sim$0.1–1 s for $N=2000$, $G=8$; dominated by the $W$ Dijkstras, so $\tau^{(0)}$ is cached and
only recomputed when $\Theta$ changed by more than `palantir_tol_theta`).
Pitfalls: a graph built on modes alone ignores within-mode dynamics (hence the mechanistic
weights); a disconnected graph (no path from a root) → cells are attached with the geometric
distance and flagged `unreached`.

### 5.2 Option 1 — Partial OT from the initial population (`options/partial_ot.py`)

The flow-based analogue of CardamomOT: waves of transport from a frontier.

```python
class PartialOT:
    mass_per_step = 0.25        # beta_k: mass transported per wave (fraction of the frontier)
    max_steps = None            # K; None = until every cell is reached
    reg = 0.05                  # entropic regularisation (relative to the median cost, as in base)
    alpha_geom = 0.2            # geometric term of the cost
    restrict = 'knn'            # candidate pool: 'knn' | 'all'
    knn_candidates = 10         # neighbours of the frontier in modes space
    phantom_cost = None         # None: partial OT with a Lagrangian mass penalty (see below)
    reuse_tau = True            # warm start: only cells within +-1 step of the frontier are re-solved
```

Per EM iteration, given $\Theta$:

1. $\mathcal{C}_0$ = roots, $\tau=0$, $P_c$ from the current proteins; $\mathcal{U}_0$ = the rest.
2. For $k=0,\dots$:
   1. **Candidates** $\mathcal{T}_k\subset\mathcal{U}_k$ (`restrict='knn'`: the union of the
      `knn_candidates` nearest unexplained cells, in modes space, of each frontier cell; `'all'`:
      the whole pool).
   2. **Cost** $C_k(c,c')$ = mechanistic cost of `plan.tex` §`sec:partialOT`:
      $C_k = \text{loss}\big(k_{on}(P_{c\to c'}),k^{\text{mix}}_{c'}\big)+\alpha\,\text{geom}(k^{\text{mix}}_c,k^{\text{mix}}_{c'})$,
      with $P_{c\to c'}$ the PDMP bridge over $[\tau_c,\tau_c+\Delta\tau]$ (`costs.bridge_cost`,
      same kernel as `my_otdistance` restricted to the network + mRNA terms).
   3. **Partial OT**: $\gamma_k$ minimising $\langle C_k,\gamma\rangle+\varepsilon\,\text{KL}(
      \gamma\,\|\,\mu\otimes\nu)$ with relaxed marginals $\sum_{c'}\gamma\le\mu$, $\sum_c\gamma\le\nu$
      and $\sum\gamma = \beta_k$. Implemented with `ot.partial.entropic_partial_wasserstein`, or —
      for the "phantom component" of `plan.tex` — an extra target column of constant cost
      $\lambda$ (`ot.partial.partial_wasserstein_lagrange`); a cell whose mass lands on the phantom
      has no descendant this wave (it waits).
   4. **Assign**: $\tau_{c'} = \sum_c \gamma_k(c,c')\,(\tau_c+\Delta\tau)/\sum_c\gamma_k(c,c')$,
      $P_{c'} = \sum_c \gamma_k(c,c')\,P_{c\to c'}/\sum_c\gamma_k$;
      $\mathcal{C}_{k+1}$ = cells with mass $>$ `min_mass` (default $10^{-3}$); stop when
      $\mathcal{U}$ is empty or $k=K$.
3. $\tau \leftarrow$ (step index + within-step offset) → absolute coordinate, normalised by $\max\tau$.

Complexity: $K$ partial-OT solves per EM iteration, each $|\mathcal{C}_k|\times|\mathcal{T}_k|$;
with `restrict='knn'` and a capped frontier ($\le$ `batch_size_traj`), comparable to one temporal
OC solve. Pitfalls: mass leaking to the phantom (all cells stuck at the frontier) → tune $\beta_k$;
a frontier that splinters into two branches is handled naturally (several descendants per source);
branching + `equal_width` bins give the two BN8 attractors as two late bins.

### 5.3 Option 2 — Path-dependent TSP (`options/tsp.py`)

```python
class PathTSP:
    init = 'palantir'      # 'palantir' | 'partial_ot' (TSP alone has no chance from scratch)
    n_moves = 2000          # 2-opt moves per EM iteration
    window = 50             # windowed 2-opt: reverse segments of length <= window
    use_path_length = True  # tau = cumulative cost / total (absolute) instead of the tour rank
```

1. Tour = cells sorted by the τ of the previous iteration (warm start; first iteration = the
   option selected by `init`).
2. Cost of a step: $\text{dist}(c_k,c_{k+1}) = \text{loss}\big(k_{on}(P_{c_k\to c_{k+1}}),k^{\text{mix}}_{c_{k+1}}\big)$,
   with $P_{c_k}$ propagated sequentially along the tour from the roots.
3. 2-opt: reverse $(c_{a+1},\dots,c_b)$ if it lowers
   $\ell_{\text{TSP}}=\sum_k \text{dist}(c_k,c_{k+1})$; because the cost is path-dependent, the
   proteins and the distances are re-propagated from $c_a$ (the re-propagation is linear in the
   segment length, which is why the moves are windowed).
4. $\tau$ = cumulative path length normalised by the total (keeps a pseudo-metric, unlike the
   rank; used for the bins and the growth).
5. Best-improvement scan of the $O(N\cdot\text{window})$ candidate moves, subsampled to
   `n_moves` per iteration (deterministic seeding via `task_seed`).

Complexity: $O(\text{window}\cdot N)$ bridge evaluations per move with the vectorised
`costs.bridge_cost`; budgeted by `n_moves` so that it stays within ~1 E-step cost.
Pitfalls (documented in `plan.tex`): asymmetric, non-metric, path-dependent ⇒ no convergence
guarantee; a single tour **cannot represent the two BN8 branches** — v1 runs it per branch after
the fate assignment of Option 3, and reports this as a known limitation.

---

## 6. Growth / proliferation in pseudotime mode

### 6.1 Reuse
`pseudotime_growth=True` calls, at the end of the cycle and unchanged:
`estimate_trajectories_given_model(growth_only=True)` per batch → `R_opt` →
`fit_birth_death_to_growth` → `refine_network_degradations` (MLP + $d_1$/$d_0$ refit) →
`simulate_network`. `self.times_data` is set to `bin_times`, so every downstream consumer
(`_transfer_weights`, `simulate_trajectories_unitary`, `report`) works without modification.

### 6.2 The pseudotime axis must be *absolute*
Growth is read from the **cell density along the axis**. Therefore τ must be a rescaled
*distance/step count* (Palantir consensus distance, partial-OT step count, TSP cumulative path
length) and **not** a rank/quantile. `tau / max(tau)` is fine; `rank(tau)/(N-1)` is not (it
destroys the density and forces $R\approx0$).

### 6.3 Equal-width bins
With `pseudotime_bin_rule='equal_count'` the number of cells per bin is constant by construction,
so the growth pass can only return $R\approx 0$. `pseudotime_growth=True` therefore implies
`equal_width` (and a warning otherwise). With equal-width bins, $N_k$ cells in bin $k$ measures the
net expansion between bins — exactly the Waddington-OT growth assumption.

### 6.4 Feedback of $R$ into the E-step (optional, `pseudotime_prolif_infer=True`)
After the first growth pass, fill `model._prolif_net_rate` per real cell (R of its bin,
interpolated) so that `_traj_marginals` reweights the OT marginals with $\exp(\pm R\Delta\tau/2)$
from the next iteration: the OT then transports *net of growth*, as in the temporal model.

---

## 7. Evaluation (`evaluate.py`)

```python
def evaluate(model, data, true_inter=None, true_time=None, outdir=None) -> dict
```

| metric | definition |
|---|---|
| `auprc`, `auroc` | `inter` (off-diagonal, stimulus row separated) vs the binarised ground truth (`uns['true_network']`, `True/inter_{r}.npy`) |
| `topk_precision` | precision of the $|\text{edges}|$ strongest inferred edges |
| `sign_accuracy` | sign agreement on the true edges |
| `spearman_tau` | Spearman between `tau` and the true simulation time (the ground truth of the whole exercise) |
| `kendall_within_branch` | BN8 only: τ order inside each attractor |
| `fate_accuracy` | BN8 only, Option 3: branch probability vs true attractor |
| `runtime`, `n_iter`, `dtau_inferred` | cost of the run, number of EM iterations, inferred total duration |

Baselines computed by the same module: **oracle τ** (true time binned → must match the temporal
model), **no-time** (`CardamomOT_method='stat'`), **random τ**, **Palantir on PCA** (non-mechanistic)
+ standard CardamomOT.

---

## 8. Milestones and test gates

| # | deliverable | gate (must pass before the next) |
|---|---|---|
| **M0** | `binning.py`, `base_pseudotime.py` skeleton, `evaluate.py`, `infer_pseudotime.py` | `pseudotime_fixed_tau=True` on `CN5/Data/data_1.h5ad` (bins = true time) reproduces the temporal CardamomOT `inter` (relative difference $<10^{-2}$); the copied basin block equals the base one on random inputs. **Done for `binning.py` + the mixture-on-bins part (§0)**; the loop and `evaluate.py` remain. |
| **M1** | `options/palantir_like.py` (plausibility weights, then mechanistic), `costs.py` | toy test: on a monotone 1-D manifold, $\tau$ is monotone (Spearman $>0.99$); on CN5, `spearman_tau` $>0.8$ and `auprc` $>$ the Palantir-on-PCA baseline. **Done for the ordering itself (§0.2)**: 14 unit tests pass, ρ beats the DPT baseline on BN8 and on the transient of every dataset; CN5 (oscillator) and the mechanistic weights at $\Theta=0$ do not reach the gate, as expected — the AUPRC needs the EM loop. |
| **M2** | `options/partial_ot.py` | CN5: no cell left behind (all `U` empty), `spearman_tau` $\ge$ M1; mass accounting test (mass conserved, phantom usage $<10\%$) |
| **M3** | duration grid + gauge (§4.3) | inferred $T_{\text{tot}}$ within a factor 2 of the simulated 96 h on CN5/BN8/FN8 |
| **M4** | `options/tsp.py` | 2-opt strictly decreases $\ell_{\text{TSP}}$; CN5 `spearman_tau` $\ge$ M1 |
| **M5** | growth + `refine_network_degradations` + simulations in pseudotime mode | $R$ estimated on CN5 > 0 with the right sign; simulated mRNA marginals within 2× of the data |
| **M6** | full benchmark, $N=5$ runs × 3 networks, report | table of AUPRC/τ-Spearman/runtime for the 3 options + 4 baselines |

Unit tests live in `tests/test_pseudotime.py` (plain `pytest`, no new dependency) and are driven by
`simulate_pseudotime_data.py --cells 200 --runs 1` for speed.

---

## 9. Default parameters (first version)

| parameter | default | note |
|---|---|---|
| `pseudotime_method` | `'palantir'` | |
| `pseudotime_bins` | `10` | same order as the simulated time points |
| `pseudotime_bin_rule` | `'equal_width'` | required by the growth pass |
| `pseudotime_duration_mode` | `'grid'` | then `'gauge'` |
| `pseudotime_growth` | `True` | per your answer; disabled automatically with `equal_count` |
| `pseudotime_refit_mixture` | `True` | mixture re-fitted on bins every `pseudotime_refit_every = 5` iterations (§10.6) |
| `force_basins` (ramp) | 0.3 → 1.0 | over `min_n_loops` iterations (vs 1.0 in the temporal model) |
| `palantir_n_neighbors` / `n_waypoints` | 20 / 50 | |
| `partial_ot_mass_per_step` | 0.25 | |
| `tsp_n_moves` / `tsp_window` | 2000 / 50 | |
| `min_n_loops` / `max_iter` / `count_max` | inherited | unchanged |
| `batch_size_traj` / `batch_size_traj_exp` | inherited | unchanged |

---

## 10. Risks and open questions

1. **τ identifiability vs $\Delta\tau$** — the gauge is only valid if $d_1$ is indeed fixed by its
   prior; if `refine_network_degradations` drags $d_1$ far from it, $T_{\text{tot}}$ and $d_1$ drift
   together. *Mitigation*: report both, and check the inferred $d_1$ against the true value in the
   simulations (we know it: 0.1/0.05/0.08 h⁻¹ for CN5/BN8/FN8).
2. **Growth from a pseudotime axis** — density changes along τ mix true growth with τ compression
   (a slow region looks over-populated). Unavoidable without absolute times; we will report the
   estimated $R$ against the true net growth (all our simulated networks are non-proliferating, so
   the target is $R\approx0$ — a good sanity check but a weak test; a proliferating simulation
   would be needed for a real validation, out of scope here).
3. **Branching** — Options 1 and 3 handle it; Option 2 does not (single tour). BN8 is the
   discriminating dataset: expect Option 2 to fail on it, and say so.
4. **Duplicated basin block** (~60 lines) — accepted to keep `base.py` untouched; guarded by the
   M0 equality test. A later refactor could extract it.
5. **Runtime** — Options 1/2 add a cost per EM iteration of the order of the E-step; a full
   $N=2000$/$G=8$ run should stay in the minutes. Option 2's `n_moves` is the knob.
6. **Mixture on bins vs pooled mixture** — *analysis done, decision = flag with a schedule*.
   `vect_t` is not cosmetic in `NegativeBinomialMixtureEM`: it makes (i) `_assign_basins` run one
   EMD per group with target masses $\nu_t$ from `_compute_nu_with_temporal_constraint` (balances
   the likelihood proportions with the mean constraint $\sum_k \nu_k k_{s,k}/c \approx \bar x_t$),
   returned as `self.pi_init[g][t]`; (ii) `_apply_temporal_constraints` calibrate the extreme modes
   on the lowest/highest-mean groups, so the modes span the range observed *along the axis*; (iii)
   `_per_cell_log_masses` vary per group instead of being a constant vector.
   **Gains of fitting on bins**: the burst-frequency distribution becomes a function of pseudotime
   $\nu_g(k)$, and it is the target marginal of the basin EMD (imposed with `force_basins`) → the
   network must now explain *how the ON-fraction evolves* along τ (distribution-level constraint,
   ~hundreds of cells per bin) instead of only per-cell residuals; the mean constraint becomes
   local (a rising gene must have a rising ON-fraction); the modes span the true dynamic range
   (matters for the CN5 cycle and the two BN8 attractors); the basin labels are coherent within a
   bin (less label noise in `sigmoid_target`); the per-bin masses give a cheap Θ-free ordering
   signal and a directly interpretable diagnostic. **Not gained**: better `ks`/`c` (still shared
   across groups — one joint fit, not T fits), nor a different number of components.
   **Risks**: circularity (in the temporal model $\nu_g(t)$ comes from independent samples; here
   the same cells define the bins, so `force_basins = 1.0` partly re-asserts the ordering) →
   ramp `force_basins` 0.3 → 1.0 and re-fit only every `pseudotime_refit_every` iterations;
   structured errors of τ are then *amplified* (deeper local minima than the pooled version);
   noisier $\nu$ with few cells per bin.
   **Decision**: `pseudotime_refit_mixture` flag, fit once from the Θ-free τ then every
   `pseudotime_refit_every = 5` iterations; the ablation with/without is part of M1 (CN5) and M2
   (BN8).
