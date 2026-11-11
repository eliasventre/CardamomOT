# Prototype: Palantir ordering + time gaps estimated inside the E-step

Status: **plan v2, decisions recorded in §10, ready for implementation** (no code yet). v1 was reviewed in
[`PROTOTYPE_REVIEW.md`](PROTOTYPE_REVIEW.md); its five proposed changes are accepted and included
here (marked *[R-n]*).
Scope: new code in `CardamomOT/pseudotime/`, tests in `tests/`. `base.py` and the rest of the base
package stay untouched. The existing options (`palantir`, `partial_ot`, `tsp`) and
`loop_pseudotime` stay as they are, so the M6 benchmark (`PLAN.md` §0.12) remains a valid reference.

---

## 0. Motivation

From the M6 benchmark (`PLAN.md` §0.12, 3 networks × 3 runs × 2 designs):

1. **The order is recovered, the spacing is not.** `order_accuracy` is 0.76–0.99, but
   `gap_ratio_within_2x` is 0.00 on CN5/FN8 for every option (best 0.30 on BN8). The pseudotime is a
   graph distance, whose increments measure *expression change*, not *time*.
2. **The time unit is a prior, not a measurement.** Every bin has the same width
   `relaxation / median(d1)`.
3. Equal-width binning of τ copies the τ spacing errors into the bins.

The prototype separates the two questions:

- **order**: mechanistic Palantir, used for the *ranks* of the cells only;
- **time**: the gap Δt_k between consecutive bins is a model parameter, chosen inside the E-step
  as the gap that minimises the mechanistic OT cost between the two bins, in hours through the
  prior on `d1`.

---

## 1. The cycle

```
prepare:  roots -> pooled NB mixture -> cell proteins at equilibrium
          -> Palantir (Theta = 0) -> ranks -> bins (sizes: roots + growth, §2.3)
          -> gaps Delta_t_k = prior, time per cell t_c
repeat n = 1..max_iter:
  (a) [every r_mix iterations]    NB mixture re-fitted on the current bins           (existing)
  (b) [every r_ord iterations]    Palantir with the current Theta / proteins -> ranks
                                  -> bins (sizes from the growth of the previous iteration)
                                  -> starting gaps from the time per cell             [R-4]
  (c) E-step, re-implemented here: for each pair k (forward), choose Delta_t_k by
      minimising the OT cost, then sample the trajectories with it   [every r_gap iterations]
  (d) M-step: network Theta                                                          (existing)
  (e) switching fractions alpha, basins                                              (existing)
  (f) [every r_growth iterations] growth pass on the current bins -> R per bin       (existing)
  (g) stop when ranks, gaps and the model error are stable
after: growth pass, refine_network_degradations, simulate_network                    (existing)
```

New: the binning rule (§2), the E-step with the gap search (§4), the time per cell, and the
schedule of the steps (§6). The mixture, M-step, alpha, basins, growth pass and downstream stages
reuse `PseudotimeNetworkModel` and the base methods.

---

## 2. Step 1 — Roots, ordering and binning

### 2.1 Roots

1. **User-defined**: `obs[pseudotime_root_key]` (default `'is_root'`, boolean or 0/1). Their number is
   n_0.
2. **Fallback**: if the column is absent, the roots are **2 % of the cells**
   (`pseudotime_root_fraction = 0.02`, so n_0 = ⌈0.02 N⌉). Choosing *which* 2 % needs a rule, because
   Palantir needs roots before any τ exists. Proposed rule (to confirm, §10 Q1): the 2 % of cells at
   one extremity of the first diffusion component of the expression kNN graph. The extremity is the
   one whose cells have the lowest mean ON-fraction of the stimulus-responsive genes, or the user
   gives the extremity through `pseudotime_root_side`. The choice is printed and stored in
   `pseudotime_diagnostics['roots']`.

### 2.2 Ordering

`PalantirLike.run` unchanged (`options/palantir_like.py`, defaults of `PLAN.md` §0.7). Only the
**ranks** of τ are used from now on. The τ values are kept for diagnostics.

### 2.3 Bin sizes: the roots, deformed by growth

A bin is the population of the previous bin after one gap of growth (proliferation − death):

$$
n_0 = \#\text{roots},\qquad n_{k+1} = n_k \exp\!\big(\bar R_k\,\Delta t_k\big),
$$

with $\bar R_k$ the mean net rate of the trajectory states of bin k and Δt_k the gap, both from the
previous iteration. Cells are cut **by rank**: bin 0 = the roots, bin 1 = the next n_1 cells in τ
order, and so on, until all N cells are assigned.

- **Without growth** ($\bar R \equiv 0$, the first target): every bin has n_0 cells (equal count).
- **First iteration**: $\bar R$ is unknown. It comes from `obs['proliferation_net_rate']` when present
  (the base convention), else 0.
- **Guards**: n_k ≥ n_min (`pseudotime_bin_min_cells`, 50) and K ∈ [K_min, K_max] = [4, 30].
  When a cap is hit, all n_k are rescaled by one common factor, which keeps the growth ratios
  n_{k+1}/n_k. The last bin takes the remainder of the cells and is merged with the previous one if
  it is below n_min.
- **Diagnostics**: n_0, the sizes n_k, K and the fraction of pairs with d1·Δt_k > 1 (more than one
  protein relaxation per gap, i.e. constraint C2 of the review is broken).

Why rank cuts: the τ spacing is unreliable (§0). The gaps carry the time, and the growth carries the
change of population size. On the uniform-sampling design without growth, equal count is equal
time width, so all gaps should be equal. On the snapshot design (10 libraries of 200 cells, the
first one = roots), the bins coincide with the libraries if the order is correct, and the gaps
should recover the geometric spacing.

**Modelling assumption** (stated in the README): the number of sampled cells along the trajectory is
proportional to the population size. This holds for a single snapshot of a population moving along
the trajectory. It does not hold for a time course where each library has a chosen number of cells.

### 2.4 Time per cell *[R-4]*

After each E-step, every cell gets the time of its bin, $t_c = \sum_{j<k(c)}\Delta t_j$. After a
re-binning (b), the starting gap of a new bin pair is the difference of the median $t_c$ of the
cells of the two new bins. Without this, a gap would stay attached to a bin index whose cells
have changed. $t_c$ is also the per-cell time output, compared with the true time.

### 2.5 Data structure

`BinScheme` gets non-uniform `bin_times` and a `sizes` field. New function
`binning.bin_by_sizes(ranks, sizes, roots, n_min, k_min, k_max) -> BinScheme`. `BinScheme.dtau`
becomes the mean gap (kept for the code paths that read it).

---

## 3. Step 2 — NB mixture

As decided: a **pooled fit on all the cells at preparation**, then a **fit on the current bins** every
`r_mix` iterations (`fit_mixture_on_bins`), with the circularity guards of `PLAN.md` §10.6. The bin
is only a group label, so non-uniform bin times are fine. Validation: test T-MIX (§8.3).

---

## 4. Step 3 — E-step with the gap search (re-implemented in `pseudotime/`)

### 4.1 Why re-implement

The gap of pair k changes the protein states that pair k+1 starts from, so the best place to choose
Δt_k is inside the E-step, after pair k−1 has been propagated and before the trajectories of pair k
are sampled. The loop of `estimate_trajectories_given_model` is in `base.py`, so it is
re-implemented in `pseudotime/estep.py`. The re-implementation is a reduced copy:

- **kept**: one sample, cost `my_otdistance` (imported), `_solve_ot`, `_traj_marginals`
  (growth-weighted marginals), `find_next_prot` and `find_next_prot_dilution` (dilution, needed
  later for growth), the alpha re-assignment, `to_keep_for_update`, `sim_real_idx`;
- **dropped**: several samples, `observed` gaps / virtual states, `alpha_pool` (test set), the
  `growth_only` pass (the existing growth pass is reused);
- **dropped in v1, to implement later** (§9): the lineage constraint (clonal barcodes) and the
  cell-type transition matrix (workbook sheet `transition_rates`);
- **not supported in v1**: `mrna_bridge = True` (raises `NotImplementedError`; default is False).

**Loop order changes.** The base loops over batches outside and pairs inside. The gap of a pair
must be the same for all batches, so here the loop is pairs outside, then (i) gap search on the
cells of **all** batches (subsampled), (ii) batches inside, each with the chosen gap. With one batch
the order of the random draws is the base one, which makes an exact equivalence test possible
(§8.1).

**Several batches are a supported case, not an edge case.** With the default
`batch_size_traj = 1024`, the simulated datasets give one batch per bin, but real datasets (and
tests with a small `batch_size_traj`) give several. Requirements:
- the gap search draws its subsample from all batches, stratified by batch (no batch dominates);
- each batch reads and writes only its own slot rows, so the order of the batches inside a pair
  does not change the result in distribution;
- the real target cells are tiled into batches as in the base (`_scaffolding`), and the search uses
  the union of the target batches;
- tested explicitly with several batches (§8.1) and in one G2 run with a small `batch_size_traj`.

### 4.2 The gap search for pair k

Inputs: the source slots of bin k (protein state P_i, modes m_i, switching fraction α_i, counts x_i),
just propagated by pair k−1; the real cells of bin k+1 (modes m_j, counts x_j); the network Θ of the
previous M-step. A subsample of at most `gap_max_cells = 300` sources and targets (seeded) is used
for the search.

For each candidate Δ on a grid of `gap_n_grid = 21` values, log-spaced on
$[0.01, 5] / \mathrm{median}(d_1)$ (15 values on $[0.05, 5]$ in G0 v1; the bottom was lowered
because the corrected curves hit it on the first, shortest pairs, §11.1):

1. end proteins $P_{ij}(\Delta)$ = `find_next_prot` (or `find_next_prot_dilution`);
2. the cost matrix $C(\Delta)$ (formulation of §5), with **scales fixed once per pair** (computed at
   the current gap), so that values at different Δ are comparable;
3. $W_k(\Delta) = $ exact EMD cost (`ot.emd2`, uniform marginals; growth-weighted marginals once
   $\bar R \neq 0$).

This gives the curve $W_k(\cdot)$ on the grid. It is stored for the diagnostics (the G0 plots).

### 4.3 Choosing the gap, with smoothing *[R-3]*

$$
\widehat{\Delta t}_k = \arg\min_{\Delta}\; W_k(\Delta)
+ \rho_1(\log\Delta - \log\Delta_{\text{prior}})^2
+ \rho_2\big[(\log\Delta - \log\Delta t_{k-1}^{(n)})^2 + (\log\Delta - \log\Delta t_{k+1}^{(n-1)})^2\big]
$$

- ρ_1 pulls towards the prior `pseudotime_relaxation / median(d1)`.
- ρ_2 pulls towards the neighbouring gaps. The previous gap was just chosen in this E-step; the
  next gap comes from the previous iteration, since it is not chosen yet.
- With ρ_2 = 0 each pair is on its own. With a large ρ_2 all gaps become equal.
- W_k is a mean cost per cell, so the penalties do not depend on the bin size.
- The minimum on the grid is refined by a parabola in log Δ.
- Damping across iterations: $\log\Delta t_k^{(n)} = (1-\eta)\log\Delta t_k^{(n-1)} + \eta\log\widehat{\Delta t}_k$.

The forward pass is an approximation of the exact joint choice: the exact choice is not tractable,
because each gap changes the starting states of the next pair. In G0, the sources are fixed by the
oracle, so the exact joint choice (dynamic programming over the stored curves) is also computed and
compared with the forward choice.

### 4.4 When the gap is not identified

- **Θ = 0** (iteration 1): the network term is constant in Δ. The search is skipped and Δt_k keeps
  the prior. The search starts at iteration `gap_start` (default 2, to be checked against 3 in G1,
  because with `intensity_prior < 1` the first E-step uses random couplings).
- **Flat curve** ($\max W_k - \min W_k$ below `gap_flat_tol`, relative): the previous value is kept,
  and the pair is flagged.
- **Boundary minimum** (at either end of the grid): flagged. Many boundary minima point to a
  problem with the objective, not to extreme gaps.

### 4.5 Sampling the trajectories

With the chosen gap, each batch of pair k runs exactly as in the base: `my_otdistance` at Δt_k (with
its usual normalisation and entropic/unbalanced regularisation), `_solve_ot`, one target sampled per
source, end proteins, alpha re-assignment. The search only decides Δt_k. The coupling that builds
the trajectories is the base one.

### 4.6 Absolute time

The flow depends on Δ only through $d_{1,g}\Delta$. With $d_1$ fixed by its prior, the gaps are in
hours, and errors in the prior pass *linearly* to the gaps and to the total duration $\sum_k\Delta t_k$.
Every reported duration states the d1 used.

### 4.7 Warm start of the switching fractions (time-aware closest state)

The switching fraction α[k, s, g] (when gene g of slot s switches mode inside step k) is estimated
after the M-step (`inference_alpha`, via `_update_alpha`). The trajectories are rebuilt at every
E-step (new OT draws), so α cannot be carried slot by slot. The base attaches it to a *state*
instead: a newly propagated state takes the α of the closest previous-iteration state at the same
bin (`base.py:1640-1645`). The current `_estep` skips this (it starts from zero proteins and draws a
new random α at every iteration), so the estimated α is never used inside the loop. That `_estep`
is **not** changed (the options are reworked later).

In the new E-step, after the slots of pair k are propagated to bin k+1:

1. **Pool of old states**: all slots of the previous iteration, each with its protein state, its
   modes, its time (the time of its bin, from the previous gaps) and its α for the step leaving
   that state.
2. **Time window**: the candidates are the old states whose time lies within the time span of the
   new bin k+1 extended by half a gap on each side (in practice, the one or two nearest old bins).
   This is robust to re-ordering and to changes of K and N_full.
3. **Closest state**: weighted L1 distance on (proteins, modes), with the base weights
   (1/G for proteins, (G−1)/G for modes). The new slot takes the α of the closest candidate.
4. **No candidate** (first iteration, empty window): α is drawn uniformly in [0.1, 0.9], as today.

The gap search of pair k+1 then uses these α for its sources. The pool also stores the old times,
so it is built after the gaps of the previous iteration are final.

---

## 5. Formulation of the cost of the gap search (open: decided by G0)

Only the term that depends on Δ moves the minimum. The other terms change which cells are paired.

| | formulation | pros | cons |
|---|---|---|---|
| **F1** | network term only: $\sum_g w_g\,\mathrm{CE}(\sigma(\Theta P_{ij}(\Delta)), \text{target}(m_j))$ | the only term that carries time through the dynamics (how long it takes for the proteins to produce the observed switches); no weight to tune | flat while Θ ≈ 0; can favour large Δ (target proteins at the target's own equilibrium, explained by a stationary Θ, review R1); the pairing is free to change with Δ, so the curve can be jagged |
| **F2** | F1 + $\lambda_P\|P_{ij}(\Delta) - P_i\|^2$ | same terms as the E-step cost, hence consistent; always gives an interior minimum | the protein term always favours Δ → 0, so the minimum can be set by λ_P rather than the data; proteins are not observed, so this term is a smoothness prior, not a measurement |
| **F3** | F2 (or F1) + $\lambda_x\|s(x_j - x_i)\|^2$ | the mRNA term does not depend on Δ but anchors the pairing to expression similarity: smoother curves, a coupling close to the E-step one | lowers the relative depth of the curve (weaker signal); one more weight |
| **F4** | the E-step cost `my_otdistance` as is | identical to the E-step | **not usable**: it rescales its terms to the median of the network term at each call, and the E-step subtracts the column minimum, so values at different Δ are not comparable. With fixed scales it becomes F3 with the E-step weights |
| **F5** | profile likelihood: for each Δ, refit Θ on the pair and keep the best fit | the most principled (the gap the network explains best after fitting), and it removes the self-consistency bias of review R2 | one M-step per candidate and per pair (×15 × K M-steps per iteration); a Θ fitted on one pair is local; for later |
| **F6** *(added after G0 v1)* | forecast: the source protein is propagated by the **network's own dynamics** $\dot P = d_1(\text{scale}\cdot k_{on}(P;\Theta) - P)$ (RK4), with no target state in the predictor; cost = CE between $k_{on}(P(\Delta))$ and the target modes | removes the cause of F1's bias (the end protein of F1 relaxes towards the target's own modes, so a target is "explained by itself" at large Δ); as Δ → ∞ the forecast goes to the network's fixed point, not to the target; the gap = time the fitted network needs for the transition; cheaper than F1 (one ODE per source) | depends more on the quality of Θ; mean-field (a switch due to bursting noise is not predicted, only partly absorbed by the OT pairing); no dilution |
| **F1 − self null** *(added after G0 v1)* | F1 curve minus the F1 curve between two disjoint halves of the target bin (sources: protein state of the cells from the previous E-step), both divided by their mean | removes the self-explanation gain measured within the pair, computable in a real run | the difference of two noisy curves is noisy (minima at the grid ends in the offline test of §11.1) |

**Decision rule.** G0 computes F1, F2 and F3 from the same flows (the extra cost is negligible) and
decides:

- if F1 has an interior minimum near the true gap on most pairs: **F1**, with F3 (small λ_x) if its
  curves are jagged;
- otherwise F2/F3, but only if the minimum is stable when λ_P changes by 4× (a minimum that follows
  λ_P is not a measurement);
- if no formulation passes, the gap search is dropped and F5 is studied separately.

---

## 6. Schedule of the steps (decoupling)

Each step runs on its own period. Running everything every iteration is not necessarily best:
re-fitting the mixture or re-ordering too often feeds the circularity (bins → mixture → bins), and
too rarely slows convergence.

| step | period | v1 default | why not every iteration |
|---|---|---|---|
| (a) mixture on bins | `r_mix` | 5 (first iteration: pooled) | circularity of the bins and the per-bin masses |
| (b) Palantir re-ordering + re-binning | `r_ord` | 1 | each re-binning moves cells between bins, and the trajectories restart |
| (c) gap search | `r_gap` (from `gap_start`) | 1 | it can oscillate with the M-step (review R2) |
| (f) growth pass (bin sizes) | `r_growth` | 1 | an extra OT pass |

The optimal periods are **not** studied in v1: the defaults above are used, and the periods are
parameters. A later experiment (§8.4) compares schedules on the validation datasets.

---

## 7. Code layout

| file | content |
|---|---|
| `pseudotime/binning.py` | + `bin_by_sizes`, `growth_sizes`; non-uniform `bin_times` and `sizes` in `BinScheme` |
| `pseudotime/roots.py` (new) | root selection: user column, 2 % fallback (§2.1) |
| `pseudotime/gaps.py` (new) | pure functions: `gap_cost_matrix(Δ, …, formulation)`, `gap_curve(…)`, `choose_gap(curve, priors, ρ_1, ρ_2)`, `joint_gaps_dp(curves, …)` (G0 only), flags |
| `pseudotime/estep.py` (new) | `estep_with_gaps(model, …)`: the reduced E-step of §4 |
| `pseudotime/prototype.py` (new) | `GapNetworkModel(PseudotimeNetworkModel)`: parameters, schedule, time per cell, `loop_prototype`, `fit_prototype` |
| `pseudotime/check_prototype.py` (new) | experiments of §8.2–§8.4, figures + metrics in `data/_prototype/` |
| `tests/test_pseudotime_prototype.py` (new) | unit tests of §8.1 |
| `pseudotime/README.md`, `PLAN.md` §0 | usage and results, once a gate passes |

New parameters (each with a one-line comment, read by the workbook):
`pseudotime_root_fraction`, `pseudotime_root_side`, `pseudotime_k_min`, `pseudotime_k_max`,
`gap_start`, `gap_grid_range`, `gap_n_grid`, `gap_max_cells`, `gap_formulation` (`'F1'` | `'F2'` |
`'F3'`), `gap_lambda_prot`, `gap_lambda_rna`, `gap_prior_weight` (ρ_1), `gap_smooth_weight` (ρ_2),
`gap_damping` (η), `gap_flat_tol`, `schedule_mix` (r_mix), `schedule_order` (r_ord), `schedule_gap`
(r_gap), `schedule_growth` (r_growth).

---

## 8. Tests and validation

Datasets *(G0 v2)*: BN8, FN8, CH20 (snapshot design, 2000 cells, 3 runs) and Schiebinger (real
data, Dox then serum, 10 real timepoints 0, 12, 36, 60, 84, 120, 168, 240, 324, 432 h, ≤ 300 cells
per timepoint drawn with the seed of the run, d1/d0 of `var`).

Datasets: simulated networks with known Θ, d1 and true times, in both designs (continuous and
`Data_grid` snapshots, `PLAN.md` §0.11). BN8, FN8 and CH20 are the main cases. CN5 is an oscillator
(τ ill-defined, §0.1) and is kept for the network only. All current networks have no growth. A
proliferating simulation is needed for §8.5.

### 8.1 Unit tests (`tests/test_pseudotime_prototype.py`, fast, synthetic inputs)

Roots and binning:
- user root column used as is; fallback gives ⌈0.02 N⌉ roots at one extremity, deterministic with a seed;
- `growth_sizes` with R = 0 gives equal sizes n_0; with a constant R it gives n_0·e^{RΣΔt}; the
  sizes sum to N; caps K_min/K_max and n_min respect the growth ratios;
- `bin_by_sizes`: bin 0 = roots, rank order kept, non-uniform `bin_times` = cumsum of the gaps;
- time per cell: after a re-binning, the starting gaps equal the median-time differences.

Gap search:
- with F1, `gap_cost_matrix` equals `costs.network_cost(bridge_end(...))`;
- scales are fixed: evaluating the grid in another order, or a single Δ alone, gives the same values;
- **recovery**: a synthetic pair generated by the flow with a known Θ and a known Δ* → the chosen gap
  is within 20 % of Δ*;
- **null**: sources and targets drawn from the same population (no time between them) → flat or
  boundary flag;
- smoothing: ρ_2 = 0 gives the per-pair argmin; a large ρ_2 gives equal gaps; `joint_gaps_dp`
  equals brute force on K = 4 pairs × 5 grid values;
- dilution: with birth rates, the flow used is `find_next_prot_dilution`.

E-step:
- **equivalence**: with the gap search off, one batch and the same seed, `estep_with_gaps` returns
  the same `y_prot`, `y_kon`, `y_rna`, `sim_real_idx` and `to_keep` as
  `NetworkModel.estimate_trajectories_given_model` on the same bins;
- with the search on, all batches of a pair use one gap, and the times passed to the next pair are
  the cumulative gaps;
- **several batches** (`batch_size_traj` small enough to give 3–4 batches per bin): one gap per pair
  shared by all batches; the search subsample contains cells of every batch; changing the order of
  the batches gives the same gaps (fixed seed for the search subsample) and statistically equivalent
  trajectories (per-bin mean protein within 3 standard errors); with the search off, per-bin mean
  proteins agree with the base E-step within 3 standard errors;
- `mrna_bridge = True` raises `NotImplementedError`.

Warm start of α (§4.7):
- with unchanged bins and gaps, the time-aware rule picks the same old states as the base rule (same
  bin), so it reproduces the base re-assignment;
- a new state identical to an old one gets that old state's α exactly;
- old states outside the time window are never selected, even if closer in state;
- K and N_full different from the previous iteration: no error, and α comes from the pool;
- empty pool (first iteration): α uniform in [0.1, 0.9];
- across iterations, the α used by the E-step at iteration n+1 are those estimated by
  `_update_alpha` at iteration n (not new random draws).

### 8.2 Gates on the gaps (`check_prototype.py`, in this order; a failed gate stops the next)

**G0 — does the cost carry time?** *[R-1]* True bins of the snapshot design (true gaps
3.0 … 24.5 h). One E-step with the gap search on, Θ frozen. Output per pair: the curves of F1, F2
and F3 (λ_P at 3 values) with the true gap marked; the forward choice and the exact joint choice
(DP). Script: `check_prototype.py`, outputs in `data/_prototype/g0/`.
- **G0a (best case)**: Θ from the oracle cycle (`loop_pseudotime` with frozen bins) on the snapshot
  bins with their true times.
- **G0b (realistic)**: Θ from the same cycle with equal gaps at the prior `1/d1` (what the loop
  starts from).
- **Null**: the G0a network, with the non-root cells shuffled between the non-root bins.
- The source states are propagated with the **prior gap** for every pair, and the cost scales are
  fixed at the prior gap. *(Corrected 2026-11-06: a first run propagated with the true gaps. That
  leaks the true gaps into the source states, and the null then still followed them. That run is
  kept in `data/_prototype/g0_true_propagation/`.)*
- Pass criteria *(revised 2026-11-06)*. The original ones (interior minimum on ≥ 80 % of the pairs,
  `gap_ratio_within_2x` ≥ 0.7) do not discriminate: the null passes both, and a *constant*
  estimate already scores 0.67 on `gap_ratio_within_2x`, because the true gaps only span a factor 8.
  The criteria are now measured against the null:
  the Spearman correlation between estimated and true gaps is high in G0a (≥ 0.8) and clearly above
  the null; G0b is reported as the realistic case; and the absolute bias (median estimated/true gap)
  is reported separately. Applies the decision rule of §5.

**G1 — does the loop move away from wrong gaps?** *[R-2]* True bins, full loop, gaps started at
0.3× the truth, at 3× the truth, and equal (10.7 h). The ratio estimated/true gap is recorded at
each iteration. Pass: the ratios move towards 1, and the final `gap_ratio_within_2x` > 0.5 (all the
current options score 0.0–0.3). Also on the continuous design: coefficient of variation of the gaps
< 0.3 (they should be equal). Also `gap_start` = 2 vs 3.

**G2 — free run.** Palantir ranks, full loop, 3 runs, both designs. Compared with the `palantir`
and `oracle` rows of §0.12: order accuracy, separation, gap2x, Spearman of t_c with the true time,
AUPRC, total duration against the true 96 h (with the d1 used).

**Bin size sensitivity** *[R-5]*: G1 and G2 with n_0 and 2 n_0 roots.

### 8.3 T-MIX — the mixture schedule

Does "pooled at start, then on bins" help, and does it create circularity?
1. **Ablation**: pooled only vs pooled + bins (r_mix = 5), on true bins and on free bins. Metrics:
   per-bin mean error, AUPRC, gap2x, Spearman of t_c.
2. **Null**: fit on randomly shuffled bins. The per-bin masses ν_g(k) must show no trend across bins
   (a slope test per gene, ≤ 5 % significant at α = 0.05). A trend means the procedure creates
   temporal signal by itself.
3. **Amplification**: start from true bins with 20 % of the cells swapped between non-adjacent bins.
   Spearman of the order with the truth after the loop, with vs without the fit on bins: does the
   fit on bins correct the error or lock it in?

### 8.4 Schedule (later, not in v1)

A grid of (r_mix, r_ord, r_gap) ∈ {1, 2, 5}³ on BN8/FN8, both designs, comparing gap2x, AUPRC and
runtime. Written down here so that the periods stay parameters in the code.

### 8.5 Growth (later, not in v1)

Needs a proliferating simulation (to add to `simulate_pseudotime_data.py`, with branching and
known birth/death rates). Checks: bin sizes against the true population sizes, R̄_k against the
true net rate, and the gaps with and without the growth-deformed sizes. The growth pass stays **on**
in v1, and its output (≈ 0 on the current networks) is reported.

---

## 9. Later (not in v1)

- **Lineage constraint**: with clonal barcodes (`obs` lineage column), cells of a clone at bin k
  may only be matched to cells of the same clone at bin k+1 (large finite penalty on cross-clone
  pairs, as in the base E-step). The gap search must use the same penalised cost.
- **Cell-type transition matrix**: type-to-type transition probabilities of the workbook
  (`transition_rates`), added as a log-penalty to the OT cost (`_transition_log_penalty`). The
  penalty depends on the gap Δ (rates × Δ), so it also enters the gap search, where it carries time
  information of its own.
- Several samples (e.g. control and KO trajectories), mRNA bridges.
- The schedule experiment (§8.4) and the growth validation (§8.5).

## 10. Decisions and open questions

Decided (2026-11-06):
1. Fallback roots: 2 % at one extremity of the first diffusion component. Acceptable as a
   suboptimal fallback; the user-defined roots are the intended input.
2. Cost formulation: decided by the results of G0 with the rule of §5.
3. Setting: a **single snapshot** of a trajectory, so the growth-deformed bins of §2.3 apply.
4. Reduced E-step as in §4.1. Lineage and cell-type transitions later (§9). Several batches
   supported and tested.

5. Switching fractions are carried across iterations in the new E-step with the time-aware
   closest-state rule (§4.7). The existing `_estep` is not changed: the options will be reworked
   later.

---

## 11. Results

### 11.1 G0 (2026-11-06)

Setting: BN8 and FN8, snapshot design (`Data_grid`), runs 1–3, seed 0, oracle cycle of 3
iterations, one E-step with the gap search (`n_iter = 4`), 15 candidate gaps on [0.05, 5] / d1
(G0 v1; outputs moved to `data/_prototype/g0_v1/`),
sources propagated with the prior gap (BN8: 20 h, d1 = 0.05; FN8: 12.5 h, d1 = 0.08). True gaps:
3.0, 3.9, 5.1, 6.6, 8.6, 11.1, 14.5, 18.8, 24.5 h. Mean over the 3 runs (`data/_prototype/g0_v1/aggregate.md`,
per-run table in `summary.md`, curves in `<NET>_r<run>_<setting>.png`). ρ = Spearman between the
estimated (raw argmin of the curve) and the true gaps; bias = median of estimated / true.

| | BN8 G0a | BN8 G0b | BN8 null | FN8 G0a | FN8 G0b | FN8 null |
|---|---|---|---|---|---|---|
| **F1** ρ / bias | **+0.93** / ×4.7 | +0.82 / ×6.5 | −0.02 / ×7.9 | **+0.92** / ×1.6 | +0.61 / ×3.8 | +0.27 / ×3.8 |
| F2 λ_P = 0.25 | +0.85 / ×0.96 | +0.62 / ×1.6 | +0.02 | +0.94 / ×0.68 | +0.89 / ×1.3 | −0.19 |
| F2 λ_P = 1 | +0.88 / ×0.29 | +0.77 / ×0.57 | −0.16 | +0.92 / ×0.29 | +0.88 / ×0.53 | −0.26 |
| F2 λ_P = 4 | +0.58 / ×0.12 | +0.66 / ×0.16 | −0.61 | +0.88 / ×0.12 | +0.88 / ×0.16 | −0.67 |
| F3 λ_x = 1 (interior) | +0.48 (0.59) | +0.55 (0.11) | +0.50 | +0.92 (1.00) | +0.61 (1.00) | +0.11 |

Findings:

1. **The network term carries the order of the gaps.** F1 ranks the 9 gaps correctly
   (ρ ≈ 0.9) with the network of the oracle cycle, against ρ ≈ 0 in the null. The null now behaves
   as expected only because the sources are propagated with the prior gap. With the true gaps
   (`g0_true_propagation/`), the null reached ρ = 0.45 (BN8) and 0.31 (FN8).
2. **F1 is biased towards large gaps (review R1, confirmed).** The estimates are 1.6× (FN8) to
   4.7× (BN8) too large in G0a, and the null estimates sit near the top of the grid (52–72 h on
   BN8). Without information from the source, the network explains each target best from its own
   equilibrium. The F1 gaps are therefore a **relative** measurement, not an absolute one.
3. **The protein term sets the scale, not the data.** F2's bias moves from ×0.96 to ×0.29 to ×0.12 when
   λ_P goes from 0.25 to 1 to 4 (BN8 G0a). That is a factor 3 per 4× of λ_P, so F2 fails the rule of §5
   (a minimum must not follow λ_P). Its near-unbiased value at λ_P = 0.25 is a coincidence of the
   weight.
4. **The mRNA term alone pushes the minimum to the boundary** on BN8 (interior 0.11–0.59): F3 with
   λ_P = 0 is rejected.
5. **Realistic case (G0b).** With a network fitted on equal prior gaps, F1 still ranks the gaps
   (ρ = 0.82 BN8, 0.61 FN8). The range is compressed (BN8 estimates 37–72 h for true 3–24 h) and
   the bias grows (×6.5, ×3.8). Whether the loop sharpens this as Θ improves, or locks it in
   (review R2), is the question of G1.
6. **The smoothing (exact joint choice, DP, ρ_1 = ρ_2 = 0.01) helps the realistic case but also
   creates trends in the null.** For F1, ρ goes from 0.61 to 0.80 on FN8 G0b, but also from −0.02
   to +0.27 (BN8) and from +0.27 to +0.43 (FN8) in the null. Any ρ obtained with smoothing must be
   compared with the null under the same smoothing.

**Gate G0 (revised criteria of §8.2): passed for F1**: ρ ≥ 0.8 in G0a on both networks, clearly
above the null. Open decision: the absolute scale of the gaps (finding 2).

### 11.2 G0 v2: corrections of F1, CH20 and Schiebinger (2026-11-06)

Same protocol as §11.1, grid of 21 gaps on [0.01, 5] / d1, datasets BN8, FN8, CH20 (snapshot
design, 2000 cells) and Schiebinger (Dox then serum, 10 real timepoints, ≤ 300 cells per timepoint,
true gaps 12, 24, 24, 24, 36, 48, 72, 84, 108 h), runs 1–3 (Schiebinger: three subsamples), seed 0,
oracle cycle of 3 iterations. Mean over the runs (`data/_prototype/g0/aggregate.md`). Each cell is
ρ / bias, with bias = median estimated / true gap.

| formulation | setting | BN8 | FN8 | CH20 | Schiebinger |
|---|---|---|---|---|---|
| F1 | G0a | +0.92 / ×4.8 | +0.93 / ×1.5 | +0.93 / ×3.0 | 0.00 / top of the grid |
| | G0b | +0.83 / ×6.6 | +0.70 / ×3.6 | +0.77 / ×5.4 | 0.00 / top of the grid |
| | null | +0.08 | +0.29 | −0.05 | 0.00 |
| F6 forecast | G0a | +0.08 / ×0.05 | **+0.96 / ×1.3** | **+0.92 / ×0.74** | −0.07 / ×0.19 |
| | G0b | −0.27 / ×1.9 | +0.67 / ×1.4 | +0.38 / ×3.0 | +0.10 / ×0.22 |
| | null | bottom of the grid | bottom | bottom | −0.22 |
| F1 − self null | G0a | −0.67 / ×0.02 | −0.04 / ×0.03 | −0.75 / ×0.01 | −0.19 / ×0.19 |
| F1 − shuffled null (offline) | G0a | +0.94 / ×1.5 | +0.95 / ×0.74 | +0.99 / ×0.83 | +0.52 / ×1.4 |
| | G0b | +0.74 / ×4.6 | +0.78 / ×1.2 | +0.93 / ×1.25 | +0.02 / ×0.28 |
| F2 λ_P = 0.25 | G0a | +0.87 / ×0.93 | +0.95 / ×0.72 | +0.88 / ×0.83 | +0.06 / ×0.60 |
| | G0b | +0.58 / ×1.7 | +0.80 / ×1.4 | +0.37 / ×1.5 | 0.00 / ×0.60 |

Findings:

1. **F1 − self null fails everywhere**: the minimum sits at the bottom of the grid with a negative
   ρ. The null built from the cells of the target bin (proteins of the previous E-step) has its own
   strong trend in Δ, which dominates the difference. Rejected.
2. **F6 (forecast) removes the bias where the network has a single attractor**: FN8 ρ = +0.96,
   ×1.3; CH20 ρ = +0.92, ×0.74 (best case). It **fails on BN8** (bistable network, two attractors):
   a mean-field forecast from one source goes to one fixed point and cannot produce both branches,
   so the best match is at Δ → 0. With the network of equal prior gaps (G0b) it degrades more than
   F1 (CH20 +0.38 against +0.77): F6 depends more on the quality of Θ. Its null goes to the bottom
   of the grid, as expected (without structure, the forecast matches best before moving).
3. **The offline correction (F1 − shuffled null) is the most consistent on the simulations**
   (ρ 0.94–0.99, bias ×0.74–1.5 in G0a; CH20 G0b +0.93 / ×1.25). It needs a network refitted on
   shuffled bins, which is not available in a real run as such. It could be made available by fitting a "null
   network" on shuffled bins at each iteration (≈ twice the M-step cost).
4. **Schiebinger: no formulation recovers the gaps.**
   - F1 decreases monotonically on every pair, to the top of the grid (pure self-explanation).
   - F6 is flat below ~10 h and rises steeply above, so its minimum falls in the flat part.
   - Only the offline correction shows a signal in G0a (+0.52), and none in G0b.
   - Two confounders on this dataset:
     - the top of the grid (5/d1 = 102 h) is below the last true gap (108 h) and close to the
       previous one (84 h);
     - the network after 3 oracle iterations on 108 genes may be too weak (not checked yet).
5. **CH20 behaves like the small networks**, with clearer signals: the size of the network is not the
   problem; the dynamics (single vs several attractors) and the real data are.

**Status of G0**: on the simulations, the gap order is carried by the network term. The absolute scale
is recovered only by F6 on single-attractor networks and by the offline shuffled-null correction.
On real data (Schiebinger), no formulation works in this setting.
