# Review of `PROTOTYPE_PLAN.md`

Status: review of plan **v1**. The five proposed changes were accepted on 2026-11-06 and are
included in plan v2. Since then the gap search has moved inside a re-implemented E-step: R3's joint
dynamic programming became a forward choice with neighbour smoothing (plan §4.3), with the exact DP
kept as a G0 comparison. Severity: **major** = can make the prototype fail
or give a wrong answer silently; **medium** = changes a design choice; **minor** = to handle
during implementation.

Summary: the structure is sound and cheap, since the base E-step already accepts unequal gaps,
so nothing in `base.py` changes. The weak point is step (f). Whether the OT cost has a
**well-placed interior minimum in Δ** is not established, and there are two mechanisms that bias it
(R1, R2). Gate G0 is designed to answer this, and it should be run before the rest of the
implementation. Five design changes are proposed (marked **→**).

---

## Problem solving

### R1 — major: the objective can be biased towards large Δ

As Δ → ∞, $P_{ij}(\Delta) \to s\,m_j$: the protein of the target reaches the equilibrium of its
*own* modes and the source is forgotten. The network term then measures how well Θ explains the
modes of cell j from cell j's own equilibrium proteins, which is the stationary criterion.

- In late bins near steady state (the simulations saturate, `PLAN.md` §0.2) this is exactly what a
  well-fitted Θ does well. So W_k(Δ) can decrease monotonically up to Δ_max, with a boundary
  minimum and a gap determined by the grid bound.
- In transient bins, the lagging proteins are what explain the switching cells, so an interior
  minimum is plausible but not guaranteed.
- The protein term λ_P‖P(Δ) − P_i‖² always favours Δ → 0. If it is what creates the interior
  minimum, Δ̂ is set by λ_P, a tuning knob, and not by the data.

**→** G0 must report, per pair, the **shape** of W_k(Δ) (curve, position of the minimum,
curvature), split into transient and plateau bins, for λ_P = 0 and λ_P > 0. An interior minimum
with λ_P = 0 is the evidence that the network term carries time information.

### R2 — major: self-consistency bias of the coordinate step

Θ is fitted (M-step) on trajectories built with the current gaps Δt^(n). At the next time step,
Θ is by construction adapted to Δt^(n), so W_k is likely to be minimal **near Δt^(n)**, whatever
the truth. The prior (ρ) and the damping (η) strengthen this anchoring. The loop could therefore
stay on its initial value (the uniform prior) and report it as an "estimate".

**→** G0/G1 must be run from **wrong starting points** (all gaps at 0.3× and 3× the truth, and
uniform gaps on the snapshot design whose true gaps differ by 8×). The criterion is that the gaps
*move towards* the truth. A gate that only starts from the prior cannot detect R2.

**→** Make G0 two-sided:
- **G0a** (optimistic): Θ from an oracle run with the true times (the `oracle` of §0.12). It is an
  upper bound, since this Θ was fitted with the right gaps.
- **G0b** (realistic): Θ from a run with true bins but **uniform** gaps. Does the Δ step recover
  the non-uniform truth from it?

The simulator's true Θ (`uns['true_inter']`) is in Harissa's parametrisation and no protein state
is stored, so the "true Θ" setting of the plan is not directly available. G0a replaces it.

### R3 — medium: bins as independent pairs make each Δt noisy

Each Δt_k is estimated from one pair of ~50–300 cells. The EMD between two small empirical
measures in G dimensions is noisy, so independent estimates will scatter even if unbiased.

**→** Replace the independent log-normal prior per pair by a **joint** estimate with a smoothness
penalty on the log-gaps: $\sum_k W_k(\Delta_k) + \rho_1 \sum_k (\log\Delta_k - \log\Delta_{\text{prior}})^2 + \rho_2 \sum_k (\log\Delta_{k+1} - \log\Delta_k)^2$.
With the grid evaluation this is a chain, solvable exactly by dynamic programming over the grid
(K × n_grid², trivial cost). ρ_2 = 0 gives back the plan.

### R4 — medium: carrying Δt across iterations when the bins change

Δt_k is attached to a bin *index*. When Palantir re-ranks the cells, bin k has different cells at
iteration n+1, and the carried value can belong to another part of the trajectory (e.g. if a branch
is re-ordered).

**→** Carry a **time per cell** instead: after step (f), each cell gets the time of its bin
$t_c = \sum_{j<k(c)} \Delta t_j$. After re-binning, the starting time of a new bin is the median of
the t_c of its cells. This also gives a per-cell time output, comparable with the true time.

### R5 — medium: the bin-size rule depends on how the roots are defined

n_0 is set by the root rule, not by the data dynamics. With the 2 % root quantile of the benchmark,
n_0 = 40 → n_b = 50 → K ≈ 40 (capped at 30). That gives many small, noisy pairs (R3). With
`is_root` given by the user it can be anything.

**→** Keep the rule (it matches the brief) but report n_0, n_b and K in the diagnostics, and run
the gates with two sizes (n_b = n_0 and n_b = 2 n_0) to see the sensitivity. Constraint C2
(d1·Δt ≲ 1) can be *checked* once Δt is estimated: the run reports the fraction of pairs with
d1·Δt_k > 1.

### R6 — minor: equal-count bins assume uniform sampling in time

Equal count = equal time only when cells are sampled uniformly in time with no growth. In real
snapshot data the libraries have different sizes, and then bins straddle libraries. Not a problem
for the simulated designs (equal size per library), but this has to be stated in the README. The
density-evolving rule (n_k = n_0 · exp(∫R)) is the right v2 direction.

---

## Mathematics

### M1 — correct: the flow and the gauge

The closed form in §4.2 is `find_next_prot` (`inference/trajectory.py:83`) and `costs.bridge_end`.
The gauge statement is right: with a single d1, only d1·Δ is identified. With gene-specific d1
(from `halflife_db`) the per-gene time scales fix the ratio, and the hours come from the prior.
Errors in the d1 prior propagate *linearly* to the gaps and the total duration. This must be
stated with every reported duration (it is the §0.10 finding 1 again).

### M2 — the switching fraction is relative to Δ

α_i is the switch time as a *fraction* of the step. In the search, α is held fixed while Δ
varies, so the absolute switch time α·Δ moves with Δ. Two consistent choices:
- keep α relative (the plan): consistent with the E-step and `inference_alpha`, which are
  re-run after the step anyway;
- or fix the absolute switch time α·Δ^(n) from the previous E-step.

The first is simpler and consistent with the base. Acceptable, but say it in the docstring.

### M3 — envelope argument for the search

$dW_k/d\Delta = \langle\gamma^*(\Delta), \partial_\Delta C(\Delta)\rangle$ (the optimal coupling is
fixed to first order). A derivative-based search is possible, but W_k is only piecewise smooth
(the EMD coupling changes discretely), so grid + local parabola is the right choice. Also
plot W_k(Δ) in G0 (R1).

### M4 — consistency with the E-step cost

The E-step minimises a *different* cost: `my_otdistance` with per-call normalisation, a decaying
mRNA weight, entropic and unbalanced regularisation. The time step minimises the fixed-scale exact
cost of §4.2. The full loop is therefore not a coordinate descent on one objective, and its
error need not decrease monotonically. Acceptable for a prototype. **→** Monitor both
(`error_after` of the M-step and Σ W_k) in the history, and do not use monotonicity as a test.

### M5 — iteration 1 and Θ = 0

Correct: the network term is constant in Δ at Θ = 0, so the step must be skipped
(`pseudotime_dt_start = 2`). With `intensity_prior < 1` the E-step cost is even zero at the first
pass (`my_otdistance`, random couplings), so the Θ after iteration 1 is weak. Starting the step at
iteration 3 may be safer. Make it a parameter and check in G1.

---

## Computation

### C1 — cost of the time step: fine

Per pair: n_grid · n_s · n_t · G flows + n_grid EMDs. At n = 300, G = 100, 15 candidates:
1.4·10^8 flow evaluations (~1 s vectorised NumPy, less with numba), 15 EMDs of 300×300
(~0.1–0.2 s). With K ≈ 10–30 pairs in parallel this stays below one E-step. Memory: build the cost
per candidate (n_s × n_t × G = 9·10^6 floats = 72 MB at G = 100). Do not stack all the candidates.

### C2 — the gap search is embarrassingly parallel

Over the pairs, with the joint DP of R3 done after all the curves W_k(·) on the grid are
computed. joblib over the pairs, as `_update_basins` already does over genes.

### C3 — downstream with unequal gaps

Checked in `base.py`:
- `estimate_trajectories_given_model`, `estimate_trajectories`, `inference_alpha`, the growth pass,
  `_transfer_weights` and the simulation all use `times[b] - times[a]`. OK.
- `_smoothing_sigma` (`base.py:3307`) smooths per-interval degradation rates over the interval
  *index*, not the duration. With unequal gaps the smoothing mixes intervals of different lengths.
  This only matters in `refine_network_degradations`. Minor; report it, no base change.
- The stimulus schedule is built from `times`. The roots have stimulus 0 and the other cells 1
  (`initialise_cell_states`), independent of the gaps. OK.

---

## Validation design

- G0 → split into G0a / G0b, with curves and wrong starting points (R1, R2).
- Add a **null test**: shuffle the cells between bins k and k+1 (no time signal). W_k(Δ) must become
  flat or hit the boundary, and the flat/boundary flags must fire. This checks that an interior
  minimum is not produced by the objective itself.
- The continuous design gives a clean secondary target: all Δt_k equal (same number of cells per
  unit of true time). Report the coefficient of variation of the Δt_k. On the snapshot design, report
  `gap_ratio_within_2x`. Both metrics already exist (`evaluate.snapshot_metrics`) or are one line.
- Report the total duration ΣΔt_k against the true 96 h, with the d1 prior used (M1).

---

## Proposed changes to the plan

| # | change | from |
|---|---|---|
| 1 | G0 split into G0a (oracle Θ) / G0b (uniform-gap Θ); report W_k(Δ) curves; λ_P = 0 and > 0 | R1, R2 |
| 2 | Start G0/G1 from wrong gaps (0.3×, 3×, uniform); pass = gaps move towards the truth | R2 |
| 3 | Joint estimate of the gaps with a smoothness penalty, exact DP over the grid | R3 |
| 4 | Carry a time per cell across iterations instead of a gap per bin index | R4 |
| 5 | Null test (shuffled bins) + sensitivity to the bin size (n_0, 2 n_0) | R5, validation |

Suggested implementation order, given R1/R2: **binning + `time_steps.py` + G0 first** (a few
hundred lines, no loop changes). The loop (`prototype.py`) comes only if G0a shows interior
minima in the network term.
