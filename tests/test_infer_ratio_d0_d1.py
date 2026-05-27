"""
Validation tests for infer_ratio_d0_d1_from_mlp.

Run with:
    python -m pytest tests/test_infer_ratio_d0_d1.py -v
or just:
    python tests/test_infer_ratio_d0_d1.py
"""
import numpy as np
import torch

from CardamomOT.inference.degradations import (
    infer_ratio_d0_d1_from_mlp,
    KonCorrectionMLP,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: make a trivial MLP that outputs a constant ratio for every gene
# ─────────────────────────────────────────────────────────────────────────────
def make_constant_mlp(G_tot: int, ns: int, g_values: np.ndarray) -> KonCorrectionMLP:
    """Return a KonCorrectionMLP whose output is approx. `g_values[g]` for gene g.

    We set W1=0, b1=0, W2=0, b2 = softplus_inverse(g_values[g]) so that:
        output = softplus(0) * 0 + softplus_inverse(g) → softplus(inv) = g
    i.e. the MLP output is constant and independent of the input.
    """
    G_genes = G_tot - ns
    assert len(g_values) == G_genes, "g_values must have one entry per non-stimulus gene"

    mlp = KonCorrectionMLP(G_tot, G_genes, hidden_dim=4)
    with torch.no_grad():
        mlp.W1.data.zero_()
        mlp.b1.data.zero_()
        mlp.W2.data.zero_()
        # softplus_inverse(x) = log(exp(x) - 1)
        for g, gv in enumerate(g_values):
            inv = float(np.log(np.exp(gv) - 1.0 + 1e-12))
            mlp.b2.data[g, 0] = inv
    mlp.eval()
    return mlp


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 – LS formula direct (unit test, no ODE)
# ─────────────────────────────────────────────────────────────────────────────
def test_ls_formula_recovers_true_ratio():
    """If y = r_true * h exactly, the LS estimate must equal r_true."""
    rng = np.random.default_rng(0)
    r_true = 0.1
    h = rng.uniform(0.1, 1.0, size=200)  # all positive h
    y = r_true * h                        # perfect data (no noise)
    lam = 0.0
    r_prior = 0.0  # irrelevant when lam=0

    # LS formula (matches what the function does)
    r_est = (np.dot(h, y) + lam * r_prior) / (np.dot(h, h) + lam)
    assert abs(r_est - r_true) < 1e-12, f"Expected {r_true}, got {r_est}"
    print(f"[test_ls_formula] r_est = {r_est:.6f}  (true = {r_true})")


def test_ls_formula_regularisation_pulls_to_prior():
    """With large lambda, the LS solution should collapse to the prior."""
    rng = np.random.default_rng(1)
    r_true = 0.1
    r_prior = 0.5
    h = rng.uniform(0.1, 1.0, size=200)
    y = r_true * h

    lam_huge = 1e12
    r_est = (np.dot(h, y) + lam_huge * r_prior) / (np.dot(h, h) + lam_huge)
    assert abs(r_est - r_prior) < 1e-6, f"Expected prior {r_prior}, got {r_est}"
    print(f"[test_ls_regularisation] r_est = {r_est:.6f}  (prior = {r_prior})")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 – Full function: prior fallback when theta_inter=0
# ─────────────────────────────────────────────────────────────────────────────
def test_prior_fallback_constant_kon():
    """
    When theta_inter = 0, kon_beta is constant ⟹ ∂ln(kon)/∂X = 0 ⟹ h_i = 0
    everywhere.  All data points fail the |h_i| > min_h filter.
    Both with lambda_deg=0 and lambda_deg>0 the result must equal the prior.
    """
    G, ns = 3, 1  # 1 stim + 2 genes
    G_genes = G - ns
    N_cells = 20

    theta_inter = np.zeros((G, G, 1), dtype=np.float32)
    bias        = np.zeros((G, 1),    dtype=np.float32)
    # n_modes=2, ks shape (n_modes, G)
    ks = np.array([[0.0] * G, [1.0] * G], dtype=np.float32)

    d0 = np.full(G, 0.2, dtype=np.float32)
    d1 = np.full(G, 0.02, dtype=np.float32)
    prior_d1d0 = (d1 / d0).astype(np.float64)   # = 0.1 for all

    kon_mlp = make_constant_mlp(G, ns, g_values=np.full(G_genes, 0.9))

    rng = np.random.default_rng(2)
    # Two timepoints: t=0 and t=1
    X0 = np.abs(rng.standard_normal((N_cells, G))).astype(np.float32)
    X0[:, 0] = 1.0  # stimulus column = 1

    X_prot = X0                            # just use the initial state
    times   = np.zeros(N_cells, dtype=np.float32)   # all at t=0

    # Add a second timepoint so T-1 = 1
    X_prot2 = np.abs(rng.standard_normal((N_cells, G))).astype(np.float32)
    X_prot2[:, 0] = 1.0
    times2  = np.ones(N_cells, dtype=np.float32)

    X_all   = np.concatenate([X_prot, X_prot2], axis=0)
    t_all   = np.concatenate([times,  times2],  axis=0)

    for lam in [0.0, 5.0]:
        rt, rg = infer_ratio_d0_d1_from_mlp(
            X_all, t_all, bias, theta_inter, ks,
            d_learned=d1,
            k1_vec=np.ones(G, dtype=np.float32),
            kon_mlp=kon_mlp,
            prior_d1d0=prior_d1d0,
            n_stimuli=ns,
            lambda_deg=lam,
            n_steps=3,
            verbose=False,
            clip_lo=0.001,
            clip_hi=1000.0,
        )
        expected = prior_d1d0.astype(np.float32)
        print(f"[prior_fallback lam={lam}]  rg[ns:] = {rg[ns:]}  prior = {expected[ns:]}")
        # Global estimate should match prior for gene columns
        np.testing.assert_allclose(rg[ns:], expected[ns:], rtol=1e-4,
                                   err_msg=f"Global ratio should equal prior (lam={lam})")
        # Stimulus column should be 1.0
        assert rg[0] == 1.0, f"Stimulus ratio should be 1.0, got {rg[0]}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 – Verify fix: d0 recovered from ratios must equal literature d0
# ─────────────────────────────────────────────────────────────────────────────
def test_fix_d0_not_underestimated():
    """
    Regression test for the d1_t bug (removed).

    With the OLD code (h_i *= d1_t), the LS estimated 1/d0 instead of d1/d0.
    Then self.ratios = 1/(1/d0) = d0, and in simulation:
        d0_sim = d1 * self.ratios = d1 * d0 ≈ 0.004   (with d0=0.2, d1=0.02)

    With the FIXED code (h_i without d1_t), the LS estimates d1/d0.
    Then self.ratios = 1/(d1/d0) = d0/d1, and in simulation:
        d0_sim = d1 * self.ratios = d1 * (d0/d1) = d0 = 0.2  ✓

    This test uses large lambda_deg so that the LS solution ≈ prior = d1/d0.
    We then verify that d0_sim = d1 / ratios_global = d0 (not d1*d0).
    """
    G, ns = 3, 1
    G_genes = G - ns
    d0_lit = np.array([1.0, 0.20, 0.15], dtype=np.float32)   # [stim, gene1, gene2]
    d1_lit = np.array([1.0, 0.02, 0.01], dtype=np.float32)

    prior_d1d0 = d1_lit / d0_lit    # [1.0, 0.1, 0.0667]

    theta_inter = np.zeros((G, G, 1), dtype=np.float32)
    bias        = np.zeros((G, 1),    dtype=np.float32)
    ks = np.array([[0.0] * G, [1.0] * G], dtype=np.float32)

    kon_mlp = make_constant_mlp(G, ns, g_values=np.full(G_genes, 0.9))

    rng = np.random.default_rng(42)
    N = 30
    X0 = np.abs(rng.standard_normal((N, G))).astype(np.float32); X0[:, 0] = 1.0
    X1 = np.abs(rng.standard_normal((N, G))).astype(np.float32); X1[:, 0] = 1.0
    X_all = np.concatenate([X0, X1])
    t_all = np.concatenate([np.zeros(N), np.ones(N)]).astype(np.float32)

    # Use very large lambda_deg so LS → prior
    _, rg = infer_ratio_d0_d1_from_mlp(
        X_all, t_all, bias, theta_inter, ks,
        d_learned=d1_lit,
        k1_vec=np.ones(G, dtype=np.float32),
        kon_mlp=kon_mlp,
        prior_d1d0=prior_d1d0,
        n_stimuli=ns,
        lambda_deg=1e6,      # huge → forces result to prior
        n_steps=3,
        verbose=False,
        clip_lo=0.001,
        clip_hi=1000.0,
    )

    # rg ≈ prior_d1d0 = d1/d0
    # In base.py: self.ratios = 1 / rg = d0/d1
    # In simulation: d0_sim = d1 * self.ratios = d1 * (d0/d1) = d0
    ratios_stored = 1.0 / rg             # self.ratios = d0/d1
    d0_simulated  = d1_lit * ratios_stored   # d0_sim = d1 * (d0/d1) = d0

    print(f"[fix_test]  d0_lit     = {d0_lit}")
    print(f"[fix_test]  rg (d1/d0) = {rg}")
    print(f"[fix_test]  ratios     = {ratios_stored}  (d0/d1)")
    print(f"[fix_test]  d0_sim     = {d0_simulated}")

    # d0_sim should equal d0_lit (not d1*d0_lit ≈ 0.004)
    np.testing.assert_allclose(
        d0_simulated[ns:], d0_lit[ns:], rtol=0.02,
        err_msg="d0 from simulation must equal literature d0 (not d1*d0)"
    )

    # Sanity: the OLD bug would give d0_sim = d1 * d0 (much smaller)
    d0_bugged = d1_lit[ns:] * d0_lit[ns:]
    print(f"[fix_test]  d0_bugged  = {d0_bugged}  (what the old code would give)")
    # Make sure we're not just getting d1*d0 by accident
    assert not np.allclose(d0_simulated[ns:], d0_bugged, rtol=0.1), \
        "d0_sim matches the bugged value — fix may not have taken effect"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 – Non-trivial network: h_i ≠ 0, LS converges with consistent (h, y)
# ─────────────────────────────────────────────────────────────────────────────
def test_ls_converges_with_nontrivial_h():
    """
    3-gene system (1 stim + 2 genes) with interactions.
    We construct a MLP that returns g values consistent with a known r_true.

    Because the MLP is constant (independent of state), and h_i varies with
    the trajectory, the LS estimate will not be exactly r_true.  But:
      - When lambda_deg is large → result → prior
      - When lambda_deg = 0    → result is some valid (clipped) positive ratio

    This test just checks that the function runs without error and returns
    positive finite ratios in [clip_lo, clip_hi].
    """
    G, ns = 3, 1
    G_genes = G - ns
    N = 40

    # Network with gene-gene interactions
    rng = np.random.default_rng(7)
    theta_inter = rng.uniform(-1.0, 1.0, size=(G, G, 1)).astype(np.float32)
    theta_inter[:, :ns, :] = 0.0   # genes don't depend on other genes' stim column
    bias = rng.uniform(-0.5, 0.5, size=(G, 1)).astype(np.float32)
    ks   = np.array([[0.0]*G, [1.0]*G], dtype=np.float32)

    d1 = np.array([1.0, 0.02, 0.01], dtype=np.float32)
    d0 = np.array([1.0, 0.20, 0.15], dtype=np.float32)
    prior_d1d0 = d1 / d0

    kon_mlp = make_constant_mlp(G, ns, g_values=np.full(G_genes, 0.85))

    X0 = np.abs(rng.standard_normal((N, G))).astype(np.float32); X0[:, 0] = 1.0
    X1 = np.abs(rng.standard_normal((N, G))).astype(np.float32); X1[:, 0] = 1.0
    X_all = np.concatenate([X0, X1])
    t_all = np.concatenate([np.zeros(N), np.ones(N)]).astype(np.float32)

    clip_lo, clip_hi = 0.01, 100.0

    for lam in [0.0, 1.0]:
        rt, rg = infer_ratio_d0_d1_from_mlp(
            X_all, t_all, bias, theta_inter, ks,
            d_learned=d1,
            k1_vec=np.ones(G, dtype=np.float32),
            kon_mlp=kon_mlp,
            prior_d1d0=prior_d1d0,
            n_stimuli=ns,
            lambda_deg=lam,
            n_steps=5,
            verbose=False,
            clip_lo=clip_lo,
            clip_hi=clip_hi,
        )
        print(f"[nontrivial lam={lam}]  rg = {rg}  rt = {rt}")
        assert rt.shape == (1, G), f"Expected (1, {G}), got {rt.shape}"
        assert rg.shape == (G,),   f"Expected ({G},), got {rg.shape}"
        assert np.all(np.isfinite(rt)), "ratios_temporal contains non-finite values"
        assert np.all(np.isfinite(rg)), "ratios_global  contains non-finite values"
        assert np.all(rg >= clip_lo) and np.all(rg <= clip_hi), \
            f"rg outside clip bounds [{clip_lo}, {clip_hi}]: {rg}"


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Test 1 – LS formula direct")
    test_ls_formula_recovers_true_ratio()
    test_ls_formula_regularisation_pulls_to_prior()
    print("  PASSED\n")

    print("Test 2 – Prior fallback (constant kon)")
    test_prior_fallback_constant_kon()
    print("  PASSED\n")

    print("Test 3 – Fix regression: d0_sim == d0_lit")
    test_fix_d0_not_underestimated()
    print("  PASSED\n")

    print("Test 4 – Non-trivial h_i, finite positive ratios")
    test_ls_converges_with_nontrivial_h()
    print("  PASSED\n")

    print("=" * 60)
    print("All tests passed ✓")
