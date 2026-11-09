
"""
Core functions for the inference of trajectories, mainly used in loop_trajectories
"""

import math
import numpy as np
import numba
from numba import njit, prange
from .network import net_cost, ordinal_consts, ordinal_targets_c, ordinal_terms, lik_levels, lik_terms, best_loglik, cost_any, sigmoid_amplitudes, sigmoid_target, gene_weights
from .sigma import sigma_pairs, kon_matrix as kon_ref_vector  # noqa: F401 (kon_ref_vector: public name)
from .sampling import stratified_choice

EPS = 1e-16



def minimal_repetition_choice(N, M, seed=None, labels=None):
    """M indices in range(N), each repeated as evenly as possible; the remainder is
    cell-type proportional if labels (length N) is given."""
    if seed is not None:
        np.random.seed(seed)

    result = []
    full_cycles = M // N
    remainder = M % N
    
    for _ in range(full_cycles):
        cycle = np.arange(N)
        np.random.shuffle(cycle)
        result.append(cycle)
    
    if remainder > 0:
        remainder_sample = stratified_choice(np.arange(N), remainder, labels)
        result.append(remainder_sample)
    
    return np.concatenate(result)


@njit(cache=True)
def find_next_prot_mixed(d1, P0, M0, M1, mode_init, mode_end, alpha, s, delta_t):
    """
    Deterministic flow interpolating between two points
    """

    P_int = (P0 - s * (M0 + (M0 - mode_init) / d1)) * np.exp(-d1 * delta_t * alpha) + s * (M0 + (mode_init - M0) * (alpha * delta_t * d1 - 1) / d1)
    return (P_int - s * (mode_end + (mode_end - M1) / d1)) * np.exp(-d1 * delta_t * (1-alpha)) + s * (mode_end + (M1 - mode_end) * ((1-alpha) * delta_t * d1 - 1) / d1)


@njit(cache=True)
def find_next_prot(d1, P0, M0, M1, mode_init, mode_end, alpha, s, delta_t):
    """
    Deterministic flow interpolating between two points
    """
    
    Pint = mode_init * s + (P0 - mode_init * s) * np.exp(-d1*delta_t*alpha)
    return mode_end * s + (Pint - mode_end * s) * np.exp(-d1*delta_t*(1-alpha))


@njit(cache=True)
def find_next_prot_dilution(d1, d1_end, P0, mode_init, mode_end, alpha, alpha_b, s, delta_t):
    """
    Flow of find_next_prot with dilution (only used with protein_dilution): the modes switch at alpha, the relaxation rate at alpha_b (birth_alpha),
    from d1 (source: d1 + its birth) to d1_end (target: d1 + its birth). mode_init / mode_end are already compressed
    by the dilution of their own state (u d1 / rate); between the two switches, the target is the mode of one state
    compressed by the rate of the other. Same birth on both sides (d1_end = d1): exactly find_next_prot.
    """
    a1 = np.minimum(alpha, alpha_b)
    a2 = np.maximum(alpha, alpha_b)
    f = 1.0 * (alpha < alpha_b)  # 1: the modes switch first
    mode_mid = f * (mode_end * d1_end / d1) + (1.0 - f) * (mode_init * d1 / d1_end)
    rate_mid = f * d1 + (1.0 - f) * d1_end
    P1 = mode_init * s + (P0 - mode_init * s) * np.exp(-d1 * delta_t * a1)
    P2 = mode_mid * s + (P1 - mode_mid * s) * np.exp(-rate_mid * delta_t * (a2 - a1))
    return mode_end * s + (P2 - mode_end * s) * np.exp(-d1_end * delta_t * (1 - a2))


@njit(cache=True)
def _exp_conv(lam, rho, h):
    # int_0^h e^{-lam s} e^{-rho (h - s)} ds, stable when rho ~ lam (rho - lam >= 0 or not small)
    dl = np.maximum(rho - lam, 1e-12) + np.minimum(rho - lam, 0.0)
    return np.exp(-lam * h) * (-np.expm1(-dl * h)) / dl


@njit(cache=True)
def _exp_conv3(a, b, c, h):
    # int_0^h e^{-c (h - s)} _exp_conv(a, b, s) ds
    dl = np.maximum(b - a, 1e-8) + np.minimum(b - a, 0.0)
    return (_exp_conv(a, c, h) - _exp_conv(a + dl, c, h)) / dl


@njit(cache=True)
def _bridge_step(d1, d0, st, m, r, h):
    """
    Advance over h, with constant mRNA level m and protein rate r, the moments of the linear (x, P) system
    (x shot noise relaxing at d0, P' = d1 x - r P), variances in burst-size units: st = (mu, V, Pf, Q, S, Cx, Cp) =
    mean and variance of x, mean of P, Cov(x, P), Var(P), and the covariances of x and P with P at an earlier time.
    """
    mu, V, Pf, Q, S, Cx, Cp = st
    rho = r + d0
    e_r, e_rho, e_2r = np.exp(-r * h), np.exp(-rho * h), np.exp(-2 * r * h)
    E_rho_2r = _exp_conv(rho, 2 * r, h)
    Pf_n = Pf * e_r + d1 * m * (-np.expm1(-r * h)) / r + d1 * (mu - m) * _exp_conv(d0, r, h)
    Q_n = Q * e_rho + d1 * m * (-np.expm1(-rho * h)) / rho + d1 * (V - m) * _exp_conv(2 * d0, rho, h)
    S_n = S * e_2r + 2 * d1 * (Q * E_rho_2r + d1 * m / rho * ((-np.expm1(-2 * r * h)) / (2 * r) - E_rho_2r)
                               + d1 * (V - m) * _exp_conv3(2 * d0, rho, 2 * r, h))
    Cp_n = Cp * e_r + d1 * Cx * _exp_conv(d0, r, h)
    return (m + (mu - m) * np.exp(-d0 * h), m + (V - m) * np.exp(-2 * d0 * h), Pf_n, Q_n, S_n,
            Cx * np.exp(-d0 * h), Cp_n)


@njit(cache=True)
def _bridge_moments(d1, d0, d0_end, r0, r1, P0, x0, m0, m1, alpha, alpha_b, T, tau):
    """Moments of _bridge_step at tau and at T (mRNA levels m0 before alpha T, m1 after; protein rates r0 and mRNA
    rates d0 before alpha_b T, r1 and d0_end after), from (x0, P0) at 0."""
    a1 = np.minimum(alpha, alpha_b) * T
    a2 = np.maximum(alpha, alpha_b) * T
    f = 1.0 * (alpha < alpha_b)  # 1: the modes switch first
    m_mid = f * m1 + (1.0 - f) * m0
    r_mid = f * r0 + (1.0 - f) * r1
    d0_mid = f * d0 + (1.0 - f) * d0_end
    zero = 0.0 * (x0 + P0 + a1 + tau)
    st = (x0 + zero, zero, P0 + zero, zero, zero, zero, zero)
    # Forward to tau
    st = _bridge_step(d1, d0, st, m0, r0, np.minimum(tau, a1))
    st = _bridge_step(d1, d0_mid, st, m_mid, r_mid, np.minimum(np.maximum(tau - a1, 0.0), a2 - a1))
    st = _bridge_step(d1, d0_end, st, m1, r1, np.maximum(tau - a2, 0.0))
    st_tau = st
    # Then to T, with the covariances to P(tau)
    st = (st[0], st[1], st[2], st[3], st[4], st[3], st[4])
    st = _bridge_step(d1, d0, st, m0, r0, np.maximum(a1 - tau, 0.0))
    st = _bridge_step(d1, d0_mid, st, m_mid, r_mid, np.maximum(a2 - np.maximum(tau, a1), 0.0))
    st = _bridge_step(d1, d0_end, st, m1, r1, T - np.maximum(tau, a2))
    return st_tau, st


@njit(cache=True)
def prot_bridge_end(d1, d0, d0_end, r0, r1, P0, x0, x1, m0, m1, alpha, alpha_b, T):
    """
    Protein at T of the linear (two-moment) bridges of the bursty mRNA from x0 (t = 0) to x1 (t = T), P' = d1 x - r P:
    conditional mean given x1. mRNA levels (kon units) m0 before alpha T, m1 after; protein rates r0 and mRNA rates
    d0 before alpha_b T, r1 and d0_end after (r0 = r1 = d1 without dilution, d0_end = d0 without mrna_dilution).
    """
    _, (muT, VT, PfT, QT, _, _, _) = _bridge_moments(d1, d0, d0_end, r0, r1, P0, x0, m0, m1, alpha, alpha_b, T, T)
    return np.maximum(PfT + QT / np.maximum(VT, 1e-12) * (x1 - muT), 0.0)


@njit(inline='always')
def _exp_conv_s(lam, rho, h):
    # Scalar _exp_conv
    dl = max(rho - lam, 1e-12) + min(rho - lam, 0.0)
    return math.exp(-lam * h) * (-math.expm1(-dl * h)) / dl


@njit(inline='always')
def _bridge_step_end(d1, d0, mu, V, Pf, Q, m, r, h):
    # Scalar _bridge_step restricted to (mu, V, Pf, Q), the moments of prot_bridge_end
    rho = r + d0
    Pf = Pf * math.exp(-r * h) + d1 * m * (-math.expm1(-r * h)) / r + d1 * (mu - m) * _exp_conv_s(d0, r, h)
    Q = Q * math.exp(-rho * h) + d1 * m * (-math.expm1(-rho * h)) / rho + d1 * (V - m) * _exp_conv_s(2 * d0, rho, h)
    return m + (mu - m) * math.exp(-d0 * h), m + (V - m) * math.exp(-2 * d0 * h), Pf, Q


@njit(cache=True)
def prot_bridge_end_row(d1, d0, d0_end, r0, r1, P0, x0, x1, fx1, m0, m1, fm1, alpha, alpha_b, T, out):
    """prot_bridge_end of one pair (1-D arrays (G,)) written into out, without temporaries (inner loop of the OT
    cost); the end levels are x1 * fx1 and m1 * fm1."""
    for g in range(out.shape[0]):
        a1 = min(alpha[g], alpha_b[g]) * T
        a2 = max(alpha[g], alpha_b[g]) * T
        f = alpha[g] < alpha_b[g]  # the modes switch first
        m_end = m1[g] * fm1
        m_mid = m_end if f else m0[g]
        r_mid = r0[g] if f else r1[g]
        d0_mid = d0[g] if f else d0_end[g]
        mu, V, Pf, Q = _bridge_step_end(d1[g], d0[g], x0[g], 0.0, P0[g], 0.0, m0[g], r0[g], a1)
        mu, V, Pf, Q = _bridge_step_end(d1[g], d0_mid, mu, V, Pf, Q, m_mid, r_mid, a2 - a1)
        mu, V, Pf, Q = _bridge_step_end(d1[g], d0_end[g], mu, V, Pf, Q, m_end, r1[g], T - a2)
        out[g] = max(Pf + Q / max(V, 1e-12) * (x1[g] * fx1 - mu), 0.0)


@njit(cache=True)
def prot_bridge(d1, d0, d0_end, r0, r1, P0, x0, x1, m0, m1, alpha, alpha_b, T, tau, P1):
    """
    Mean protein at tau in [0, T] of the bridges of prot_bridge_end, conditioned on x1 and on the protein P1 at T
    (P1 = prot_bridge_end).
    """
    (_, _, Pf, _, _, _, _), (muT, VT, PfT, QT, ST, Cx, Cp) = _bridge_moments(
        d1, d0, d0_end, r0, r1, P0, x0, m0, m1, alpha, alpha_b, T, tau)
    VT = np.maximum(VT, 1e-12)
    det = VT * ST - QT * QT
    ok = 1.0 * (det > 1e-10 * (VT * ST + 1e-300))
    det = np.maximum(det, 1e-300)
    # Weights of (x1, P1) in the conditional mean; P1 dropped where P(T) is (nearly) determined by x(T)
    wx = ok * (ST * Cx - QT * Cp) / det + (1.0 - ok) * Cx / VT
    wp = ok * (VT * Cp - QT * Cx) / det
    return np.maximum(Pf + wx * (x1 - muT) + wp * (P1 - PfT), 0.0)


@njit(inline='always')
def _bridge_step_s(d1, d0, mu, V, Pf, Q, S, Cx, Cp, m, r, h):
    # Scalar _bridge_step (all seven moments)
    rho = r + d0
    E_rho_2r = _exp_conv_s(rho, 2 * r, h)
    dl = max(rho - 2 * d0, 1e-8) + min(rho - 2 * d0, 0.0)
    E3 = (_exp_conv_s(2 * d0, 2 * r, h) - _exp_conv_s(2 * d0 + dl, 2 * r, h)) / dl
    S = S * math.exp(-2 * r * h) + 2 * d1 * (Q * E_rho_2r + d1 * m / rho * ((-math.expm1(-2 * r * h)) / (2 * r) - E_rho_2r)
                                         + d1 * (V - m) * E3)
    Cp = Cp * math.exp(-r * h) + d1 * Cx * _exp_conv_s(d0, r, h)
    mu, V, Pf, Q = _bridge_step_end(d1, d0, mu, V, Pf, Q, m, r, h)
    return mu, V, Pf, Q, S, Cx * math.exp(-d0 * h), Cp


@njit(cache=True)
def prot_bridge_row(d1, d0, d0_end, r0, r1, P0, x0, x1, fx1, m0, m1, fm1, alpha, alpha_b, T, tau, P1, out):
    """prot_bridge of one pair (1-D arrays (G,), tau (G,)) written into out, without temporaries (inner loops of the
    OT cost); the end levels are x1 * fx1 and m1 * fm1."""
    for g in range(out.shape[0]):
        a1 = min(alpha[g], alpha_b[g]) * T
        a2 = max(alpha[g], alpha_b[g]) * T
        f = alpha[g] < alpha_b[g]  # the modes switch first
        m_end = m1[g] * fm1
        m_mid = m_end if f else m0[g]
        r_mid = r0[g] if f else r1[g]
        d0_mid = d0[g] if f else d0_end[g]
        t = tau[g]
        # Forward to tau
        mu, V, Pf, Q, S, _, _ = _bridge_step_s(d1[g], d0[g], x0[g], 0.0, P0[g], 0.0, 0.0, 0.0, 0.0, m0[g], r0[g],
                                               min(t, a1))
        mu, V, Pf, Q, S, _, _ = _bridge_step_s(d1[g], d0_mid, mu, V, Pf, Q, S, 0.0, 0.0, m_mid, r_mid,
                                               min(max(t - a1, 0.0), a2 - a1))
        mu, V, Pf, Q, S, _, _ = _bridge_step_s(d1[g], d0_end[g], mu, V, Pf, Q, S, 0.0, 0.0, m_end, r1[g],
                                               max(t - a2, 0.0))
        Pf_tau = Pf
        # Then to T, with the covariances to P(tau)
        mu, V, Pf, Q, S, Cx, Cp = _bridge_step_s(d1[g], d0[g], mu, V, Pf, Q, S, Q, S, m0[g], r0[g],
                                                 max(a1 - t, 0.0))
        mu, V, Pf, Q, S, Cx, Cp = _bridge_step_s(d1[g], d0_mid, mu, V, Pf, Q, S, Cx, Cp, m_mid, r_mid,
                                                 max(a2 - max(t, a1), 0.0))
        mu, V, Pf, Q, S, Cx, Cp = _bridge_step_s(d1[g], d0_end[g], mu, V, Pf, Q, S, Cx, Cp, m_end, r1[g],
                                                 T - max(t, a2))
        V = max(V, 1e-12)
        det = V * S - Q * Q
        # Weights of (x1, P1) in the conditional mean; P1 dropped where P(T) is (nearly) determined by x(T)
        if det > 1e-10 * (V * S + 1e-300):
            wx, wp = (S * Cx - Q * Cp) / det, (V * Cp - Q * Cx) / det
        else:
            wx, wp = Cx / V, 0.0
        out[g] = max(Pf_tau + wx * (x1[g] * fx1 - mu) + wp * (P1[g] - Pf), 0.0)


def birth_alpha(alpha, kon_init, kon_end, tol=0.6):
    """
    Switching fractions of the birth rate (dilution) in the equation of each gene (find_next_prot_dilution), rows = steps (..., G):
    alpha_b = w alpha + (1 - w) mean_alpha, with w the weight of the change of mode of the gene (as in inference_alpha)
    and mean_alpha the mean fraction of the genes that change mode (weighted by w; 1/2 if none). A gene that changes
    mode switches its birth rate with its modes, the others at the time of the global change of state. The modes
    still switch at alpha (their mRNA-shrunk levels differ between the two ends even without change of mode), and
    alpha itself is unchanged.
    """
    w = np.minimum((np.abs(kon_end - kon_init) / max(tol, EPS)) ** (1 / tol), 1)
    w_sum = w.sum(axis=-1, keepdims=True)
    mean_alpha = np.where(w_sum > 0, (w * alpha).sum(axis=-1, keepdims=True) / np.maximum(w_sum, EPS), 0.5)
    return w * alpha + (1 - w) * mean_alpha


@njit(cache=True)
def _birth_alpha_row(alpha, kon_init, kon_end, tol):
    """birth_alpha for one step (1-D arrays), inside numba loops."""
    w = np.minimum((np.abs(kon_end - kon_init) / max(tol, 1e-12)) ** (1 / tol), 1.0)
    w_sum = w.sum()
    mean_alpha = (w * alpha).sum() / w_sum if w_sum > 0 else 0.5
    return w * alpha + (1 - w) * mean_alpha


def count_errors(vect_prot, vect_kon, ks, Y, X, n_stimuli=1, spec=None, mode=0, mid=None, w=None):
    """
    Network cost (cost_any of code mode; central modes mid (G - n_stimuli,) for the ordinal cost) between the sigmoid
    of the network (basal Y, interactions X; spec: families of sigma) and the sigmoid target of the kon vect_kon, mean
    over the cells and over the genes with the weights w (G - n_stimuli,) (None: the genes of two distinct modes,
    equally weighted).
    """
    N, G = vect_prot.shape
    ns = n_stimuli
    sigma = sigma_pairs(Y, X, vect_prot, spec)[:, ns:, 1]
    kb = sigmoid_amplitudes(ks)[ns:]
    w = (gene_weights(ks)[ns:] > 0).astype(float) if w is None else np.asarray(w, dtype=float)
    ce = cost_any(sigma, sigmoid_target(vect_kon[:, ns:], kb[:, 0], kb[:, 1]), mode, mid)
    return float(np.sum(ce * w)) / (N * max(float(np.sum(w)), EPS))


def ks_of(ks, s_idx):
    """Mixture amplitudes of sample s_idx: per-sample when ks is 3-D (S, G, n_modes), else shared."""
    return ks[min(s_idx, ks.shape[0] - 1)] if np.ndim(ks) == 3 else ks


def inter_of(inter, s_idx):
    """Interactions of sample s_idx: per-sample when inter is 3-D (S, G, G) (network conditions), else shared (G, G)."""
    return inter[min(s_idx, inter.shape[0] - 1)] if np.ndim(inter) == 3 else inter


def s1_of(s1, s_idx):
    """mRNA-to-protein factor of sample s_idx: per-sample when s1 is 2-D (S, G), else shared."""
    return s1[min(s_idx, s1.shape[0] - 1)] if np.ndim(s1) == 2 else s1


def s1_rows(s1, samples_data):
    """s1 for each row (cell) of samples_data (sample positions as in _kon_per_sample)."""
    if np.ndim(s1) < 2 or samples_data is None:
        return s1
    pos = np.searchsorted(np.sort(np.unique(samples_data)), samples_data)
    return s1[np.minimum(pos, s1.shape[0] - 1)]


def _kon_per_sample(y_prot, ks, inter, basal, samples_data=None, spec=None):
    """
    kon_ref_vector with per-sample basal and mixture support (spec: families of sigma, None = logistic).

    When neither basal is 2-D (n_samples, G) nor ks is 3-D (n_samples, G, n_modes)
    nor inter is 3-D (n_samples, G, G), or samples_data is None, delegates to
    kon_ref_vector directly. Otherwise routes each cell to its sample's slices via samples_data.
    """
    if (basal.ndim < 2 and np.ndim(ks) < 3 and np.ndim(inter) < 3) or samples_data is None:
        if np.ndim(inter) == 3:
            raise ValueError("per-sample interactions (network conditions) need samples_data")
        return kon_ref_vector(y_prot, ks, inter, basal, spec)
    samples_id = np.sort(np.unique(samples_data))
    out = np.zeros((y_prot.shape[0], y_prot.shape[1]))
    for s_idx, s in enumerate(samples_id):
        mask = (samples_data == s)
        if not np.any(mask):
            continue
        basal_s = basal[min(s_idx, basal.shape[0] - 1)] if basal.ndim == 2 else basal
        out[mask] = kon_ref_vector(y_prot[mask], ks_of(ks, s_idx), inter_of(inter, s_idx), basal_s, spec)
    return out

    
@njit(cache=True)
def _pair_state(f, kind, d1, d0_i, d0_end_j, d1_i, d1_end_j, prot_init_i, x0, x1, mode_init_raw, mode_init_i,
                mode_end_j, alpha_i, alpha_b_j, delta_t, prot_end_j, rna_init_i, rna_end_j, c_i, s1):
    """Proteins (G,) of the pair (source i, target j) at the fractions f (G,) of the interval (one per gene), with the same
    bridge or flow as its end state (kind 0: mRNA bridge, 1: dilution, 2: flow of find_next_prot); f -> 0: start state."""
    f = np.maximum(f, 1e-9)
    if kind == 0:
        out = np.empty(f.shape[0])
        prot_bridge_row(d1, d0_i, d0_end_j, d1_i, d1_end_j, prot_init_i, x0, x1, 1.0, mode_init_raw, mode_end_j, 1.0,
                        alpha_i, alpha_b_j, delta_t, f * delta_t, prot_end_j, out)
        return out
    if kind == 1:
        return find_next_prot_dilution(d1_i, d1_end_j, prot_init_i, mode_init_i, mode_end_j * (d1 / d1_end_j),
                                       np.minimum(alpha_i / f, 1.0), np.minimum(alpha_b_j / f, 1.0), s1, delta_t * f)
    return find_next_prot(d1_i, prot_init_i, rna_init_i, rna_end_j, mode_init_i, mode_end_j * c_i,
                          np.minimum(alpha_i / f, 1.0), s1, delta_t * f)


@njit(fastmath=True, parallel=True, cache=True)
def my_otdistance(vect_kon_init, vect_kon_end, vect_prot_init, vect_rna_init, vect_rna_end,
                            mode_init, mode_end, alpha, s1, d1, delta_t, basal, inter, tau_end, w_genes,
                            n_iter=1, intensity_prior=1,
                            n_stimuli=1, stim_vals=np.ones(1), scale_proteins=1, weight_fixed=-1.0,
                            b_init=np.zeros(0), b_end=np.zeros(0), alpha_tol=0.6,
                            d0=np.zeros(0), rna_to_x=1.0, mrna_dil=False,
                            g_active=np.zeros(0), spec=None, mem_frac=np.zeros(0),
                            mem_w=np.zeros((0, 0)), net_mode=0, net_mid=np.zeros(0), lik_ll=np.zeros((0, 0, 0)),
                            lik_tl=np.zeros((0, 0, 0)), lik_kap=np.zeros((0, 0)),
                            tau_end2=np.zeros((0, 0)), lik_ok=np.zeros((0, 0, 0)), g_init=np.zeros(0),
                            g_end=np.zeros(0)) -> np.ndarray:
    """
    OT cost between n1 trajectory ends and n2 real cells: w A + (1 - w) [(1 - 1/G) B + (1/G) P], with
    A = |Δ mRNA s1|² (mRNA in kon units, as the modes and proteins: k_z / k1 at the mean of mode z, up to the
    constant fact_simple), B = network term, P = |Δ protein|², A and P brought to the median of B over the (n1, n2) pairs
    (the OT regularizations are relative to the cost, see estimate_trajectories_given_model), and w = weight_fixed if >= 0, else decreasing with n_iter while n_iter <= intensity_prior.
    At the first training pass (n_iter = 1, weight_fixed < 0) with intensity_prior < 1, the cost is zero (random
    couplings). b_init (n1,), b_end (n2,): birth
    rates of the sources and of the targets (dilution: rate d1 + b, mode targets times d1 / (d1 + b)), switched in the
    equation of each gene at its birth_alpha (alpha_tol: threshold of inference_alpha); empty = no dilution. g_init (n1,),
    g_end (n2,): with them, protein synthesis factors of the sources and targets (mRNA content of growing cells: mode
    targets times g, and for the bridges the mRNA and kon levels times g, switched with the modes at alpha); empty = 1. Only the cost is returned: the (n1, n2, G) end states are never
    stored; recompute them with find_next_prot / find_next_prot_dilution for the sampled pairs.
    Network term: network cost of code net_mode (fit_ot_cost: cross-entropy, euclidean, or ordinal with the central
    modes net_mid (G,)) between the sigmoid of the network at the end states and tau_end (n2, G), the sigmoid targets
    of the real cells (sigmoid_target), summed over the genes with the weights w_genes (G,); ordinal cost with
    tau_end2 (n2, G) given: tau_end and tau_end2 are the warped targets (u1, u2) of the real cells (soft targets:
    expectations under their posteriors, ordinal_cost_w); otherwise the warps of tau_end, computed once per call
    with the constants of the warps of each gene. lik_ll (n2, G, L) given
    (fit_likelihood 'network+ot'): the network term is instead the likelihood of the counts of each real cell under
    the prior exp(-kappa C) of the cost of code net_mode on its levels, at the sigma of the end state
    (lik_terms, shifted by the best log-likelihood of each gene: log-likelihoods lik_ll at the levels lik_tl (n2, G,
    L) of the mask lik_ok (n2, G, L) (1: level of the gene; lik_ll finite on it), precisions lik_kap (n2, G); the
    targets of the levels computed once per call, lik_levels).
    d0 (G,) given: end proteins from the mRNA bridges (prot_bridge_end), mode_init / mode_end being then the kon levels
    of the modes and the mRNA rna * s1 * rna_to_x in the same units; empty: find_next_prot on shrunk modes. mrna_dil
    (mrna_dilution, with b_init / b_end): mRNA of the bridges relaxing at d0 + b, switched at birth_alpha as the protein
    rate. g_active (G,): 1 for the genes of the sample's condition, 0 for the others (left out of
    the mRNA and protein terms; the network term already ignores them through w_genes); empty = every gene.
    spec: families of the GRN sigma (model_learning_grn), None = logistic.
    mem_frac (Q,), mem_w (Q + 1, G) given (targets_accounting_late_mrna): network term on sigma averaged along the mRNA
    memory of each target, w_0 sigma(end) + sum_q w_q sigma(P(mem_frac_q delta_t)), the proteins at the nodes from the
    same bridge or flow as the end state.
    """
    n1, G = vect_rna_init.shape
    gmask = g_active if g_active.shape[0] == G else np.ones(G)
    G_eff = max(np.sum(gmask), 1.0)
    bridge = d0.shape[0] == G
    n2 = vect_rna_end.shape[0]
    ns = n_stimuli
    # First pass with intensity_prior < 1 (training): zero cost, i.e. random couplings (independent coupling)
    if weight_fixed < 0 and n_iter == 1 and intensity_prior < 1:
        return np.zeros((n1, n2))

    # Preallocation (important for Numba): the three terms of the cost
    cost_rna = np.zeros((n1, n2))
    cost_net = np.zeros((n1, n2))
    cost_prot = np.zeros((n1, n2))
    # mRNA in kon units (s1 = fact_simple c / k1 of the sample)
    kon_rna_end = vect_rna_end * s1
    kon_rna_init = vect_rna_init * s1

    weight_init: float = weight_fixed if weight_fixed >= 0 else (n_iter <= intensity_prior) * (1 / n_iter)**(1 - 1/n_iter)

    # Computed once, before the parallel loop (arrays allocated unconditionally: a parfor may read the first
    # definition of an array redefined in a branch). Ordinal cost: constants of the warps of each gene (net_mid,
    # unused for the other costs) and warped targets of the real cells (u1, u2)
    ord_c = ordinal_consts(np.ascontiguousarray(net_mid))
    n_ord = n2 if net_mode == 2 and lik_ll.shape[0] == 0 else 0
    u1_end, u2_end = np.zeros((n_ord, G)), np.zeros((n_ord, G))
    for j in range(n_ord):
        if tau_end2.shape[0] > 0:  # soft targets: their expected warps
            u1_end[j], u2_end[j] = tau_end[j], tau_end2[j]
        else:
            w1, w2 = ordinal_targets_c(np.ascontiguousarray(tau_end[j]), ord_c)
            u1_end[j], u2_end[j] = w1, w2
    # Likelihood: targets of the levels of the real cells and their best log-likelihoods (shift >= 0)
    n_lik, n_lev = lik_ll.shape[0], lik_ll.shape[2]
    lik_u1, lik_u2, lik_ent = np.zeros((n_lik, G, n_lev)), np.zeros((n_lik, G, n_lev)), np.zeros((n_lik, G, n_lev))
    lik_best = np.zeros(n_lik)
    for j in range(n_lik):
        a1, a2, a3 = lik_levels(np.ascontiguousarray(lik_tl[j]), np.ascontiguousarray(lik_ok[j]), net_mode, ord_c)
        lik_u1[j], lik_u2[j], lik_ent[j] = a1, a2, a3
        lik_best[j] = np.sum(w_genes * best_loglik(np.ascontiguousarray(lik_ll[j]), np.ascontiguousarray(lik_ok[j])))

    # mRNA memory read before the parallel loop: inside it, the invariant mem_w[0] and mem_frac[0] would be hoisted out
    # of their conditions (out of bounds without memory). single: one node of weight 1 (late_mrna_nodes = 1)
    n_mem = mem_frac.shape[0]
    mem_w0, f0 = np.zeros(G), np.zeros(G)
    if n_mem > 0:
        mem_w0[:] = mem_w[0]
        f0[:] = mem_frac[0]
    single = n_mem == 1 and np.max(mem_w0) == 0.0

    for i in prange(n1):  # parallelize cell-by-cell

        prot_init_i = vect_prot_init[i].copy()
        kon_rna_init_i = kon_rna_init[i]
        rna_init_i = vect_rna_init[i]
        mode_init_i = mode_init[i]
        kon_init_i = vect_kon_init[i]
        alpha_i = alpha[i]
        # Dilution at the birth rate of the source before alpha: rate d1 + b, targets compressed by d1 / (d1 + b) and
        # synthesised at g
        g_i = g_init[i] if g_init.shape[0] == n1 else 1.0
        if b_init.shape[0] == n1:
            d1_i = d1 + b_init[i]
            c_i = g_i * d1 / d1_i
        else:
            d1_i = d1.copy()
            c_i = np.ones_like(d1)
        mode_init_i = mode_init_i * c_i
        # Bridges: mRNA and kon levels of the source times g
        x0_i = kon_rna_init_i * rna_to_x * g_i
        mode_raw_i = mode_init[i] * g_i
        g_end_i = np.ones(n2)
        # mRNA of the bridges (mrna_dilution): rate d0 + b of the source before alpha_b, of the target after
        dil0 = mrna_dil and bridge and b_init.shape[0] == n1
        d0_i = d0 + b_init[i] if dil0 else d0.copy()

        prot_end_i = np.zeros((n2, G))
        alpha_b_i = np.zeros((n2, G))
        d1_end_i = np.zeros((n2, G))
        d0_end_i = np.zeros((n2, G))
        # Full end states (stimuli + genes) of cell i, for the kon computation
        prot_full_i = np.empty((n2, G + ns))
        prot_full_i[:, :ns] = stim_vals * scale_proteins

        # --- Loop over target cells j (birth rate of the target after alpha) ---
        for j in range(0, n2):
            if bridge:
                r_j = d1 + b_end[j] if b_end.shape[0] == n2 else d1_i
                d0_j = d0 + b_end[j] if (dil0 and b_end.shape[0] == n2) else d0_i
                alpha_b = _birth_alpha_row(alpha_i, kon_init_i, vect_kon_end[j], alpha_tol) \
                    if b_end.shape[0] == n2 else alpha_i
                g_j = g_end[j] if g_end.shape[0] == n2 else g_i
                prot_bridge_end_row(d1, d0_i, d0_j, d1_i, r_j, prot_init_i, x0_i, kon_rna_end[j], rna_to_x * g_j,
                                    mode_raw_i, mode_end[j], g_j, alpha_i, alpha_b, delta_t, prot_end_i[j])
                alpha_b_i[j], d1_end_i[j], d0_end_i[j], g_end_i[j] = alpha_b, r_j, d0_j, g_j
            elif b_end.shape[0] == n2:
                d1_j = d1 + b_end[j]
                g_j = g_end[j] if g_end.shape[0] == n2 else 1.0
                alpha_b = _birth_alpha_row(alpha_i, kon_init_i, vect_kon_end[j], alpha_tol)
                prot_end_i[j, :] = find_next_prot_dilution(d1_i, d1_j, prot_init_i, mode_init_i,
                                                           mode_end[j] * (g_j * d1 / d1_j), alpha_i, alpha_b, s1, delta_t)
                alpha_b_i[j], d1_end_i[j], g_end_i[j] = alpha_b, d1_j, g_j
            else:
                prot_end_i[j, :] = find_next_prot(d1_i, prot_init_i, rna_init_i, vect_rna_end[j], mode_init_i,
                                                  mode_end[j] * c_i, alpha_i, s1, delta_t)
        prot_full_i[:, ns:] = prot_end_i
        kind = 0 if bridge else (1 if b_end.shape[0] == n2 else 2)

        # --- Terms of the cost; network term = cross-entropy on the sigmoid of the genes, weighted by w_genes ---
        if single:  # single node of weight 1 (late_mrna_nodes = 1): sigma at the lagged proteins only
            prot_q = np.empty((n2, G + ns))
            prot_q[:, :ns] = stim_vals * scale_proteins
            for j in range(0, n2):
                prot_q[j, ns:] = _pair_state(f0, kind, d1, d0_i, d0_end_i[j], d1_i, d1_end_i[j], prot_init_i,
                                             x0_i, kon_rna_end[j] * rna_to_x * g_end_i[j], mode_raw_i,
                                             mode_init_i, mode_end[j] * g_end_i[j], alpha_i, alpha_b_i[j], delta_t,
                                             prot_end_i[j], rna_init_i, vect_rna_end[j], c_i, s1)
            sigma = sigma_pairs(basal, inter, prot_q, spec)
        else:
            sigma = sigma_pairs(basal, inter, prot_full_i, spec)
        if n_mem > 0 and not single:
            # sigma averaged along the mRNA memory of each target (nodes of the same bridge / flow)
            sig_bar = mem_w0 * sigma[:, ns:, 1]
            prot_q = np.empty((n2, G + ns))
            prot_q[:, :ns] = stim_vals * scale_proteins
            for q in range(n_mem):
                f = np.full(G, mem_frac[q])
                for j in range(0, n2):
                    prot_q[j, ns:] = _pair_state(f, kind, d1, d0_i, d0_end_i[j], d1_i, d1_end_i[j], prot_init_i,
                                                 x0_i, kon_rna_end[j] * rna_to_x * g_end_i[j], mode_raw_i,
                                                 mode_init_i, mode_end[j] * g_end_i[j], alpha_i, alpha_b_i[j], delta_t,
                                                 prot_end_i[j], rna_init_i, vect_rna_end[j], c_i, s1)
                sig_bar = sig_bar + mem_w[q + 1] * sigma_pairs(basal, inter, prot_q, spec)[:, ns:, 1]
            for j in range(0, n2):
                sigma[j, ns:, 1] = np.minimum(np.maximum(sig_bar[j], 0.0), 1.0)
        for j in range(0, n2):
            diff_prot = gmask * (prot_end_i[j] - prot_init_i) / scale_proteins
            diff_rna = gmask * (kon_rna_end[j] - kon_rna_init_i)
            cost_rna[i, j] = np.sum(diff_rna * diff_rna)
            if lik_ll.shape[0] > 0:
                cost_net[i, j] = lik_best[j] + np.sum(w_genes * lik_terms(
                    np.ascontiguousarray(sigma[j, ns:, 1]), lik_ll[j], lik_ok[j], lik_u1[j], lik_u2[j], lik_ent[j],
                    lik_kap[j], net_mode, ord_c, False)[0])
            elif net_mode == 2:  # warped targets of the real cells
                cost_net[i, j] = np.sum(w_genes * ordinal_terms(np.ascontiguousarray(sigma[j, ns:, 1]), u1_end[j],
                                                                u2_end[j], ord_c, False)[0])
            else:
                cost_net[i, j] = np.sum(w_genes * net_cost(sigma[j, ns:, 1], tau_end[j], net_mode))
            cost_prot[i, j] = np.sum(diff_prot * diff_prot)

    # mRNA and protein terms brought to the median of the network term, then mRNA term (weight_init) handing over to it
    median_ref = _median_scale(cost_net)
    cost_rna *= median_ref / _median_scale(cost_rna)
    cost_net *= median_ref /_median_scale(cost_net)
    cost_prot *= median_ref /_median_scale(cost_prot)
    dist = (cost_rna * weight_init + (1 - weight_init) *
            ((1.0 - 1.0 / G_eff) * cost_net + (1.0 / G_eff) * cost_prot))
    return dist


@njit(cache=True)
def _median_scale(cost):
    """Median of a cost term over the pairs (its mean if the median is 0, 1 if the term is 0 everywhere)."""
    m = np.median(cost)
    if m <= 1e-12:
        m = np.mean(cost)
    return m if m > 1e-12 else 1.0


def mrna_step_delay(d0, L):
    """Delay s of the step of the mRNA level equivalent to its exponential relaxation (rate d0) after a switch of kon,
    over the remaining time L: same integral of the mRNA up to L as the relaxation, the step reaching the (partially
    relaxed) level at L: s = L (1 / x - 1 / (e^x - 1)), x = d0 L; s <= L / 2, s -> 1 / d0 for x large."""
    x = np.maximum(np.asarray(d0, dtype=float) * L, 0.0)
    small = x < 1e-4
    xs = np.where(small, 1.0, x)
    return L * np.where(small, 0.5 - x / 12.0, 1.0 / xs - 1.0 / np.expm1(xs))


def kon_switch_fraction(alpha, d0, delta_t, n_iter=40):
    """Fraction t of the interval at which kon switches when the mRNA-level forcing switches at alpha
    (alpha = t + mrna_step_delay(d0, (1 - t) delta_t) / delta_t, increasing in t; bisection, 0 below its value at 0)."""
    alpha = np.asarray(alpha, dtype=float)
    d0 = np.broadcast_to(np.asarray(d0, dtype=float), alpha.shape)
    lo, hi = np.zeros(alpha.shape), np.minimum(alpha, 1.0)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        above = mid + mrna_step_delay(d0, (1.0 - mid) * delta_t) / delta_t > alpha
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    return 0.5 * (lo + hi)


def inference_alpha(d1, s1, alpha_init, y_kon_init_true, y_kon_init, y_prot_init, y_rna_init,
                    y_kon_end_true, y_kon_end, y_prot_end, y_rna_end, mode_init, mode_end,
                    basal, inter, ks, delta_t, tol=0.6, n_pas=25, samples_data=None, stim_vals=np.ones(1), scale_proteins=1,
                    d1_end=None, d0=None, d1_prod=None, rna_to_x=1.0, spec=None, d0_mem=None, d0_end=None):
    """
    Switching fractions: for each gene whose mode changes, the earliest fraction of the step at which the network
    commits it to its new mode (blended with the previous value by the size of the change). d0_mem (N, G_genes) given
    (targets_accounting_late_mrna, forcing = mRNA level): the mRNA follows the burst frequency with its memory, so the
    forcing switches after the commitment by the delay of the equivalent step (mrna_step_delay over the rest of the
    step, at most half of it); kon forcings (bridges, modes) switch at the commitment. d1 / d1_end: relaxation
    rates before / after the switch of the birth rate (d1 + birth of the source / of the target; d1_end=None: no
    dilution switch, find_next_prot),
    which happens at birth_alpha in the equation of each gene (alpha itself is not changed by it).
    d0 given: proteins from the mRNA bridges (prot_bridge, production d1_prod) conditioned on the end proteins
    y_prot_end, mode_init / mode_end being the kon levels and the mRNA y_rna * s1 * rna_to_x; d0_end: mRNA rate after
    the switch of the birth rate (mrna_dilution), None = d0.
    """
    stim_vals = np.asarray(stim_vals, dtype=float)
    ns = stim_vals.shape[-1]  # (ns,) or one row per cell (per-sample schedules)
    s1 = s1_rows(s1, samples_data)
    y_prot = np.ones_like(y_rna_end)
    y_prot[:, :ns] = stim_vals * scale_proteins
    diff_kon = np.abs(y_kon_end_true[:, ns:] - y_kon_init_true[:, ns:]) / max(tol, EPS)
    weights = np.minimum(diff_kon ** (1/tol), 1)

    alpha = alpha_init.copy()
    cnt = np.zeros_like(alpha_init, dtype=int)
    t = 1 / n_pas

    if d0 is not None:
        x_init, x_end = y_rna_init[:, ns:] * s1 * rna_to_x, y_rna_end[:, ns:] * s1 * rna_to_x
        r_end = d1 if d1_end is None else d1_end
        d0_e = d0 if d0_end is None else d0_end

    while t < 1 and np.min(cnt) < 1:
        tp = t
        if d0 is not None:
            alpha_b = alpha if d1_end is None else birth_alpha(alpha, y_kon_init_true[:, ns:], y_kon_end_true[:, ns:], tol)
            y_prot[:, ns:] = prot_bridge(d1_prod, d0, d0_e, d1, r_end, y_prot_init[:, ns:], x_init, x_end,
                                         mode_init, mode_end, alpha, alpha_b, delta_t, tp * delta_t, y_prot_end[:, ns:])
        elif d1_end is None:
            y_prot[:, ns:] = find_next_prot(d1, y_prot_init[:, ns:], y_rna_init[:, ns:], y_rna_end[:, ns:],
                        mode_init, mode_end, np.minimum(alpha / tp, 1.0), s1, tp * delta_t)
        else:
            alpha_b = birth_alpha(alpha, y_kon_init_true[:, ns:], y_kon_end_true[:, ns:], tol)
            y_prot[:, ns:] = find_next_prot_dilution(d1, d1_end, y_prot_init[:, ns:], mode_init, mode_end,
                        np.minimum(alpha / tp, 1.0), np.minimum(alpha_b / tp, 1.0), s1, tp * delta_t)
        kon_new = _kon_per_sample(y_prot, ks, inter, basal, samples_data=samples_data, spec=spec)  # shape (N, G)

        # Condition: if abs(kon_new[i] - y_kon_init[i]) > abs(kon_new[i] - y_kon_end[i])
        # Only apply where cnt < 1
        diff_init = np.abs(kon_new[:, ns:] - y_kon_init[:, ns:])
        diff_end = np.abs(kon_new[:, ns:] - y_kon_end[:, ns:])
        bool_var = (cnt == 0) & (diff_init >= diff_end)
        # Switch of the forcing: the commitment, delayed by the equivalent step of the mRNA (forcing = mRNA level)
        t_switch = t if d0_mem is None else np.minimum(t + mrna_step_delay(d0_mem, (1 - t) * delta_t) / delta_t, 1.0)
        t_switch = np.broadcast_to(t_switch, alpha.shape)
        alpha[bool_var] = alpha[bool_var] * (1 - weights[bool_var]) + t_switch[bool_var] * weights[bool_var]
        cnt[bool_var] = 1

        t += 1 / n_pas

    return alpha


def inference_alpha_1thread(*args, **kwargs):
    """inference_alpha with numba on one thread: joblib task over intervals, without oversubscription."""
    n_threads = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        return inference_alpha(*args, **kwargs)
    finally:
        numba.set_num_threads(n_threads)
