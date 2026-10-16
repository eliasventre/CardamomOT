"""
Degradation rates refined inside the trajectory loop of fit_network (fit_degradations_in_loop): d1 by the network term
at the end of each step (theta, alpha and couplings fixed). fit_d0_mrna (likelihood of the mRNA along the couplings) is
kept for experiments but unused: the end modes being assigned from the mRNA itself, it always favours a larger d0.
"""

import numpy as np
import torch
from scipy.optimize import minimize

EPS_CE = 1e-6


def fit_d0_mrna(x_a, x_b, m_a, m_b, alpha, dt, beta, nu, d0_prior, lam, span=10.0, n_grid=81, use_cov=False):
    """
    d0 (G,) maximizing the Gaussian (two-moment) likelihood of the mRNA x_b at the end of each step given x_a at its
    start, along the couplings, plus lam (log d0 - log d0_prior)^2. Steps (n, G): mRNA x (kon units), mode levels m_a
    before alpha dt and m_b after (shot noise of mean m, variance beta m at equilibrium), Poisson counting noise nu m
    at both ends; grid search over [d0_prior / span, d0_prior * span]. use_cov=False: x_b independent of x_a given the
    modes (the OT couplings pair cells by mode, not by mRNA level within a mode), d0 then set by the relaxation of the
    mean and variance after the switch.
    """
    grid = np.exp(np.linspace(-np.log(span), np.log(span), n_grid))  # factors of d0_prior
    h = (1.0 - alpha) * dt[:, None]  # time after the switch
    var_a = (beta + nu) * m_a
    obj = np.zeros((n_grid, len(d0_prior)))
    for k, fac in enumerate(grid):
        d0 = d0_prior * fac
        e1, eT = np.exp(-d0 * h), np.exp(-d0 * dt[:, None])
        mu_b = m_b + (m_a - m_b) * e1
        var_b = beta * (m_b + (m_a - m_b) * e1 ** 2) + nu * mu_b
        cov = eT * beta * m_a * use_cov
        c_mean = mu_b + cov / var_a * (x_a - m_a)
        c_var = np.maximum(var_b - cov ** 2 / var_a, 1e-12)
        obj[k] = (0.5 * (np.log(2 * np.pi * c_var) + (x_b - c_mean) ** 2 / c_var)).mean(axis=0) \
            + lam * np.log(fac) ** 2
    return d0_prior * grid[np.argmin(obj, axis=0)]


def _flow_torch(rate_a, rate_b, P0, mode_init, mode_end, alpha, alpha_b, s, dt):
    """find_next_prot_dilution in torch (rates and compressed modes already given)."""
    a1, a2 = torch.minimum(alpha, alpha_b), torch.maximum(alpha, alpha_b)
    f = (alpha < alpha_b).to(P0.dtype)
    mode_mid = f * (mode_end * rate_b / rate_a) + (1 - f) * (mode_init * rate_a / rate_b)
    rate_mid = f * rate_a + (1 - f) * rate_b
    P1 = mode_init * s + (P0 - mode_init * s) * torch.exp(-rate_a * dt * a1)
    P2 = mode_mid * s + (P1 - mode_mid * s) * torch.exp(-rate_mid * dt * (a2 - a1))
    return mode_end * s + (P2 - mode_end * s) * torch.exp(-rate_b * dt * (1 - a2))


def fit_d1_network(P_a, mode_a, mode_b, alpha, alpha_b, s, dt, b_a, b_b, stim, basal_rows, inter, tau, w,
                   d1_cur, d1_prior, lam, lb, ub, ns=1):
    """
    d1 (G,) minimizing the network term at the end of the steps, mean over the steps and genes (weights w) of the
    cross-entropy between sigmoid(basal + theta . P_b(d1)) and the targets tau of the observed end modes, plus
    lam mean_g (log d1 - log d1_prior)^2, with P_b(d1) the protein flow of each step from P_a (find_next_prot_dilution,
    birth rates b_a / b_b; switch of the modes at alpha, of the rates at alpha_b). The own protein of each target gene
    is kept at d1_cur (no self-loop shortcut). Bounds lb <= d1 <= ub (L-BFGS-B on log d1).
    """
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))
    P_a, mode_a, mode_b, alpha, alpha_b, s = map(t, (P_a, mode_a, mode_b, alpha, alpha_b, s))
    dt, b_a, b_b = t(dt)[:, None], t(b_a)[:, None], t(b_b)[:, None]
    stim, basal_rows, inter, tau, w = map(t, (stim, basal_rows, inter, tau, w))
    log_prior = t(np.log(d1_prior))
    diag = torch.diagonal(inter)[ns:]

    def prot_end(d1):
        rate_a, rate_b = d1 + b_a, d1 + b_b
        return _flow_torch(rate_a, rate_b, P_a, mode_a * d1 / rate_a, mode_b * d1 / rate_b, alpha, alpha_b, s, dt)

    P_cur = prot_end(t(d1_cur))

    def objective(log_d1):
        u = torch.tensor(log_d1, dtype=torch.float64, requires_grad=True)
        P = prot_end(torch.exp(u))
        Z = basal_rows + torch.cat([stim, P], dim=1) @ inter
        Z = Z[:, ns:] + diag * (P_cur - P)  # own protein of each target at d1_cur
        sig = torch.sigmoid(Z).clamp(EPS_CE, 1 - EPS_CE)
        tc = tau.clamp(EPS_CE, 1 - EPS_CE)
        ce = -(tau * torch.log(sig / tc) + (1 - tau) * torch.log((1 - sig) / (1 - tc)))
        loss = (ce * w).sum() / (ce.shape[0] * max(float(w.sum()), 1e-12)) + lam * ((u - log_prior) ** 2).mean()
        loss.backward()
        return float(loss), u.grad.numpy().copy()

    x0 = np.clip(np.log(d1_cur), np.log(lb), np.log(ub))
    res = minimize(objective, x0, jac=True, method='L-BFGS-B', bounds=list(zip(np.log(lb), np.log(ub))))
    return np.exp(res.x)
