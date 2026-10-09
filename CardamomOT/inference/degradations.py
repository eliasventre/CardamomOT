"""
Utilities for degradation rate inference and temporal epsilon estimation.

This module provides PyTorch models and helper functions used by the
CARDAMOM pipeline when learning gene-specific degradation parameters
from protein dynamics.  It includes the
:class:`GeneRegulatoryODE_softmax` neural ODE model and the
:func:`infer_ratio_d0_d1_unitary` routine among other utilities.
"""

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import logging
from typing import Any

from CardamomOT.logging import get_logger
from .sampling import stratified_choice

# Initialize module-level logger
logger = get_logger(__name__)

# ---------------------------
# Helpers and small utilities
# ---------------------------

def _get_device_from_module(module) -> torch.device:
    """Return the preferred device for a given PyTorch module.

    The function inspects the module and attempts to figure out which device
    its parameters or buffers reside on. It follows this order:

    1. First parameter of the module.
    2. First buffer of the module.
    3. Defaults to ``cpu`` if neither are available.

    Args:
        module: Any object implementing ``parameters()`` and/or ``buffers()``
                (typically an ``nn.Module``).

    Returns:
        ``torch.device`` indicating the device where module data lives.
    """
    try:
        p = next(module.parameters())
        return p.device
    except StopIteration:
        try:
            b = next(module.buffers())
            return b.device
        except StopIteration:
            return torch.device("cpu")


def _stim_at(stim_schedule, t, sample, ns):
    """Stimulus values at time t for a sample (its own schedule when per-sample overrides exist)."""
    if not stim_schedule:
        return np.ones(ns, dtype=np.float32)
    if hasattr(stim_schedule, 'at'):
        return np.asarray(stim_schedule.at(t, sample), dtype=np.float32)
    return np.asarray(stim_schedule[t], dtype=np.float32)


def _ks_sample(ks, s_idx):
    """Amplitudes (n_modes, G) of sample s_idx: ks is per-sample (S, n_modes, G) when 3-D, else shared."""
    ks = np.asarray(ks)
    return ks[min(s_idx, ks.shape[0] - 1)] if ks.ndim == 3 else ks


def _theta_sample(theta_inter, s_idx):
    """Interactions (G, G, n_nets) of sample s_idx: per-sample (S, G, G, n_nets) when 4-D (network conditions), else shared."""
    theta_inter = np.asarray(theta_inter)
    return theta_inter[min(s_idx, theta_inter.shape[0] - 1)] if theta_inter.ndim == 4 else theta_inter


def build_kon_fn(ks, theta_inter, bias, device="cpu"):
    """
    Return a function kon(X_numpy_or_torch) -> numpy array (batch, G)
    The function accepts either numpy arrays or torch tensors; it returns numpy.
    """
    ks_t: torch.Tensor = torch.tensor(ks, dtype=torch.float32, device=device)
    theta_t: torch.Tensor = torch.tensor(theta_inter, dtype=torch.float32, device=device)
    bias_t: torch.Tensor = torch.tensor(bias, dtype=torch.float32, device=device)
    n_modes: int = ks_t.shape[0]
    G: int = ks_t.shape[1]

    def kon(X):
        # X can be torch tensor or numpy -> ensure torch on device
        is_numpy: bool = isinstance(X, np.ndarray)
        if is_numpy:
            X_t: torch.Tensor = torch.tensor(X, dtype=torch.float32, device=device)
        else:
            X_t = X.to(device).float()

        if X_t.dim() == 1:
            X_t = X_t.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        Z: torch.Tensor = torch.zeros((X_t.shape[0], G, n_modes), device=device)
        for k in range(n_modes - 1):
            Z[:, :, k + 1] = X_t @ theta_t[:, :, k] + bias_t[:, k]
        base_kon: torch.Tensor = torch.softmax(Z, dim=-1)
        kon_t: torch.Tensor = torch.sum(base_kon * ks_t.T.unsqueeze(0), dim=-1)

        if squeeze_output:
            kon_t: torch.Tensor = kon_t.squeeze(0)
        return kon_t.cpu().numpy()

    return kon


class GeneRegulatoryODE_softmax(nn.Module):
    """
    ODE model for gene regulatory dynamics with generalized softmax-based kon.
    Learns gene-specific degradation rates (d) and scale factors.
    Optionally multiplies kon by a per-cell ratio g(t) (mRNA-driven proteins, see set_g_interpolation).
    """

    def __init__(self, G, d_init, ks, theta_inter, bias, n_stimuli=1, stim_vals=None,
                 device="cpu", lambda_scale=1e3) -> None:
        """
        Args:
            G           : number of genes (total, including stimuli)
            d_init      : initial degradation rates (array of size G)
            ks          : array of shape (n_modes, G)
            theta_inter : array of shape (G, G, n_modes-1)
            bias        : array of shape (G, n_modes-1)
            n_stimuli   : number of stimulus columns (default 1)
            stim_vals   : fixed stimulus values shape (n_stimuli,); defaults to ones
        """
        super().__init__()
        self.G = int(G)
        self.n_stimuli = int(n_stimuli)
        self.device = torch.device(device)

        # ----- d parameter (degradation rates) -----
        d_init = np.asarray(d_init, dtype=np.float32)
        inv_softplus = np.log(np.exp(d_init) - 1.0 + 1e-8)
        self.d_param = nn.Parameter(torch.tensor(inv_softplus, dtype=torch.float32))

        # ----- scale parameter -----
        self.scale_param = nn.Parameter(torch.ones(G, dtype=torch.float32))

        # ----- static network parameters -----
        self.register_buffer("ks", torch.tensor(np.asarray(ks, dtype=np.float32)))
        self.register_buffer("theta_inter", torch.tensor(np.asarray(theta_inter, dtype=np.float32)))
        self.register_buffer("bias", torch.tensor(np.asarray(bias, dtype=np.float32)))

        if stim_vals is None:
            stim_vals = np.ones(self.n_stimuli, dtype=np.float32)
        self.register_buffer("stim_vals", torch.tensor(np.asarray(stim_vals, dtype=np.float32)))

        self.n_modes = int(self.ks.shape[0])
        self.lambda_scale = float(lambda_scale)

        # Per-cell ratio g = kon_beta_nonscaled / kon_beta at both ends of the interval (None: no correction)
        self._g0 = self._g1 = None
        self._t0 = self._t1 = None
        # Per-cell birth rate (dilution of the proteins, None: no dilution)
        self._birth = None

    def set_birth(self, birth=None):
        """Birth rate b (batch,) of the rows of X in the next odeint calls: dilution term - b P of the genes
        (dP/dt = d (kon - P) - b P = d kon - (d + b) P); None clears it."""
        self._birth = None if birth is None else torch.as_tensor(np.asarray(birth, dtype=np.float32))

    def set_g_interpolation(self, t0, t1, g0=None, g1=None):
        """Per-cell ratios g0, g1 (batch, G_genes) at t0 and t1, linearly interpolated in time
        to multiply kon[:, ns:] in the next odeint calls (rows = rows of X); None clears them."""
        if g0 is None:
            self._g0 = self._g1 = None
            return
        self._t0, self._t1 = float(t0), float(t1)
        self._g0 = torch.as_tensor(np.asarray(g0, dtype=np.float32))
        self._g1 = torch.as_tensor(np.asarray(g1, dtype=np.float32))

    def forward(self, t, X):
        """
        Compute dX/dt for a given state X at time t.
        Includes learned scaling of theta_inter and bias, and the per-cell
        ratio g(t) set by ``set_g_interpolation`` (mRNA-driven proteins).
        """
        squeeze_output = False
        if X.dim() == 1:
            X = X.unsqueeze(0)
            squeeze_output = True

        ns = self.n_stimuli
        X = X.clone()
        X[:, :ns] = self.stim_vals.to(X.device)

        batch_size, G = X.shape[0], self.G
        n_modes: int = self.n_modes

        # ----- compute scale -----
        # First ns elements (stimuli) are fixed at 1; ns: elements are learned
        scale_raw = torch.nn.functional.softplus(self.scale_param)
        scale = torch.cat([
            torch.ones(ns, device=X.device, dtype=scale_raw.dtype),
            scale_raw[ns:]
        ])

        # scale theta_inter and bias
        theta_scaled = self.theta_inter * scale[None, :, None]  # scale each row g
        bias_scaled = self.bias * scale[:, None]                # scale each gene’s bias

        # compute softmax activations
        Z: torch.Tensor = torch.zeros((batch_size, G, n_modes), dtype=torch.float32, device=X.device)
        for k in range(n_modes - 1):
            Z[:, :, k + 1] = X @ theta_scaled[:, :, k] + bias_scaled[:, k]

        base_kon: torch.Tensor = torch.softmax(Z, dim=-1)
        ks_expanded = self.ks.T.unsqueeze(0)  # (1, G, n_modes)
        kon: torch.Tensor = torch.sum(base_kon * ks_expanded.to(X.device), dim=-1)

        # ----- per-cell ratio g(t), linear between the interval ends -----
        if self._g0 is not None:
            t_f = float(t.detach()) if torch.is_tensor(t) else float(t)
            alpha = min(1.0, max(0.0, (t_f - self._t0) / max(self._t1 - self._t0, 1e-10)))
            g = (1.0 - alpha) * self._g0.to(X.device) + alpha * self._g1.to(X.device)
            kon = torch.cat([kon[:, :ns], kon[:, ns:] * g], dim=-1)

        # degradation and ODE dynamics
        d_eff: torch.Tensor = torch.nn.functional.softplus(self.d_param.to(X.device))
        dXdt = d_eff * (kon - X)
        if self._birth is not None:  # dilution at the birth rate of each cell
            dXdt = dXdt - self._birth.to(X.device)[:, None] * X
        mask = torch.ones(self.G, device=X.device)
        mask[:ns] = 0.0
        dXdt = dXdt * mask.unsqueeze(0)

        if squeeze_output:
            dXdt = dXdt.squeeze(0)

        return dXdt


# ---------------------------
# fit_scale_theta
# ---------------------------

def fit_scale_theta(X_prot, kon_beta, bias, theta_inter, ks, ns, samples_data=None):
    """Find a single scale factor minimising the total MSE across all genes jointly.

    Args:
        X_prot      : ``(N, G)`` protein levels.
        kon_beta    : ``(N, G)`` mixture-inferred kon values.
        bias        : ``(G, n_modes-1)`` or ``(n_samples, G, n_modes-1)`` GRN basal.
        theta_inter : ``(G, G, n_modes-1)`` GRN interaction tensor, or per sample ``(n_samples, G, G, n_modes-1)``.
        ks          : ``(n_modes, G)`` softmax burst-rate amplitudes.
        ns          : number of stimulus columns.
        samples_data: ``(N,)`` per-cell sample index when bias is 3-D, else None.

    Returns:
        scale_theta : float, the jointly optimal scale (same for all genes).
    """
    from scipy.optimize import minimize_scalar

    X_prot      = np.asarray(X_prot,    dtype=np.float64)
    kon_beta    = np.asarray(kon_beta,  dtype=np.float64)
    bias        = np.asarray(bias,      dtype=np.float64)
    theta_inter = np.asarray(theta_inter, dtype=np.float64)
    ks          = np.asarray(ks,        dtype=np.float64)

    G       = X_prot.shape[1]
    N       = X_prot.shape[0]
    n_modes = ks.shape[-2]

    per_sample = (bias.ndim == 3 and samples_data is not None)
    if (ks.ndim == 3 or theta_inter.ndim == 4) and not per_sample:
        raise ValueError('per-sample ks / interactions need a 3-D bias and samples_data')

    # Pre-compute all per-gene inputs before optimisation.
    if per_sample:
        sd = np.asarray(samples_data)
        unique_s = np.sort(np.unique(sd))
        gene_data = []
        for g in range(ns, G):
            sample_data_g = []
            for s_idx, s_label in enumerate(unique_s):
                mask_s = (sd == s_label)
                if not np.any(mask_s):
                    continue
                bias_s = bias[min(s_idx, bias.shape[0] - 1)]
                A_g_s  = X_prot[mask_s] @ _theta_sample(theta_inter, s_idx)[:, g, :] + bias_s[g, :]
                sample_data_g.append((mask_s, A_g_s, _ks_sample(ks, s_idx)[:, g]))
            gene_data.append((kon_beta[:, g], sample_data_g))

        def _total_loss(log_s):
            s = np.exp(log_s)
            total = 0.0
            for target, sample_data_g in gene_data:
                pred = np.zeros(N)
                for mask_s_, A_g_s_, ks_g in sample_data_g:
                    N_s = int(mask_s_.sum())
                    Z   = np.zeros((N_s, n_modes))
                    Z[:, 1:] = s * A_g_s_
                    Z -= Z.max(axis=1, keepdims=True)
                    exp_Z = np.exp(Z)
                    sigma = exp_Z / exp_Z.sum(axis=1, keepdims=True)
                    pred[mask_s_] = (sigma * ks_g).sum(axis=1)
                total += np.mean((pred - target) ** 2)
            return total
    else:
        gene_data = [
            (kon_beta[:, g], ks[:, g],
             X_prot @ theta_inter[:, g, :] + bias[g, :])
            for g in range(ns, G)
        ]

        def _total_loss(log_s):
            s = np.exp(log_s)
            total = 0.0
            for target, ks_g, A_g in gene_data:
                Z = np.zeros((N, n_modes))
                Z[:, 1:] = s * A_g
                Z -= Z.max(axis=1, keepdims=True)
                exp_Z = np.exp(Z)
                sigma = exp_Z / exp_Z.sum(axis=1, keepdims=True)
                total += np.mean(((sigma * ks_g).sum(axis=1) - target) ** 2)
            return total

    res = minimize_scalar(_total_loss, bounds=(-5, 5), method='bounded')
    return float(np.exp(res.x))


# ---------------------------
# infer_ratio_d0_d1_unitary
# ---------------------------

def infer_ratio_d0_d1_unitary(
    X_prot, times, bias, theta_inter, ks,
    d_learned_temporal, k1_vec,
    n_stimuli=1, stim_schedule=None,
    samples_data=None,
    method="dopri5", rtol=1e-6, atol=1e-8,
    lambda_deg=0.0, prior_eps=None,
    outlier_quantile=0.95,
    min_kon=1e-6, eps_min=0.01, eps_max=100.0,
    verbose=True,
    scale=1.0,
    two_stage=False,
    birth=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ε_i = d1_i/d0_i from ODE residuals (``birth``: (N,) birth rate of the rows, dilution in the
    mean-field prediction; None: no dilution) (bursty-PDMP variance matching).

    **Theoretical background.**

    In the bursty PDMP actually simulated (see :class:`BurstyPDMP`), the burst
    parameters at interval ``cnt`` are:

    * burst **rate**  : ``λ_i = k1_i · kon_norm_i(P) · d0_i``
    * burst **size**  : ``Exp(k1_i · d0_i / (scale · d1_i))``
      i.e. mean burst = ``scale · d1_i / (k1_i · d0_i)``

    This ensures the mean-field ODE limit is ``dP_i/dt = d1_i(kon_i/k1_i − P_i)``,
    where ``kon_i`` denotes the *true*, un-normalised kon of the generative
    process. Note that ``kon_i(P_c)`` below (and the ``ks``/``kon_avg`` computed
    from it) is already the ``kon_i/k1_i``-normalised version used throughout
    this codebase (NB-inferred kon/d0, rescaled by the NB-inferred k1/d0) — i.e.
    it already carries one implicit factor of ``1/k1_i`` relative to the true kon.

    By Campbell's theorem (leading term for small dt), in terms of the true kon:

    .. math::

        \\operatorname{Var}[\\Delta X_i] \\approx
        \\lambda_i \\cdot E[B_i^2] \\cdot dt
        = 2\\,dt\\cdot k_{on,i}^{\\mathrm{true}}(P)\\cdot\\mathrm{scale}\\,
          \\frac{d_{1,i}^2}{k_{1,i}^2\\,d_{0,i}}
        = \\varepsilon_i \\cdot h_{ic}

    where :math:`\\varepsilon_i = d_{1,i}/d_{0,i}` and, substituting
    :math:`k_{on,i}^{\\mathrm{true}}(P) = k_{1,i}\\cdot k_{on,i}(P_c)`
    (``kon_i(P_c)`` being the already-normalised kon actually computed below):

    .. math::

        h_{ic} = 2\\,dt\\cdot\\mathrm{scale}\\cdot
                 \\frac{d_{1,i}}{k_{1,i}}\\cdot k_{on,i}(P_c)

    The regularised method-of-moments estimator is:

    .. math::

        \\varepsilon_i =
        \\frac{\\sum_c (X_{\\mathrm{pred},c} - X_{1,c})^2
               + \\lambda\\,\\varepsilon_{\\mathrm{prior},i}}
              {\\sum_c h_{ic} + \\lambda}

    Args:
        X_prot             : ``(N, G)`` protein observations (all timepoints).
        times              : ``(N,)`` time label for each row.
        bias               : GRN bias — ``(T-1, G, n_nets)`` or
                             ``(T-1, n_samples, G, n_nets)``.
        theta_inter        : ``(T-1, G, G, n_nets)`` GRN interaction tensor.
        ks                 : ``(n_modes, G)`` softmax burst-rate amplitudes
                             (already multiplied by ``scale_proteins``).
        d_learned_temporal : ``(T-1, G)`` learned protein degradation rates d1.
        k1_vec             : ``(G,)`` max burst rate × scale (= ``k1 * scale``).
        n_stimuli          : Number of stimulus columns (default 1).
        stim_schedule      : ``{float: np.ndarray(ns)}`` stimulus values per time.
        samples_data       : ``(N,)`` per-cell sample index (optional).
        method, rtol, atol : ODE solver settings.
        lambda_deg         : Tikhonov weight; 0 = pure MoM, large → prior.
        prior_eps          : ``(G,)`` prior for ε = d1/d0. Defaults to all-ones.
        outlier_quantile   : Fraction of cells to keep (sorted by residual).
        min_kon            : Minimum kon value for a cell to contribute.
        eps_min, eps_max   : Output clipping bounds for ε = d1/d0.
        verbose            : Log per-interval diagnostics.
        scale              : Protein scale (``self.scale_proteins``). Default 1.0.
        two_stage          : True for the Harissa PDMP (explicit mRNA): the mRNA filters the protein
                             noise by d0/(d0+d1), i.e. Var = ε/(1+ε)·h instead of ε·h.

    Returns:
        eps_temporal : ``(T-1, G)`` per-interval ε = d1/d0 estimates.
        eps_global   : ``(G,)`` global ε pooled over all intervals.
    """
    device = "cpu"
    X_prot   = np.asarray(X_prot,  dtype=np.float32)
    # float64 on purpose: time values are used as keys of stim_schedule and
    # ratio dicts built from float64 times; a float32 cast turns e.g. 5.55 into
    # 5.550000190734863 and the lookup misses.
    times    = np.asarray(times,   dtype=np.float64)
    k1_vec   = np.asarray(k1_vec,  dtype=np.float32)
    d_learned_temporal = np.asarray(d_learned_temporal, dtype=np.float32)
    ks_np    = np.asarray(ks,      dtype=np.float32)   # (n_modes, G)

    unique_times = np.sort(np.unique(times))
    T: int   = len(unique_times)
    ns: int  = int(n_stimuli)
    G: int   = X_prot.shape[1]

    assert d_learned_temporal.shape[0] == T - 1, "d_learned_temporal must have (T-1, G)"
    assert k1_vec.shape[-1] == G

    # ── Prior for ε = d1/d0 (used by regularisation) ────────────────────────
    r_prior = (
        np.ones(G, dtype=np.float64)
        if prior_eps is None
        else np.asarray(prior_eps, dtype=np.float64)
    )
    lam = float(lambda_deg)

    # ── LS accumulators (float64) ────────────────────────────────────────────
    # Bursty-PDMP model (Campbell's theorem for Exp bursts):
    #   Var(ΔX_i) = ε_i · h_{ic}
    # where  ε_i = d1_i/d0_i  and  kon_i(P_c) is the k1-normalised kon (kon_true/k1)
    #        h_{ic} = 2 · dt · scale · d1_i/k1_i · kon_i(P_c)
    # Regularised MoM: ε_i = (Σ res²_{ci} + λ·ε_prior) / (Σ h_{ic} + λ)
    num_t   = np.zeros((T - 1, G), dtype=np.float64)   # Σ residuals²
    denom_t = np.zeros((T - 1, G), dtype=np.float64)   # Σ h
    num_g   = np.zeros(G,          dtype=np.float64)
    denom_g = np.zeros(G,          dtype=np.float64)

    for cnt in range(T - 1):
        t0 = float(unique_times[cnt])
        t1 = float(unique_times[cnt + 1])
        dt = t1 - t0
        if dt <= 0:
            continue

        mask0_t = times == unique_times[cnt]
        # Each sample with its own cells, basal and mixture (sample index = value of samples_data),
        # from t0 to its next observed time (samples may miss timepoints)
        sd = np.asarray(samples_data).astype(int) if samples_data is not None else np.zeros(len(times), dtype=int)
        n_valid_cnt = 0
        for s_int in np.unique(sd[mask0_t]):
            later = np.unique(times[(sd == s_int) & (times > t0)])
            if not len(later):
                continue
            t1s = float(later[0])
            dts = t1s - t0
            k_end = int(np.searchsorted(unique_times, t1s))   # grid intervals cnt..k_end-1 covered
            mask0 = mask0_t & (sd == s_int)
            mask1 = (times == later[0]) & (sd == s_int)
            X0_np = X_prot[mask0]
            X1_np = X_prot[mask1]
            b0_np = None if birth is None else np.asarray(birth, dtype=np.float32)[mask0]
            if len(X0_np) == 0 or len(X1_np) == 0:
                continue

            # Pair cells at t0 with cells at t1 (arbitrary but consistent)
            n_pairs = min(len(X0_np), len(X1_np))
            X0_np = X0_np[:n_pairs]
            X1_np = X1_np[:n_pairs]

            stim0 = _stim_at(stim_schedule, t0, s_int, ns) * scale
            stim1 = _stim_at(stim_schedule, t1s, s_int, ns) * scale

            # ── Select per-interval bias / theta ─────────────────────────────────
            bias_np  = np.asarray(bias,         dtype=np.float32)
            theta_np = np.asarray(theta_inter,  dtype=np.float32)

            ks_cnt = _ks_sample(ks_np, s_int)
            # Support 3-D bias (T-1, G, nets) and 4-D (T-1, n_samples, G, nets)
            if bias_np.ndim == 3:
                bias_cnt  = bias_np[cnt]                         # (G, n_nets)
            elif bias_np.ndim == 4:
                bias_cnt = bias_np[cnt, min(s_int, bias_np.shape[1] - 1)]
            else:
                bias_cnt = bias_np

            if theta_np.ndim == 4:
                theta_cnt = theta_np[cnt]                        # (G, G, n_nets)
            elif theta_np.ndim == 5:
                theta_cnt = theta_np[cnt, min(s_int, theta_np.shape[1] - 1)]
            else:
                theta_cnt = theta_np

            d_param_vec = d_learned_temporal[cnt]                # (G,)
            k1_i        = (k1_vec[min(s_int, len(k1_vec) - 1)] if k1_vec.ndim == 2 else k1_vec) / float(scale)  # (G,) raw max burst rate

            # ── Build ODE and simulate ────────────────────────────────────────────
            ode = GeneRegulatoryODE_softmax(
                G, d_param_vec, ks_cnt, theta_cnt, bias_cnt,
                n_stimuli=ns,
                stim_vals=stim1,
                device=device,
            ).to(device)
            ode.eval()
            ode.set_birth(None if b0_np is None else b0_np[:n_pairs])  # dilution of the start states

            X0_t = torch.tensor(X0_np, dtype=torch.float32)
            X0_t[:, :ns] = torch.tensor(stim0)
            t_span = torch.tensor([t0, t1s], dtype=torch.float32)

            with torch.no_grad():
                traj = odeint(ode, X0_t, t_span, method=method, rtol=rtol, atol=atol)
            X_pred = traj[-1].cpu().numpy()       # (n_pairs, G)
            X_pred[:, :ns] = stim1

            # ── Compute kon at initial and predicted states ───────────────────────
            kon_fn = build_kon_fn(ks_cnt, theta_cnt, bias_cnt, device=device)
            kon_0 = kon_fn(X0_np)                # (n_pairs, G) in protein units
            kon_1 = kon_fn(X_pred)

            # Midpoint average of kon over the interval
            kon_avg = 0.5 * (kon_0 + kon_1)   # (n_pairs, G)

            # ── Residuals and denominator ─────────────────────────────────────────
            residuals = (X_pred - X1_np) ** 2   # (n_pairs, G)

            # h_{ic} = 2 · dt · scale · d1_i/k1_i · kon_i(P_c), kon_i(P_c) = kon_avg already = kon_true/k1_i
            # Derived from Campbell's theorem: burst rate=k1·kon_norm·d0, burst size~Exp(k1·d0/(scale·d1))
            #   Var(ΔP_i) = burst_rate · E[B²] · dt = ε_i · 2·dt·scale·(d1_i/k1_i)·kon_i
            h_mat = (2.0 * dts * float(scale)) * (d_param_vec / k1_i)[None, :] * kon_avg   # (n_pairs, G)
            h_mat[:, :ns] = 0.0   # stimuli don't get ε estimated

            # ── Outlier filtering: exclude top (1 - outlier_quantile) per gene ────
            valid_mask = np.ones(n_pairs, dtype=bool)   # start with all cells
            if outlier_quantile < 1.0:
                # Per-gene sum of squared residuals; filter cells with very large total
                res_sum = residuals.sum(axis=1)
                q_thresh = np.quantile(res_sum, outlier_quantile)
                valid_mask = res_sum <= q_thresh

            res_v   = residuals[valid_mask]   # (n_valid, G)
            h_v     = h_mat[valid_mask]       # (n_valid, G)
            kon_v   = kon_avg[valid_mask]     # (n_valid, G) — for min_kon filter

            # Per-gene: only cells where kon > min_kon
            for g in range(ns, G):
                cell_ok = kon_v[:, g] > min_kon
                if cell_ok.sum() == 0:
                    continue
                contrib_num   = float(res_v[cell_ok, g].sum())
                contrib_denom = float(h_v[cell_ok, g].sum())
                num_t[cnt:k_end, g]   += contrib_num    # every grid interval of the observed pair
                denom_t[cnt:k_end, g] += contrib_denom
                num_g[g]        += contrib_num
                denom_g[g]      += contrib_denom

        if verbose:
            with np.errstate(invalid="ignore", divide="ignore"):
                eps_cnt = num_t[cnt, ns:] / np.where(denom_t[cnt, ns:] > 0,
                                                      denom_t[cnt, ns:], np.nan)
                if two_stage:
                    eps_cnt = eps_cnt / (1.0 - eps_cnt)
            logger.info(
                "[eps_temporal cnt=%d: t=%.3g→%.3g]  cells=%d  "
                "mean_ε(genes)=%.3g",
                cnt, t0, t1, n_valid_cnt,
                float(np.nanmean(eps_cnt)),
            )

    # ── Solve regularised MoM for q = Var/h, then ε = d1/d0 ──────────────────
    # q_i = (Σ res²_ci  +  λ · q_prior_i) / (Σ h_ic  +  λ); q = ε (one stage) or ε/(1+ε) (two stage)
    def to_q(e):
        return e / (1.0 + e) if two_stage else e

    def to_eps(q):
        if not two_stage:
            return q
        q = np.minimum(q, to_q(eps_max))
        return q / (1.0 - q)

    q_prior = to_q(r_prior)
    eff_denom_t = denom_t + lam
    eff_num_t   = num_t   + lam * q_prior[None, :]
    has_t       = eff_denom_t > 0
    eps_temporal_raw = to_eps(np.where(has_t,
                                       eff_num_t / np.where(has_t, eff_denom_t, 1.0),
                                       q_prior[None, :]))
    eps_temporal = np.clip(eps_temporal_raw, eps_min, eps_max).astype(np.float32)
    eps_temporal[:, :ns] = 1.0

    eff_denom_g = denom_g + lam
    eff_num_g   = num_g   + lam * q_prior
    has_g       = eff_denom_g > 0
    eps_global_raw = to_eps(np.where(has_g,
                                     eff_num_g / np.where(has_g, eff_denom_g, 1.0),
                                     q_prior))
    eps_global = np.clip(eps_global_raw, eps_min, eps_max).astype(np.float32)
    eps_global[:ns] = 0.2

    return eps_temporal, eps_global


# ---------------------------
# inference_degradation_prot
# ---------------------------

def _draw_minibatch(n_cells, batch_size, labels=None):
    """All rows if batch_size is None or >= n_cells, else batch_size rows (cell-type proportional if labels)."""
    if batch_size is None:
        return np.arange(n_cells)
    return stratified_choice(np.arange(n_cells), batch_size, labels)


def inference_degradation_prot(
    X_prot, times, bias, theta_inter, ks, d=None,
    n_epochs=500, lr=1e-2, method="dopri5",
    rtol=1e-6, atol=1e-8, print_every=50,
    batch_size=None, verbose=True,
    n_stimuli=1, stim_schedule=None,
    scale_proteins=1.0,
    samples_data=None,
    strata=None,
    lambda_scale=1e3,
    lambda_deg=0.0,
    g_ratio=None,
    birth=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate degradation rates and scaling factors from protein time-course data.
    ``birth``: (N,) birth rate of the rows of X_prot (dilution of the proteins, rate of the start state of each
    interval: dP/dt = d (kon - P) - b P), or None (no dilution).

    When ``bias`` is 3-D (n_samples, G, n_modes-1) and ``samples_data`` is provided,
    one ODE module is created per sample with its own bias while ``d_param`` and
    ``scale_param`` are shared, so a single optimizer refines the shared kinetics
    from all samples jointly.

    ``batch_size``: trajectories drawn at random per interval at each optimizer
    step (one step per epoch); None uses every trajectory at each step.
    ``strata``: per-row cell types (same indexing as X_prot) or None; minibatches
    are then cell-type proportional at the start time of each interval.
    ``g_ratio``: ``(N, G - ns)`` per-row ratio kon_beta_nonscaled / kon_beta (rows of X_prot) or None;
    interpolated per trajectory over each interval to multiply kon (mRNA-driven proteins).

    Returns:
        ``d_learned``: Learned degradation rates, shape (G,).
        ``scale_learned``: Learned scaling parameters, shape (G,).
    """
    device = "cpu"
    ns: int = int(n_stimuli)
    X_prot = np.asarray(X_prot, dtype=np.float32)
    # float64 on purpose: time values are used as keys of stim_schedule and
    # ratio dicts built from float64 times; a float32 cast turns e.g. 5.55 into
    # 5.550000190734863 and the lookup misses.
    times  = np.asarray(times,  dtype=np.float64)
    bias   = np.asarray(bias,   dtype=np.float32)
    g_ratio = None if g_ratio is None else np.asarray(g_ratio, dtype=np.float32)
    birth = None if birth is None else np.asarray(birth, dtype=np.float32)

    G: int = X_prot.shape[1]

    if d is None:
        d_init = np.ones(G, dtype=np.float32)
    else:
        d_init = np.asarray(d, dtype=np.float32)

    per_sample_mode = (bias.ndim == 3 and samples_data is not None)

    if per_sample_mode:
        # ── Per-sample mode: one ODE per sample, shared d_param / scale_param ──
        n_samples = bias.shape[0]
        samples_data_arr = np.asarray(samples_data)
        unique_s = np.unique(samples_data_arr)

        # Create one ODE module per sample
        all_unique_times = np.sort(np.unique(times))
        if stim_schedule and len(all_unique_times) >= 2:
            t1_first = float(all_unique_times[1])
            stim_first = np.asarray(stim_schedule[t1_first], dtype=np.float32) * scale_proteins
        else:
            stim_first = np.ones(ns, dtype=np.float32) * scale_proteins
        ode_funcs = [
            GeneRegulatoryODE_softmax(
                G, d_init, _ks_sample(ks, s_idx), _theta_sample(theta_inter, s_idx), bias[s_idx],
                n_stimuli=ns, stim_vals=stim_first, device=device,
                lambda_scale=lambda_scale,
            ).to(device)
            for s_idx in range(n_samples)
        ]
        # Share d_param and scale_param: assign the same nn.Parameter objects
        for s_idx in range(1, n_samples):
            ode_funcs[s_idx].d_param     = ode_funcs[0].d_param
            ode_funcs[s_idx].scale_param = ode_funcs[0].scale_param

        optimizer = torch.optim.Adam([ode_funcs[0].d_param, ode_funcs[0].scale_param], lr=lr)
        mse = nn.MSELoss(reduction="mean")

        # Build per-sample pairs: (s_idx, t0, t1, X0, X1, stim0, stim1)
        all_pairs = []
        for s in unique_s:
            s_idx = int(s)  # sample index (row of bias), also when some samples are absent from the rows
            mask_s = (samples_data_arr == s)
            X_s  = X_prot[mask_s]
            t_s  = times[mask_s]
            g_s  = None if g_ratio is None else g_ratio[mask_s]
            b_s  = None if birth is None else birth[mask_s]
            unique_t = np.sort(np.unique(t_s))
            for ti in range(len(unique_t) - 1):
                t0, t1 = float(unique_t[ti]), float(unique_t[ti + 1])
                X0_np = X_s[t_s == unique_t[ti]]
                X1_np = X_s[t_s == unique_t[ti + 1]]
                stim0 = _stim_at(stim_schedule, t0, s, ns) * scale_proteins
                stim1 = _stim_at(stim_schedule, t1, s, ns) * scale_proteins
                n_p = min(len(X0_np), len(X1_np))
                if n_p > 0:
                    lab0 = None if strata is None else np.asarray(strata)[mask_s][t_s == unique_t[ti]][:n_p]
                    g01 = (None, None) if g_s is None else (g_s[t_s == unique_t[ti]][:n_p], g_s[t_s == unique_t[ti + 1]][:n_p])
                    b0 = None if b_s is None else b_s[t_s == unique_t[ti]][:n_p]
                    all_pairs.append((s_idx, t0, t1, X0_np[:n_p], X1_np[:n_p], stim0, stim1, lab0, g01, b0))

        old_loss  = 1e16

        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            total_loss, total_count = 0.0, 0

            for (s_idx, t0, t1, X0_full, X1_full, stim0, stim1, lab0, (g0, g1), b0) in all_pairs:
                ode = ode_funcs[s_idx]
                ode.stim_vals.copy_(torch.tensor(stim1, dtype=torch.float32))
                idxs = _draw_minibatch(X0_full.shape[0], batch_size, lab0)
                ode.set_g_interpolation(t0, t1, None if g0 is None else g0[idxs], None if g1 is None else g1[idxs])
                ode.set_birth(None if b0 is None else b0[idxs])
                X0 = torch.tensor(X0_full[idxs], dtype=torch.float32, device=device)
                X1 = torch.tensor(X1_full[idxs], dtype=torch.float32, device=device)
                X0[:, :ns] = torch.tensor(stim0, dtype=torch.float32)
                X1[:, :ns] = torch.tensor(stim1, dtype=torch.float32)
                t_span = torch.tensor([t0, t1], dtype=torch.float32, device=device)
                X_pred = odeint(ode, X0, t_span, method=method, rtol=rtol, atol=atol)[-1]
                X_pred[:, :ns] = torch.tensor(stim1, dtype=torch.float32, device=device)
                loss_batch = mse(X_pred, X1)
                loss_batch.backward()
                total_loss  += loss_batch.item() * len(idxs)
                total_count += len(idxs)

            # Regularization: penalize deviation of scale[ns:] from 1 and d from d_init
            if lambda_scale > 0:
                scale_genes = torch.nn.functional.softplus(ode_funcs[0].scale_param[ns:])
                (lambda_scale * torch.mean((scale_genes - 1.0) ** 2)).backward()
            if lambda_deg > 0:
                d_init_t = torch.tensor(d_init, dtype=torch.float32)
                d_now = torch.nn.functional.softplus(ode_funcs[0].d_param)
                (lambda_deg * torch.mean((d_now - d_init_t) ** 2)).backward()

            optimizer.step()
            loss: float = total_loss / total_count if total_count > 0 else 0.0
            loss_ema = loss if epoch == 1 else 0.9 * loss_ema + 0.1 * loss
            if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == n_epochs):
                scale_now = torch.nn.functional.softplus(ode_funcs[0].scale_param[ns:]).detach().cpu().numpy()
                logger.info(f"[Epoch {epoch}/{n_epochs}] loss = {loss_ema:.6e}  max_scale[ns:] = {scale_now.max():.3e}")
                if abs(loss_ema - old_loss) < 1e-4:
                    break
                old_loss = loss_ema

        d_learned     = torch.nn.functional.softplus(ode_funcs[0].d_param).detach().cpu().numpy()
        scale_learned = torch.nn.functional.softplus(ode_funcs[0].scale_param).detach().cpu().numpy()
        return d_learned, scale_learned

    # ── Single-bias mode (original behaviour) ────────────────────────────
    unique_times = np.sort(np.unique(times))
    pairs = []
    for idx in range(len(unique_times) - 1):
        t0, t1 = float(unique_times[idx]), float(unique_times[idx + 1])
        mask0, mask1 = (times == t0), (times == t1)
        X0_np, X1_np = X_prot[mask0], X_prot[mask1]
        stim0 = stim_schedule[t0] if stim_schedule else np.ones(ns, dtype=np.float32)
        stim1 = stim_schedule[t1] if stim_schedule else np.ones(ns, dtype=np.float32)
        stim0, stim1 = stim0*scale_proteins, stim1*scale_proteins
        n_pairs: int = min(len(X0_np), len(X1_np))
        if n_pairs > 0:
            lab0 = None if strata is None else np.asarray(strata)[mask0][:n_pairs]
            g01 = (None, None) if g_ratio is None else (g_ratio[mask0][:n_pairs], g_ratio[mask1][:n_pairs])
            b0 = None if birth is None else birth[mask0][:n_pairs]
            pairs.append((float(t0), float(t1), X0_np[:n_pairs], X1_np[:n_pairs], stim0, stim1, lab0, g01, b0))

    stim_first = pairs[0][5] if pairs else np.ones(ns, dtype=np.float32) * scale_proteins
    if np.ndim(ks) == 3 or np.ndim(theta_inter) == 4:
        raise ValueError('per-sample ks / interactions need a 3-D bias and samples_data')
    ode_func: GeneRegulatoryODE_softmax = GeneRegulatoryODE_softmax(
        G, d_init, ks, theta_inter, bias,
        n_stimuli=ns, stim_vals=stim_first, device=device,
        lambda_scale=lambda_scale,
    ).to(device)

    optimizer = torch.optim.Adam([ode_func.d_param, ode_func.scale_param], lr=lr)
    mse = nn.MSELoss(reduction="mean")

    old_loss = 1e16

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        total_loss, total_count = 0.0, 0

        for (t0, t1, X0_full, X1_full, stim0, stim1, lab0, (g0, g1), b0) in pairs:
            ode_func.stim_vals.copy_(torch.tensor(stim1, dtype=torch.float32))
            idxs = _draw_minibatch(X0_full.shape[0], batch_size, lab0)
            ode_func.set_g_interpolation(t0, t1, None if g0 is None else g0[idxs], None if g1 is None else g1[idxs])
            ode_func.set_birth(None if b0 is None else b0[idxs])
            X0: torch.Tensor = torch.tensor(X0_full[idxs], dtype=torch.float32, device=device)
            X1: torch.Tensor = torch.tensor(X1_full[idxs], dtype=torch.float32, device=device)

            X0[:, :ns] = torch.tensor(stim0, dtype=torch.float32)
            X1[:, :ns] = torch.tensor(stim1, dtype=torch.float32)

            t_span: torch.Tensor = torch.tensor([t0, t1], dtype=torch.float32, device=device)

            X_pred_traj = odeint(
                ode_func, X0, t_span,
                method=method, rtol=rtol, atol=atol
            )
            X_pred = X_pred_traj[-1]
            X_pred[:, :ns] = torch.tensor(stim1, dtype=torch.float32, device=device)

            loss_batch = mse(X_pred, X1)
            loss_batch.backward()

            total_loss += loss_batch.item() * len(idxs)
            total_count += len(idxs)

        # Regularization: penalize deviation of scale[ns:] from 1 and d from d_init
        if lambda_scale > 0:
            scale_genes = torch.nn.functional.softplus(ode_func.scale_param[ns:])
            (lambda_scale * torch.mean((scale_genes - 1.0) ** 2)).backward()
        if lambda_deg > 0:
            d_init_t = torch.tensor(d_init, dtype=torch.float32)
            d_now = torch.nn.functional.softplus(ode_func.d_param)
            (lambda_deg * torch.mean((d_now - d_init_t) ** 2)).backward()

        optimizer.step()
        loss: float | Any = total_loss / total_count if total_count > 0 else 0.0
        loss_ema = loss if epoch == 1 else 0.9 * loss_ema + 0.1 * loss

        if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == n_epochs):
            scale_now = torch.nn.functional.softplus(ode_func.scale_param[ns:]).detach().cpu().numpy()
            logger.info(f"[Epoch {epoch}/{n_epochs}] loss = {loss_ema:.6e}  max_scale[ns:] = {scale_now.max():.3e}")
            if abs(loss_ema - old_loss) < 1e-4:
                break
            old_loss: float | Any = loss_ema

    d_learned = torch.nn.functional.softplus(ode_func.d_param).detach().cpu().numpy()
    scale_learned = torch.nn.functional.softplus(ode_func.scale_param).detach().cpu().numpy()

    return d_learned, scale_learned


# ---------------------------
# Prediction & comparison utils
# ---------------------------

def predict_trajectory(ode_func, X0, t_span, method="dopri5", rtol=1e-6, atol=1e-8, stim_vals=None):
    """
    Simulate a trajectory given an initial state and trained ODE model.
    """
    try:
        device = _get_device_from_module(ode_func)
    except Exception:
        device = torch.device("cpu")

    ns: int = getattr(ode_func, "n_stimuli", 1)
    if stim_vals is None:
        stim_vals = np.ones(ns, dtype=np.float32)
    stim_t = torch.tensor(stim_vals, dtype=torch.float32, device=device)

    X0_tensor: torch.Tensor = torch.tensor(X0, dtype=torch.float32, device=device)
    t_span_tensor: torch.Tensor = torch.tensor(t_span, dtype=torch.float32, device=device)

    if X0_tensor.dim() == 1:
        X0_tensor[:ns] = stim_t
    else:
        X0_tensor[:, :ns] = stim_t

    with torch.no_grad():
        traj = odeint(
            ode_func, X0_tensor, t_span_tensor,
            method=method, rtol=rtol, atol=atol
        )
        if traj.dim() == 3:
            traj[:, :, :ns] = stim_t
        elif traj.dim() == 2:
            traj[:, :ns] = stim_t

    return traj.cpu().numpy()


def compare_trajectories_umap(ode_func, X_prot, times, method="dopri5"):
    """
    Compare real and simulated trajectories using UMAP projection.
    """
    import umap

    X_prot = np.asarray(X_prot, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    unique_times = np.sort(np.unique(times))

    X_pred_full, time_pred_full = [], []
    for i, t in enumerate(unique_times[:-1]):
        mask = times == t
        X_at_t = X_prot[mask]
        if X_at_t.size == 0:
            continue
        t_next = unique_times[i + 1]

        traj = predict_trajectory(ode_func, X_at_t, [t, t_next], method=method)
        X_pred_next = traj[-1]

        X_pred_full.append(X_pred_next)
        time_pred_full.extend([t_next] * X_pred_next.shape[0])

    if len(X_pred_full) == 0:
        raise RuntimeError("No predicted points generated - check your input times/data.")

    X_pred_concat = np.vstack(X_pred_full)
    time_pred_concat = np.array(time_pred_full)

    X_combined = np.vstack([X_prot, X_pred_concat])
    labels_combined = np.concatenate([np.zeros(len(X_prot)), np.ones(len(X_pred_concat))])
    time_combined = np.concatenate([times, time_pred_concat])

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_combined)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    real_mask = labels_combined == 0
    pred_mask = labels_combined == 1

    axes[0].scatter(embedding[real_mask, 0], embedding[real_mask, 1],
                    c="blue", alpha=0.6, s=30, label="Real")
    axes[0].scatter(embedding[pred_mask, 0], embedding[pred_mask, 1],
                    c="red", alpha=0.6, s=30, label="Simulated")
    axes[0].set_title("Real vs Simulated")
    axes[0].legend()

    sc = axes[1].scatter(embedding[:, 0], embedding[:, 1],
                         c=time_combined, cmap="viridis", alpha=0.7, s=30)
    axes[1].set_title("Colored by time")
    plt.colorbar(sc, ax=axes[1], label="Time")

    plt.tight_layout()
    plt.show()

    return embedding, labels_combined, time_combined


# ---------------------------
# Example usage (test)
# ---------------------------

if __name__ == "__main__":
    # quick smoke test
    N_cells, G = 100, 5
    times = np.repeat(np.arange(10), N_cells // 10)
    X_prot = np.random.rand(N_cells, G).astype(np.float32)

    n_modes = 3
    bias = np.random.randn(G, n_modes - 1).astype(np.float32) * 0.1
    theta_inter = np.random.randn(G, G, n_modes - 1).astype(np.float32) * 0.1
    ks = np.random.rand(n_modes, G).astype(np.float32)

    d_init = np.ones(G, dtype=np.float32) * 0.5

    d_learned, scale_learned = inference_degradation_prot(
        X_prot, times, bias, theta_inter, ks, d=d_init, n_epochs=10, print_every=5
    )

    logger.info("Learned degradation rates = %s", d_learned)
    logger.info("Learned scale = %s", scale_learned)
