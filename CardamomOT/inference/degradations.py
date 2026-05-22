"""
Utilities for degradation rate inference and temporal epsilon estimation.

This module provides PyTorch models and helper functions used by the
CARDAMOM pipeline when learning gene-specific degradation parameters
from protein dynamics.  It includes the
:class:`GeneRegulatoryODE_softmax` neural ODE model and the
:func:`inference_epsilon_temporal` routine among other utilities.
"""

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import logging
from typing import Any

from CardamomOT.logging import get_logger

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


class KonCorrectionMLP(nn.Module):
    """
    Small MLP learning a per-gene multiplicative correction g(X) ∈ ℝ_{>0}^{G_genes}.

    Used in the Harissa branch to capture the deviation between continuous
    kon_beta_harissa estimates and the discrete-mode network prediction:
        kon_corrected(X) = kon_network(X) * g(X)

    Input  : protein vector X of shape (..., G_genes)
    Output : per-gene correction factor of shape (..., G_genes), forced > 0 via softplus
    """

    def __init__(self, G_genes: int, hidden_dim: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(G_genes, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, G_genes))
        self.net = nn.Sequential(*layers)
        self.G_genes = G_genes

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.net(X))


def train_kon_correction_mlp(
    y_prot: np.ndarray,
    kon_harissa: np.ndarray,
    kon_network: np.ndarray,
    ns: int,
    hidden_dim: int = 32,
    n_layers: int = 2,
    n_epochs: int = 300,
    lr: float = 1e-3,
    device: str = "cpu",
) -> "KonCorrectionMLP":
    """
    Train a KonCorrectionMLP to learn g(X) ≈ kon_harissa / kon_network.

    Args:
        y_prot      : protein data (N*T, G_tot)
        kon_harissa : continuous burst-rate estimates (N*T, G_tot)
        kon_network : discrete-mode network predictions (N*T, G_tot)
        ns          : number of stimulus dimensions to skip
        hidden_dim  : hidden layer width
        n_layers    : number of hidden layers
        n_epochs    : training epochs
        lr          : Adam learning rate
        device      : torch device string

    Returns:
        Trained KonCorrectionMLP (on CPU).
    """
    G_genes = y_prot.shape[1] - ns
    X_t = torch.tensor(y_prot[:, ns:].astype(np.float32), device=device)

    ratio = kon_harissa[:, ns:] / np.clip(kon_network[:, ns:], 1e-8, None)
    ratio = np.clip(ratio, 0.05, 20.0).astype(np.float32)
    y_t = torch.tensor(ratio, device=device)

    mlp = KonCorrectionMLP(G_genes, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(mlp(X_t), y_t)
        loss.backward()
        optimizer.step()

    return mlp.cpu().eval()


class GeneRegulatoryODE_softmax(nn.Module):
    """
    ODE model for gene regulatory dynamics with generalized softmax-based kon.
    Learns gene-specific degradation rates (d) and scale factors.
    Optionally applies a KonCorrectionMLP multiplicative correction (Harissa branch).
    """

    def __init__(self, G, d_init, ks, theta_inter, bias, n_stimuli=1, stim_vals=None,
                 device="cpu", kon_mlp=None) -> None:
        """
        Args:
            G           : number of genes (total, including stimuli)
            d_init      : initial degradation rates (array of size G)
            ks          : array of shape (n_modes, G)
            theta_inter : array of shape (G, G, n_modes-1)
            bias        : array of shape (G, n_modes-1)
            n_stimuli   : number of stimulus columns (default 1)
            stim_vals   : fixed stimulus values shape (n_stimuli,); defaults to ones
            kon_mlp     : optional KonCorrectionMLP; if provided, kon[:, ns:] *= g(X[:, ns:])
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

        # ----- optional harissa correction MLP -----
        self.kon_mlp = kon_mlp  # KonCorrectionMLP or None; not registered as submodule (frozen)

    def forward(self, t, X):
        """
        Compute dX/dt for a given state X at time t.
        Includes learned scaling of theta_inter and bias.
        When kon_mlp is set, applies a frozen per-gene multiplicative correction.
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
        scale: torch.Tensor = torch.nn.functional.softplus(self.scale_param)  # ensures positivity
        scale[:ns] = 1

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

        # ----- apply harissa correction (frozen, no grad) -----
        if self.kon_mlp is not None:
            with torch.no_grad():
                g = self.kon_mlp(X[:, ns:].to("cpu")).to(X.device)
            kon[:, ns:] = kon[:, ns:] * g

        # degradation and ODE dynamics
        d_eff: torch.Tensor = torch.nn.functional.softplus(self.d_param.to(X.device))
        dXdt = d_eff * (kon - X)
        dXdt[:, :ns] = 0.0

        if squeeze_output:
            dXdt = dXdt.squeeze(0)

        return dXdt


# ---------------------------
# inference_epsilon_temporal
# ---------------------------

def inference_epsilon_temporal(
    X_prot, times, bias, theta_inter, ks,
    d_learned_temporal, k1_vec, ratios_init, alpha,
    method="dopri5", rtol=1e-6, atol=1e-8,
    min_x=1e-8, eps_min=1e-2, verbose=True,
    n_stimuli=1, stim_schedule=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate epsilon from residuals given per-interval learned d (d_learned_temporal),
    using a diffusion-like term f(X) = combination * / ratios_init.

    Returns:
        eps_per_interval: numpy array (T-1, G)
        eps_global: numpy array (G,)
        diagnostics: dict with numerators/denominators and pairs used
    """
    device = "cpu"
    X_prot = np.asarray(X_prot, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    k1_vec = np.asarray(k1_vec, dtype=np.float32)

    unique_times = np.unique(times)
    T: int = len(unique_times)
    assert d_learned_temporal.shape[0] == T - 1, "d_learned_temporal must have shape (T-1, G)"
    assert ratios_init.shape[0] == T - 1, "ratios_init must have shape (T-1, G)"

    ns: int = int(n_stimuli)
    G: int = X_prot.shape[1]
    assert k1_vec.shape[0] == G, "k1_vec must have shape (G,)"

    # Containers
    eps_per_interval = np.zeros((T - 1, G), dtype=np.float32)
    numerators_pool = np.zeros(G, dtype=np.float64)
    denominators_pool = np.zeros(G, dtype=np.float64)

    # Build pairs
    pairs = []
    for idx in range(T - 1):
        t0, t1 = float(unique_times[idx]), float(unique_times[idx + 1])
        mask0, mask1 = (times == t0), (times == t1)
        X0_np, X1_np = X_prot[mask0], X_prot[mask1]
        n_pairs: int = min(len(X0_np), len(X1_np))
        if n_pairs > 0:
            pairs.append((idx, t0, t1, X0_np[:n_pairs], X1_np[:n_pairs]))

    # For each interval, simulate and compute residuals
    for (interval_idx, t0, t1, X0_np, X1_np) in pairs:
        stim0 = stim_schedule[t0] if stim_schedule else np.ones(ns, dtype=np.float32)
        stim1 = stim_schedule[t1] if stim_schedule else np.ones(ns, dtype=np.float32)
        dt = float(t1 - t0)
        if dt <= 0:
            raise ValueError("Non positive dt encountered in times")

        d_param_vec = d_learned_temporal[interval_idx]  # shape (G,)
        ratios_init_vec = ratios_init[interval_idx]  # shape (G,)

        # local ODE to compute kon(X)
        stim1_t = torch.tensor(stim1, dtype=torch.float32)
        class LocalODE_for_eps(nn.Module):
            def __init__(self, d_vec) -> None:
                super().__init__()
                self.register_buffer("d_eff_buf", torch.tensor(d_vec, dtype=torch.float32))
                self.register_buffer("stim_buf", stim1_t)

            def forward(self, t, X):
                squeeze_output = False
                if X.dim() == 1:
                    X = X.unsqueeze(0)
                    squeeze_output = True
                X = X.clone()
                X[:, :ns] = self.stim_buf.to(X.device)
                n_modes = ks.shape[0]
                Z: torch.Tensor = torch.zeros((X.shape[0], G, n_modes), device=X.device)
                for k in range(n_modes - 1):
                    Z[:, :, k + 1] = X @ torch.tensor(theta_inter[interval_idx, :, :, k], dtype=torch.float32, device=X.device) \
                                     + torch.tensor(bias[interval_idx, :, k], dtype=torch.float32, device=X.device)
                base_kon: torch.Tensor = torch.softmax(Z, dim=-1)
                kon: torch.Tensor = torch.sum(base_kon * torch.tensor(ks, dtype=torch.float32, device=X.device).T.unsqueeze(0), dim=-1)
                d_eff = self.d_eff_buf.to(X.device)
                dXdt = d_eff * (kon - X)
                dXdt[:, :ns] = 0.0
                if squeeze_output:
                    dXdt = dXdt.squeeze(0)
                return dXdt

        ode_module: LocalODE_for_eps = LocalODE_for_eps(d_param_vec).to(device)

        X0: torch.Tensor = torch.tensor(X0_np, dtype=torch.float32, device=device)
        X1: torch.Tensor = torch.tensor(X1_np, dtype=torch.float32, device=device)
        X0[:, :ns] = torch.tensor(stim0, dtype=torch.float32)
        X1[:, :ns] = torch.tensor(stim1, dtype=torch.float32)
        t_span: torch.Tensor = torch.tensor([t0, t1], dtype=torch.float32, device=device)

        # simulate
        with torch.no_grad():
            X_pred_traj = odeint(ode_module, X0, t_span, method=method, rtol=rtol, atol=atol)
            X_pred = X_pred_traj[-1].cpu().numpy()  # shape (batch, G)
            X_pred[:, :ns] = stim1

        # compute kon(X_pred) and kon(X0)
        kon_fn = build_kon_fn(ks, theta_inter[interval_idx], bias[interval_idx], device=device)
        kon_1 = kon_fn(X_pred)
        kon_0 = kon_fn(X0_np)

        # compute residuals
        residuals = (X_pred - X1.cpu().numpy()) ** 2  # (batch, G)

        d_vec = np.asarray(d_param_vec, dtype=np.float64)
        switch_ratio = alpha[interval_idx]
        fX = np.ones_like(X_pred)
        fX[:, ns:] = ((1 - switch_ratio) * kon_1[:, ns:] + switch_ratio * kon_0[:, ns:]) / ratios_init_vec[None, ns:]
        fX = np.clip(fX, a_min=min_x, a_max=None)

        to_denom_g = 2.0 * (d_vec / k1_vec)[None, :] * fX * float(dt)

        # numerator and denominator per gene
        num_g = np.sum(residuals, axis=0)
        denom_g = np.sum(to_denom_g, axis=0)
        # filtered numerator and denominator per gene
        num_g_filtered, denom_g_filtered = np.zeros_like(num_g), np.zeros_like(denom_g)
        for g in range(G):
            quant = np.quantile(residuals[:, g], .9)  # The variability expalins the worst 5%
            indices_filtered = (residuals[:, g] <= quant)
            num_g_filtered[g] = np.sum(residuals[indices_filtered, g])
            denom_g_filtered[g] = np.sum(to_denom_g[indices_filtered, g])

        eps_interval = num_g / np.where(denom_g > 0, denom_g, np.nan)
        eps_interval = np.where(np.isfinite(eps_interval), eps_interval, float(eps_min))
        eps_interval = np.clip(eps_interval, a_min=float(eps_min), a_max=None)

        eps_interval_filtered = num_g_filtered / np.where(denom_g_filtered > 0, denom_g_filtered, np.nan)
        eps_interval_filtered = np.where(np.isfinite(eps_interval_filtered), eps_interval_filtered, float(eps_min))
        eps_interval_filtered = np.clip(eps_interval_filtered, a_min=float(eps_min), a_max=None)

        for g in range(len(eps_interval)):
            if eps_interval[g] > eps_interval_filtered[g]:
                eps_per_interval[interval_idx, g] = (eps_interval[g] - eps_interval_filtered[g]).astype(np.float32)
            else: 
                eps_per_interval[interval_idx, g] = 1e-16

        numerators_pool += num_g
        denominators_pool += denom_g

        if verbose:
            logger.debug(f"[interval {interval_idx}] dt={dt:.3g} mean_eps_interval={np.mean(eps_interval):.3e}")

    eps_global = numerators_pool / np.where(denominators_pool > 0, denominators_pool, np.nan)
    eps_global = np.where(np.isfinite(eps_global), eps_global, float(eps_min))
    eps_global = np.clip(eps_global, a_min=float(eps_min), a_max=None)

    diagnostics = {
        "numerators_pool": numerators_pool,
        "denominators_pool": denominators_pool,
        "pairs_used": len(pairs)
    }

    return eps_per_interval, eps_global.astype(np.float32)


# ---------------------------
# inference_degradation_prot
# ---------------------------

def inference_degradation_prot(
    X_prot, times, bias, theta_inter, ks, d=None,
    n_epochs=500, lr=1e-2, method="dopri5",
    rtol=1e-6, atol=1e-8, print_every=50,
    batch_size=None, verbose=True,
    n_stimuli=1, stim_schedule=None,
    samples_data=None,
    kon_mlp=None,
) -> tuple[np.ndarray, np.ndarray, float, "GeneRegulatoryODE_softmax"]:
    """
    Estimate degradation rates and scaling factors from protein time-course data.

    When ``bias`` is 3-D (n_samples, G, n_modes-1) and ``samples_data`` is provided,
    one ODE module is created per sample with its own bias while ``d_param`` and
    ``scale_param`` are shared, so a single optimizer refines the shared kinetics
    from all samples jointly.

    Returns:
        ``d_learned``: Learned degradation rates, shape (G,).
        ``scale_learned``: Learned scaling parameters, shape (G,).
    """
    device = "cpu"
    ns: int = int(n_stimuli)
    X_prot = np.asarray(X_prot, dtype=np.float32)
    times  = np.asarray(times,  dtype=np.float32)
    bias   = np.asarray(bias,   dtype=np.float32)

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
        stim_first = np.ones(ns, dtype=np.float32)
        ode_funcs = [
            GeneRegulatoryODE_softmax(
                G, d_init, ks, theta_inter, bias[s_idx],
                n_stimuli=ns, stim_vals=stim_first, device=device,
                kon_mlp=kon_mlp,
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
        for s_idx, s in enumerate(unique_s):
            mask_s = (samples_data_arr == s)
            X_s  = X_prot[mask_s]
            t_s  = times[mask_s]
            unique_t = np.unique(t_s)
            for ti in range(len(unique_t) - 1):
                t0, t1 = float(unique_t[ti]), float(unique_t[ti + 1])
                X0_np = X_s[t_s == unique_t[ti]]
                X1_np = X_s[t_s == unique_t[ti + 1]]
                stim0 = stim_schedule[t0] if stim_schedule else np.ones(ns, dtype=np.float32)
                stim1 = stim_schedule[t1] if stim_schedule else np.ones(ns, dtype=np.float32)
                n_p = min(len(X0_np), len(X1_np))
                if n_p > 0:
                    all_pairs.append((s_idx, t0, t1, X0_np[:n_p], X1_np[:n_p], stim0, stim1))

        old_loss  = 1e16
        final_loss = None

        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            total_loss, total_count = 0.0, 0

            for (s_idx, t0, t1, X0_full, X1_full, stim0, stim1) in all_pairs:
                ode = ode_funcs[s_idx]
                ode.stim_vals.copy_(torch.tensor(stim1, dtype=torch.float32))
                n_cells = X0_full.shape[0]
                cur_bs  = n_cells if batch_size is None else batch_size
                perm    = np.random.permutation(n_cells)
                for start in range(0, n_cells, cur_bs):
                    idxs = perm[start:start + cur_bs]
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

            optimizer.step()
            loss: float = total_loss / total_count if total_count > 0 else 0.0
            if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == n_epochs):
                logger.info(f"[Epoch {epoch}/{n_epochs}] loss = {loss:.6e}")
                if abs(loss - old_loss) < 1e-4:
                    break
                old_loss = loss

        d_learned     = torch.nn.functional.softplus(ode_funcs[0].d_param).detach().cpu().numpy()
        scale_learned = torch.nn.functional.softplus(ode_funcs[0].scale_param).detach().cpu().numpy()
        return d_learned, scale_learned

    # ── Single-bias mode (original behaviour) ────────────────────────────
    unique_times = np.unique(times)
    pairs = []
    for idx in range(len(unique_times) - 1):
        t0, t1 = unique_times[idx], unique_times[idx + 1]
        mask0, mask1 = (times == t0), (times == t1)
        X0_np, X1_np = X_prot[mask0], X_prot[mask1]
        stim0 = stim_schedule[float(t0)] if stim_schedule else np.ones(ns, dtype=np.float32)
        stim1 = stim_schedule[float(t1)] if stim_schedule else np.ones(ns, dtype=np.float32)
        n_pairs: int = min(len(X0_np), len(X1_np))
        if n_pairs > 0:
            pairs.append((float(t0), float(t1), X0_np[:n_pairs], X1_np[:n_pairs], stim0, stim1))

    stim_first = pairs[0][4] if pairs else np.ones(ns, dtype=np.float32)
    ode_func: GeneRegulatoryODE_softmax = GeneRegulatoryODE_softmax(
        G, d_init, ks, theta_inter, bias,
        n_stimuli=ns, stim_vals=stim_first, device=device,
        kon_mlp=kon_mlp,
    ).to(device)

    optimizer = torch.optim.Adam([ode_func.d_param, ode_func.scale_param], lr=lr)
    mse = nn.MSELoss(reduction="mean")

    old_loss = 1e16

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        total_loss, total_count = 0.0, 0

        for (t0, t1, X0_full, X1_full, stim0, stim1) in pairs:
            ode_func.stim_vals.copy_(torch.tensor(stim1, dtype=torch.float32))
            n_cells = X0_full.shape[0]
            current_batch_size = n_cells if batch_size is None else batch_size

            perm = np.random.permutation(n_cells)
            for start in range(0, n_cells, current_batch_size):
                idxs = perm[start:start + current_batch_size]
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

        optimizer.step()
        loss: float | Any = total_loss / total_count if total_count > 0 else 0.0

        if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == n_epochs):
            logger.info(f"[Epoch {epoch}/{n_epochs}] loss = {loss:.6e} "
                        f"max scale = {np.max(torch.nn.functional.softplus(ode_func.scale_param).detach().cpu().numpy()[1:]):.3e}")
            if abs(loss - old_loss) < 1e-4:
                break
            old_loss: float | Any = loss

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
    unique_times = np.unique(times)

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

    d_learned, final_loss, ode_func = inference_degradation_prot(
        X_prot, times, bias, theta_inter, ks, d=d_init, n_epochs=10, print_every=5
    )

    logger.info("Learned degradation rates = %s", d_learned)
    logger.info("Final loss = %s", final_loss)

    embedding, labels, times_comb = compare_trajectories_umap(ode_func, X_prot, times)
