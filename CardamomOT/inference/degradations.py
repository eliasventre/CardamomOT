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


class GeneRegulatoryODE_softmax(nn.Module):
    """
    ODE model for gene regulatory dynamics with generalized softmax-based kon.
    Learns gene-specific degradation rates (d) and scale factors.
    """

    def __init__(self, G, d_init, ks, theta_inter, bias, device="cpu") -> None:
        """
        Args:
            G : number of genes
            d_init : initial degradation rates (array of size G)
            ks : array of shape (n_modes, G) -> base kon rates for each mode
            theta_inter : array of shape (G, G, n_modes-1) -> regulatory interactions
            bias : array of shape (G, n_modes-1) -> bias for each gene and mode
        """
        super().__init__()
        self.G = int(G)
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

        self.n_modes = int(self.ks.shape[0])

    def forward(self, t, X):
        """
        Compute dX/dt for a given state X at time t.
        Includes learned scaling of theta_inter and bias.
        """
        squeeze_output = False
        if X.dim() == 1:
            X = X.unsqueeze(0)
            squeeze_output = True

        X = X.clone()
        X[:, 0] = 1.0  # external input = constant

        batch_size, G = X.shape[0], self.G
        n_modes: int = self.n_modes

        # ----- compute scale -----
        scale: torch.Tensor = torch.nn.functional.softplus(self.scale_param)  # ensures positivity
        scale[0] = 1

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

        # degradation and ODE dynamics
        d_eff: torch.Tensor = torch.nn.functional.softplus(self.d_param.to(X.device))
        dXdt = d_eff * (kon - X)
        dXdt[:, 0] = 0.0  # no dynamics for external input

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
    min_x=1e-8, eps_min=1e-2, verbose=True
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
        dt = float(t1 - t0)
        if dt <= 0:
            raise ValueError("Non positive dt encountered in times")

        d_param_vec = d_learned_temporal[interval_idx]  # shape (G,)
        ratios_init_vec = ratios_init[interval_idx]  # shape (G,)

        # local ODE to compute kon(X)
        class LocalODE_for_eps(nn.Module):
            def __init__(self, d_vec) -> None:
                super().__init__()
                # register buffers (not params) because we don't optimize here
                self.register_buffer("d_eff_buf", torch.tensor(d_vec, dtype=torch.float32))

            def forward(self, t, X):
                squeeze_output = False
                if X.dim() == 1:
                    X = X.unsqueeze(0)
                    squeeze_output = True
                X = X.clone()
                X[:, 0] = 1.0
                n_modes = ks.shape[0]
                Z: torch.Tensor = torch.zeros((X.shape[0], G, n_modes), device=X.device)
                for k in range(n_modes - 1):
                    # here we create small tensors on the fly but it's acceptable for inference
                    Z[:, :, k + 1] = X @ torch.tensor(theta_inter[interval_idx, :, :, k], dtype=torch.float32, device=X.device) \
                                     + torch.tensor(bias[interval_idx, :, k], dtype=torch.float32, device=X.device)
                base_kon: torch.Tensor = torch.softmax(Z, dim=-1)
                kon: torch.Tensor = torch.sum(base_kon * torch.tensor(ks, dtype=torch.float32, device=X.device).T.unsqueeze(0), dim=-1)
                d_eff = self.d_eff_buf.to(X.device)
                dXdt = d_eff * (kon - X)
                dXdt[:, 0] = 0.0
                if squeeze_output:
                    dXdt = dXdt.squeeze(0)
                return dXdt

        ode_module: LocalODE_for_eps = LocalODE_for_eps(d_param_vec).to(device)

        X0: torch.Tensor = torch.tensor(X0_np, dtype=torch.float32, device=device)
        X1: torch.Tensor = torch.tensor(X1_np, dtype=torch.float32, device=device)
        X0[:, 0] = 1.0
        X1[:, 0] = 1.0
        t_span: torch.Tensor = torch.tensor([t0, t1], dtype=torch.float32, device=device)

        # simulate
        with torch.no_grad():
            X_pred_traj = odeint(ode_module, X0, t_span, method=method, rtol=rtol, atol=atol)
            X_pred = X_pred_traj[-1].cpu().numpy()  # shape (batch, G)
            X_pred[:, 0] = 1.0

        # compute kon(X_pred) and kon(X0)
        kon_fn = build_kon_fn(ks, theta_inter[interval_idx], bias[interval_idx], device=device)
        kon_1 = kon_fn(X_pred)
        kon_0 = kon_fn(X0_np)

        # compute residuals
        residuals = (X_pred - X1.cpu().numpy()) ** 2  # (batch, G)

        d_vec = np.asarray(d_param_vec, dtype=np.float64)
        switch_ratio = alpha[interval_idx]
        fX = np.ones_like(X_pred)
        # for genes > 0 use the combination as in original logic; assume gene 0 is stimulus and stays 1
        fX[:, 1:] = ((1 - switch_ratio) * kon_1[:, 1:] + switch_ratio * kon_0[:, 1:]) / ratios_init_vec[None, 1:]
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
    batch_size=None, verbose=True
) -> tuple[np.ndarray, np.ndarray, float, "GeneRegulatoryODE_softmax"]:
    """
    Estimate degradation rates and scaling factors from protein time-course data.

    This function fits a neural ODE model (``GeneRegulatoryODE_softmax``) to the
    observed protein expression trajectories.  Both the gene-specific degradation
    rates ``d`` and per-gene scaling factors are optimized by minimizing mean-
    squared error between observations and ODE predictions.

    Args:
        X_prot: Protein expression matrix (cells × genes).
        times: Vector of measurement times.
        bias: Bias parameters for the regulatory network.
        theta_inter: Interaction parameters for the ODE model.
        ks: Burst frequency modes.
        d: Optional initial degradation rates (length = G).  If None,
           starts with ones.
        n_epochs: Number of optimization epochs.
        lr: Learning rate for the Adam optimizer.
        method: Integration method for ``odeint``.
        rtol: Relative tolerance for solver.
        atol: Absolute tolerance for solver.
        print_every: Frequency (in epochs) of status logging.
        batch_size: Number of cells per batch during optimization.
        verbose: If True, training progress is logged.

    Returns:
        ``d_learned``: Learned degradation rates, shape (G,).
        ``scale_learned``: Learned scaling parameters, shape (G,).
        ``final_loss``: Final training loss (float).
        ``ode_func``: Trained ``GeneRegulatoryODE_softmax`` instance.
    """
    device = "cpu"
    X_prot = np.asarray(X_prot, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)

    G: int = X_prot.shape[1]

    if d is None:
        d_init = np.ones(G, dtype=np.float32)
    else:
        d_init = np.asarray(d, dtype=np.float32)

    # Build time pairs (t0 -> t1)
    unique_times = np.unique(times)
    pairs = []
    for idx in range(len(unique_times) - 1):
        t0, t1 = unique_times[idx], unique_times[idx + 1]
        mask0, mask1 = (times == t0), (times == t1)
        X0_np, X1_np = X_prot[mask0], X_prot[mask1]

        n_pairs: int = min(len(X0_np), len(X1_np))
        if n_pairs > 0:
            pairs.append((float(t0), float(t1), X0_np[:n_pairs], X1_np[:n_pairs]))

    # Instantiate ODE model
    ode_func: GeneRegulatoryODE_softmax = GeneRegulatoryODE_softmax(G, d_init, ks, theta_inter, bias, device=device).to(device)

    # Optimizer now learns both d and scale
    optimizer = torch.optim.Adam([ode_func.d_param, ode_func.scale_param], lr=lr)
    mse = nn.MSELoss(reduction="mean")

    old_loss = 1e16
    final_loss = None

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        total_loss, total_count = 0.0, 0

        for (t0, t1, X0_full, X1_full) in pairs:
            n_cells = X0_full.shape[0]
            current_batch_size = n_cells if batch_size is None else batch_size

            perm = np.random.permutation(n_cells)
            for start in range(0, n_cells, current_batch_size):
                idxs = perm[start:start + current_batch_size]
                X0: torch.Tensor = torch.tensor(X0_full[idxs], dtype=torch.float32, device=device)
                X1: torch.Tensor = torch.tensor(X1_full[idxs], dtype=torch.float32, device=device)

                # enforce constant external stimulus
                X0[:, 0] = 1.0
                X1[:, 0] = 1.0

                t_span: torch.Tensor = torch.tensor([t0, t1], dtype=torch.float32, device=device)

                X_pred_traj = odeint(
                    ode_func, X0, t_span,
                    method=method, rtol=rtol, atol=atol
                )
                X_pred = X_pred_traj[-1]
                X_pred[:, 0] = 1.0

                loss_batch = mse(X_pred, X1)
                loss_batch.backward()

                total_loss += loss_batch.item() * len(idxs)
                total_count += len(idxs)

        optimizer.step()
        loss: float | Any = total_loss / total_count if total_count > 0 else 0.0
        final_loss: float | Any = loss

        if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == n_epochs):
            logger.info(f"[Epoch {epoch}/{n_epochs}] loss = {loss:.6e} "
                        f"max scale = {np.max(torch.nn.functional.softplus(ode_func.scale_param).detach().cpu().numpy()[1:]):.3e}")
            if abs(loss - old_loss) < 1e-4:
                break
            old_loss: float | Any = loss

    # Return learned parameters
    d_learned = torch.nn.functional.softplus(ode_func.d_param).detach().cpu().numpy()
    scale_learned = torch.nn.functional.softplus(ode_func.scale_param).detach().cpu().numpy()

    return d_learned, scale_learned


# ---------------------------
# Prediction & comparison utils
# ---------------------------

def predict_trajectory(ode_func, X0, t_span, method="dopri5", rtol=1e-6, atol=1e-8):
    """
    Simulate a trajectory given an initial state and trained ODE model.
    """
    # Determine device robustly
    try:
        device = _get_device_from_module(ode_func)
    except Exception:
        device = torch.device("cpu")

    X0_tensor: torch.Tensor = torch.tensor(X0, dtype=torch.float32, device=device)
    t_span_tensor: torch.Tensor = torch.tensor(t_span, dtype=torch.float32, device=device)

    if X0_tensor.dim() == 1:
        X0_tensor[0] = 1.0
    else:
        X0_tensor[:, 0] = 1.0

    with torch.no_grad():
        traj = odeint(
            ode_func, X0_tensor, t_span_tensor,
            method=method, rtol=rtol, atol=atol
        )
        # ensure stimulus column stays at 1
        if traj.dim() == 3:
            traj[:, :, 0] = 1.0
        elif traj.dim() == 2:
            traj[:, 0] = 1.0

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
