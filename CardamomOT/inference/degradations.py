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


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

# def nb_log_likelihood_stable(
#     k: torch.Tensor,
#     mu: torch.Tensor,
#     r: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     NB log-likelihood, paramétrisation mean/dispersion.
#     Stable numériquement grâce aux log directs.
#     variance = mu + mu²/r  (r → ∞ : Poisson)
#     """
#     log_p = (
#         torch.lgamma(k + r)
#         - torch.lgamma(r)
#         - torch.lgamma(k + 1)
#         + r * (torch.log(r) - torch.log(r + mu))
#         + k * (torch.log(mu) - torch.log(r + mu))
#     )
#     return log_p.sum()


# class KonCorrectionMLP(nn.Module):
#     """
#     Learns delta_mu = kon_harissa / kon_beta from [y_prot, kon_beta].

#     At init: delta_mu = 1 everywhere (prior, no correction).

#     Call directly with numpy:
#         ratio = mlp(y_prot, kon_beta_genes)   # np.ndarray (N, G_genes)
#     """

#     def __init__(
#         self,
#         G_tot: int,
#         G_genes: int,
#         hidden_dim: int = 64,
#         n_layers: int = 2,
#     ) -> None:
#         super().__init__()
#         self.G_tot   = G_tot
#         self.G_genes = G_genes

#         input_dim = G_tot + G_genes

#         trunk: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
#         for _ in range(n_layers - 1):
#             trunk += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
#         self.trunk = nn.Sequential(*trunk)

#         for layer in self.trunk:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight, gain=0.1)
#                 nn.init.zeros_(layer.bias)

#         self.head = nn.Linear(hidden_dim, G_genes)
#         nn.init.zeros_(self.head.weight)
#         nn.init.constant_(self.head.bias, 0.541)  # softplus(0.541) ≈ 1.0

#     def _forward_tensor(
#         self, y_prot: torch.Tensor, kon_beta: torch.Tensor
#     ) -> torch.Tensor:
#         x = torch.cat([y_prot, kon_beta], dim=-1)
#         return torch.nn.functional.softplus(self.head(self.trunk(x)))

#     def forward(self, y_prot: np.ndarray, kon_beta: np.ndarray) -> np.ndarray:
#         """
#         Prédit delta_mu = kon_harissa / kon_beta.

#         Args:
#             y_prot   : (N, G_tot)   ou (G_tot,)
#             kon_beta : (N, G_genes) ou (G_genes,)   colonnes ns: seulement

#         Returns:
#             delta_mu : (N, G_genes) ou (G_genes,)   toujours > 0
#         """
#         squeeze = y_prot.ndim == 1
#         X  = torch.tensor(np.atleast_2d(y_prot).astype(np.float32))
#         kb = torch.tensor(np.atleast_2d(kon_beta).astype(np.float32))

#         with torch.no_grad():
#             delta_mu = self._forward_tensor(X, kb)

#         result = delta_mu.numpy()
#         return result[0] if squeeze else result


# def train_kon_correction_mlp(
#     y_prot: np.ndarray,
#     kon_harissa: np.ndarray,
#     kon_beta: np.ndarray,
#     ns: int,
#     hidden_dim: int = 64,
#     n_layers: int = 2,
#     n_epochs: int = 1000,
#     lr: float = 1e-2,
#     device: str = "cpu",
#     seuil: float = 1e-8,
#     fixed_r: float = 10.0,    # grand → proche Poisson, pas appris
# ) -> KonCorrectionMLP:
#     """
#     Maximise la NB log-likelihood avec r fixe (proche Poisson) :

#         kon_harissa_g | X  ~  NB( mean = kon_beta_g * delta_mu_g(X), r=fixed_r )

#     Le log dans la NB-NLL écrase naturellement les grandes valeurs
#     et force le réseau à expliquer la moyenne plutôt que la variance.
#     """
#     N, G_tot = y_prot.shape
#     G_genes  = G_tot - ns

#     hidden_dim = max(hidden_dim, G_tot) 

#     X_t  = torch.tensor(y_prot.astype(np.float32),               device=device)
#     kb_t = torch.tensor(kon_beta[:, ns:].astype(np.float32),     device=device).clamp(seuil, None)
#     kh_t = torch.tensor(kon_harissa[:, ns:].astype(np.float32),  device=device).clamp(seuil, None)
#     r    = torch.full_like(kh_t, fixed_r)   # constante, pas de grad

#     mlp = KonCorrectionMLP(G_tot, G_genes, hidden_dim, n_layers).to(device)
#     optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

#     for epoch in range(n_epochs):
#         optimizer.zero_grad()
#         delta_mu = mlp._forward_tensor(X_t, kb_t)
#         mu       = kb_t * delta_mu
#         loss     = -nb_log_likelihood_stable(kh_t, mu, r)
#         loss.backward()
#         optimizer.step()

#         if epoch % 1000 == 0:
#             with torch.no_grad():
#                 rmse = ((delta_mu * kb_t - kh_t) ** 2).mean().item() ** 0.5
#             print(f"epoch {epoch:5d}  NB-NLL/obs = {loss.item()/(N*G_genes):.4f}  RMSE(ratio) = {rmse:.4f}")

#     return mlp.cpu().eval()

from joblib import Parallel, delayed

def _train_single_gene(
    g_idx: int,
    P_g: np.ndarray,       # (N,) protéine du gène g
    kb_g: np.ndarray,      # (N,) kon_beta du gène g
    kh_g: np.ndarray,      # (N,) kon_harissa du gène g
    hidden_dim: int,
    n_epochs: int,
    lr: float,
    fixed_r: float,
    seuil: float,
) -> np.ndarray:
    """Entraîne un MLP pour un seul gène, retourne ses poids."""
    import torch
    import torch.nn as nn

    P_t  = torch.tensor(P_g.astype(np.float32)).unsqueeze(-1)   # (N, 1)
    kb_t = torch.tensor(kb_g.astype(np.float32)).unsqueeze(-1).clamp(seuil, None)  # (N, 1)
    kh_t = torch.tensor(kh_g.astype(np.float32)).clamp(seuil, None)   # (N,)
    r    = torch.full_like(kh_t, fixed_r)

    x = torch.cat([P_t, kb_t], dim=-1)  # (N, 2)

    W1 = nn.Parameter(torch.empty(2, hidden_dim))
    b1 = nn.Parameter(torch.zeros(hidden_dim))
    W2 = nn.Parameter(torch.zeros(hidden_dim, 1))
    b2 = nn.Parameter(torch.full((1,), 0.541))
    nn.init.xavier_uniform_(W1, gain=0.1)

    optimizer = torch.optim.Adam([W1, b1, W2, b2], lr=lr)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        h       = torch.tanh(x @ W1 + b1)              # (N, hidden)
        out     = torch.nn.functional.softplus(h @ W2 + b2).squeeze(-1)  # (N,)
        mu      = (kb_t.squeeze(-1) * out).clamp(seuil)
        log_p   = (
            torch.lgamma(kh_t + r) - torch.lgamma(r) - torch.lgamma(kh_t + 1)
            + r * (torch.log(r) - torch.log(r + mu))
            + kh_t * (torch.log(mu) - torch.log(r + mu))
        )
        loss = -log_p.sum()
        loss.backward()
        optimizer.step()

    return W1.detach(), b1.detach(), W2.detach(), b2.detach()


class KonCorrectionMLP(nn.Module):
    """
    G_genes MLPs indépendants entraînés en parallèle via joblib.
    Gène i prend [P_i, kon_beta_i] → delta_mu_i.
    At init: delta_mu = 1 everywhere.
    """

    def __init__(self, G_tot: int, G_genes: int, hidden_dim: int = 8) -> None:
        super().__init__()
        self.G_tot    = G_tot
        self.G_genes  = G_genes
        self.ns       = G_tot - G_genes
        self.hidden_dim = hidden_dim

        # stocke les poids comme buffers après entraînement
        self.W1 = nn.Parameter(torch.zeros(G_genes, 2, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(G_genes, hidden_dim))
        self.W2 = nn.Parameter(torch.zeros(G_genes, hidden_dim, 1))
        self.b2 = nn.Parameter(torch.full((G_genes, 1), 0.541))

    def _forward_tensor(
        self, y_prot: torch.Tensor, kon_beta: torch.Tensor
    ) -> torch.Tensor:
        ns = self.ns
        P  = y_prot[:, ns:]                      # (N, G_genes)
        x  = torch.stack([P, kon_beta], dim=-1)  # (N, G_genes, 2)
        x  = x.permute(1, 0, 2)                  # (G_genes, N, 2)
        h   = torch.tanh(torch.bmm(x, self.W1) + self.b1.unsqueeze(1))
        out = torch.bmm(h, self.W2) + self.b2.unsqueeze(1)
        return torch.nn.functional.softplus(out.squeeze(-1).permute(1, 0))

    def forward(self, y_prot: np.ndarray, kon_beta: np.ndarray) -> np.ndarray:
        squeeze = y_prot.ndim == 1
        X  = torch.tensor(np.atleast_2d(y_prot).astype(np.float32))
        kb = torch.tensor(np.atleast_2d(kon_beta).astype(np.float32))
        with torch.no_grad():
            delta_mu = self._forward_tensor(X, kb)
        result = delta_mu.numpy()
        return result[0] if squeeze else result


def train_kon_correction_mlp(
    y_prot: np.ndarray,
    kon_harissa: np.ndarray,
    kon_beta: np.ndarray,
    ns: int,
    hidden_dim: int = 8, 
    n_epochs: int = 1000,
    lr: float = 1e-2,
    seuil: float = 1e-8,
    fixed_r: float = 10.0,
    n_jobs: int = -1,
) -> KonCorrectionMLP:

    N, G_tot = y_prot.shape
    G_genes  = G_tot - ns

    P_all  = y_prot[:, ns:]          # (N, G_genes)
    kb_all = kon_beta[:, ns:]        # (N, G_genes)
    kh_all = kon_harissa[:, ns:]     # (N, G_genes)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_train_single_gene)(
            g, P_all[:, g], kb_all[:, g], kh_all[:, g],
            hidden_dim, n_epochs, lr, fixed_r, seuil
        )
        for g in range(G_genes)
    )

    mlp = KonCorrectionMLP(G_tot, G_genes, hidden_dim)
    for g, (W1, b1, W2, b2) in enumerate(results):
        mlp.W1.data[g] = W1
        mlp.b1.data[g] = b1
        mlp.W2.data[g] = W2
        mlp.b2.data[g] = b2

    return mlp.eval()


class GeneRegulatoryODE_softmax(nn.Module):
    """
    ODE model for gene regulatory dynamics with generalized softmax-based kon.
    Learns gene-specific degradation rates (d) and scale factors.
    Optionally applies a KonCorrectionMLP multiplicative correction (Harissa branch).
    """

    def __init__(self, G, d_init, ks, theta_inter, bias, n_stimuli=1, stim_vals=None,
                 device="cpu", kon_mlp=None, lambda_scale=1e3) -> None:
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
        self.lambda_scale = float(lambda_scale)

        # ----- optional harissa correction MLP -----
        self.kon_mlp = kon_mlp  # KonCorrectionMLP or None; not registered as submodule (frozen)

        # ----- lambda_mlp interpolation state (set per interval via set_ratio_interpolation) -----
        self._lambda_mlp   = 0.0   # mix weight; 0 = pure MLP (original behaviour)
        self._ratio_t0     = None  # (G_genes,) mean observed g at interval start
        self._ratio_t1     = None  # (G_genes,) mean observed g at interval end
        self._t_interval_start = None  # float
        self._t_interval_end   = None  # float

    def set_ratio_interpolation(self, t_start, t_end, ratio_t0, ratio_t1, lambda_mlp):
        """Configure time-interpolated ratio mix for the next odeint call.

        Args:
            t_start, t_end : boundaries of the current interval (floats).
            ratio_t0, ratio_t1 : mean observed g = kon_harissa/kon_beta at
                                  ``t_start`` / ``t_end``, shape ``(G_genes,)``.
            lambda_mlp     : mix weight (1 = pure interpolation, 0 = pure MLP).
        """
        self._lambda_mlp        = float(lambda_mlp)
        self._t_interval_start  = float(t_start)
        self._t_interval_end    = float(t_end)
        self._ratio_t0 = torch.tensor(np.asarray(ratio_t0, dtype=np.float32))
        self._ratio_t1 = torch.tensor(np.asarray(ratio_t1, dtype=np.float32))

    def forward(self, t, X):
        """
        Compute dX/dt for a given state X at time t.
        Includes learned scaling of theta_inter and bias.
        When kon_mlp is set, applies a frozen per-gene multiplicative correction.
        If ``set_ratio_interpolation`` has been called, the correction is mixed
        with the linearly interpolated observed training ratios at weight
        ``lambda_mlp``.
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

        # ----- apply harissa correction (frozen, no grad) -----
        if self.kon_mlp is not None:
            g = torch.tensor(
                self.kon_mlp(
                    X.detach().cpu().numpy(),
                    kon[:, ns:].detach().cpu().numpy(),
                ),
                device=X.device,
            )   # (batch, G_genes)

            # lambda_mlp mix: blend with linearly interpolated observed ratios
            if self._lambda_mlp > 0.0 and self._ratio_t0 is not None:
                t_s   = self._t_interval_start
                t_e   = self._t_interval_end
                alpha = float((float(t) - t_s) / (t_e - t_s + 1e-10))
                alpha = max(0.0, min(1.0, alpha))
                r0    = self._ratio_t0.to(X.device).unsqueeze(0)  # (1, G_genes)
                r1    = self._ratio_t1.to(X.device).unsqueeze(0)
                g_interp = (1.0 - alpha) * r0 + alpha * r1        # (1, G_genes) → broadcast
                a_alpha = 4 * (1 - self._lambda_mlp)
                lamb_alpha = 1 + a_alpha * alpha * (alpha - 1)
                g = lamb_alpha * g_interp + (1.0 - lamb_alpha) * g

            kon = torch.cat([kon[:, :ns], kon[:, ns:] * g], dim=-1)

        # degradation and ODE dynamics
        d_eff: torch.Tensor = torch.nn.functional.softplus(self.d_param.to(X.device))
        dXdt = d_eff * (kon - X)
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
        theta_inter : ``(G, G, n_modes-1)`` GRN interaction tensor.
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
    n_modes = ks.shape[0]

    per_sample = (bias.ndim == 3 and samples_data is not None)

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
                A_g_s  = X_prot[mask_s] @ theta_inter[:, g, :] + bias_s[g, :]
                sample_data_g.append((mask_s, A_g_s))
            gene_data.append((kon_beta[:, g], ks[:, g], sample_data_g))

        def _total_loss(log_s):
            s = np.exp(log_s)
            total = 0.0
            for target, ks_g, sample_data_g in gene_data:
                pred = np.zeros(N)
                for mask_s_, A_g_s_ in sample_data_g:
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
    verbose=True, kon_mlp=None,
    scale=1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ε_i = d1_i/d0_i from ODE residuals (bursty-PDMP variance matching).

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
        kon_mlp            : Optional :class:`KonCorrectionMLP` (Harissa branch).
        scale              : Protein scale (``self.scale_proteins``). Default 1.0.

    Returns:
        eps_temporal : ``(T-1, G)`` per-interval ε = d1/d0 estimates.
        eps_global   : ``(G,)`` global ε pooled over all intervals.
    """
    device = "cpu"
    X_prot   = np.asarray(X_prot,  dtype=np.float32)
    times    = np.asarray(times,   dtype=np.float32)
    k1_vec   = np.asarray(k1_vec,  dtype=np.float32)
    d_learned_temporal = np.asarray(d_learned_temporal, dtype=np.float32)
    ks_np    = np.asarray(ks,      dtype=np.float32)   # (n_modes, G)

    unique_times = np.sort(np.unique(times))
    T: int   = len(unique_times)
    ns: int  = int(n_stimuli)
    G: int   = X_prot.shape[1]

    assert d_learned_temporal.shape[0] == T - 1, "d_learned_temporal must have (T-1, G)"
    assert k1_vec.shape[0] == G

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

        mask0 = times == unique_times[cnt]
        mask1 = times == unique_times[cnt + 1]
        X0_np = X_prot[mask0]
        X1_np = X_prot[mask1]
        if len(X0_np) == 0 or len(X1_np) == 0:
            continue

        # Pair cells at t0 with cells at t1 (arbitrary but consistent)
        n_pairs = min(len(X0_np), len(X1_np))
        X0_np = X0_np[:n_pairs]
        X1_np = X1_np[:n_pairs]

        stim0 = np.asarray(stim_schedule[t0] if stim_schedule else np.ones(ns), dtype=np.float32)
        stim1 = np.asarray(stim_schedule[t1] if stim_schedule else np.ones(ns), dtype=np.float32)
        stim0, stim1 = stim0*scale, stim1*scale

        # ── Select per-interval bias / theta ─────────────────────────────────
        bias_np  = np.asarray(bias,         dtype=np.float32)
        theta_np = np.asarray(theta_inter,  dtype=np.float32)

        # Support 3-D bias (T-1, G, nets) and 4-D (T-1, n_samples, G, nets)
        if bias_np.ndim == 3:
            bias_cnt  = bias_np[cnt]                         # (G, n_nets)
        elif bias_np.ndim == 4:
            # Use the sample of the first cell in this interval
            if samples_data is not None:
                s_idx = int(np.asarray(samples_data)[mask0][0])
                bias_cnt = bias_np[cnt, s_idx]
            else:
                bias_cnt = bias_np[cnt, 0]                   # fallback: first sample
        else:
            bias_cnt = bias_np

        if theta_np.ndim == 4:
            theta_cnt = theta_np[cnt]                        # (G, G, n_nets)
        elif theta_np.ndim == 5:
            s_idx = int(np.asarray(samples_data)[mask0][0]) if samples_data is not None else 0
            theta_cnt = theta_np[cnt, s_idx]
        else:
            theta_cnt = theta_np

        d_param_vec = d_learned_temporal[cnt]                # (G,)
        k1_i        = k1_vec / float(scale)                 # (G,) — raw max burst rate

        # ── Build ODE and simulate ────────────────────────────────────────────
        ode = GeneRegulatoryODE_softmax(
            G, d_param_vec, ks_np, theta_cnt, bias_cnt,
            n_stimuli=ns,
            stim_vals=stim1,
            device=device,
            kon_mlp=kon_mlp,
        ).to(device)
        ode.eval()

        X0_t = torch.tensor(X0_np, dtype=torch.float32)
        X0_t[:, :ns] = torch.tensor(stim0)
        t_span = torch.tensor([t0, t1], dtype=torch.float32)

        with torch.no_grad():
            traj = odeint(ode, X0_t, t_span, method=method, rtol=rtol, atol=atol)
        X_pred = traj[-1].cpu().numpy()       # (n_pairs, G)
        X_pred[:, :ns] = stim1

        # ── Compute kon at initial and predicted states ───────────────────────
        kon_fn = build_kon_fn(ks_np, theta_cnt, bias_cnt, device=device)
        kon_0 = kon_fn(X0_np)                # (n_pairs, G) in protein units
        kon_1 = kon_fn(X_pred)
        if kon_mlp is not None:
            kon_0 = np.concatenate([
                kon_0[:, :ns],
                kon_0[:, ns:] * kon_mlp(X0_np, kon_0[:, ns:]),
            ], axis=-1)
            kon_1 = np.concatenate([
                kon_1[:, :ns],
                kon_1[:, ns:] * kon_mlp(X_pred, kon_1[:, ns:]),
            ], axis=-1)

        # Midpoint average of kon over the interval
        kon_avg = 0.5 * (kon_0 + kon_1)   # (n_pairs, G)

        # ── Residuals and denominator ─────────────────────────────────────────
        residuals = (X_pred - X1_np) ** 2   # (n_pairs, G)

        # h_{ic} = 2 · dt · scale · d1_i/k1_i · kon_i(P_c), kon_i(P_c) = kon_avg already = kon_true/k1_i
        # Derived from Campbell's theorem: burst rate=k1·kon_norm·d0, burst size~Exp(k1·d0/(scale·d1))
        #   Var(ΔP_i) = burst_rate · E[B²] · dt = ε_i · 2·dt·scale·(d1_i/k1_i)·kon_i
        h_mat = (2.0 * dt * float(scale)) * (d_param_vec / k1_i)[None, :] * kon_avg   # (n_pairs, G)
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
            num_t[cnt, g]   += contrib_num
            denom_t[cnt, g] += contrib_denom
            num_g[g]        += contrib_num
            denom_g[g]      += contrib_denom

        if verbose:
            with np.errstate(invalid="ignore", divide="ignore"):
                eps_cnt = num_t[cnt, ns:] / np.where(denom_t[cnt, ns:] > 0,
                                                      denom_t[cnt, ns:], np.nan)
            logger.info(
                "[eps_temporal cnt=%d: t=%.3g→%.3g]  cells=%d  "
                "mean_ε(genes)=%.3g",
                cnt, t0, t1, int(valid_mask.sum()),
                float(np.nanmean(eps_cnt)),
            )

    # ── Solve regularised MoM for ε = d1/d0 ─────────────────────────────────
    # ε_i = (Σ res²_ci  +  λ · ε_prior_i) / (Σ h_ic  +  λ)
    eff_denom_t = denom_t + lam
    eff_num_t   = num_t   + lam * r_prior[None, :]
    has_t       = eff_denom_t > 0
    eps_temporal_raw = np.where(has_t,
                                eff_num_t / np.where(has_t, eff_denom_t, 1.0),
                                r_prior[None, :])
    eps_temporal = np.clip(eps_temporal_raw, eps_min, eps_max).astype(np.float32)
    eps_temporal[:, :ns] = 1.0

    eff_denom_g = denom_g + lam
    eff_num_g   = num_g   + lam * r_prior
    has_g       = eff_denom_g > 0
    eps_global_raw = np.where(has_g,
                              eff_num_g / np.where(has_g, eff_denom_g, 1.0),
                              r_prior)
    eps_global = np.clip(eps_global_raw, eps_min, eps_max).astype(np.float32)
    eps_global[:ns] = 0.2

    return eps_temporal, eps_global


# ---------------------------
# inference_degradation_prot
# ---------------------------

def inference_degradation_prot(
    X_prot, times, bias, theta_inter, ks, d=None,
    n_epochs=500, lr=1e-2, method="dopri5",
    rtol=1e-6, atol=1e-8, print_every=50,
    batch_size=None, verbose=True,
    n_stimuli=1, stim_schedule=None,
    scale_proteins=1.0,
    samples_data=None,
    kon_mlp=None,
    lambda_scale=1e3,
    lambda_deg=0.0,
    lambda_mlp=0.0,
    g_obs_train=None,
) -> tuple[np.ndarray, np.ndarray]:
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
        all_unique_times = np.sort(np.unique(times))
        if stim_schedule and len(all_unique_times) >= 2:
            t1_first = float(all_unique_times[1])
            stim_first = np.asarray(stim_schedule[t1_first], dtype=np.float32) * scale_proteins
        else:
            stim_first = np.ones(ns, dtype=np.float32) * scale_proteins
        ode_funcs = [
            GeneRegulatoryODE_softmax(
                G, d_init, ks, theta_inter, bias[s_idx],
                n_stimuli=ns, stim_vals=stim_first, device=device,
                kon_mlp=kon_mlp, lambda_scale=lambda_scale,
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
            unique_t = np.sort(np.unique(t_s))
            for ti in range(len(unique_t) - 1):
                t0, t1 = float(unique_t[ti]), float(unique_t[ti + 1])
                X0_np = X_s[t_s == unique_t[ti]]
                X1_np = X_s[t_s == unique_t[ti + 1]]
                stim0 = stim_schedule[t0] if stim_schedule else np.ones(ns, dtype=np.float32)
                stim1 = stim_schedule[t1] if stim_schedule else np.ones(ns, dtype=np.float32)
                stim0, stim1 = stim0*scale_proteins, stim1*scale_proteins
                n_p = min(len(X0_np), len(X1_np))
                if n_p > 0:
                    all_pairs.append((s_idx, t0, t1, X0_np[:n_p], X1_np[:n_p], stim0, stim1))

        # ── Pre-compute observed training ratios per (sample, timepoint) ────────
        # g_obs_train (N, G_genes) = kon_beta_harissa[:, ns:] / kon_beta[:, ns:]
        # already computed by the caller — same row-indexing as X_prot / times.
        lam_mlp = float(lambda_mlp)
        ratio_obs_persample = {}  # key: (s_idx, t_val_float) → (G_genes,) ndarray
        if kon_mlp is not None and lam_mlp > 0 and g_obs_train is not None:
            g_obs_arr = np.asarray(g_obs_train, dtype=np.float32)
            for s_idx_pre, s_pre in enumerate(unique_s):
                mask_sp = (samples_data_arr == s_pre)
                t_sp    = times[mask_sp]
                g_sp    = g_obs_arr[mask_sp]
                for t_val in np.unique(t_sp):
                    mask_t = (t_sp == t_val)
                    if mask_t.sum() == 0:
                        continue
                    ratio_obs_persample[(s_idx_pre, float(t_val))] = np.mean(g_sp[mask_t], axis=0)

        old_loss  = 1e16

        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            total_loss, total_count = 0.0, 0

            for (s_idx, t0, t1, X0_full, X1_full, stim0, stim1) in all_pairs:
                ode = ode_funcs[s_idx]
                ode.stim_vals.copy_(torch.tensor(stim1, dtype=torch.float32))
                # Apply lambda_mlp mix in forward() for this interval
                if kon_mlp is not None and lam_mlp > 0:
                    _ones = np.ones(G - ns, dtype=np.float32)
                    r0 = ratio_obs_persample.get((s_idx, t0), _ones)
                    r1 = ratio_obs_persample.get((s_idx, t1), _ones)
                    ode.set_ratio_interpolation(t0, t1, r0, r1, lam_mlp)
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
            if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == n_epochs):
                scale_now = torch.nn.functional.softplus(ode_funcs[0].scale_param[ns:]).detach().cpu().numpy()
                logger.info(f"[Epoch {epoch}/{n_epochs}] loss = {loss:.6e}  max_scale[ns:] = {scale_now.max():.3e}")
                if abs(loss - old_loss) < 1e-4:
                    break
                old_loss = loss

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
            pairs.append((float(t0), float(t1), X0_np[:n_pairs], X1_np[:n_pairs], stim0, stim1))

    stim_first = pairs[0][-1] if pairs else np.ones(ns, dtype=np.float32) * scale_proteins
    ode_func: GeneRegulatoryODE_softmax = GeneRegulatoryODE_softmax(
        G, d_init, ks, theta_inter, bias,
        n_stimuli=ns, stim_vals=stim_first, device=device,
        kon_mlp=kon_mlp, lambda_scale=lambda_scale,
    ).to(device)

    optimizer = torch.optim.Adam([ode_func.d_param, ode_func.scale_param], lr=lr)
    mse = nn.MSELoss(reduction="mean")

    # ── Pre-compute observed training ratios at each training timepoint ──────
    # g_obs_train (N, G_genes) = kon_beta_harissa[:, ns:] / kon_beta[:, ns:]
    # already computed by the caller — same row-indexing as X_prot / times.
    lam_mlp = float(lambda_mlp)
    ratio_obs = {}  # t_val_float → (G_genes,) ndarray
    if kon_mlp is not None and lam_mlp > 0 and g_obs_train is not None:
        g_obs_arr = np.asarray(g_obs_train, dtype=np.float32)
        for t_val in unique_times:
            mask_t = (times == t_val)
            if mask_t.sum() == 0:
                continue
            ratio_obs[float(t_val)] = np.mean(g_obs_arr[mask_t], axis=0)

    old_loss = 1e16

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        total_loss, total_count = 0.0, 0

        for (t0, t1, X0_full, X1_full, stim0, stim1) in pairs:
            ode_func.stim_vals.copy_(torch.tensor(stim1, dtype=torch.float32))
            # Apply lambda_mlp mix in forward() for this interval
            if kon_mlp is not None and lam_mlp > 0:
                _ones = np.ones(G - ns, dtype=np.float32)
                ode_func.set_ratio_interpolation(
                    t0, t1,
                    ratio_obs.get(t0, _ones),
                    ratio_obs.get(t1, _ones),
                    lam_mlp,
                )
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

        if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == n_epochs):
            scale_now = torch.nn.functional.softplus(ode_func.scale_param[ns:]).detach().cpu().numpy()
            logger.info(f"[Epoch {epoch}/{n_epochs}] loss = {loss:.6e}  max_scale[ns:] = {scale_now.max():.3e}")
            if abs(loss - old_loss) < 1e-4:
                break
            old_loss: float | Any = loss

    d_learned = torch.nn.functional.softplus(ode_func.d_param).detach().cpu().numpy()
    scale_learned = torch.nn.functional.softplus(ode_func.scale_param).detach().cpu().numpy()

    return d_learned, scale_learned


# ---------------------------
# infer_ratio_d0_d1_full
# ---------------------------

def infer_ratio_d0_d1_full(
    X_prot,
    times,
    bias,
    theta_inter,
    ks,
    d_learned,
    k1_vec,
    kon_mlp,
    prior_d1d0=None,
    n_stimuli=1,
    stim_schedule=None,
    samples_data=None,
    method="dopri5",
    rtol=1e-5,
    atol=1e-7,
    n_steps=10,
    lambda_deg=0.0,
    lambda_mlp=0.5,
    g_obs_train=None,
    min_h=1e-4,
    verbose=True,
    clip_lo=0.01,
    clip_hi=100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer d1/d0 from full simulated trajectories via regularised LS.

    **Theoretical background.**

    To first order in ``d1/d0``, the MLP correction factor satisfies:

    .. math::

        1 - g_i(P(t)) \\approx \\frac{d_{1,i}}{d_{0,i}} \\cdot h_i(P(t))

    where ``g_i = kon_harissa_i / kon_beta_i`` (learned by *kon_mlp*) and

    .. math::

        h_i(P) = \\sum_j \\frac{\\partial\\ln kon_i}{\\partial X_j}
                 \\cdot (kon^{\\rm eff}_j - X_j)

    where the ``d_{1,j}`` factor is *not* included in ``h_i`` (uniform-d1
    approximation: ``d_{1,j} \\approx d_{1,i}`` for all j, so it becomes
    part of the ratio ``d_{1,i}/d_{0,i}`` that the LS estimates directly).

    **Fitting strategy.**

    Instead of evaluating only at the terminal state, this function simulates
    full trajectories with ``n_steps`` intermediate evaluation points and fits
    ``r_i = d1_i/d0_i`` by minimising over all *(cell, timestep)* pairs:

    .. math::

        \\mathcal{L}(r_i) = \\sum_{c,t} \\bigl[y_{ict} - r_i\\,h_{ict}\\bigr]^2
                           + \\lambda\\,(r_i - r_{\\rm prior,i})^2

    with ``y_{ict} = 1 - g_i(P_c(t))``.  The closed-form solution is:

    .. math::

        r_i^* = \\frac{\\sum h\\,y + \\lambda\\,r_{\\rm prior}}{\\sum h^2 + \\lambda}

    Only *(cell, timestep)* pairs where ``h_i`` and ``y_i`` have the same
    sign (i.e. where the formula predicts ``r_i > 0``) and where
    ``|h_i| > min_h`` are included.

    **Usage in base.py**::

        ratios_temporal, ratios_global = infer_ratio_d0_d1_full(...)
        # temporal (one ratio per interval):
        self.ratios[cnt] = 1.0 / ratios_temporal[cnt]
        # global (one shared ratio):
        self.ratios[:] = (1.0 / ratios_global)[None, :]

    Args:
        X_prot       : ``(N, G)`` protein observations across all timepoints.
        times        : ``(N,)`` time label for each observation.
        bias         : GRN bias — ``(G, n_modes-1)`` or ``(n_samples, G, n_modes-1)``.
        theta_inter  : GRN interactions — ``(G, G, n_modes-1)`` or
                       ``(n_samples, G, G, n_modes-1)``.
        ks           : Mode amplitudes, shape ``(n_modes, G)``.
        d_learned    : Protein degradation rates, shape ``(G,)`` (constant) or
                       ``(T-1, G)`` (temporal). When 2-D, interval ``cnt``
                       uses ``d_learned[cnt]`` for the ODE simulation.
        k1_vec       : Kept for interface compatibility (not used internally).
        kon_mlp      : Trained :class:`KonCorrectionMLP`.
        prior_d1d0   : ``(G,)`` prior for d1/d0 used when ``lambda_deg > 0``.
                       Defaults to all-ones (quasi-stationary limit).
        n_stimuli    : Number of stimulus columns (default 1).
        stim_schedule: Dict ``{float_time: np.ndarray(ns,)}``; defaults to
                       all-ones when ``None``.
        samples_data : ``(N,)`` per-cell sample index.
        method       : ODE solver method (default ``"dopri5"``).
        rtol, atol   : ODE solver tolerances.
        n_steps      : Number of intermediate ODE evaluation points per
                       interval (default 10; total = n_steps+1 incl. t0, t1).
        lambda_deg   : Tikhonov regularisation weight — same role as in
                       :func:`inference_degradation_prot`.  ``0`` = pure LS;
                       large values → prior.
        lambda_mlp   : Mix weight between the linearly interpolated observed
                       training ratios and the MLP prediction.  For a simulated
                       state at time ``t`` in ``[t1, t2]`` the effective ratio
                       used as ``y_i = 1 - g_eff`` is:

                       .. math::

                           g_{\\rm eff} = \\lambda_{\\rm mlp}
                               \\Bigl[
                                   \\frac{t_2-t}{t_2-t_1}\\,r_1
                                 + \\frac{t-t_1}{t_2-t_1}\\,r_2
                               \\Bigr]
                             + (1-\\lambda_{\\rm mlp})\\,g(P,\\,k_{\\rm on}(P))

                       where ``r_1`` / ``r_2`` are the mean observed MLP ratios
                       at the training timepoints ``t_1`` / ``t_2``
                       (shape ``(G_genes,)``).
                       ``1.0`` = pure linear interpolation of observed g;
                       ``0.0`` = pure MLP prediction (original behaviour).
        min_h        : Minimum ``|h_i|`` to include a *(cell, timestep)* pair.
        verbose      : Log per-interval diagnostics.
        clip_lo, clip_hi : Final clamp on the estimated d1/d0.

    Returns:
        ratios_temporal : ``(T-1, G)`` per-interval d1/d0 estimates.
        ratios_global   : ``(G,)``    global d1/d0 pooled over all intervals.

        Stimulus columns (``[:ns]``) are fixed to 1.0 in both outputs.
    """
    device = "cpu"
    X_prot   = np.asarray(X_prot,       dtype=np.float32)
    times    = np.asarray(times,        dtype=np.float32)
    bias_np  = np.asarray(bias,         dtype=np.float32)
    theta_np = np.asarray(theta_inter,  dtype=np.float32)
    d_arr    = np.asarray(d_learned,    dtype=np.float32)   # (G,) or (T-1, G)
    _ = k1_vec   # interface compat; X is already in normalised units

    ns      = int(n_stimuli)
    G       = X_prot.shape[1]
    G_genes = G - ns

    unique_times = np.sort(np.unique(times))
    T = len(unique_times)

    ks_t    = torch.tensor(np.asarray(ks, dtype=np.float32))
    n_modes = int(ks_t.shape[0])

    # Prior for d1/d0 (quasi-stationary limit = 1 by default)
    r_prior = (
        np.ones(G, dtype=np.float64)
        if prior_d1d0 is None
        else np.asarray(prior_d1d0, dtype=np.float64)
    )

    # Accumulators for the regularised LS  — float64 for numerical stability
    # Per-interval
    num_t   = np.zeros((T - 1, G), dtype=np.float64)
    denom_t = np.zeros((T - 1, G), dtype=np.float64)
    # Global (pooled over all intervals)
    num_g   = np.zeros(G, dtype=np.float64)
    denom_g = np.zeros(G, dtype=np.float64)

    kon_mlp.eval()

    # ── Pre-compute mean observed training ratios at each timepoint ───────────
    # g_obs_train (N, G_genes) = kon_beta_harissa[:, ns:] / kon_beta[:, ns:]
    # already computed by the caller — same row-indexing as X_prot / times.
    lam_mlp = float(lambda_mlp)
    if lam_mlp > 0.0 and g_obs_train is not None:
        g_obs_arr = np.asarray(g_obs_train, dtype=np.float32)
        training_ratios_obs = np.zeros((T, G_genes), dtype=np.float32)
        for tidx, t_val in enumerate(unique_times):
            mask_t = times == t_val
            if mask_t.sum() == 0:
                continue
            training_ratios_obs[tidx] = np.mean(g_obs_arr[mask_t], axis=0)
    else:
        training_ratios_obs = None   # not used when lam_mlp == 0 or no data

    for cnt in range(T - 1):
        t0 = float(unique_times[cnt])
        t1 = float(unique_times[cnt + 1])

        mask0 = times == unique_times[cnt]
        X0_np = X_prot[mask0]
        if len(X0_np) == 0:
            continue

        stim0 = np.asarray(
            stim_schedule[t0] if stim_schedule else np.ones(ns), dtype=np.float32
        )
        stim1 = np.asarray(
            stim_schedule[t1] if stim_schedule else np.ones(ns), dtype=np.float32
        )

        # ── Select bias / theta ──────────────────────────────────────────────
        # Supported layouts (mirroring infer_ratio_d0_d1_unitary):
        #   bias : (G, n_nets)                    — constant
        #          (T-1, G, n_nets)               — time-indexed
        #          (T-1, n_samples, G, n_nets)    — time + sample indexed
        #   theta: (G, G, n_nets)                 — constant
        #          (T-1, G, G, n_nets)            — time-indexed
        #          (T-1, n_samples, G, G, n_nets) — time + sample indexed
        if bias_np.ndim == 3:
            bias_cnt = bias_np[cnt]                                 # (G, n_nets)
        elif bias_np.ndim == 4:
            s_idx    = int(np.asarray(samples_data)[mask0][0]) if samples_data is not None else 0
            bias_cnt = bias_np[cnt, s_idx]                         # (G, n_nets)
        else:
            bias_cnt = bias_np                                      # (G, n_nets)

        if theta_np.ndim == 4:
            theta_cnt = theta_np[cnt]                              # (G, G, n_nets)
        elif theta_np.ndim == 5:
            s_idx     = int(np.asarray(samples_data)[mask0][0]) if samples_data is not None else 0
            theta_cnt = theta_np[cnt, s_idx]                      # (G, G, n_nets)
        else:
            theta_cnt = theta_np                                   # (G, G, n_nets)

        bias_t      = torch.tensor(bias_cnt.astype(np.float32))
        theta_cnt_t = torch.tensor(theta_cnt.astype(np.float32))

        # ── 1. Simulate full trajectory (n_steps+1 evaluation points) ────────
        d_cnt = d_arr[cnt] if d_arr.ndim == 2 else d_arr
        d1_t  = torch.tensor(d_cnt, dtype=torch.float32)   # (G,) d1_j for all genes
        ode = GeneRegulatoryODE_softmax(
            G, d_cnt, ks, theta_cnt, bias_cnt,
            n_stimuli=ns, stim_vals=stim1,
            device=device, kon_mlp=kon_mlp,
        ).to(device)
        ode.eval()

        X0_t = torch.tensor(X0_np, dtype=torch.float32)
        X0_t[:, :ns] = torch.tensor(stim0)
        N_cells = len(X0_np)

        # linspace includes t0 and t1 → n_steps+1 points
        t_eval = torch.linspace(t0, t1, n_steps + 1, dtype=torch.float32)

        with torch.no_grad():
            traj = odeint(ode, X0_t, t_eval, method=method, rtol=rtol, atol=atol)
        # traj: (n_steps+1, N_cells, G) — fix stimuli for all steps
        traj[:, :, :ns] = torch.tensor(stim1)

        # ── 2. Batch all timesteps for efficient autograd ────────────────────
        # (n_steps+1, N_cells, G)  →  (n_ts * N_cells, G)
        n_ts = n_steps + 1
        X_batch_np = traj.reshape(n_ts * N_cells, G).cpu().numpy()

        # ── 3. Evaluate g_i = MLP ratio for every state (no grad) ───────────
        with torch.no_grad():
            X_tmp = torch.tensor(X_batch_np, dtype=torch.float32)
            Z_tmp = torch.zeros(n_ts * N_cells, G, n_modes)
            for k in range(n_modes - 1):
                Z_tmp[:, :, k + 1] = X_tmp @ theta_cnt_t[:, :, k] + bias_t[:, k]
            kb_tmp = (torch.softmax(Z_tmp, dim=-1) * ks_t.T.unsqueeze(0)).sum(dim=-1)

        g_batch_np = kon_mlp(X_batch_np, kb_tmp[:, ns:].numpy())   # (n_ts*N, G_genes)

        # ── lambda_mlp mix: blend MLP prediction with linear interpolation ───
        # of the mean observed ratios at the bracketing training timepoints.
        # For time t in [t0, t1]:
        #   alpha       = (t - t0) / (t1 - t0)
        #   g_interp    = (1-alpha)*ratio_obs[cnt] + alpha*ratio_obs[cnt+1]
        #   g_eff       = lam_mlp * g_interp + (1-lam_mlp) * g_mlp
        # The batch is ordered as [t_eval[0]*N_cells, t_eval[1]*N_cells, ...].
        if lam_mlp > 0.0:
            alphas = (t_eval.numpy() - t0) / (t1 - t0)          #   (n_ts,)
            r1 = training_ratios_obs[cnt]                         # (G_genes,)
            r2 = training_ratios_obs[cnt + 1]                     # (G_genes,)
            # Interpolated ratio at each timestep: (n_ts, G_genes)
            g_interp = (1.0 - alphas[:, None]) * r1 + alphas[:, None] * r2
            # Expand to match batch: each of the n_ts steps has N_cells rows
            g_interp_batch = np.repeat(g_interp, N_cells, axis=0)  # (n_ts*N, G_genes)
            a_alpha = 4 * (1 - lam_mlp)
            lamb_alpha = 1 + a_alpha * alphas * (alphas - 1)
            lamb_alpha_batch = np.repeat(lamb_alpha, N_cells)[:, None]  # (n_ts*N, 1)
            g_batch_np = lamb_alpha_batch * g_interp_batch + (1.0 - lamb_alpha_batch) * g_batch_np

        g_batch    = torch.tensor(g_batch_np, dtype=torch.float32)

        # ── 4. Compute h_i via autograd over the full batch ──────────────────
        X_batch_t = torch.tensor(X_batch_np, dtype=torch.float32, requires_grad=True)

        Z = torch.zeros(n_ts * N_cells, G, n_modes, dtype=torch.float32)
        for k in range(n_modes - 1):
            Z[:, :, k + 1] = X_batch_t @ theta_cnt_t[:, :, k] + bias_t[:, k]
        kon_beta = (torch.softmax(Z, dim=-1) * ks_t.T.unsqueeze(0)).sum(dim=-1)

        # Effective kon — g is detached so autograd only sees ∂kon_beta/∂X
        kon_eff = torch.cat([
            kon_beta[:, :ns].detach(),
            kon_beta[:, ns:].detach() * g_batch,
        ], dim=-1)

        # Driving force  drive_j = kon_eff_j − X_j  (fully detached)
        drive = (kon_eff - X_batch_t.detach()).detach()
        drive[:, :ns] = 0.0   # stimuli do not evolve

        # ── 5. Accumulate LS numerators and denominators ─────────────────────
        for gene_idx in range(G_genes):
            i      = ns + gene_idx
            retain = gene_idx < G_genes - 1

            jac_i = torch.autograd.grad(
                kon_beta[:, i].sum(), X_batch_t,
                retain_graph=retain, create_graph=False,
            )[0].detach()   # (n_ts*N, G)

            ln_jac_i = jac_i / (kon_beta[:, i:i+1].detach() + 1e-10)   # (n_ts*N, G)

            # h_i = Σ_j (∂ ln kon_i / ∂X_j) · (d1_j/d1_i) · drive_j
            #
            # Exact derivation (first-order mRNA lag):
            #   1 - g_i  =  (1/d0_i) · d(ln kon_i)/dt
            #             =  (1/d0_i) · Σ_j ∂(ln kon_i)/∂X_j · d1_j · drive_j
            #             =  (d1_i/d0_i) · Σ_j ∂(ln kon_i)/∂X_j · (d1_j/d1_i) · drive_j
            #                              └────────────────── h_i ──────────────────────┘
            # No uniform-d1 approximation needed: we weight each term by d1_j/d1_i.
            d1_i_val  = d1_t[i].clamp(min=1e-10)
            d1_weight = (d1_t / d1_i_val).unsqueeze(0)          # (1, G)
            h_i = (ln_jac_i * drive * d1_weight).sum(dim=-1)   # (n_ts*N,)
            y_i = 1.0 - g_batch[:, gene_idx]               # (n_ts*N,)  y = 1 − g

            # Keep only points that are not near-equilibrium AND where the
            # sign is consistent with a positive d1/d0.
            valid = (h_i.abs() > min_h) & (h_i * y_i > 0)

            if valid.sum() == 0:
                continue

            h_v = h_i[valid].detach().numpy().astype(np.float64)
            y_v = y_i[valid].detach().numpy().astype(np.float64)

            contrib_num   = float(np.dot(h_v, y_v))
            contrib_denom = float(np.dot(h_v, h_v))

            num_t[cnt, i]   += contrib_num
            denom_t[cnt, i] += contrib_denom
            num_g[i]        += contrib_num
            denom_g[i]      += contrib_denom

        if verbose:
            n_genes_with_data = int((denom_t[cnt, ns:] > 0).sum())
            logger.info(
                "[cnt=%d: t=%.3g→%.3g]  genes with LS data: %d / %d",
                cnt, t0, t1, n_genes_with_data, G_genes,
            )

    # ── 6. Solve regularised LS ───────────────────────────────────────────────
    # r_i* = (Σ h*y  +  λ * r_prior_i) / (Σ h²  +  λ)
    # When no data available for a gene/interval → fall back to prior.

    lam = float(lambda_deg)

    # Per-interval
    eff_denom_t = denom_t + lam                            # (T-1, G)
    eff_num_t   = num_t   + lam * r_prior[None, :]        # (T-1, G)
    has_data_t  = eff_denom_t > 0
    r_t = np.where(has_data_t, eff_num_t / np.where(has_data_t, eff_denom_t, 1.0),
                   r_prior[None, :])
    ratios_temporal = np.clip(r_t, clip_lo, clip_hi).astype(np.float32)
    ratios_temporal[:, :ns] = 1.0   # stimuli: neutral

    # Global
    eff_denom_g = denom_g + lam                            # (G,)
    eff_num_g   = num_g   + lam * r_prior                  # (G,)
    has_data_g  = eff_denom_g > 0
    r_g = np.where(has_data_g, eff_num_g / np.where(has_data_g, eff_denom_g, 1.0),
                   r_prior)
    ratios_global = np.clip(r_g, clip_lo, clip_hi).astype(np.float32)
    ratios_global[:ns] = 0.2   # stimuli: neutral

    return ratios_temporal, ratios_global


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
