"""
Proliferation rate inference utilities for CardamomOT.

Provides a lightweight MLP that maps protein levels to a net proliferation
rate R = b - d (birth minus death). It is fitted so that the integral of R
along each inferred trajectory, between two timepoints, matches the log mass
gain of that trajectory state given by the growth OT pass (R_opt), in the
spirit of the unbalanced probability flow matching objective (Maddu et al. 2026).
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import PchipInterpolator


class ProliferationMLP(nn.Module):
    """
    Two-hidden-layer MLP: [stimuli, protein levels] → net proliferation rate R (inputs
    standardised). The first n_stim inputs are the inference stimuli (0 = proteins only).
    With two_heads, a second small network regresses the prior birth and death rates (b, δ) of the states on
    the same inputs: the birth dilutes the proteins in the simulations (predict_birth), as the prior birth of
    the real cells in the inferred trajectories. R alone drives the branching: the net growth of the OT
    marginals also absorbs composition changes (e.g. a rising quiescent fraction), not a division rate.
    """

    def __init__(self, n_inputs: int, hidden_size: int = 64, n_stim: int = 0, two_heads: bool = False) -> None:
        super().__init__()
        self.register_buffer('x_mean', torch.zeros(n_inputs))
        self.register_buffer('x_std', torch.ones(n_inputs))
        self.register_buffer('n_stim', torch.tensor(int(n_stim)))  # loaded from the state dict
        # Output scale: R = r_mean + r_scale * net (targets are ~1e-3 per hour; identity for older nets)
        self.register_buffer('r_mean', torch.tensor(0.0))
        self.register_buffer('r_scale', torch.tensor(1.0))
        self.register_buffer('two_heads', torch.tensor(int(two_heads)))
        self.register_buffer('split_scale', torch.tensor(1.0))  # prior rates = split_scale * softplus(z)
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        if two_heads:
            self.split_net = nn.Sequential(
                nn.Linear(n_inputs, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 2),
            )

    def prior_rates(self, x: torch.Tensor):
        """(b0, δ0): prior birth and death rates regressed on the inputs (two_heads only)."""
        z = self.split_net((x - self.x_mean) / self.x_std)
        sp = self.split_scale * torch.nn.functional.softplus(z)
        return sp[..., 0], sp[..., 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.r_mean + self.r_scale * self.net((x - self.x_mean) / self.x_std).squeeze(-1)

    def predict(self, prot: np.ndarray, stim=None) -> np.ndarray:
        """prot: (..., n_proteins), stim: (..., n_stim) or broadcastable (ignored if n_stim = 0) → R (...)."""
        prot = np.asarray(prot, dtype=np.float32)
        k = int(self.n_stim)
        if k > 0:
            stim = np.zeros(k) if stim is None else np.asarray(stim, dtype=np.float32)[..., :k]
            prot = np.concatenate([np.broadcast_to(stim, prot.shape[:-1] + (k,)), prot], axis=-1)
        self.eval()
        with torch.no_grad():
            return self.forward(torch.as_tensor(prot, dtype=torch.float32)).numpy()

    def _inputs(self, prot, stim=None):
        prot = np.asarray(prot, dtype=np.float32)
        k = int(self.n_stim)
        if k > 0:
            stim = np.zeros(k) if stim is None else np.asarray(stim, dtype=np.float32)[..., :k]
            prot = np.concatenate([np.broadcast_to(stim, prot.shape[:-1] + (k,)), prot], axis=-1)
        return torch.as_tensor(prot, dtype=torch.float32)

    def predict_rates(self, prot: np.ndarray, stim=None):
        """(birth, death) of the states: regressed prior rates (two heads), else (max(R, 0), max(−R, 0))."""
        self.eval()
        with torch.no_grad():
            x = self._inputs(prot, stim)
            if int(self.two_heads):
                return tuple(v.numpy() for v in self.prior_rates(x))
            R = self.forward(x).numpy()
        return np.maximum(R, 0.0), np.maximum(-R, 0.0)

    def predict_birth(self, prot: np.ndarray, stim=None) -> np.ndarray:
        """Birth rate b of the states (dilution)."""
        return self.predict_rates(prot, stim)[0]

    def predict_death(self, prot: np.ndarray, stim=None) -> np.ndarray:
        """Death rate δ of the states."""
        return self.predict_rates(prot, stim)[1]


def load_proliferation_mlp(path, n_inputs):
    """Saved ProliferationMLP (one or two heads, from the state dict); strict=False keeps the networks saved
    before input standardisation."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = ProliferationMLP(int(n_inputs), two_heads=any(k.startswith('split_net.') for k in state))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def interval_stimulus(prot, times_data, ns=1):
    """
    Stimulus applied over the interval that starts at each state: that of the next timepoint
    of the same trajectory slot (as in the simulation); own value at the last time.
    """
    times = np.sort(np.unique(times_data))
    T = len(times)
    N = len(times_data) // T
    S = np.asarray(prot)[:T * N, :ns].reshape(T, N, ns)
    return np.concatenate([S[1:], S[-1:]], axis=0).reshape(T * N, ns)


def quadrature(n_nodes: int):
    """Relative nodes in [0, 1] and trapezoidal weights (summing to 1) on one interval."""
    u = np.linspace(0.0, 1.0, max(2, n_nodes))
    w = np.full(len(u), 1.0 / (len(u) - 1))
    w[[0, -1]] /= 2
    return u, w


def growth_path_states(prot, times_data, ns=1, n_nodes=5, with_stim=False):
    """
    Protein states along each trajectory at the quadrature nodes of every interval.

    Trajectories are the slots of prot (time blocks of N rows, slot n = trajectory n),
    interpolated across all timepoints by a shape-preserving PCHIP spline.
    Returns states (T-1, Q, N, G) and the interval lengths (T-1,); with_stim prepends the
    ns stimulus values of each interval (constant along it, those of its end timepoint).
    """
    times = np.sort(np.unique(times_data))
    T = len(times)
    N = len(times_data) // T
    P = prot[:, ns:].reshape(T, N, -1)
    u, _ = quadrature(n_nodes)
    dt = np.diff(times)
    nodes = times[:-1, None] + dt[:, None] * u[None, :]
    states = np.maximum(PchipInterpolator(times, P, axis=0)(nodes.ravel()), 0).reshape(T - 1, len(u), N, -1)
    if with_stim and ns > 0:
        S = prot[:T * N, :ns].reshape(T, N, ns)[1:]                        # (T-1, N, ns)
        states = np.concatenate([np.broadcast_to(S[:, None], (T - 1, len(u), N, ns)), states], axis=-1)
    return states, dt


def train_proliferation_mlp(
    prot: np.ndarray,
    R_opt: np.ndarray,
    times_data: np.ndarray,
    ns: int = 1,
    n_nodes: int = 5,
    hidden_size: int = 64,
    n_epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    lambda_mass: float = 10.0,
    n_mass: int = 128,
    with_stim: bool = True,
    val_fraction: float = 0.2,
    patience: int = 50,
    seed: int = 0,
    verb: bool = True,
    birth_prior: np.ndarray = None,
    death_prior: np.ndarray = None,
) -> ProliferationMLP:
    """
    Fit R(u, P) so that, for each trajectory n and interval k,
        ∫_{t_k}^{t_k+1} R(u_k, P_n(s)) ds  ≈  R_opt[k, n] · Δt_k   (log mass gain),
    the integral being a trapezoidal sum over n_nodes states of the PCHIP path, u_k the
    inference stimuli over the interval (if with_stim). The residuals are divided by Δt_k so
    that all intervals weigh as rates. A total-mass term (weight lambda_mass, as in
    unbalanced PFM) makes the population growth of each interval, log mean_n exp(∫R), match
    the one of the targets: the pathwise least squares alone fit the mean log gain and
    underestimate it (Jensen).

    With birth_prior and death_prior ((T * N,) prior birth and death rates of the states), R is fitted as above
    (the data only see the net growth), then a small network regresses (b, δ) on the states (held-out
    trajectories for early stopping), giving the birth (dilution) of the simulated cells (predict_birth).

    A fraction val_fraction of the trajectories is held out: training stops when their loss
    has not improved for `patience` epochs and the best weights are kept. The fit quality is
    stored in model.diagnostics (R² of the log mass gains, population growth per interval).

    Parameters
    ----------
    prot : (T * N, G_tot) protein trajectories (time blocks, stimulus dims first).
    R_opt : (T * N,) net growth rate of each state over the next interval (NaN at last time).
    times_data : (T * N,) time of each state.
    ns : number of stimulus dimensions of prot.
    """
    n_stim = ns if with_stim else 0
    states, dt = growth_path_states(prot, times_data, ns=ns, n_nodes=n_nodes, with_stim=with_stim)
    Km1, Q, N, G = states.shape
    _, w = quadrature(n_nodes)
    w_t = torch.tensor(w, dtype=torch.float32)
    target = R_opt.reshape(Km1 + 1, N)[:-1] * dt[:, None]
    ok = np.isfinite(target)

    # One sample per (interval, trajectory): its Q path states, Δt and log mass gain
    X = np.transpose(states, (0, 2, 1, 3))[ok].astype(np.float32)          # (n_pairs, Q, G)
    D = np.broadcast_to(dt[:, None], (Km1, N))[ok].astype(np.float32)
    Y = target[ok].astype(np.float32)
    K = np.broadcast_to(np.arange(Km1)[:, None], (Km1, N))[ok]
    traj = np.broadcast_to(np.arange(N)[None, :], (Km1, N))[ok]
    two_heads = birth_prior is not None and death_prior is not None

    # Held-out trajectories (whole paths, so that validation pairs are not neighbours of training ones)
    rng = np.random.default_rng(seed)
    n_val = int(round(val_fraction * N)) if N >= 10 else 0
    val_traj = rng.choice(N, n_val, replace=False) if n_val else np.array([], dtype=int)
    is_val = np.isin(traj, val_traj)
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    X_all, D_all, Y_all = torch.tensor(X), torch.tensor(D), torch.tensor(Y)
    intervals = np.unique(K)

    def groups_of(idx):
        return [idx[K[idx] == i] for i in intervals if np.any(K[idx] == i)]

    def log_mean_exp(v):
        return float(np.log(np.mean(np.exp(v.astype(float)))))

    groups_tr, groups_va = groups_of(tr), groups_of(va)
    # Interval length of each group: the mass residual is divided by it, in rate units as the path term
    dt_tr = torch.tensor([float(D[g[0]]) for g in groups_tr], dtype=torch.float32)
    dt_va = torch.tensor([float(D[g[0]]) for g in groups_va], dtype=torch.float32)
    mass_tr = torch.tensor([log_mean_exp(Y[g]) for g in groups_tr], dtype=torch.float32)
    mass_va = torch.tensor([log_mean_exp(Y[g]) for g in groups_va], dtype=torch.float32)

    def integral(idx):
        return (model(X_all[idx]) * w_t).sum(dim=1) * D_all[idx]

    def mass_loss(groups, mass_target, dt_groups, sample=True):
        # Population growth of the model per interval, on up to n_mass random pairs (all if not sample)
        lme = []
        for g in groups:
            sel = g if (not sample or len(g) <= n_mass) else rng.choice(g, n_mass, replace=False)
            integ = integral(torch.as_tensor(sel))
            lme.append(torch.logsumexp(integ, 0) - np.log(len(sel)))
        return (((torch.stack(lme) - mass_target) / dt_groups) ** 2).mean()

    model = ProliferationMLP(G, hidden_size=hidden_size, n_stim=n_stim, two_heads=two_heads)
    flat = X[tr].reshape(-1, G)
    sd = flat.std(axis=0)
    model.x_mean.copy_(torch.tensor(flat.mean(axis=0)))
    model.x_std.copy_(torch.tensor(np.where(sd > 1e-6, sd, 1.0)))   # constant inputs (e.g. stimulus) left as is
    rates = Y[tr] / D[tr]
    model.r_mean.fill_(float(rates.mean()))
    model.r_scale.fill_(float(max(rates.std(), 1e-8)))
    with torch.no_grad():
        # Stimulus inputs start with no effect; constant ones get no gradient and keep none
        model.net[0].weight[:, :n_stim] = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.as_tensor(tr)),
                                         batch_size=batch_size, shuffle=True)

    def val_loss():
        if len(va) == 0:
            return np.nan
        with torch.no_grad():
            loss = (((integral(torch.as_tensor(va)) - Y_all[va]) / D_all[va]) ** 2).mean()
            if lambda_mass > 0 and groups_va:
                loss = loss + lambda_mass * mass_loss(groups_va, mass_va, dt_va, sample=False)
        return float(loss)

    best, best_state, best_epoch, wait = np.inf, None, 0, 0
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for (ib,) in loader:
            loss = (((integral(ib) - Y_all[ib]) / D_all[ib]) ** 2).mean()
            if lambda_mass > 0:
                loss = loss + lambda_mass * mass_loss(groups_tr, mass_tr, dt_tr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(ib)
        model.eval()
        vl = val_loss()
        if np.isfinite(vl):
            if vl < best - 1e-9:
                best, best_epoch, wait = vl, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
        if verb and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"[ProliferationMLP] epoch {epoch:4d}/{n_epochs}  "
                  f"loss={epoch_loss / max(len(tr), 1):.6f}  val={vl:.6f}")
        if wait >= patience:
            if verb:
                print(f"[ProliferationMLP] early stop at epoch {epoch} (best validation epoch {best_epoch})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Fit quality: R² of the log mass gains (held-out and training paths), growth per interval
    with torch.no_grad():
        pred = integral(torch.arange(len(Y))).numpy()

    def r2(idx):
        # On the mean rates over the intervals (log gain / Δt), as the training loss
        if len(idx) < 2:
            return float('nan')
        y, p = Y[idx] / D[idx], pred[idx] / D[idx]
        ss = np.sum((y - y.mean()) ** 2)
        return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float('nan')

    all_idx = np.arange(len(Y))
    model.diagnostics = dict(
        r2_train=r2(tr), r2_val=r2(va), n_train_paths=int(N - n_val), n_val_paths=int(n_val),
        best_epoch=int(best_epoch), with_stimulus=bool(n_stim),
        growth_target=[log_mean_exp(Y[g]) for g in groups_of(all_idx)],
        growth_mlp=[log_mean_exp(pred[g]) for g in groups_of(all_idx)],
        interval_start=[float(np.sort(np.unique(times_data))[i]) for i in intervals],
        two_heads=bool(two_heads),
    )
    if two_heads:
        _fit_split(model, prot, times_data, ns, with_stim, birth_prior, death_prior, n_val, seed, verb)
    if verb:
        print(f"[ProliferationMLP] R² of the path mean rates: train {model.diagnostics['r2_train']:.3f}, "
              f"held-out {model.diagnostics['r2_val']:.3f}")
    return model


def _fit_split(model, prot, times_data, ns, with_stim, birth_prior, death_prior, n_val, seed=0, verb=True,
               n_epochs=300, lr=1e-2, patience=30):
    """Regression of the prior birth and death rates of the states on their inputs (split_net of model), held-out
    trajectories for early stopping; diagnostics: mean regressed and prior birth / death."""
    T = len(np.unique(times_data))
    N = len(times_data) // T
    X = np.asarray(prot, dtype=np.float32)
    if with_stim and ns > 0:
        X = np.concatenate([interval_stimulus(prot, times_data, ns), X[:, ns:]], axis=1).astype(np.float32)
    else:
        X = X[:, ns:]
    B = np.nan_to_num(np.asarray(birth_prior, dtype=np.float32))
    D = np.nan_to_num(np.asarray(death_prior, dtype=np.float32))
    rng = np.random.default_rng(None if seed is None else seed + 1)
    val_traj = rng.choice(N, n_val, replace=False) if n_val else np.array([], dtype=int)
    is_val = np.isin(np.tile(np.arange(N), T), val_traj)
    Xt, Bt, Dt = torch.tensor(X), torch.tensor(B), torch.tensor(D)
    s = float(max(B.mean(), D.mean(), 1e-6))
    model.split_scale.fill_(s)
    opt = torch.optim.Adam(model.split_net.parameters(), lr=lr)

    def loss(idx):  # relative to the scale of the rates
        b, d = model.prior_rates(Xt[idx])
        return (((b - Bt[idx]) / s) ** 2 + ((d - Dt[idx]) / s) ** 2).mean()

    tr, va = torch.as_tensor(np.flatnonzero(~is_val)), torch.as_tensor(np.flatnonzero(is_val))
    best, best_state, wait = np.inf, None, 0
    for _ in range(n_epochs):
        opt.zero_grad()
        l = loss(tr)
        l.backward()
        opt.step()
        with torch.no_grad():
            vl = float(loss(va)) if len(va) else float(l)
        if vl < best - 1e-9:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.split_net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.split_net.load_state_dict(best_state)
    stim = interval_stimulus(prot, times_data, ns) if (with_stim and ns > 0) else None
    b, d = model.predict_rates(np.asarray(prot)[:, ns:], stim)
    model.diagnostics.update(two_heads=True, birth_mlp=float(np.mean(b)), death_mlp=float(np.mean(d)),
                             birth_prior=float(B.mean()), death_prior=float(D.mean()), split_val_loss=float(best))
    if verb:
        print(f"[ProliferationMLP] Birth / death regressed on the states: mean birth "
              f"{np.mean(b):.4f} (prior {B.mean():.4f}), mean death {np.mean(d):.4f} (prior {D.mean():.4f}) h^-1")
