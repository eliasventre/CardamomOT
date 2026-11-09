"""
Proliferation rate inference utilities for CardamomOT.

Provides a lightweight MLP that maps protein levels (and the inference stimuli) to a net proliferation
rate R. It is fitted so that the integral of R along each inferred trajectory, between two timepoints,
matches the log mass gain of that trajectory state (R_opt of the growth OT pass, or the prior net rates:
use_recomputed_proliferation), in the spirit of the unbalanced probability flow matching objective
(Maddu et al. 2026). A second head regresses the birth and death rates of the states consistent with that
net growth (birth of the dilution).
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import PchipInterpolator


class ProliferationMLP(nn.Module):
    """
    Two-hidden-layer MLP: [stimuli, protein levels] → birth b and death δ rates of a cell, (b, δ) = scale ·
    softplus(z) (inputs standardised; one scale per head, the mean prior rates of the states, so that the death, often
    ~10x below the birth, weighs as much). b dilutes the proteins, the net rate R = b − δ drives the branching. The
    first n_stim inputs are the inference stimuli (0 = proteins only).
    """

    def __init__(self, n_inputs: int, hidden_size: int = 64, n_stim: int = 0) -> None:
        super().__init__()
        self.register_buffer('x_mean', torch.zeros(n_inputs))
        self.register_buffer('x_std', torch.ones(n_inputs))
        self.register_buffer('n_stim', torch.tensor(int(n_stim)))  # loaded from the state dict
        self.register_buffer('scale', torch.ones(2))  # (birth, death) = scale * softplus(z)
        self.register_buffer('natural', torch.tensor(0))  # 1: natural rates (inference stimuli constant), effects added outside
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )
        with torch.no_grad():  # each rate at its scale at the start (softplus = 1)
            self.net[-1].bias.fill_(float(np.log(np.e - 1.0)))

    def rates(self, x: torch.Tensor):
        """(b, δ) of the inputs x (..., n_inputs): [stimuli, proteins]."""
        sp = self.scale * torch.nn.functional.softplus(self.net((x - self.x_mean) / self.x_std))
        return sp[..., 0], sp[..., 1]

    def rates_of(self, prot: torch.Tensor, stim=None):
        """(b, δ) of the protein states prot (..., n_proteins) under the stimuli stim (tensor broadcastable to
        (..., n_stim), or None: 0), differentiable in the parameters."""
        k = int(self.n_stim)
        if k > 0:
            st = torch.zeros(k, dtype=prot.dtype) if stim is None else stim[..., :k].to(prot.dtype)
            prot = torch.cat([st.expand(*prot.shape[:-1], k), prot], dim=-1)
        return self.rates(prot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d = self.rates(x)
        return b - d

    def _inputs(self, prot, stim=None):
        prot = np.asarray(prot, dtype=np.float32)
        k = int(self.n_stim)
        if k > 0:
            stim = np.zeros(k) if stim is None else np.asarray(stim, dtype=np.float32)[..., :k]
            prot = np.concatenate([np.broadcast_to(stim, prot.shape[:-1] + (k,)), prot], axis=-1)
        return torch.as_tensor(prot, dtype=torch.float32)

    def predict(self, prot: np.ndarray, stim=None, samples=None) -> np.ndarray:
        """prot: (..., n_proteins), stim: (..., n_stim) or broadcastable (ignored if n_stim = 0) → R = b − δ (...);
        samples (sample index of the rows) only used by ConditionProliferation."""
        b, d = self.predict_rates(prot, stim)
        return b - d

    def predict_rates(self, prot: np.ndarray, stim=None, samples=None):
        """(birth, death) of the states."""
        self.eval()
        with torch.no_grad():
            return tuple(v.numpy() for v in self.rates(self._inputs(prot, stim)))

    def predict_birth(self, prot: np.ndarray, stim=None, samples=None) -> np.ndarray:
        """Birth rate b of the states (dilution)."""
        return self.predict_rates(prot, stim)[0]

    def predict_death(self, prot: np.ndarray, stim=None, samples=None) -> np.ndarray:
        """Death rate δ of the states."""
        return self.predict_rates(prot, stim)[1]


class ConditionProliferation:
    """
    One ProliferationMLP per network condition (ProliferationMLP_per_condition): the rows (first axis of prot) are
    routed by the condition of their sample (samples, required), each MLP reading the proteins of its condition's
    genes (cols[c], indices among the proteins without stimuli). Same predict interface as ProliferationMLP.
    """

    def __init__(self, nets, cols, sample_cond, names=None):
        self.nets, self.cols = list(nets), [np.asarray(c, dtype=int) for c in cols]
        self.sample_cond, self.names = np.asarray(sample_cond, dtype=int), names

    @property
    def diagnostics(self):
        return {str(self.names[c] if self.names else c): getattr(n, 'diagnostics', None)
                for c, n in enumerate(self.nets) if n is not None}

    def net_of(self, sample):
        """(MLP or None, columns of its genes among the proteins) of the condition of a sample (index)."""
        c = int(self.sample_cond[min(int(sample), len(self.sample_cond) - 1)])
        return self.nets[c], self.cols[c]

    def _route(self, fn, prot, stim, samples, n_out=1):
        if samples is None:
            raise ValueError("ConditionProliferation: the sample of each row is needed (samples)")
        prot = np.asarray(prot, dtype=np.float32)
        cond = self.sample_cond[np.minimum(np.asarray(samples, dtype=int), len(self.sample_cond) - 1)]
        outs = [np.zeros(prot.shape[:-1]) for _ in range(n_out)]
        st = None if stim is None else np.asarray(stim, dtype=np.float32)
        if st is not None:
            st = np.broadcast_to(st, prot.shape[:-1] + st.shape[-1:])
        for c, net in enumerate(self.nets):
            rows = cond == c
            if net is None or not rows.any():
                continue
            res = fn(net, prot[rows][..., self.cols[c]], None if st is None else st[rows])
            for o, r in zip(outs, res if n_out > 1 else (res,)):
                o[rows] = r
        return outs[0] if n_out == 1 else tuple(outs)

    def predict(self, prot, stim=None, samples=None):
        return self._route(lambda n, p, s: n.predict(p, s), prot, stim, samples)

    def predict_rates(self, prot, stim=None, samples=None):
        return self._route(lambda n, p, s: n.predict_rates(p, s), prot, stim, samples, n_out=2)

    def predict_birth(self, prot, stim=None, samples=None):
        return self.predict_rates(prot, stim, samples)[0]

    def predict_death(self, prot, stim=None, samples=None):
        return self.predict_rates(prot, stim, samples)[1]

    def save(self, cardamom_dir):
        """prolif_network_<c>.pt per condition and prolif_network_conditions.json (genes, sample conditions)."""
        import json
        import os
        for c, net in enumerate(self.nets):
            if net is not None:
                torch.save(net.state_dict(), os.path.join(cardamom_dir, f'prolif_network_{c}.pt'))
        with open(os.path.join(cardamom_dir, 'prolif_network_conditions.json'), 'w') as fh:
            json.dump({'names': self.names, 'cols': [c.tolist() for c in self.cols],
                       'sample_cond': self.sample_cond.tolist(),
                       'n_inputs': [None if n is None else int(n.net[0].in_features) for n in self.nets]}, fh)


def natural_rows(prolif, samples):
    """(n,) bool: the proliferation model of the sample of each row (samples (n,) sample indices) learned the natural
    rates (inference stimuli constant in its condition: their effects are added by the refine SDE and the simulations
    through the nearest trajectory states)."""
    samples = np.asarray(samples, dtype=int)
    if not isinstance(prolif, ConditionProliferation):
        return np.full(len(samples), bool(int(getattr(prolif, 'natural', 0))))
    nat = np.array([n is not None and bool(int(getattr(n, 'natural', 0))) for n in prolif.nets])
    return nat[prolif.sample_cond[np.minimum(samples, len(prolif.sample_cond) - 1)]]


def load_proliferation(cardamom_dir):
    """Proliferation model saved by infer_network_simul: ConditionProliferation (prolif_network_conditions.json) or a
    single ProliferationMLP (prolif_network.pt); None if absent."""
    import json
    import os
    js = os.path.join(cardamom_dir, 'prolif_network_conditions.json')
    if os.path.exists(js):
        info = json.load(open(js))
        nets = [None if n is None else load_proliferation_mlp(os.path.join(cardamom_dir, f'prolif_network_{c}.pt'), n)
                for c, n in enumerate(info['n_inputs'])]
        return ConditionProliferation(nets, info['cols'], info['sample_cond'], info.get('names'))
    pt, npt = os.path.join(cardamom_dir, 'prolif_network.pt'), os.path.join(cardamom_dir, 'prolif_network_n_proteins.npy')
    if os.path.exists(pt) and os.path.exists(npt):
        return load_proliferation_mlp(pt, int(np.load(npt)[0]))
    return None


def load_proliferation_mlp(path, n_inputs):
    """Saved ProliferationMLP (state dict)."""
    model = ProliferationMLP(int(n_inputs))
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def interval_stimulus(prot, times_data, ns=1, valid=None):
    """
    Stimulus applied over the interval that starts at each state: that of the next timepoint
    of the same trajectory slot (as in the simulation); own value at the last time. valid (T * N,): that of the
    next valid state of the slot (next observed time of its sample), own value after its last one.
    """
    times = np.sort(np.unique(times_data))
    T = len(times)
    N = len(times_data) // T
    S = np.asarray(prot)[:T * N, :ns].reshape(T, N, ns)
    if valid is None:
        return np.concatenate([S[1:], S[-1:]], axis=0).reshape(T * N, ns)
    V = np.asarray(valid, dtype=bool)[:T * N].reshape(T, N)
    out, nxt, has = S.copy(), np.zeros((N, ns)), np.zeros(N, dtype=bool)
    for t in range(T - 1, -1, -1):
        out[t] = np.where(has[:, None], nxt, S[t])
        nxt, has = np.where(V[t][:, None], S[t], nxt), has | V[t]
    return out.reshape(T * N, ns)


def birth_heads(net):
    """
    Numpy parameters of the birth / death rates of a ProliferationMLP or ConditionProliferation (light enough to be
    sent to the simulation workers): (heads, sample_cond), one head per network condition (None if absent),
    sample_cond the condition of each sample (None: a single head). See head_rates.
    """
    def one(m, cols=None):
        if m is None:
            return None
        layers = [(l.weight.detach().double().numpy(), l.bias.detach().double().numpy())
                  for l in m.net if isinstance(l, nn.Linear)]
        return dict(x_mean=m.x_mean.double().numpy(), x_std=m.x_std.double().numpy(), n_stim=int(m.n_stim),
                    cols=cols, layers=layers, scale=m.scale.double().numpy())
    if isinstance(net, ConditionProliferation):
        return [one(m, c) for m, c in zip(net.nets, net.cols)], net.sample_cond
    return [one(net)], None


def head_rates(head, prot, stim=None):
    """(birth, death) of one state from birth_heads parameters, as predict_rates; prot (n_proteins,) without the
    stimuli, stim (n_stim,) or None."""
    x = np.asarray(prot, dtype=float)
    if head['cols'] is not None:
        x = x[head['cols']]
    k = head['n_stim']
    if k > 0:
        x = np.concatenate([np.zeros(k) if stim is None else np.asarray(stim, dtype=float)[:k], x])
    z = (x - head['x_mean']) / head['x_std']
    for i, (W, b) in enumerate(head['layers']):
        z = W @ z + b
        if i < len(head['layers']) - 1:
            z = np.tanh(z)
    sp = head['scale'] * np.logaddexp(0.0, z)  # softplus
    return float(sp[0]), float(sp[1])


def quadrature(n_nodes: int):
    """Relative nodes in [0, 1] and trapezoidal weights (summing to 1) on one interval."""
    u = np.linspace(0.0, 1.0, max(2, n_nodes))
    w = np.full(len(u), 1.0 / (len(u) - 1))
    w[[0, -1]] /= 2
    return u, w


def growth_path_states(prot, times_data, ns=1, n_nodes=5, with_stim=False, valid=None):
    """
    Protein states along each trajectory at the quadrature nodes of every interval.

    Trajectories are the slots of prot (time blocks of N rows, slot n = trajectory n),
    interpolated across all timepoints by a shape-preserving PCHIP spline.
    Returns states (T-1, Q, N, G) and the interval lengths (T-1,); with_stim prepends the
    ns stimulus values of each interval (constant along it, those of its end timepoint; with valid, of the next
    valid state of the slot, interval_stimulus).
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
        S = interval_stimulus(prot, times_data, ns, valid=valid).reshape(T, N, ns)[:-1]   # (T-1, N, ns)
        states = np.concatenate([np.broadcast_to(S[:, None], (T - 1, len(u), N, ns)), states], axis=-1)
    return states, dt


def _started(valid, T, N):
    """(T, N) bool: states at or after the first observed time of their trajectory (the virtual copies before it, of
    the first observed state with growth 0, are left out of the fits); all True without valid."""
    if valid is None:
        return np.ones((T, N), dtype=bool)
    return np.cumsum(np.asarray(valid, dtype=bool).reshape(T, N), axis=0) > 0


class ProliferationAnchor:
    """
    Anchor of the rates of a ProliferationMLP on the trajectory states: its pretraining loss, and its rappel when it
    is learned in the refine SDE. For each trajectory n and interval k (inputs: the stimuli of the interval and the
    proteins along the PCHIP path of the trajectory, n_nodes states, trapezoids):
      - path term: ∫ (b − δ) ds ≈ R[k, n] Δt_k, the net growth of the state over its interval, residuals per Δt in
        units of the sd of the target rates;
      - mass term (lambda_mass): population growth of each interval log mean_n exp(∫ (b − δ)) ≈ that of the targets
        (the path term alone fits the mean log gain and underestimates the growth: Jensen), in the same units;
      - split term (lambda_split): (b, δ) of each state ≈ its prior birth and death (consistent with R, the data
        only seeing the net growth), each rate in units of its mean.
    Intervals and states before the first observed time of a trajectory (valid) are left out; a fraction
    val_fraction of the trajectories is held out (early stopping of the pretraining, diagnostics).
    """

    def __init__(self, prot, R, times_data, ns=1, n_nodes=5, with_stim=True, valid=None, birth_prior=None,
                 death_prior=None, lambda_mass=10.0, lambda_split=1.0, n_mass=128, val_fraction=0.2, seed=0):
        states, dt = growth_path_states(prot, times_data, ns=ns, n_nodes=n_nodes, with_stim=with_stim, valid=valid)
        Km1, Q, N, G = states.shape
        self.n_stim, self.n_inputs = (ns if with_stim else 0), G
        self.w = torch.tensor(quadrature(n_nodes)[1], dtype=torch.float32)
        R = np.asarray(R, dtype=float)
        started = _started(valid, Km1 + 1, N)
        target = R.reshape(Km1 + 1, N)[:-1] * dt[:, None]
        ok = np.isfinite(target) & started[:-1]
        # One pair per (interval, trajectory): its Q path states, Δt and log mass gain
        self.X = torch.tensor(np.transpose(states, (0, 2, 1, 3))[ok], dtype=torch.float32)
        self.D = torch.tensor(np.broadcast_to(dt[:, None], (Km1, N))[ok], dtype=torch.float32)
        self.Y = torch.tensor(target[ok], dtype=torch.float32)
        self.K = np.broadcast_to(np.arange(Km1)[:, None], (Km1, N))[ok]
        traj = np.broadcast_to(np.arange(N)[None, :], (Km1, N))[ok]
        self.intervals = np.unique(self.K)
        self.interval_start = [float(np.sort(np.unique(times_data))[k]) for k in self.intervals]
        # States of the split term: inputs [stimuli of the interval, proteins], prior birth and death
        if birth_prior is None or death_prior is None:  # without birth: max(R, 0), max(-R, 0)
            Rn = np.nan_to_num(R)
            birth_prior, death_prior = np.maximum(Rn, 0.0), np.maximum(-Rn, 0.0)
        Xs = np.asarray(prot, dtype=float)[:, ns:]
        if self.n_stim:
            Xs = np.concatenate([interval_stimulus(prot, times_data, ns, valid=valid), Xs], axis=1)
        B, Dd = np.asarray(birth_prior, dtype=float), np.asarray(death_prior, dtype=float)
        use = started.ravel() & np.isfinite(B) & np.isfinite(Dd)
        self.Xs = torch.tensor(Xs[use], dtype=torch.float32)
        self.B, self.Dd = torch.tensor(B[use], dtype=torch.float32), torch.tensor(Dd[use], dtype=torch.float32)
        self.scale = (float(max(B[use].mean(), 1e-6)) if use.any() else 1.0,
                      float(max(Dd[use].mean(), 1e-6)) if use.any() else 1.0)
        # Unit of the net-rate residuals: sd of the target rates, at least a tenth of the level of the rates
        rates = (self.Y / self.D).numpy()
        self.r_unit = float(max(rates.std() if len(rates) > 1 else 0.0, 0.1 * max(self.scale), 1e-6))
        # Held-out trajectories (whole paths, so that held-out pairs are not neighbours of training ones)
        rng = np.random.default_rng(seed)
        self.n_paths = N
        self.n_val_paths = int(round(val_fraction * N)) if N >= 10 else 0
        val_traj = rng.choice(N, self.n_val_paths, replace=False) if self.n_val_paths else np.array([], dtype=int)
        is_val, is_val_s = np.isin(traj, val_traj), np.isin(np.tile(np.arange(N), Km1 + 1)[use], val_traj)
        self.parts = {'tr': (np.flatnonzero(~is_val), np.flatnonzero(~is_val_s)),
                      'va': (np.flatnonzero(is_val), np.flatnonzero(is_val_s)),
                      'all': (np.arange(len(self.Y)), np.arange(len(self.B)))}
        self.groups = {k: [v[0][self.K[v[0]] == i] for i in self.intervals if np.any(self.K[v[0]] == i)]
                       for k, v in self.parts.items()}
        Yn = self.Y.numpy().astype(float)
        self.mass = {k: torch.tensor([np.log(np.mean(np.exp(Yn[g]))) for g in gs], dtype=torch.float32)
                     for k, gs in self.groups.items()}
        self.dt_groups = {k: torch.tensor([float(self.D[g[0]]) for g in gs], dtype=torch.float32)
                          for k, gs in self.groups.items()}
        self.lambda_mass, self.lambda_split, self.n_mass = float(lambda_mass), float(lambda_split), int(n_mass)
        self.rng = np.random.default_rng(seed + 1)

    def init_model(self, model):
        """Input standardisation (training paths) and output scales (mean prior birth and death) of a new MLP; the
        stimulus inputs start with no effect (constant ones get no gradient and keep none)."""
        flat = self.X[self.parts['tr'][0]].reshape(-1, self.n_inputs).numpy()
        sd = flat.std(axis=0)
        model.x_mean.copy_(torch.tensor(flat.mean(axis=0)))
        model.x_std.copy_(torch.tensor(np.where(sd > 1e-6, sd, 1.0)))
        model.scale.copy_(torch.tensor(self.scale))
        with torch.no_grad():
            model.net[0].weight[:, :self.n_stim] = 0.0

    def integral(self, model, idx):
        """∫ (b − δ) along the paths of the pairs idx."""
        b, d = model.rates(self.X[idx])
        return ((b - d) * self.w).sum(dim=1) * self.D[idx]

    def loss(self, model, part='tr', batch=None, log_units=False):
        """Anchor loss on the pairs and states of part ('tr' training, 'va' held-out, 'all'): path + lambda_mass mass
        + lambda_split split terms; batch: random subsets of that many pairs and states (mass term on up to n_mass
        pairs per interval), None: all. log_units: the path and mass residuals in log mass gain over the interval
        (the scale of the branching log weights; rappel in the refine SDE), else in rates (per Δt) in units of r_unit
        (pretraining: dimensionless loss, independent of the weight decay)."""
        pairs, rows = self.parts[part]
        if not len(pairs):
            return torch.zeros((), dtype=torch.float32)
        pick = lambda v, n: v if n is None or len(v) <= n else self.rng.choice(v, n, replace=False)
        idx = torch.as_tensor(pick(pairs, batch))
        unit = (lambda dt: 1.0) if log_units else (lambda dt: dt * self.r_unit)
        loss = (((self.integral(model, idx) - self.Y[idx]) / unit(self.D[idx])) ** 2).mean()
        if self.lambda_mass > 0 and self.groups[part]:
            lme = []
            for g in self.groups[part]:
                sel = torch.as_tensor(pick(g, None if batch is None else self.n_mass))
                lme.append(torch.logsumexp(self.integral(model, sel), 0) - np.log(len(sel)))
            res = (torch.stack(lme) - self.mass[part]) / unit(self.dt_groups[part])
            loss = loss + self.lambda_mass * (res ** 2).mean()
        if self.lambda_split > 0 and len(rows):
            r = torch.as_tensor(pick(rows, batch))
            b, d = model.rates(self.Xs[r])
            loss = loss + self.lambda_split * (((b - self.B[r]) / self.scale[0]) ** 2
                                               + ((d - self.Dd[r]) / self.scale[1]) ** 2).mean()
        return loss

    def diagnostics(self, model):
        """R² of the path mean rates (training and held-out paths), population growth per interval (targets and MLP),
        mean birth and death (MLP and prior)."""
        with torch.no_grad():
            pred = self.integral(model, torch.arange(len(self.Y))).numpy().astype(float)
            b, d = model.rates(self.Xs)
        Y, D = self.Y.numpy().astype(float), self.D.numpy().astype(float)

        def r2(idx):
            if len(idx) < 2:
                return float('nan')
            y, p = Y[idx] / D[idx], pred[idx] / D[idx]
            ss = np.sum((y - y.mean()) ** 2)
            return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float('nan')
        lme = lambda v: float(np.log(np.mean(np.exp(v))))
        return dict(r2_train=r2(self.parts['tr'][0]), r2_val=r2(self.parts['va'][0]),
                    n_train_paths=int(self.n_paths - self.n_val_paths), n_val_paths=int(self.n_val_paths),
                    with_stimulus=bool(self.n_stim), interval_start=self.interval_start,
                    growth_target=[lme(Y[g]) for g in self.groups['all']],
                    growth_mlp=[lme(pred[g]) for g in self.groups['all']],
                    birth_mlp=float(b.mean()), death_mlp=float(d.mean()),
                    birth_prior=float(self.B.mean()), death_prior=float(self.Dd.mean()))


def train_proliferation_mlp(
    prot: np.ndarray,
    R: np.ndarray,
    times_data: np.ndarray,
    ns: int = 1,
    n_nodes: int = 5,
    hidden_size: int = 64,
    n_epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    lambda_mass: float = 10.0,
    lambda_split: float = 1.0,
    n_mass: int = 128,
    with_stim: bool = True,
    val_fraction: float = 0.2,
    patience: int = 50,
    seed: int = 0,
    verb: bool = True,
    birth_prior: np.ndarray = None,
    death_prior: np.ndarray = None,
    valid: np.ndarray = None,
) -> ProliferationMLP:
    """
    Pretraining of the birth and death heads (b, δ)(u, P) on the trajectory states (ProliferationAnchor): for each
    trajectory n and interval k,
        ∫_{t_k}^{t_k+1} (b − δ)(u_k, P_n(s)) ds  ≈  R[k, n] · Δt_k   (log mass gain),
    u_k the inference stimuli over the interval (if with_stim), with the mass and split terms; Adam on minibatches of
    the training paths, early stopping on the held-out ones (patience epochs, best weights kept). The anchor is kept
    as model.anchor (rappel of the MLP when it is learned in the refine SDE) and the fit quality in model.diagnostics.

    Parameters
    ----------
    prot : (T * N, G_tot) protein trajectories (time blocks, stimulus dims first).
    R : (T * N,) net growth rate of each state over the next interval (NaN at last time).
    times_data : (T * N,) time of each state.
    ns : number of stimulus dimensions of prot.
    birth_prior, death_prior : (T * N,) prior birth and death of the states (consistent with R), None: max(±R, 0).
    valid : (T * N,) observed states or None: stimulus of the interval of each state (interval_stimulus); the
        intervals before the first observed time of a trajectory are left out.
    """
    anchor = ProliferationAnchor(prot, R, times_data, ns=ns, n_nodes=n_nodes, with_stim=with_stim, valid=valid,
                                 birth_prior=birth_prior, death_prior=death_prior, lambda_mass=lambda_mass,
                                 lambda_split=lambda_split, n_mass=n_mass, val_fraction=val_fraction, seed=seed)
    model = ProliferationMLP(anchor.n_inputs, hidden_size=hidden_size, n_stim=anchor.n_stim)
    anchor.init_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_tr = len(anchor.parts['tr'][0])
    n_steps = max(1, int(np.ceil(n_tr / batch_size)))
    best, best_state, best_epoch, wait = np.inf, None, 0, 0
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(n_steps):
            loss = anchor.loss(model, 'tr', batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        model.eval()
        with torch.no_grad():
            vl = float(anchor.loss(model, 'va')) if anchor.n_val_paths else np.nan
        if np.isfinite(vl):
            if vl < best - 1e-9:
                best, best_epoch, wait = vl, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
        if verb and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"[ProliferationMLP] epoch {epoch:4d}/{n_epochs}  loss={epoch_loss / n_steps:.6f}  val={vl:.6f}")
        if wait >= patience:
            if verb:
                print(f"[ProliferationMLP] early stop at epoch {epoch} (best validation epoch {best_epoch})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.anchor = anchor
    model.diagnostics = dict(anchor.diagnostics(model), best_epoch=int(best_epoch), stage='pretraining')
    if verb:
        d = model.diagnostics
        print(f"[ProliferationMLP] R² of the path mean rates: train {d['r2_train']:.3f}, held-out {d['r2_val']:.3f}; "
              f"mean birth {d['birth_mlp']:.4f} (prior {d['birth_prior']:.4f}), death {d['death_mlp']:.4f} "
              f"(prior {d['death_prior']:.4f}) h^-1")
    return model
