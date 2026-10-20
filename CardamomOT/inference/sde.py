"""
Free-running fit (no teacher forcing) of the simulation model on the observed mode distributions.

The model of the simulations is integrated from the trajectory states of a starting time over the next intervals
and compared, at each reached time, with the kon of the observed modes (debiased Sinkhorn on kon, per sample).
Horizon curriculum: windows of 1 interval first (close to the teacher-forced fits), then 2, ... up to the whole
time course, so that the errors accumulated along the simulations are corrected.

Models of the simulators, with dilution at the birth rate b of each cell (bursts over each integration step as
Gamma increments with the mean and variance of the bursts, or their diffusion limit with noise='gaussian'):
- 'ode':  dP = (d1 (kon(P) - P) - b P) dt
- 'sde1': protein-only bursty PDMP, bursts of rate kon k1 d0 and mean size d1 / (d0 k1):
          dP = (d1 (kon(P) - P) - b P) dt + sqrt(2 kon d1 eps / k1) dW, eps = d1 / d0
- 'sde2': explicit mRNA (normalised m = M c / k1), mRNA bursts of rate kon k1 d0 and mean size 1 / c:
          dm = d0 (kon(P) - m) dt + sqrt(2 kon d0 / k1) dW,  dP = (d1 m - (d1 + b) P) dt, eps = d1 / d0;
          the observed mRNA (normalised as m) are then compared too, the modes alone not setting the mRNA speed d0
Bursty PDMP and its diffusion limit: Herbach et al. 2017 (BMC Syst. Biol.); free-running training of a neural
SDE: Kidger et al. 2021 (ICML), with Sinkhorn divergences (Feydy et al. 2019, AISTATS; geomloss).
"""

import time

import numpy as np
import torch
from geomloss import SamplesLoss

SCALE_BOUNDS = (0.2, 5.0)  # scale of theta per target gene: within [1/5, 5] (network neither erased nor blown up)


def fit_free_running(groups, times, ns, theta, basal, d1, eps, scale, mode='sde1', learn_theta=True,
                     learn_d=True, learn_eps=True, learn_scale=True, temporal=True, lam_theta=1.0, lam_rates=1.0,
                     eps_bounds=(1 / 50, 1 / 1.5), epochs=30, lr=1e-2, n_sub=20, batch_size=256, n_rep=2,
                     max_horizon=9, n_windows=10, n_cond=1, d1_ref=None, eps_ref=None, only_max_horizon=False,
                     blur=0.05, noise='gamma', scale_p=1.0, seed=0, verb=False):
    """
    groups: one dict per sample s with
        'X' (T, n, G) trajectory states (starts of the windows; mode 'sde2' also 'M' (T, n, G) normalised mRNA
        and 'M_targets' list of T arrays of observed normalised mRNA),
        'targets' list of T arrays (n_t, G) of observed kon (None: time not observed), 'k_lo', 'k_hi', 'k1' (G,),
        'stim' (T - 1, ns) stimulus of each interval, 'idx' index of its basal / network rows, 'cond' index of its
        kinetics (network condition, 0 .. n_cond - 1: d1, eps and scale per condition),
        'start_ok' (T,) bool times observed for the sample (windows start only there, virtual states being copies),
        'birth' f(P (n, G), t, slots) -> (n,) or None, 'rate' f(P (n, G), t) -> (n,) or None (branching weights).
    theta (G, G) or (S, G, G), basal (S, G): starting network (support and signs kept, rappel lam_theta);
    d1, eps, scale (T - 1, G): starting rates (rappel lam_rates on the logs towards d1_ref, eps_ref, default the
    starting values, and towards a scale of 1); only_max_horizon: no curriculum, windows of max_horizon only (with
    T - 1 <= max_horizon, the whole time course from the first timepoint only); temporal: one value per interval,
    else shared; max_horizon: longest window (intervals) of the curriculum; n_windows: window starts drawn at each
    step (None: all; the cost of a step grows as n_windows x horizon instead of T^2), integrated together in one
    batch; scale_p: protein level of a kon of 1 (scale_proteins, also for the normalised mRNA). Returns
    (theta, basal, d1, eps, scale (n_cond, T - 1, G), history).
    """
    seed = 0 if seed is None else int(seed)  # no model seed: fixed draws
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=float)
    T, G = len(times), basal.shape[-1]
    T1 = T - 1
    f64 = torch.float64
    gmask = torch.ones(G, dtype=f64)
    gmask[:ns] = 0.0

    theta0 = np.asarray(theta, dtype=float)
    mask = torch.tensor(theta0 != 0, dtype=f64)
    sgn = torch.tensor(np.sign(theta0), dtype=f64)
    th = torch.tensor(np.abs(theta0), dtype=f64, requires_grad=learn_theta)
    ba = torch.tensor(np.asarray(basal, dtype=float), dtype=f64, requires_grad=learn_theta)

    def log_par(v, lo, hi, learn):
        v = np.log(np.clip(np.asarray(v, dtype=float), lo, hi))
        v = v if temporal else v.mean(axis=0, keepdims=True)
        return torch.tensor(np.tile(v, (n_cond, 1, 1)), dtype=f64, requires_grad=learn)  # (n_cond, T1 or 1, G)
    ld = log_par(d1, 1e-8, np.inf, learn_d)
    le = log_par(eps, *eps_bounds, learn_eps)
    ls = log_par(scale, *SCALE_BOUNDS, learn_scale)
    ls_lo, ls_hi = np.log(SCALE_BOUNDS[0]), np.log(SCALE_BOUNDS[1])
    le_lo, le_hi = np.log(eps_bounds[0]), np.log(eps_bounds[1])
    ref = [p.detach().clone() for p in (th, ba, ld, le, ls)]
    if d1_ref is not None:  # rappel targets other than the starting values (e.g. the literature)
        ref[2] = log_par(d1_ref, 1e-8, np.inf, False)
    if eps_ref is not None:
        ref[3] = log_par(eps_ref, *eps_bounds, False)
    ref[4] = torch.zeros_like(ls)  # scale rappel towards 1

    # Per-gene units of the kon distances (floor: constant genes)
    allk = np.concatenate([tg[:, ns:] for g in groups for tg in g['targets'] if tg is not None and len(tg)])
    sdk = torch.tensor(np.maximum(allk.std(axis=0), 0.05), dtype=f64)
    if mode == 'sde2':
        allm = np.concatenate([tg[:, ns:] for g in groups for tg in g['M_targets'] if tg is not None and len(tg)])
        sdm = torch.tensor(np.maximum(allm.std(axis=0), 0.05), dtype=f64)
    sink = SamplesLoss('sinkhorn', p=2, blur=blur, scaling=0.5, debias=True)  # geomloss default annealing
    tens = [dict(k_lo=torch.tensor(g['k_lo'], dtype=f64), k_hi=torch.tensor(g['k_hi'], dtype=f64),
                 k1=torch.tensor(np.maximum(g['k1'], 1e-8), dtype=f64)) for g in groups]

    def theta_of(i):
        t_ = sgn * torch.relu(th) * mask
        return t_[i] if t_.dim() == 3 else t_

    def rates_of(tw, c):
        """d1, eps, scale (W, G) of kinetics c at the intervals tw (W,) of the windows."""
        i = tw if temporal else np.zeros_like(tw)
        return (torch.exp(ld[c, i]), torch.exp(le[c, i].clamp(le_lo, le_hi)), torch.exp(ls[c, i].clamp(ls_lo, ls_hi)))

    def kon_of(P, gi, s):
        """kon (W, n, G) of the states P (W, n, G) of group gi, scale of theta s (W, G) of each window."""
        g, c = groups[gi], tens[gi]
        z = torch.einsum('wng,gj->wnj', P, theta_of(g['idx'])) * s[:, None, :] + (ba[g['idx']] * s)[:, None, :]
        return c['k_lo'] + (c['k_hi'] - c['k_lo']) * torch.sigmoid(z)

    def bursts(rate, size, h):
        """Bursts over a step h: Gamma increments with the mean (rate h size) and variance (2 rate h size^2) of the
        compound Poisson of exponential sizes (positive, skewed, reparametrised); 'gaussian': diffusion limit."""
        if noise == 'gaussian':
            return rate * h * size + torch.sqrt(torch.clamp(2 * rate * h, min=1e-12)) * size * torch.randn_like(rate)
        conc = torch.clamp(rate * h / 2, min=1e-6)
        return torch.distributions.Gamma(conc, 1.0 / (2 * size)).rsample()

    def run_interval(P, Mn, gi, tw, b):
        """Integration of the windows over their intervals [times[tw], times[tw + 1]] (n_sub steps each, exact
        relaxation over a step); P, Mn (W, n, G), b (W, n) birth rates."""
        d, e, s = rates_of(tw, groups[gi].get('cond', 0))
        d, e = d[:, None, :], e[:, None, :]
        k1 = tens[gi]['k1']
        h = torch.tensor((times[tw + 1] - times[tw]) / n_sub, dtype=f64)[:, None, None]
        dp = d + b[..., None]                     # protein decay with dilution
        decay_p = torch.exp(-dp * h)
        keep = gmask.bool()
        for _ in range(n_sub):
            kon = kon_of(P, gi, s)
            if mode == 'sde2':
                d0 = d / e
                # mRNA bursts of rate kon k1 d0 and mean size 1 / k1 (normalised), proteins translated from them
                Mn = torch.relu(Mn * torch.exp(-d0 * h) + bursts(kon * k1 * d0, scale_p / k1, h)) * gmask
                P_new = P * decay_p + d * Mn * (1 - decay_p) / dp
            elif mode == 'sde1':
                # protein bursts of rate kon k1 d0 and mean size d1 / (d0 k1) = eps / k1
                P_new = P * decay_p + bursts(kon * k1 * d / e, scale_p * e / k1, h) * (1 - decay_p) / (dp * h)
            else:
                P_new = P * decay_p + d * scale_p * kon * (1 - decay_p) / dp
            P = torch.relu(torch.where(keep, P_new, P))
        return P, Mn, s

    def windows_loss(a_sel, H, full=False):
        """Mean loss of the windows starting at a_sel (H intervals each), all windows of a group in one batch: one
        batched Sinkhorn per interval (targets resampled to a common size)."""
        loss, n_terms = 0.0, 0
        for gi, g in enumerate(groups):
            a_ok = np.array([a for a in a_sel if g['start_ok'][a]], dtype=int)  # virtual states are no start
            if not len(a_ok):
                continue
            W, n = len(a_ok), g['X'].shape[1]
            slots = np.stack([np.arange(n) if full or n <= batch_size else rng.choice(n, batch_size, replace=False)
                              for _ in range(W)])
            slots = np.repeat(slots, n_rep, axis=1)                       # (W, n_p)
            P = torch.tensor(g['X'][a_ok[:, None], slots], dtype=f64)
            Mn = torch.tensor(g['M'][a_ok[:, None], slots], dtype=f64) if mode == 'sde2' else None
            logw = np.zeros(slots.shape)
            stim = torch.tensor(g['stim'], dtype=f64)
            for k in range(H):
                tw = a_ok + k
                P = P.clone()
                P[..., :ns] = stim[tw][:, None, :]
                Pn = P.detach().numpy()
                b = np.zeros(slots.shape) if g['birth'] is None else \
                    np.stack([np.asarray(g['birth'](Pn[w], tw[w], slots[w]), float) for w in range(W)])
                R0 = None if g['rate'] is None else np.stack([np.asarray(g['rate'](Pn[w], tw[w]), float)
                                                              for w in range(W)])
                P, Mn, s = run_interval(P, Mn, gi, tw, torch.tensor(b, dtype=f64))
                P = P.clone()
                P[..., :ns] = stim[tw][:, None, :]
                if R0 is not None:  # branching: weights exp(int R) along the window
                    Pn = P.detach().numpy()
                    R1 = np.stack([np.asarray(g['rate'](Pn[w], tw[w]), float) for w in range(W)])
                    logw = logw + 0.5 * (R0 + R1) * (times[tw + 1] - times[tw])[:, None]
                obs = [w for w in range(W) if g['targets'][tw[w] + 1] is not None and len(g['targets'][tw[w] + 1])]
                if not obs:
                    continue
                M = len(g['targets'][tw[obs[0]] + 1]) if full else batch_size
                sel = [rng.choice(len(g['targets'][tw[w] + 1]), M, replace=len(g['targets'][tw[w] + 1]) < M)
                       if not full else np.arange(M) for w in obs]
                Y = torch.tensor(np.stack([g['targets'][tw[w] + 1][i][:, ns:] for w, i in zip(obs, sel)]), dtype=f64)
                wts = torch.softmax(torch.tensor(logw[obs], dtype=f64), dim=1)
                u = torch.full((len(obs), M), 1.0 / M, dtype=f64)
                kon = kon_of(P[obs], gi, s[obs])[..., ns:]
                # Genes of the group's condition only (select_genes_per_condition): the others are not compared
                gk = sdk if g.get('gmask') is None else sdk / torch.tensor(g['gmask'][ns:], dtype=f64).clamp(min=1e-12)
                loss = loss + sink(wts, kon / gk, u, Y / gk).sum()
                if mode == 'sde2':  # observed mRNA (same cells)
                    Ym = torch.tensor(np.stack([g['M_targets'][tw[w] + 1][i][:, ns:] for w, i in zip(obs, sel)]),
                                      dtype=f64)
                    gm_ = sdm if g.get('gmask') is None else sdm / torch.tensor(g['gmask'][ns:], dtype=f64).clamp(min=1e-12)
                    loss = loss + sink(wts, Mn[obs][..., ns:] / gm_, u, Ym / gm_).sum()
                n_terms += len(obs)
        return loss / max(n_terms, 1)

    def penalty():
        th_r, ba_r, ld_r, le_r, ls_r = ref
        p = 0.0
        if learn_theta:
            p = p + lam_theta * ((th - th_r) ** 2 * mask).sum() / ((th_r ** 2 * mask).sum() + 1e-12)
            p = p + lam_theta * ((ba - ba_r)[..., ns:] ** 2).sum() / ((ba_r[..., ns:] ** 2).sum() + 1e-12)
        for par, r_, on in ((ld, ld_r, learn_d), (le, le_r, learn_eps), (ls, ls_r, learn_scale)):
            if on:
                p = p + lam_rates * ((par - r_)[..., ns:] ** 2).mean()
        return p

    def full_loss():
        st = torch.get_rng_state()
        torch.manual_seed(seed + 1)
        with torch.no_grad():
            v = float(windows_loss([0], T1, full=True))
        torch.set_rng_state(st)
        return v

    params = [p for p in (th, ba, ld, le, ls) if p.requires_grad]
    hist = {'before': full_loss()}
    if params:
        opt = torch.optim.Adam(params, lr=lr)
        H_max = T1 if max_horizon is None else max(1, min(int(max_horizon), T1))
        t_start = time.time()
        for H in ([H_max] if only_max_horizon else range(1, H_max + 1)):
            starts = np.arange(T - H)
            for _ in range(int(epochs)):
                a_sel = starts if n_windows is None or len(starts) <= n_windows else \
                    np.sort(rng.choice(starts, int(n_windows), replace=False))
                opt.zero_grad()
                loss = windows_loss(a_sel, H) + penalty()
                loss.backward()
                opt.step()
            if verb:
                print(f"[fit_free_running] horizon {H}/{H_max}: loss {float(loss):.4f} "
                      f"({time.time() - t_start:.0f} s)", flush=True)
    hist['after'] = full_loss()

    full_t = lambda p, lo, hi: np.broadcast_to(np.exp(np.clip(p.detach().numpy(), lo, hi)), (n_cond, T1, G)).copy()
    theta_new = (sgn * torch.relu(th) * mask).detach().numpy()
    return (theta_new, ba.detach().numpy(), full_t(ld, -np.inf, np.inf), full_t(le, le_lo, le_hi),
            full_t(ls, ls_lo, ls_hi), hist)
