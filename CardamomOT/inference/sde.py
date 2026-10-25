"""
Free-running fit (no teacher forcing) of the simulation model on the observed mode distributions.

The model of the simulations is integrated from the trajectory states of a starting time over the next intervals
and compared, at each reached time, with the kon of the observed modes (debiased Sinkhorn on kon, per sample); with
the mRNA memory (targets_accounting_late_mrna, group 'lag'), the kon of each target at the proteins the lag of its
class earlier (mean of the memory kernel truncated at the start of the interval), the modes being read on the mRNA,
which follows the burst frequency with this delay (the lag is that of fit_network: no gradient through d0 = d1 / eps).
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

from .sigma import sigma_torch

SCALE_BOUNDS = (0.2, 5.0)  # scale of theta per target gene: within [1/5, 5] (network neither erased nor blown up)


def fit_free_running(groups, times, ns, theta, basal, d1, eps, scale, mode='sde1', learn_theta=True,
                     learn_d=True, learn_eps=True, learn_scale=True, temporal=True, lam_theta=1.0, lam_rates=1.0,
                     eps_bounds=(1 / 50, 1 / 1.5), epochs=30, lr=1e-2, n_sub=20, batch_size=256, n_rep=2,
                     max_horizon=9, n_windows=10, n_cond=1, d1_ref=None, eps_ref=None, only_max_horizon=False,
                     blur=0.05, noise='gamma', scale_p=1.0, seed=0, verb=False, spec=None, support_tol=0.0,
                     memory_filter=False, ce_weight=0.0):
    """
    groups: one dict per sample s with
        'X' (T, n, G) trajectory states (starts of the windows; mode 'sde2' also 'M' (T, n, G) normalised mRNA
        and 'M_targets' list of T arrays of observed normalised mRNA),
        'targets' list of T arrays (n_t, G) of observed kon (None: time not observed), 'k_lo', 'k_hi', 'k1' (G,),
        optionally 'post' / 'levels' lists of T arrays (n_t, G, L): posteriors and levels of each observed cell, the
        targets being then drawn from them at every minibatch (their mean 'targets' would contract the distribution),
        optionally 'tau' (T, n, G) sigmoid targets of the network fit of each trajectory state, 'tau_ok' (T, n) observed
        states and 'gw' (G,) gene weights: deterministic model ('ode'), cross-entropy of each trajectory against its
        targets (the pairing of the trajectories), added with weight ce_weight to the Sinkhorn between distributions,
        optionally 'binary' (G,) bool: genes of two modes, whose simulated kon is compared as a Bernoulli draw of
        sigma (probability of the high mode) between k_lo and k_hi,
        'stim' (T - 1, ns) stimulus of each interval, 'idx' index of its basal / network rows, 'cond' index of its
        kinetics (network condition, 0 .. n_cond - 1: d1, eps and scale per condition),
        'start_ok' (T,) bool times observed for the sample (windows start only there, virtual states being copies),
        'birth' f(P (n, G), kon (n, G), t, slots) -> (n,) birth rates of the dilution or None, re-evaluated at every
        integration step (proliferation MLP),
        'rate' f(P (n, G), t) -> (n,) or None: net rate of the branching weights, integrated along the steps (MLP),
        'nearest' f(P, kon, t, slots) -> (birth (n,) or None, net rate (n,) or None) or None: rates evaluated at the
        start of each interval only and kept over it (rates of the nearest trajectory states, without the MLP),
        'lag' (optional, (T - 1, K) hours per interval and class of targets) and 'lag_w' ((K, G) one-hot class of
        each gene): the observed modes of a target are compared with its kon at the proteins the lag of its class
        before each reached time (integration step nearest to it, the start of the interval at most).
    theta (G, G) or (S, G, G), basal (S, G): starting network (support |theta| > support_tol and signs kept,
    the other entries set to 0; rappel lam_theta on the support);
    d1, eps, scale (T - 1, G): starting rates (rappel lam_rates on the logs towards d1_ref, eps_ref, default the
    starting values, and towards a scale of 1); only_max_horizon: no curriculum, windows of max_horizon only (with
    T - 1 <= max_horizon, the whole time course from the first timepoint only); temporal: one value per interval,
    else shared; max_horizon: longest window (intervals) of the curriculum; n_windows: window starts drawn at each
    step (None: all; the cost of a step grows as n_windows x horizon instead of T^2), integrated together in one
    batch; scale_p: protein level of a kon of 1 (scale_proteins, also for the normalised mRNA); spec: families of the
    GRN sigma (sigma.py, model_learning_grn), None = logistic. memory_filter: the observed modes are compared with
    the kon read on the mRNA, Kb' = d0 (kon - Kb) at the d0 = d1 / eps of the model, carried along the windows from
    the kon of the start states (groups 'K0' (T, n, G)), instead of the lagged proteins. Returns
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

    # Support: |theta| > support_tol (residues of the smoothed L1 and of the averaged fits are absent edges)
    theta0 = np.asarray(theta, dtype=float)
    theta0 = np.where(np.abs(theta0) > support_tol, theta0, 0.0)
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

    # Per-gene units of the kon distances (floor: constant genes); with posteriors, total variance of the draws
    # (from the posteriors alone, whatever the forcing modes in 'targets')
    if all(g.get('post') is not None for g in groups):
        pl = [(p_[:, ns:], lv[:, ns:]) for g in groups for p_, lv in zip(g['post'], g['levels'])
              if p_ is not None and len(p_)]
        m1 = np.concatenate([(p_ * lv).sum(-1) for p_, lv in pl])
        m2 = np.concatenate([(p_ * lv ** 2).sum(-1) for p_, lv in pl])
        var_k = np.maximum(m2.mean(axis=0) - m1.mean(axis=0) ** 2, 0.0)
    else:
        allk = np.concatenate([tg[:, ns:] for g in groups for tg in g['targets'] if tg is not None and len(tg)])
        var_k = allk.var(axis=0)
    sdk = torch.tensor(np.maximum(np.sqrt(var_k), 0.05), dtype=f64)
    if mode == 'sde2':
        allm = np.concatenate([tg[:, ns:] for g in groups for tg in g['M_targets'] if tg is not None and len(tg)])
        sdm = torch.tensor(np.maximum(allm.std(axis=0), 0.05), dtype=f64)
    sink = SamplesLoss('sinkhorn', p=2, blur=blur, scaling=0.5, debias=True)  # geomloss default annealing
    tens = [dict(k_lo=torch.tensor(g['k_lo'], dtype=f64), k_hi=torch.tensor(g['k_hi'], dtype=f64),
                 k1=torch.tensor(np.maximum(g['k1'], 1e-8), dtype=f64),
                 binary=None if g.get('binary') is None or not np.any(g['binary'][ns:]) else
                 torch.tensor(np.asarray(g['binary'][ns:], dtype=bool)),
                 lag_w=None if g.get('lag') is None else torch.tensor(np.asarray(g['lag_w'], dtype=float)))
            for g in groups]

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
        return c['k_lo'] + (c['k_hi'] - c['k_lo']) * sigma_torch(P, theta_of(g['idx']), ba[g['idx']], s, spec)

    def kon_read(P_obs, gi, s):
        """kon (W, n, G) compared with the observed modes: at the proteins P_obs (W, n, G), or with the mRNA memory
        P_obs (K, W, n, G) at the lagged node of the class of each target."""
        lw = tens[gi]['lag_w']
        if lw is None:
            return kon_of(P_obs, gi, s)
        return sum(lw[k] * kon_of(P_obs[k], gi, s) for k in range(lw.shape[0]))

    def bursts(rate, size, h):
        """Bursts over a step h: Gamma increments with the mean (rate h size) and variance (2 rate h size^2) of the
        compound Poisson of exponential sizes (positive, skewed, reparametrised); 'gaussian': diffusion limit."""
        if noise == 'gaussian':
            return rate * h * size + torch.sqrt(torch.clamp(2 * rate * h, min=1e-12)) * size * torch.randn_like(rate)
        conc = torch.clamp(rate * h / 2, min=1e-6)
        return torch.distributions.Gamma(conc, 1.0 / (2 * size)).rsample()

    def run_interval(P, Mn, gi, tw, sl, Kb=None):
        """Integration of the windows over their intervals [times[tw], times[tw + 1]] (n_sub steps each, exact
        relaxation over a step); P, Mn (W, n, G), sl (W, n) trajectory slots of the cells. Birth rates of the
        dilution re-evaluated on the current states (every step, or at the start of the interval), branching net
        rate integrated along the steps (trapezoids). Kb (W, n, G) given (memory_filter): kon read on the mRNA,
        relaxed towards kon at the d0 = d1 / eps of the model. Returns (P, Mn, s, increment of the log-weights (W, n),
        proteins compared with the observed modes: P without lag, else the proteins (K, W, n, G) the lag of each class
        before the end of each interval, Kb)."""
        g = groups[gi]
        W = P.shape[0]
        d, e, s = rates_of(tw, g.get('cond', 0))
        d, e = d[:, None, :], e[:, None, :]
        k1 = tens[gi]['k1']
        h_np = (times[tw + 1] - times[tw]) / n_sub
        h = torch.tensor(h_np, dtype=f64)[:, None, None]
        b = torch.zeros(P.shape[:2], dtype=f64)
        rate_of = None if g['rate'] is None else \
            (lambda X: np.stack([np.asarray(g['rate'](X[w], tw[w]), float) for w in range(W)]))
        R_prev = None if rate_of is None else rate_of(P.detach().numpy())
        dlogw = np.zeros(P.shape[:2])
        if g.get('nearest') is not None:  # rates of the nearest trajectory states, constant over the interval
            Pn, kn = P.detach().numpy(), kon_of(P, gi, s).detach().numpy()
            out = [g['nearest'](Pn[w], kn[w], tw[w], sl[w]) for w in range(W)]
            if out[0][0] is not None:
                b = torch.tensor(np.stack([np.asarray(o[0], float) for o in out]), dtype=f64)
            if out[0][1] is not None:
                dlogw = np.stack([np.asarray(o[1], float) for o in out]) * (times[tw + 1] - times[tw])[:, None]
        keep = gmask.bool()
        lag = g.get('lag') if Kb is None else None
        hist = [P] if lag is not None else None  # states at the steps, for the lagged comparison
        for k in range(n_sub):
            kon = kon_of(P, gi, s)
            if Kb is not None:  # mean mRNA (kon units) given the kon path: memory kernel of the model's d0
                Kb = kon + (Kb - kon) * torch.exp(-(d / e) * h)
            if g['birth'] is not None:
                Pn, kn = P.detach().numpy(), kon.detach().numpy()
                b = torch.tensor(np.stack([np.asarray(g['birth'](Pn[w], kn[w], tw[w], sl[w]), float)
                                           for w in range(W)]), dtype=f64)
            dp = d + b[..., None]                     # protein decay with dilution
            decay_p = torch.exp(-dp * h)
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
            if hist is not None:
                hist.append(P)
            if rate_of is not None:  # branching: weights exp(int R) along the steps
                R_new = rate_of(P.detach().numpy())
                dlogw = dlogw + 0.5 * (R_prev + R_new) * h_np[:, None]
                R_prev = R_new
        if hist is None:
            return P, Mn, s, dlogw, P, Kb
        k_lag = np.clip(np.rint(np.asarray(lag)[tw] / h_np[:, None]).astype(int), 0, n_sub)  # (W, K) steps
        P_lag = torch.stack([torch.stack([hist[n_sub - k_lag[w, k]][w] for w in range(W)])
                             for k in range(k_lag.shape[1])])
        return P, Mn, s, dlogw, P_lag, Kb

    def observed_kon(kon, gi, r):
        """Simulated kon (W, n, G - ns) read as the observed modes: genes of two modes ('binary') as a Bernoulli draw
        of sigma, the probability of the high mode (as in the network fit), straight-through gradient through sigma."""
        b = tens[gi]['binary']
        if b is None:
            return kon
        lo, hi = tens[gi]['k_lo'][ns:], tens[gi]['k_hi'][ns:]
        sig = ((kon - lo) / torch.clamp(hi - lo, min=1e-12)).clamp(0.0, 1.0)
        z = torch.tensor(r.random(tuple(kon.shape)) < sig.detach().numpy(), dtype=f64)
        return torch.where(b, lo + (hi - lo) * (z + sig - sig.detach()), kon)

    def kon_targets(g, t, i, r):
        """Observed kon (len(i), G - ns) of the cells i at time t: levels drawn from their posteriors (generator r),
        else their 'targets'."""
        if g.get('post') is None:
            return g['targets'][t][i][:, ns:]
        P, lev = g['post'][t][i][:, ns:], g['levels'][t][i][:, ns:]
        k = (np.cumsum(P, axis=-1) < r.random(P.shape[:2])[..., None]).sum(-1).clip(max=P.shape[-1] - 1)
        return np.take_along_axis(lev, k[..., None], axis=-1)[..., 0]

    def windows_loss(a_sel, H, full=False):
        """Mean loss of the windows starting at a_sel (H intervals each), all windows of a group in one batch: one
        batched Sinkhorn per interval (targets resampled to a common size; full: fixed draws of the targets)."""
        loss, n_terms = 0.0, 0
        r_tg = np.random.default_rng(seed + 2) if full else rng
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
            # kon read on the mRNA (memory_filter): from the kon of the start states
            Kb = torch.tensor(g['K0'][a_ok[:, None], slots], dtype=f64) if memory_filter else None
            logw = np.zeros(slots.shape)
            stim = torch.tensor(g['stim'], dtype=f64)
            for k in range(H):
                tw = a_ok + k
                P = P.clone()
                P[..., :ns] = stim[tw][:, None, :]
                P, Mn, s, dlogw, P_obs, Kb = run_interval(P, Mn, gi, tw, slots, Kb)
                P = P.clone()
                P[..., :ns] = stim[tw][:, None, :]
                P_obs = P_obs.clone()
                P_obs[..., :ns] = stim[tw][:, None, :]
                logw = logw + dlogw
                obs = [w for w in range(W) if g['targets'][tw[w] + 1] is not None and len(g['targets'][tw[w] + 1])]
                if not obs:
                    continue
                if g.get('tau') is not None and ce_weight > 0:  # ODE: + cross-entropy of each trajectory (its mode)
                    P_sel = P_obs[obs] if tens[gi]['lag_w'] is None else P_obs[:, obs]
                    lo, hi = tens[gi]['k_lo'][ns:], tens[gi]['k_hi'][ns:]
                    sig = ((kon_read(P_sel, gi, s[obs])[..., ns:] - lo) / torch.clamp(hi - lo, min=1e-12))
                    sig = sig.clamp(1e-6, 1 - 1e-6)
                    tau = torch.tensor(np.stack([g['tau'][tw[w] + 1][slots[w]] for w in obs])[..., ns:], dtype=f64)
                    ok = torch.tensor(np.stack([g['tau_ok'][tw[w] + 1][slots[w]] for w in obs]), dtype=f64)
                    gw = torch.tensor(g['gw'][ns:], dtype=f64) * (1.0 if g.get('gmask') is None else
                                                                torch.tensor(g['gmask'][ns:], dtype=f64))
                    ce = -(tau * torch.log(sig) + (1 - tau) * torch.log(1 - sig))
                    per = (ce * gw).sum(-1) / gw.sum().clamp(min=1e-12)
                    wo = torch.softmax(torch.tensor(logw[obs], dtype=f64), dim=1) * ok
                    loss = loss + ce_weight * ((per * wo).sum(1) / wo.sum(1).clamp(min=1e-12)).sum()
                M = len(g['targets'][tw[obs[0]] + 1]) if full else batch_size
                sel = [rng.choice(len(g['targets'][tw[w] + 1]), M, replace=len(g['targets'][tw[w] + 1]) < M)
                       if not full else np.arange(M) for w in obs]
                Y = torch.tensor(np.stack([kon_targets(g, tw[w] + 1, i, r_tg) for w, i in zip(obs, sel)]), dtype=f64)
                wts = torch.softmax(torch.tensor(logw[obs], dtype=f64), dim=1)
                u = torch.full((len(obs), M), 1.0 / M, dtype=f64)
                if Kb is not None:
                    kon = observed_kon(Kb[obs][..., ns:], gi, r_tg)
                else:
                    P_sel = P_obs[obs] if tens[gi]['lag_w'] is None else P_obs[:, obs]
                    kon = observed_kon(kon_read(P_sel, gi, s[obs])[..., ns:], gi, r_tg)
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
