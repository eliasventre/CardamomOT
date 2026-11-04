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
          the observed mRNA (normalised as m) are then compared too, the modes alone not setting the mRNA speed d0;
          with mrna_dilution, d0 + b in place of d0 in the mRNA equation (same levels kon, d0 a pure decay rate)
Bursty PDMP and its diffusion limit: Herbach et al. 2017 (BMC Syst. Biol.); free-running training of a neural
SDE: Kidger et al. 2021 (ICML), with Sinkhorn divergences (Feydy et al. 2019, AISTATS; geomloss).
"""

import time

import numpy as np
import torch
from geomloss import SamplesLoss

from .sigma import sigma_torch

try:  # geomloss 'auto' switches to the KeOps 'online' backend above 5000² pairs: needs pykeops
    import pykeops.torch  # noqa: F401
    SINKHORN_BACKEND = 'auto'
except Exception:
    SINKHORN_BACKEND = 'tensorized'

SCALE_BOUNDS = (0.2, 5.0)  # scale of theta per target gene: within [1/5, 5] (network neither erased nor blown up)


def fit_free_running(groups, times, ns, theta, basal, d1, eps, scale, mode='sde1', learn_theta=True,
                     learn_d=True, learn_eps=True, learn_scale=True, temporal=True, lam_theta=1.0, lam_rates=1.0,
                     eps_bounds=(1 / 50, 1 / 1.5), epochs=30, lr=1e-2, n_sub=20, batch_size=256, n_rep=2,
                     max_horizon=None, horizon_growth=2.0, n_windows=10, n_cond=1, d1_ref=None, eps_ref=None, only_max_horizon=False,
                     blur=0.05, noise='gamma', scale_p=1.0, seed=0, verb=False, spec=None, support_tol=0.0,
                     memory_filter=False, mrna_dilution=False, prot_weight=0.0, kon_weight=1.0,
                     mrna_weight=1.0, free_theta=False, lam_l1=0.0, first_start_only=False,
                     qut_level=None, qut_reps=20, null_epochs=10, sink_scaling=0.9, mode_cost='ce',
                     ce_eps=0.05, init_weight=None, lr_rates=None, lr_decay=1.0, lr_min_frac=1.0, bptt=None,
                     adaptive=False):
    """
    groups: one dict per sample s with
        'X' (T, n, G) trajectory states (starts of the windows; mode 'sde2' also 'M' (T, n, G) normalised mRNA
        and 'M_targets' list of T arrays of observed normalised mRNA),
        'targets' list of T arrays (n_t, G) of observed kon (None: time not observed), 'k_lo', 'k_hi', 'k1' (G,),
        optionally 'post' / 'levels' lists of T arrays (n_t, G, L): posteriors and levels of each observed cell, the
        targets being then drawn from them at every minibatch (their mean 'targets' would contract the distribution),
        optionally 'mode_levels' (G, L) levels of the mixture of each gene (normalised kon, grid padded by
        repetition): the simulated kon is read as a mode, drawn between the two levels around it with the mean kept
        (Bernoulli of sigma for two modes), straight-through gradient; else 'binary' (G,) bool, genes of two modes so
        read between k_lo and k_hi,
        'stim' (T - 1, ns) stimulus of each interval, 'stim0' (ns,) stimulus at the first observed time, 'idx' index of its basal / network rows, 'cond' index of its
        kinetics (network condition, 0 .. n_cond - 1: d1, eps and scale per condition),
        'start_ok' (T,) bool times observed for the sample (windows start only there and span its observed intervals,
        virtual states being copies),
        'birth' f(P (n, G), kon (n, G), t, slots) -> (n,) birth rates of the dilution or None, re-evaluated at every
        integration step (proliferation MLP),
        'rate' f(P (n, G), t) -> (n,) or None: net rate of the branching weights, integrated along the steps (MLP),
        'nearest' f(P, kon, t, slots) -> (birth (n,) or None, net rate (n,) or None) or None: rates evaluated at the
        start of each interval only and kept over it, added to those of 'birth' and 'rate' (rates of the nearest
        trajectory states without the MLP; effects of the stimulus with an MLP of the natural rates),
        'lag' (optional, (T - 1, K) hours per interval and class of targets) and 'lag_w' ((K, G) one-hot class of
        each gene): the observed modes of a target are compared with its kon at the proteins the lag of its class
        before each reached time (integration step nearest to it, the start of the interval at most).
    theta (G, G) or (S, G, G), basal (S, G): starting network (support |theta| > support_tol and signs kept,
    the other entries set to 0; rappel lam_theta on the support);
    d1, eps, scale (T - 1, G): starting rates (rappel lam_rates on the logs towards d1_ref, eps_ref, default the
    starting values, and towards a scale of 1); only_max_horizon: no curriculum, windows of the whole time
    course only; temporal: one value per interval, else shared; max_horizon: longest window of the curriculum, in observed
    intervals of each group (None: whole time course; a group with fewer takes its whole time course);
    horizon_growth: geometric curriculum, horizons 1, then each the previous one x horizon_growth (at least + 1), up
    to max_horizon (1: every horizon); n_windows: window starts drawn at each
    step (None: all; the cost of a step grows as n_windows x horizon instead of T^2), integrated together in one
    batch; scale_p: protein level of a kon of 1 (scale_proteins, also for the normalised mRNA); spec: families of the
    GRN sigma (sigma.py, model_learning_grn), None = logistic. memory_filter: the observed modes are compared with
    the kon read on the mRNA, Kb' = d0 (kon - Kb) at the d0 = d1 / eps of the model, carried along the windows from
    the kon of the start states (groups 'K0' (T, n, G)), instead of the lagged proteins. mrna_dilution: the mRNA
    ('sde2' and the memory filter) relaxes at d0 + b, the bursts at rate kon k1 (d0 + b) (same levels), d0 = d1 / eps
    staying a pure decay rate. prot_weight > 0: + prot_weight x debiased Sinkhorn, at the last time of each window,
    between the simulated proteins and the trajectory states 'X' of that time (reference proteins: reconstructed by
    fit_network, unitary for the ODE, or observed), per gene in units of their standard deviation. kon_weight,
    mrna_weight: weights of the Sinkhorn on the kon of the observed modes and ('sde2') on the observed mRNA, both in
    units of the highest mode of each gene (unit modes; mRNA positioned on the modes, as in the OT of fit_network).
    free_theta: every entry of theta learned with its sign (network from scratch, e.g. null or random start; rappel
    lam_theta as a mean squared deviation; no scale of theta), else support and signs of theta kept. lam_l1: L1 on theta - theta_null
    (theta_null = 0 with free_theta, else the starting network), in units of the loss gradient, by a proximal step
    after each Adam step preconditioned as that step (soft threshold lr lam / (sqrt(v_hat) + eps)): exact zeros;
    replaces the L2 rappel on theta. qut_level: lam per target by the quantile universal threshold instead, the
    qut_level-quantile over qut_reps replicates of max_g |dL/dtheta_gj| at the null model (theta_null, basal fitted
    with theta_null fixed over null_epochs steps per horizon with free_theta), the observed modes and mRNA being
    replaced by draws of the null model from the initial states (parametric bootstrap). sink_scaling: annealing
    ratio of the Sinkhorn iterations (geomloss 'scaling'): 0.9 brings the gradients within ~10 % of the converged
    ones (0.5 can flip their sign). mode_cost: 'euclidean', debiased Sinkhorn on the unit modes (simulated kon read on
    the grid of levels); 'ce', entropic OT whose ground cost is the cross-entropy of fit_network between the continuous
    sigma of the simulated states and the positions tau of the observed modes (gene weights 1 - k_lo / k_hi),
    regularisation ce_eps (annealed down to it), gradient through the optimal plan. init_weight: weight of the initial
    term (None: that of the kon term). lr_rates: Adam learning rate of the rates and
    scales (None: lr); schedule of both: warm restarts at each horizon of the curriculum, peak x lr_decay at each new
    horizon, cosine decay down to lr_min_frac x the peak within the horizon (1: constant). bptt: truncated
    backpropagation, the simulated states detached every bptt intervals of a window (None: through the whole window). first_start_only: every window
    starts at the first observed time of each group (only the initial states of the trajectories are used), the
    curriculum lengthening it up to max_horizon. adaptive (free_theta with qut_level): second pass, adaptive lasso
    on the support of the first one, weights 1 / |theta_1| (warm start), at its own QUT per target j: qut_level-quantile
    of max_g |theta_1,gj| |dL/dtheta_gj| at the null model of j given the first pass (column j of theta set to 0, its
    basal refitted, the other targets as fitted), whose simulated cells are heterogeneous, unlike the global null
    model (all sigma equal: entropic couplings that only see the mean of each regulator per time).
    Always: + the cross-entropy (as in fit_network) between the sigma of the initial states (first observed time of
    each group, stimulus 'stim0' of the schedule) and the modes of their own cells (groups 'valid0' (n,) bool: slots
    of the observed cells at that time), with the weight of the kon term (init_weight): it sets the basal at the initial
    stimulus, hence the effect of the stimulus.
    Returns (theta, basal, d1, eps, scale (n_cond, T - 1, G), history).
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
    mask = torch.tensor(np.ones_like(theta0) if free_theta else theta0 != 0, dtype=f64)
    sgn = torch.tensor(np.sign(theta0), dtype=f64)
    th = torch.tensor(theta0 if free_theta else np.abs(theta0), dtype=f64, requires_grad=learn_theta)
    ba = torch.tensor(np.asarray(basal, dtype=float), dtype=f64, requires_grad=learn_theta)

    def log_par(v, lo, hi, learn):
        v = np.log(np.clip(np.asarray(v, dtype=float), lo, hi))
        v = v if temporal else v.mean(axis=0, keepdims=True)
        return torch.tensor(np.tile(v, (n_cond, 1, 1)), dtype=f64, requires_grad=learn)  # (n_cond, T1 or 1, G)
    ld = log_par(d1, 1e-8, np.inf, learn_d)
    le = log_par(eps, *eps_bounds, learn_eps)
    ls = log_par(scale, *SCALE_BOUNDS, learn_scale and not free_theta)  # free_theta: scale redundant with theta
    ls_lo, ls_hi = np.log(SCALE_BOUNDS[0]), np.log(SCALE_BOUNDS[1])
    le_lo, le_hi = np.log(eps_bounds[0]), np.log(eps_bounds[1])
    ref = [p.detach().clone() for p in (th, ba, ld, le, ls)]
    if d1_ref is not None:  # rappel targets other than the starting values (e.g. the literature)
        ref[2] = log_par(d1_ref, 1e-8, np.inf, False)
    if eps_ref is not None:
        ref[3] = log_par(eps_ref, *eps_bounds, False)
    ref[4] = torch.zeros_like(ls)  # scale rappel towards 1

    if prot_weight > 0:  # per-gene units of the protein distances (reference states at the observed times)
        allp = np.concatenate([g['X'][t][:, ns:] for g in groups for t in np.flatnonzero(g['start_ok'])])
        sdp = torch.tensor(np.maximum(allp.std(axis=0), 0.05 * scale_p), dtype=f64)
    w_term = dict(kon=kon_weight, mrna=mrna_weight,  # weights of the terms (initial: default as the modes)
                  init=kon_weight if init_weight is None else init_weight)
    # Annealing close to convergence: with geomloss's default (0.5) the gradients are biased, up to their sign
    sink = SamplesLoss('sinkhorn', p=2, blur=blur, scaling=sink_scaling, debias=True, backend=SINKHORN_BACKEND)
    tens = [dict(k_lo=torch.tensor(g['k_lo'], dtype=f64), k_hi=torch.tensor(g['k_hi'], dtype=f64),
                 k1=torch.tensor(np.maximum(g['k1'], 1e-8), dtype=f64),
                 binary=None if g.get('binary') is None or not np.any(g['binary'][ns:]) else
                 torch.tensor(np.asarray(g['binary'][ns:], dtype=bool)),
                 lag_w=None if g.get('lag') is None else torch.tensor(np.asarray(g['lag_w'], dtype=float)))
            for g in groups]

    def units(gi, g, scale=1.0):
        """Per-gene units of the kon (scale 1) and normalised mRNA (scale_p) distances: the highest mode of each gene
        (unit modes kon / k_hi, i.e. sigma (1 - k_lo / k_hi) up to a constant; mRNA positioned on the modes), genes
        out of the group's condition (gmask) left out."""
        u = scale * tens[gi]['k_hi'][ns:].clamp(min=1e-12)
        return u if g.get('gmask') is None else u / torch.tensor(g['gmask'][ns:], dtype=f64).clamp(min=1e-12)

    def signed_theta():
        return th * mask if free_theta else sgn * torch.relu(th) * mask

    def theta_of(i):
        t_ = signed_theta()
        return t_[i] if t_.dim() == 3 else t_

    def rates_of(tw, c):
        """d1, eps, scale (W, G) of kinetics c at the intervals tw (W,) of the windows."""
        i = tw if temporal else np.zeros_like(tw)
        return (torch.exp(ld[c, i]), torch.exp(le[c, i].clamp(le_lo, le_hi)), torch.exp(ls[c, i].clamp(ls_lo, ls_hi)))

    def kon_of(P, gi, s):
        """kon (W, n, G) of the states P (W, n, G) of group gi, scale of theta s (W, G) of each window."""
        g, c = groups[gi], tens[gi]
        return c['k_lo'] + (c['k_hi'] - c['k_lo']) * sigma_torch(P, theta_of(g['idx']), ba[g['idx']], s, spec, all_edges=free_theta)

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
                b = torch.clamp(torch.tensor(np.stack([np.asarray(o[0], float) for o in out]), dtype=f64), min=0.0)
            if out[0][1] is not None:
                dlogw = np.stack([np.asarray(o[1], float) for o in out]) * (times[tw + 1] - times[tw])[:, None]
        keep = gmask.bool()
        b_const = b.clone()
        lag = g.get('lag') if Kb is None else None
        hist = [P] if lag is not None else None  # states at the steps, for the lagged comparison
        for k in range(n_sub):
            kon = kon_of(P, gi, s)
            if Kb is not None and not mrna_dilution:  # mean mRNA (kon units) given the kon path: memory kernel of d0
                Kb = kon + (Kb - kon) * torch.exp(-(d / e) * h)
            if g['birth'] is not None:  # MLP birth at the current states (+ the constant part of 'nearest')
                Pn, kn = P.detach().numpy(), kon.detach().numpy()
                b = torch.clamp(b_const + torch.tensor(np.stack([np.asarray(g['birth'](Pn[w], kn[w], tw[w], sl[w]), float)
                                                               for w in range(W)]), dtype=f64), min=0.0)
            if Kb is not None and mrna_dilution:  # memory kernel of d0 + b (current birth rate)
                Kb = kon + (Kb - kon) * torch.exp(-(d / e + b[..., None]) * h)
            dp = d + b[..., None]                     # protein decay with dilution
            decay_p = torch.exp(-dp * h)
            if mode == 'sde2':
                d0 = d / e + b[..., None] if mrna_dilution else d / e  # mRNA relaxation (mrna_dilution: d0 + b)
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

    def level_grid(ml):
        """Sorted distinct levels of each target gene (Gt, L), padded with their highest level, and their counts."""
        lv = [np.unique(np.round(v, 12)) for v in np.asarray(ml, dtype=float)[ns:]]
        L = max(len(v) for v in lv)
        return np.stack([np.pad(v, (0, L - len(v)), mode='edge') for v in lv]), np.array([len(v) for v in lv])
    levels = [None if g.get('mode_levels') is None else level_grid(g['mode_levels']) for g in groups]

    def observed_kon(kon, gi, r):
        """Simulated kon (W, n, G - ns) read as the observed modes: drawn between the two levels of the mixture around
        it, the upper one with probability its relative position (mean kept; Bernoulli of sigma for two modes),
        straight-through gradient; without levels, genes of two modes ('binary') only."""
        if levels[gi] is not None:
            lv, nl = levels[gi]
            k = kon.detach().numpy()
            x = np.clip(k, lv[:, 0], lv[np.arange(len(nl)), nl - 1])
            i = np.clip((lv[:, 1:] <= x[..., None]).sum(-1), 0, np.maximum(nl - 2, 0))  # lower level index
            lo = np.take_along_axis(np.broadcast_to(lv, k.shape + lv.shape[-1:]), i[..., None], -1)[..., 0]
            hi = np.take_along_axis(np.broadcast_to(lv, k.shape + lv.shape[-1:]),
                                    np.minimum(i + 1, nl - 1)[..., None], -1)[..., 0]
            lo, hi = torch.tensor(lo, dtype=f64), torch.tensor(hi, dtype=f64)
            span = torch.clamp(hi - lo, min=1e-12)
            p = ((kon - lo) / span).clamp(0.0, 1.0)
            z = torch.tensor(r.random(tuple(kon.shape)) < p.detach().numpy(), dtype=f64)
            return lo + (hi - lo) * (z + p - p.detach())
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

    def init_targets(gi, g):
        """Initial states paired with their cells (slots 'valid0' at the first observed time) and their sigmoid targets
        tau (n0, G - ns): position of their mode between the lowest and highest ones, expected under the posteriors."""
        t_f = obs_t[gi][0]
        sl = np.flatnonzero(g['valid0']) if g.get('valid0') is not None else np.arange(len(g['targets'][t_f]))
        lo, hi = g['k_lo'][ns:], g['k_hi'][ns:]
        pos = lambda k: np.clip((k - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0)
        if g.get('post') is not None:
            tau = (g['post'][t_f][:, ns:] * pos(np.moveaxis(g['levels'][t_f][:, ns:], -1, 0)).transpose(1, 2, 0)).sum(-1)
        else:
            tau = pos(g['targets'][t_f][:, ns:])
        return sl, tau

    def init_term(gi, g, full):
        """Cross-entropy (as in fit_network) between the sigma of the initial states (stimulus 'stim0') and the
        sigmoid targets of their cells, gene weights 1 - k_lo / k_hi, mean over the cells."""
        sl, tau = g['tau0'] if g.get('tau0') is not None else init_tg[gi]
        i0 = np.arange(len(sl)) if full or len(sl) <= batch_size else rng.choice(len(sl), batch_size, replace=False)
        P0 = torch.tensor(g['X'][obs_t[gi][0]][sl[i0]], dtype=f64)[None]
        P0[..., :ns] = torch.tensor(np.asarray(g['stim0'], dtype=float), dtype=f64)
        s0 = rates_of(np.array([min(obs_t[gi][0], T1 - 1)]), g.get('cond', 0))[2]
        sig = sigma_torch(P0, theta_of(g['idx']), ba[g['idx']], s0, spec, all_edges=free_theta)[0, :, ns:]
        sig = sig.clamp(1e-6, 1 - 1e-6)
        t_ = torch.tensor(tau[i0], dtype=f64)
        ce = -(t_ * torch.log(sig) + (1 - t_) * torch.log(1 - sig))
        lo, hi = g['k_lo'][ns:], g['k_hi'][ns:]
        gw = np.clip(1.0 - lo / np.maximum(hi, 1e-12), 0.0, 1.0) * (1.0 if g.get('gmask') is None else g['gmask'][ns:])
        gw = torch.tensor(gw, dtype=f64)
        return (ce * gw).sum(-1).mean() / gw.sum().clamp(min=1e-12)

    def ot_ce(a, sig, b, tau, gw):
        """Entropic OT (B batches) between sigma (B, N, Gt) of weights a (B, N) and targets tau (B, M, Gt) of weights
        b (B, M), ground cost the weighted cross-entropy (gw (Gt,) summing to 1); value <pi, C>, gradient pi."""
        ls, l1s = torch.log(sig.clamp(1e-6, 1 - 1e-6)), torch.log1p(-sig.clamp(1e-6, 1 - 1e-6))
        C = -(ls @ (gw * tau).transpose(1, 2) + l1s @ (gw * (1 - tau)).transpose(1, 2))  # (B, N, M)
        la, lb = torch.log(a.clamp(min=1e-300)), torch.log(b)
        with torch.no_grad():
            Cd = C.detach()
            f, g_ = torch.zeros_like(la), torch.zeros_like(lb)
            e = max(float(Cd.max()), ce_eps)
            while True:
                g_ = -e * torch.logsumexp((f[:, :, None] - Cd) / e + la[:, :, None], dim=1)
                f = -e * torch.logsumexp((g_[:, None, :] - Cd) / e + lb[:, None, :], dim=2)
                if e <= ce_eps:
                    break
                e = max(e * sink_scaling, ce_eps)
            for _ in range(20):
                g_ = -e * torch.logsumexp((f[:, :, None] - Cd) / e + la[:, :, None], dim=1)
                f = -e * torch.logsumexp((g_[:, None, :] - Cd) / e + lb[:, None, :], dim=2)
            pi = torch.exp((f[:, :, None] + g_[:, None, :] - Cd) / e + la[:, :, None] + lb[:, None, :])
        return (pi * C).sum(dim=(1, 2))

    def mode_weights(gi, g):
        """Gene weights of the cross-entropy, 1 - k_lo / k_hi on the genes of the group's condition, summing to 1."""
        lo, hi = g['k_lo'][ns:], g['k_hi'][ns:]
        w = np.clip(1.0 - lo / np.maximum(hi, 1e-12), 0.0, 1.0) * (1.0 if g.get('gmask') is None else g['gmask'][ns:])
        return torch.tensor(w / max(w.sum(), 1e-12), dtype=f64)

    def windows_loss(p_sel, H, full=False):
        """Mean loss of the windows starting at the observed times of rank p_sel of each group and spanning H of its
        observed intervals (all of them if fewer), all windows of a group in one batch: one batched Sinkhorn per
        interval (targets resampled to a common size; full: fixed draws of the targets)."""
        loss, n_terms = 0.0, 0
        r_tg = np.random.default_rng(seed + 2) if full else rng
        for gi, g in enumerate(groups):
            ob = obs_t[gi]
            if w_term['init'] > 0 and len(ob):  # kon of the initial states (initial stimulus) against the initial modes
                loss = loss + w_term['init'] * init_term(gi, g, full)
                n_terms += 1
            Hg = min(H, len(ob) - 1)
            p_ok = np.array([p for p in p_sel if p + Hg < len(ob)], dtype=int) if Hg >= 1 else np.zeros(0, int)
            if not len(p_ok):
                continue
            a_ok, L = ob[p_ok], ob[p_ok + Hg] - ob[p_ok]  # starts and lengths (intervals of the time grid)
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
            for k in range(int(L.max())):
                if np.any(L <= k):  # finished windows leave the batch
                    keep = L > k
                    kt = torch.from_numpy(keep)
                    P, slots, logw, a_ok, L = P[kt], slots[keep], logw[keep], a_ok[keep], L[keep]
                    Mn = None if Mn is None else Mn[kt]
                    Kb = None if Kb is None else Kb[kt]
                    W = len(a_ok)
                tw = a_ok + k
                if bptt is not None and k > 0 and k % int(bptt) == 0:  # truncated backpropagation: states detached
                    P = P.detach()
                    Mn = None if Mn is None else Mn.detach()
                    Kb = None if Kb is None else Kb.detach()
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
                M = len(g['targets'][tw[obs[0]] + 1]) if full else batch_size
                sel = [rng.choice(len(g['targets'][tw[w] + 1]), M, replace=len(g['targets'][tw[w] + 1]) < M)
                       if not full else np.arange(M) for w in obs]
                Y = torch.tensor(np.stack([kon_targets(g, tw[w] + 1, i, r_tg) for w, i in zip(obs, sel)]), dtype=f64)
                wts = torch.softmax(torch.tensor(logw[obs], dtype=f64), dim=1)
                u = torch.full((len(obs), M), 1.0 / M, dtype=f64)
                if Kb is not None:
                    kon_c = Kb[obs][..., ns:]
                else:
                    P_sel = P_obs[obs] if tens[gi]['lag_w'] is None else P_obs[:, obs]
                    kon_c = kon_read(P_sel, gi, s[obs])[..., ns:]
                if w_term['kon'] > 0 and mode_cost == 'ce':  # continuous sigma against the positions of the modes
                    lo, hi = tens[gi]['k_lo'][ns:], tens[gi]['k_hi'][ns:]
                    span = torch.clamp(hi - lo, min=1e-12)
                    sig = ((kon_c - lo) / span).clamp(0.0, 1.0)
                    tau = ((Y - lo) / span).clamp(0.0, 1.0)
                    loss = loss + w_term['kon'] * ot_ce(wts, sig, u, tau, mode_weights(gi, g)).sum()
                elif w_term['kon'] > 0:
                    # Genes of the group's condition only (select_genes_per_condition): the others are not compared
                    gk = units(gi, g)
                    kon = observed_kon(kon_c, gi, r_tg)
                    loss = loss + w_term['kon'] * sink(wts, kon / gk, u, Y / gk).sum()
                fin = [j for j, w in enumerate(obs) if L[w] == k + 1]  # windows ending at this time
                if prot_weight > 0 and fin:  # proximity to the reference proteins at the end of the window
                    nX, wf = g['X'].shape[1], [obs[j] for j in fin]
                    sel_p = [rng.choice(nX, M, replace=nX < M) if not full else np.arange(nX) for _ in wf]
                    Yp = torch.tensor(np.stack([g['X'][tw[w] + 1][i][:, ns:] for w, i in zip(wf, sel_p)]), dtype=f64)
                    up = torch.full(Yp.shape[:2], 1.0 / Yp.shape[1], dtype=f64)
                    gp = sdp if g.get('gmask') is None else sdp / torch.tensor(g['gmask'][ns:], dtype=f64).clamp(min=1e-12)
                    loss = loss + prot_weight * sink(wts[fin], P[wf][..., ns:] / gp, up, Yp / gp).sum()
                if mode == 'sde2' and w_term['mrna'] > 0:  # observed mRNA (same cells)
                    Ym = torch.tensor(np.stack([g['M_targets'][tw[w] + 1][i][:, ns:] for w, i in zip(obs, sel)]),
                                      dtype=f64)
                    gm_ = units(gi, g, scale_p)
                    loss = loss + w_term['mrna'] * sink(wts, Mn[obs][..., ns:] / gm_, u, Ym / gm_).sum()
                n_terms += len(obs)
        return loss / max(n_terms, 1)

    def penalty():
        th_r, ba_r, ld_r, le_r, ls_r = ref
        p = 0.0
        if learn_theta:
            if lam_th is None:  # L2 rappel on theta, unless the L1
                den = mask.sum() if free_theta else (th_r ** 2 * mask).sum() + 1e-12
                p = p + lam_theta * ((th - th_r) ** 2 * mask).sum() / den
            den_b = ba[..., ns:].numel() if free_theta else (ba_r[..., ns:] ** 2).sum() + 1e-12
            p = p + lam_theta * ((ba - ba_r)[..., ns:] ** 2).sum() / den_b
        for par, r_, on in ((ld, ld_r, learn_d), (le, le_r, learn_eps), (ls, ls_r, ls.requires_grad)):
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

    # Observed times of each group: windows start there and count their observed intervals (virtual states are copies)
    obs_t = [np.flatnonzero(g['start_ok']) for g in groups]
    init_tg = [init_targets(gi, g) for gi, g in enumerate(groups)]
    n_obs = max(len(ob) for ob in obs_t)
    params = [p for p in (th, ba, ld, le, ls) if p.requires_grad]
    H_max = max(1, n_obs - 1 if max_horizon is None else min(int(max_horizon), n_obs - 1))
    horizons = [1]  # geometric curriculum up to the whole time course
    while horizons[-1] < H_max:
        horizons.append(min(H_max, max(horizons[-1] + 1, int(np.ceil(horizons[-1] * horizon_growth)))))
    starts_of = lambda H: np.zeros(1, int) if first_start_only else np.arange(n_obs - H)  # ranks of the starts

    def adam(net, rates):
        """Adam on the learned network parameters (lr) and rates / scales (lr_rates)."""
        groups_ = [{'params': [p for p in net if p.requires_grad], 'lr': lr},
                   {'params': [p for p in rates if p.requires_grad], 'lr': lr if lr_rates is None else lr_rates}]
        return torch.optim.Adam([pg for pg in groups_ if pg['params']])

    def train(opt, n_ep, prox=None):
        """Horizon curriculum (n_ep steps per horizon), proximal L1 step on theta after each step (prox)."""
        t_start = time.time()
        peak = [pg['lr'] for pg in opt.param_groups]
        hs = [H_max] if only_max_horizon else horizons
        for h, H in enumerate(hs):
            starts = starts_of(H)
            for i in range(int(n_ep)):
                f = lr_decay ** h * (lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + np.cos(np.pi * i / max(int(n_ep), 1))))
                for pg, p0 in zip(opt.param_groups, peak):
                    pg['lr'] = p0 * f
                a_sel = starts if n_windows is None or len(starts) <= n_windows else \
                    np.sort(rng.choice(starts, int(n_windows), replace=False))
                opt.zero_grad()
                loss = windows_loss(a_sel, H) + penalty()
                loss.backward()
                opt.step()
                if prox is not None:
                    prox(opt)
            if verb:
                print(f"[fit_free_running] horizon {H}/{H_max}: loss {float(loss):.4f} "
                      f"({time.time() - t_start:.0f} s)", flush=True)

    def null_draws(r):
        """Observed modes and mRNA of each group drawn from the current model, integrated from its initial states
        (all trajectories, resampled by their branching weights to the sizes of the observations); the modes of each
        cell read as in the fit (observed_kon)."""
        out = []
        with torch.no_grad():
            for gi, g in enumerate(groups):
                ob = obs_t[gi]
                as_modes = lambda kon: observed_kon(kon, gi, r)[0].numpy()
                t_f, n = ob[0], g['X'].shape[1]
                sl = np.arange(n)[None]
                P = torch.tensor(g['X'][t_f][None], dtype=f64)
                Mn = torch.tensor(g['M'][t_f][None], dtype=f64) if mode == 'sde2' else None
                Kb = torch.tensor(g['K0'][t_f][None], dtype=f64) if memory_filter else None
                logw = np.zeros((1, n))
                tg = [None] * T
                mt = [None] * T if mode == 'sde2' else None

                def record(t, kon, Mn_, logw):
                    m = len(g['targets'][t])
                    w = np.exp(logw[0] - logw[0].max())
                    i = r.choice(n, m, replace=True, p=w / w.sum())
                    k = np.zeros((m, G))
                    k[:, ns:] = kon[i]
                    tg[t] = k
                    if mt is not None:
                        mt[t] = Mn_[0][i].numpy()
                P0 = P.clone()
                P0[..., :ns] = torch.tensor(np.asarray(g['stim0'], dtype=float), dtype=f64)
                s0 = rates_of(np.array([min(t_f, T1 - 1)]), g.get('cond', 0))[2]
                record(t_f, as_modes(kon_of(P0, gi, s0)[..., ns:]), Mn, logw)
                sl0 = init_tg[gi][0]
                k0 = as_modes(kon_of(P0[:, sl0], gi, s0)[..., ns:])
                lo, hi = g['k_lo'][ns:], g['k_hi'][ns:]
                tau0 = (sl0, np.clip((k0 - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0))
                stim = torch.tensor(g['stim'], dtype=f64)
                for t in range(t_f, ob[-1]):
                    tw = np.array([t])
                    P = P.clone()
                    P[..., :ns] = stim[tw][:, None, :]
                    P, Mn, s_, dlogw, P_obs, Kb = run_interval(P, Mn, gi, tw, sl, Kb)
                    logw = logw + dlogw
                    if g['targets'][t + 1] is None or not len(g['targets'][t + 1]):
                        continue
                    P_obs = P_obs.clone()
                    P_obs[..., :ns] = stim[tw][:, None, :]
                    kon = Kb[..., ns:] if Kb is not None else kon_read(P_obs, gi, s_)[..., ns:]
                    record(t + 1, as_modes(kon), Mn, logw)
                out.append(dict(targets=tg, post=None, M_targets=mt, tau0=tau0))
        return out

    def qut_lambda(weight=None, seed_off=3):
        """lam (G,) per target: qut_level-quantile of max_g weight_gj |dL/dtheta_gj| (weight: 1, or |theta_1| of the
        adaptive lasso) at the current (null) model on null draws."""
        r = np.random.default_rng(seed + seed_off)
        saved = list(groups)
        cand = mask.detach().numpy() > 0
        cand[..., :ns] = False
        w = 1.0 if weight is None else weight
        stats = np.zeros((int(qut_reps), G))
        for k in range(int(qut_reps)):
            for gi, d in enumerate(null_draws(r)):
                groups[gi] = dict(saved[gi], **d)
            loss = windows_loss(starts_of(H_max), H_max, full=True)
            (gr,) = torch.autograd.grad(loss, th)
            a = np.where(cand, w * np.abs(gr.numpy()), 0.0)
            stats[k] = a.reshape(-1, G, G).max(axis=(0, 1))
        groups[:] = saved
        return np.quantile(stats, qut_level, axis=0)

    def adaptive_qut(w1):
        """lam (G,) of the adaptive lasso of weights 1 / w1 (w1 = |theta_1|): per target j, null model of j given the
        first pass (column j of theta at 0, basal of j refitted over null_epochs steps of the longest windows, the
        rest fixed), QUT of max_g w1_gj |dL/dtheta_gj| on its draws."""
        lam2 = np.zeros(G)
        th1, ba1 = th.detach().clone(), ba.detach().clone()
        for j in range(ns, G):
            if not np.any(w1[..., j] > 0):
                continue
            with torch.no_grad():
                th[..., j] = 0.0
            onehot = torch.zeros_like(ba)
            onehot[..., j] = 1.0
            hook = ba.register_hook(lambda gr_, m=onehot: gr_ * m)
            opt = torch.optim.Adam([ba], lr=lr)
            for _ in range(int(null_epochs)):
                opt.zero_grad()
                windows_loss(starts_of(H_max), H_max).backward()
                opt.step()
            hook.remove()
            th.grad = None
            wj = np.zeros_like(w1)
            wj[..., j] = w1[..., j]
            lam2[j] = qut_lambda(wj, seed_off=5 + j)[j]
            with torch.no_grad():
                th.copy_(th1)
                ba.copy_(ba1)
        return lam2

    def term_diag():
        """Diagnostic at the current model: gradient of each term (initial, modes, mRNA) on the stimulus edges and the
        basals, and mean unit modes / mRNA per observed time, drawn from the model against the observations."""
        saved = dict(w_term)
        p2 = lambda v: np.array2string(np.asarray(v), precision=4, max_line_width=200)
        for name in ('init', 'kon', 'mrna'):
            if saved[name] <= 0 or (name == 'mrna' and mode != 'sde2'):
                continue
            w_term.update({k: (saved[k] if k == name else 0.0) for k in w_term})
            gt, gb = torch.autograd.grad(windows_loss(starts_of(H_max), H_max, full=True), [th, ba])
            print(f"[term_diag] {name}: dL/dtheta stimulus {p2(gt.reshape(-1, G, G)[0, :ns, ns:].numpy())}, "
                  f"dL/dbasal {p2(gb.reshape(-1, G)[0, ns:].numpy())}", flush=True)
        w_term.update(saved)
        r = np.random.default_rng(seed + 4)
        for gi, (g, d) in enumerate(zip(groups, null_draws(r))):
            uk, um = units(gi, g).numpy(), units(gi, g, scale_p).numpy()
            for t in obs_t[gi]:
                obs_k = kon_targets(g, t, np.arange(len(g['targets'][t])), r).mean(0) / uk
                line = f"[term_diag] group {gi} t={times[t]:g}: modes sim {p2(d['targets'][t][:, ns:].mean(0) / uk)} " \
                       f"obs {p2(obs_k)}"
                if mode == 'sde2':
                    line += f" | mRNA sim {p2(d['M_targets'][t][:, ns:].mean(0) / um)} " \
                            f"obs {p2(g['M_targets'][t][:, ns:].mean(0) / um)}"
                print(line, flush=True)

    lam_th = None
    if learn_theta and (qut_level is not None or lam_l1 > 0):
        if qut_level is not None:
            if free_theta:  # null model: theta_null fixed, basal and rates fitted
                th.requires_grad_(False)
                train(adam([ba], [ld, le, ls]), null_epochs)
                th.requires_grad_(True)
            lam_th = qut_lambda()
            if verb:  # gradient of the observed data at the null model, against the threshold
                (gr,) = torch.autograd.grad(windows_loss(starts_of(H_max), H_max, full=True), th)
                a = np.where(mask.detach().numpy() > 0, np.abs(gr.numpy()), 0.0).reshape(-1, G, G).max(axis=0)
                a[:, :ns] = 0.0
                print(f"[fit_free_running] QUT (level {qut_level}) per target: "
                      + np.array2string(lam_th[ns:], precision=4) + "; data max |grad|: "
                      + np.array2string(a.max(axis=0)[ns:], precision=4)
                      + "; data |grad| stimulus: " + np.array2string(a[:ns, ns:].max(axis=0), precision=4), flush=True)
                if verb > 1:
                    term_diag()
        else:
            lam_th = np.full(G, float(lam_l1))
    th_null = torch.zeros_like(th) if free_theta else th.detach().clone()
    lam_t = None if lam_th is None else torch.tensor(lam_th, dtype=f64)
    lam_now = [lam_t]  # current L1 level per target, or per entry (adaptive lasso)

    def prox_l1(opt):
        """Soft threshold of theta - theta_null, preconditioned as the Adam step (exact zeros)."""
        st = opt.state[th]
        beta2, eps_a = opt.param_groups[0]['betas'][1], opt.param_groups[0]['eps']
        step = float(st['step'])
        lr_th = next(pg['lr'] for pg in opt.param_groups if any(q is th for q in pg['params']))
        thr = lr_th * lam_now[0] / (torch.sqrt(st['exp_avg_sq'] / (1 - beta2 ** step)) + eps_a)
        with torch.no_grad():
            dv = th - th_null
            th.copy_(th_null + torch.sign(dv) * torch.relu(dv.abs() - thr))
            if not free_theta:
                th.clamp_(min=0.0)

    hist = {'before': full_loss()}
    if params:
        train(adam([th, ba], [ld, le, ls]), epochs, prox=prox_l1 if lam_t is not None and learn_theta else None)
        if adaptive and free_theta and qut_level is not None and learn_theta:
            # Second pass: adaptive lasso on the support of the first one, at its QUT given the first pass
            w1 = np.abs(signed_theta().detach().numpy())
            w1[..., :ns] = 0.0
            lam2 = adaptive_qut(w1)
            sup = torch.tensor(w1 > 0, dtype=f64)
            with torch.no_grad():
                mask.copy_(sup)
                th.mul_(sup)
            lam_now[0] = torch.tensor(np.where(w1 > 0, lam2 / np.maximum(w1, 1e-12), 1e30), dtype=f64)
            hist['first'] = full_loss()
            if verb:
                print(f"[fit_free_running] adaptive lasso: {int((w1[..., ns:] > 0).sum())} edges of the first pass, "
                      f"QUT per target " + np.array2string(lam2[ns:], precision=4), flush=True)
            train(adam([th, ba], [ld, le, ls]), epochs, prox=prox_l1)
    hist['after'] = full_loss()

    full_t = lambda p, lo, hi: np.broadcast_to(np.exp(np.clip(p.detach().numpy(), lo, hi)), (n_cond, T1, G)).copy()
    theta_new = signed_theta().detach().numpy()
    return (theta_new, ba.detach().numpy(), full_t(ld, -np.inf, np.inf), full_t(le, le_lo, le_hi),
            full_t(ls, ls_lo, ls_hi), hist)
