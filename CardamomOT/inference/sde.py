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
With the 'kappa' of a group (sample), the protein synthesis of each model is multiplied by g = exp(kappa b) (mRNA
content of growing cells: d1 g kon, protein bursts g times larger, translation d1 g m).
Bursty PDMP and its diffusion limit: Herbach et al. 2017 (BMC Syst. Biol.); free-running training of a neural
SDE: Kidger et al. 2021 (ICML), with Sinkhorn divergences (Feydy et al. 2019, AISTATS; geomloss).
"""

import time

import numpy as np
import torch
from geomloss import SamplesLoss

from .sigma import sigma_torch
from .network import ORD_H, NULL_DEV_SD, ordinal_params, null_deviance, cost_code

try:  # geomloss 'auto' switches to the KeOps 'online' backend above 5000² pairs: needs pykeops
    import pykeops.torch  # noqa: F401
    SINKHORN_BACKEND = 'auto'
except Exception:
    SINKHORN_BACKEND = 'tensorized'

SCALE_BOUNDS = (0.2, 5.0)  # scale of theta per target gene: within [1/5, 5] (network neither erased nor blown up)


def fit_free_running(groups, times, ns, theta, basal, d1, eps, scale, mode='sde1', learn_theta=True,
                     learn_d=True, learn_eps=True, learn_scale=True, temporal=True, lam_theta=0.0, lam_rates=1.0,
                     eps_bounds=(1 / 50, 1 / 1.5), epochs=30, lr=1e-2, n_sub=20, batch_size=256, n_rep=2,
                     max_horizon=None, horizon_growth=2.0, n_windows=10, n_cond=1, d1_ref=None, eps_ref=None, only_max_horizon=False,
                     blur=0.05, noise='gamma', scale_p=1.0, seed=0, verb=False, spec=None, support_tol=0.0,
                     memory_filter=False, mrna_dilution=False, prot_weight=0.0, kon_weight=1.0,
                     mrna_weight=1.0, free_theta=False, lam_l1=0.0, first_start_only=False,
                     qut_level=None, qut_reps=20, null_epochs=10, sink_scaling=0.9, mode_cost='ce',
                     ce_eps=0.05, init_weight=None, lr_rates=None, lr_decay=1.0, lr_min_frac=1.0, bptt=None,
                     adaptive=False, qut_reps_adaptive=10, null_epochs_adaptive=3, qut_cells=None,
                     null_dev_weights=False, prolif_params=None, prolif_anchor=None, prolif_weight=1.0,
                     lr_prolif=1e-3):
    """
    groups: one dict per sample s with
        'X' (T, n, G) trajectory states (starts of the windows; mode 'sde2' also 'M' (T, n, G) normalised mRNA
        and 'M_targets' list of T arrays of observed normalised mRNA),
        'targets' list of T arrays (n_t, G) of observed kon (None: time not observed; the modes, or their mean under
        the posteriors: no draw, the simulated kon compared as it is), 'w0' (optional, (T, n)) log weights of the
        trajectory states as window starts (default 0), 'init_idx' (optional, (n0,)) row in 'targets' of the first
        observed time of the cell of each initial slot of 'valid0' (default: the rows are those slots), 'hard'
        (optional, as 'targets') the hard modes
        (basins) of the same cells, for the units / weights of null_dev_weights, 'wtargets' (optional, ordinal cost,
        list of T arrays (n_t, 2, G)) their expected warped targets (u1, u2) under the posteriors (soft targets),
        'k_lo', 'k_hi', 'k1' (G,),
        'stim' (T - 1, ns) stimulus of each interval, 'stim0' (ns,) stimulus at the first observed time, 'idx' index of its basal / network rows, 'cond' index of its
        kinetics (network condition, 0 .. n_cond - 1: d1, eps and scale per condition),
        'start_ok' (T,) bool times observed for the sample (windows start only there and span its observed intervals,
        virtual states being copies),
        'prolif' (or None): proliferation MLP of the sample, dict(fn: (P (W, n, G) tensor, tw (W,)) -> (birth, death)
        (W, n) tensors differentiable in its parameters, dil / br: birth in the dilution / net rate b - delta in the
        branching weights, lo / hi: bounds of its birth in the dilution), re-evaluated at every integration step,
        'nearest' f(P, kon, t, slots) -> (birth (n,) or None, net rate (n,) or None) or None: rates evaluated at the
        start of each interval only and kept over it, added to those of 'prolif' (rates of the nearest
        trajectory states without the MLP; effects of the stimulus with an MLP of the natural rates); kon at the
        stimulus of the time of the start states ('stim_own' (T, ns), as their observed modes) for the window starts,
        then the kon compared with the observed modes at the end of the previous interval (lagged, memory filter),
        'lag' (optional, (T - 1, K) hours per interval and class of targets) and 'lag_w' ((K, G) one-hot class of
        each gene): the observed modes of a target are compared with its kon at the proteins the lag of its class
        before each reached time (integration step nearest to it, the start of the interval at most).
    theta (G, G) or (S, G, G), basal (S, G): starting network (support |theta| > support_tol and signs kept,
    the other entries set to 0; the basal a starting value only, without rappel; lam_theta: anchor of theta to it,
    relative L2 sum (theta - theta_start)^2 / sum theta_start^2 on its support, with the L1);
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
    mrna_weight: weights of the Sinkhorn on the kon of the observed modes and ('sde2') on the observed mRNA, the gene
    weights of the end of ot_gene_weight_schedule in fit_network: positions tau = (kon - k_lo) / (k_hi - k_lo), genes
    equally weighted (mRNA in the same units, x scale_p); null_dev_weights (ot_gene_weight_schedule), in units of the
    standard deviation of the gene in each group (positions of the hard modes: 1 / (2 Var(tau)); normalised mRNA).
    free_theta: every entry of theta learned with its sign (network from scratch, e.g. null or random start; no
    scale of theta), else support and signs of theta kept. lam_l1: L1 on theta - theta_null
    (theta_null = 0 with free_theta, else the starting network), in units of the loss gradient, by a proximal step
    after each Adam step preconditioned as that step (soft threshold lr lam / (sqrt(v_hat) + eps)): exact zeros;
    replaces the L2 rappel on theta. qut_level: lam per target by the quantile universal threshold instead, as in
    fit_network: qut_level-quantile over qut_reps replicates of max_g |dL/dtheta_gj| at the null network (theta 0,
    basal of the null network: closed form, then null_epochs steps on the longest windows; with free_theta, basal and
    rates over the curriculum, start of the fit; else the starting network and basal restored after the QUT), the observed modes and mRNA of each target permuted across the observed
    cells of each group, all times together; the windows start from the same states. qut_cells: the QUT computed on
    a fixed subsample of qut_cells trajectories per window and observed cells per time (all window starts), its
    level rescaled by sqrt(qut_cells / n) (n the median number of observed cells per time; null noise of the
    gradient of a mean loss in 1 / sqrt(n)); None: all cells. sink_scaling: annealing
    ratio of the Sinkhorn iterations (geomloss 'scaling'): 0.9 brings the gradients within ~10 % of the converged
    ones (0.5 can flip their sign). mode_cost: 'euclidean', debiased Sinkhorn on the modes in the units of units();
    'ce' / 'ordinal', entropic OT whose ground cost is the network cost of fit_network (cross-entropy, or the ordinal cost with the
    central modes 'mid' (G,) of each group, -1 without) between the continuous sigma of the simulated states and the
    positions tau of the observed modes, on the scale of the euclidean cost (mode_weights: genes of the group's
    condition equally weighted; null_dev_weights, 1 / (2 null deviance) of the gene in that cost over the hard modes of
    the group, null_deviance), regularisation ce_eps (annealed down to it), gradient through the optimal plan. init_weight: weight of the initial
    term (None: that of the kon term). lr_rates: Adam learning rate of the rates and
    scales (None: lr); schedule of both: warm restarts at each horizon of the curriculum, peak x lr_decay at each new
    horizon, cosine decay down to lr_min_frac x the peak within the horizon (1: constant). bptt: truncated
    backpropagation, the simulated states detached every bptt intervals of a window (None: through the whole window). first_start_only: every window
    starts at the first observed time of each group (only the initial states of the trajectories are used), the
    curriculum lengthening it up to max_horizon. adaptive (free_theta with qut_level): second pass, adaptive lasso
    on the support of the first one, weights 1 / |theta_1| (warm start), at its own QUT per target j: qut_level-quantile
    of max_g |theta_1,gj| |dL/dtheta_gj| at the null model of j given the first pass (column j of theta set to 0, its
    basal refitted, the other targets as fitted; the observations of j permuted), whose simulated cells are
    heterogeneous, unlike the global null model (all sigma equal: entropic couplings that only see the mean of each
    regulator per time); the null models of all targets simulated together, as copies of the windows,
    qut_reps_adaptive replicates, basals refitted over null_epochs_adaptive steps.
    Always: + the cross-entropy (as in fit_network) between the sigma of the initial states (first observed time of
    each group, stimulus 'stim0' of the schedule) and the modes of their own cells (groups 'valid0' (n,) bool: slots
    of the observed cells at that time), with the weight of the kon term (init_weight): it sets the basal at the initial
    stimulus, hence the effect of the stimulus.
    prolif_params: parameters of the proliferation MLPs of the groups ('prolif'), learned with the rest (branching
    weights: gradient through the transport plans of the modes, the dual potentials in the weights; dilution: through
    the simulated proteins; the MLP inputs, the simulated states, are detached), prolif_anchor() their anchor (torch
    scalar, the pretraining loss of the MLPs on the trajectory states: net growth and prior birth / death), of weight
    prolif_weight (Adam learning rate lr_prolif, that of its pretraining); None: MLP fixed.
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
                 lag_w=None if g.get('lag') is None else torch.tensor(np.asarray(g['lag_w'], dtype=float)))
            for g in groups]

    def units(gi, g, mrna=False):
        """Per-gene units of the kon (mrna False) and normalised mRNA (mrna True) distances of group gi, as the gene
        weights of the end of ot_gene_weight_schedule: the span k_hi - k_lo (positions tau, genes equally weighted;
        mRNA positioned on the modes, x scale_p), with null_dev_weights the standard deviation of the gene in the
        group (sd_k / sd_m); genes out of the group's condition (gmask) left out."""
        if null_dev_weights:
            u = sd_m[gi] if mrna else sd_k[gi]
        else:
            u = (scale_p if mrna else 1.0) * (tens[gi]['k_hi'] - tens[gi]['k_lo'])[ns:].clamp(min=1e-12)
        return u if g.get('gmask') is None else u / torch.tensor(g['gmask'][ns:], dtype=f64).clamp(min=1e-12)

    def signed_theta():
        return th * mask if free_theta else sgn * torch.relu(th) * mask

    # Null models of the QUT: 'Z' (as th) added to theta (gradient at the null network); null copies of the adaptive
    # QUT: 'cols' (W,) target of each window, 'D' (G, G), 'b' (as ba)
    nctx = {}

    def theta_of(i):
        t_ = signed_theta() if nctx.get('Z') is None else signed_theta() + nctx['Z']
        return t_[i] if t_.dim() == 3 else t_

    def rates_of(tw, c):
        """d1, eps, scale (W, G) of kinetics c at the intervals tw (W,) of the windows."""
        i = tw if temporal else np.zeros_like(tw)
        return (torch.exp(ld[c, i]), torch.exp(le[c, i].clamp(le_lo, le_hi)), torch.exp(ls[c, i].clamp(ls_lo, ls_hi)))

    def kon_of(P, gi, s):
        """kon (W, n, G) of the states P (W, n, G) of group gi, scale of theta s (W, G) of each window."""
        g, c = groups[gi], tens[gi]
        sig = sigma_torch(P, theta_of(g['idx']), ba[g['idx']], s, spec, all_edges=free_theta or 'Z' in nctx)
        cols = nctx.get('cols')
        if cols is not None and len(cols) == P.shape[0]:  # null copies: column cols[w] of window w from (D, b)
            sig0 = sigma_torch(P, nctx['D'], nctx['b'][g['idx']], s, spec, all_edges=True)
            M = torch.zeros((P.shape[0], 1, G), dtype=f64)
            M[np.arange(P.shape[0]), 0, cols] = 1.0
            sig = sig * (1 - M) + sig0 * M
        return c['k_lo'] + (c['k_hi'] - c['k_lo']) * sig

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

    def run_interval(P, Mn, gi, tw, sl, Kb=None, Km=None):
        """Integration of the windows over their intervals [times[tw], times[tw + 1]] (n_sub steps each, exact
        relaxation over a step); P, Mn (W, n, G), sl (W, n) trajectory slots of the cells. Birth rate of the dilution
        and net rate of the branching: the proliferation MLP ('prolif', torch, differentiable in its parameters, inputs
        the current states detached) at every step, the net rate b - delta integrated along the steps (trapezoids),
        plus the rates of the nearest trajectory states ('nearest', constant over the interval). Kb (W, n, G) given
        (memory_filter): kon read on the mRNA, relaxed towards kon at the d0 = d1 / eps of the model. Km (W, n, G): kon
        of the start states matched with the trajectory states ('nearest'; default the kon of P at the stimulus of the
        interval). Returns (P, Mn, s, increment of the log-weights (W, n) (tensor), proteins compared with the observed
        modes: P without lag, else the proteins (K, W, n, G) the lag of each class before the end of each interval,
        Kb)."""
        g = groups[gi]
        W = P.shape[0]
        d, e, s = rates_of(tw, g.get('cond', 0))
        d, e = d[:, None, :], e[:, None, :]
        k1 = tens[gi]['k1']
        h_np = (times[tw + 1] - times[tw]) / n_sub
        h = torch.tensor(h_np, dtype=f64)[:, None, None]
        b_const, R_const = (torch.zeros(P.shape[:2], dtype=f64) for _ in range(2))
        dil = br = False
        if g.get('nearest') is not None:  # rates of the nearest trajectory states, constant over the interval
            Pn = P.detach().numpy()
            kn = (kon_of(P, gi, s) if Km is None else Km).detach().numpy()
            out = [g['nearest'](Pn[w], kn[w], tw[w], sl[w]) for w in range(W)]
            if out[0][0] is not None:
                b_const, dil = torch.tensor(np.stack([np.asarray(o[0], float) for o in out]), dtype=f64), True
            if out[0][1] is not None:
                R_const, br = torch.tensor(np.stack([np.asarray(o[1], float) for o in out]), dtype=f64), True
        pf = g.get('prolif')
        if pf is not None:
            dil, br = dil or pf['dil'], br or pf['br']

        def cell_rates(P_):
            """(birth of the dilution, net rate of the branching) at the states P_."""
            if pf is None:
                return b_const, R_const
            bm, dm = pf['fn'](P_.detach(), tw)
            return bm.clamp(pf['lo'], pf['hi']) + b_const, bm - dm + R_const
        b_cur, R_cur = cell_rates(P)
        dlogw = torch.zeros(P.shape[:2], dtype=f64)
        zero_b = torch.zeros(P.shape[:2], dtype=f64)
        keep = gmask.bool()
        lag = g.get('lag') if Kb is None else None
        hist = [P] if lag is not None else None  # states at the steps, for the lagged comparison
        for k in range(n_sub):
            kon = kon_of(P, gi, s)
            if Kb is not None and not mrna_dilution:  # mean mRNA (kon units) given the kon path: memory kernel of d0
                Kb = kon + (Kb - kon) * torch.exp(-(d / e) * h)
            b = torch.clamp(b_cur, min=0.0) if dil else zero_b
            if Kb is not None and mrna_dilution:  # memory kernel of d0 + b (current birth rate)
                Kb = kon + (Kb - kon) * torch.exp(-(d / e + b[..., None]) * h)
            dp = d + b[..., None]                     # protein decay with dilution
            decay_p = torch.exp(-dp * h)
            # Protein synthesis factor of growing cells (scalar 1 without: same graph, bit-identical gradients)
            gk = torch.exp(g['kappa'] * b)[..., None] if g.get('kappa') else 1.0
            if mode == 'sde2':
                d0 = d / e + b[..., None] if mrna_dilution else d / e  # mRNA relaxation (mrna_dilution: d0 + b)
                # mRNA bursts of rate kon k1 d0 and mean size 1 / k1 (normalised), proteins translated from them
                Mn = torch.relu(Mn * torch.exp(-d0 * h) + bursts(kon * k1 * d0, scale_p / k1, h)) * gmask
                P_new = P * decay_p + d * gk * Mn * (1 - decay_p) / dp
            elif mode == 'sde1':
                # protein bursts of rate kon k1 d0 and mean size d1 / (d0 k1) = eps / k1 (times g)
                P_new = P * decay_p + bursts(kon * k1 * d / e, scale_p * e / k1 * gk, h) * (1 - decay_p) / (dp * h)
            else:
                P_new = P * decay_p + d * gk * scale_p * kon * (1 - decay_p) / dp
            P = torch.relu(torch.where(keep, P_new, P))
            if hist is not None:
                hist.append(P)
            if pf is not None or br:
                b_new, R_new = cell_rates(P)
                if br:  # branching: weights exp(int R) along the steps
                    dlogw = dlogw + 0.5 * (R_cur + R_new) * h[..., 0]
                b_cur, R_cur = b_new, R_new
        if hist is None:
            return P, Mn, s, dlogw, P, Kb
        k_lag = np.clip(np.rint(np.asarray(lag)[tw] / h_np[:, None]).astype(int), 0, n_sub)  # (W, K) steps
        P_lag = torch.stack([torch.stack([hist[n_sub - k_lag[w, k]][w] for w in range(W)])
                             for k in range(k_lag.shape[1])])
        return P, Mn, s, dlogw, P_lag, Kb

    def kon_targets(g, t, i):
        """Observed kon (len(i), G - ns) of the cells i at time t."""
        return g['targets'][t][i][:, ns:]

    def warped_targets(g, t, i):
        """Expected warped targets (len(i), 2, G - ns) of the ordinal cost of the cells i at time t ('wtargets'),
        None without."""
        return None if mode_cost != 'ordinal' or g.get('wtargets') is None else g['wtargets'][t][i][..., ns:]

    def init_targets(gi, g):
        """Initial states paired with their cells (slots 'valid0' at the first observed time) and their sigmoid targets
        tau (n0, G - ns): position of their mode between the lowest and highest ones; ordinal cost with 'wtargets',
        their expected warped targets (n0, 2, G - ns)."""
        t_f = obs_t[gi][0]
        sl = np.flatnonzero(g['valid0']) if g.get('valid0') is not None else np.arange(len(g['targets'][t_f]))
        rows = g['init_idx'] if g.get('init_idx') is not None else slice(None)
        wt = warped_targets(g, t_f, rows)
        if wt is not None:
            return sl, np.array(wt, dtype=float)
        lo, hi = g['k_lo'][ns:], g['k_hi'][ns:]
        pos = lambda k: np.clip((k - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0)
        return sl, pos(g['targets'][t_f][rows][:, ns:])

    def init_term(gi, g, full, copies=None):
        """Initial states (stimulus 'stim0') against the modes of their own cells, with the ground cost of the modes
        at the other times, as the mean cost of this known pairing (the scale of the transport cost <pi, C> of the
        other times, pi summing to 1): mode_cost 'ce' / 'ordinal', network cost (as in fit_network) between sigma and
        the sigmoid targets, gene weights mode_weights; 'euclidean', |kon - Y|^2 / 2 in the units of units() (Y the
        observed kon; geomloss cost of p = 2).
        copies: every column from the null copies (D, b), each against the targets of its own copy (sum of the
        initial terms of the copies)."""
        sl, tau = g['tau0'] if g.get('tau0') is not None else init_tg[gi]
        if copies is not None and g.get('copy_draws') is not None:
            tau = tau.copy()
            for c, j in enumerate(copies):
                tau[..., j - ns] = g['copy_draws'][c]['tau0'][1][..., j - ns]
        i0 = np.arange(len(sl)) if full or len(sl) <= batch_size else rng.choice(len(sl), batch_size, replace=False)
        P0 = torch.tensor(g['X'][obs_t[gi][0]][sl[i0]], dtype=f64)[None]
        P0[..., :ns] = torch.tensor(np.asarray(g['stim0'], dtype=float), dtype=f64)
        s0 = rates_of(np.array([min(obs_t[gi][0], T1 - 1)]), g.get('cond', 0))[2]
        if copies is None:
            sig = sigma_torch(P0, theta_of(g['idx']), ba[g['idx']], s0, spec,
                              all_edges=free_theta or 'Z' in nctx)[0, :, ns:]
        else:
            sig = sigma_torch(P0, nctx['D'], nctx['b'][g['idx']], s0, spec, all_edges=True)[0, :, ns:]
        if mode_cost == 'euclidean':
            lo_t, hi_t = tens[gi]['k_lo'][ns:], tens[gi]['k_hi'][ns:]
            span = hi_t - lo_t
            kon0, Y = lo_t + span * sig, lo_t + span * torch.tensor(tau[i0], dtype=f64)
            return 0.5 * (((kon0 - Y) / units(gi, g)) ** 2).sum(-1).mean()
        ce = 0.0
        tg_ = tau[i0]
        for sg, tg, wk in warped(gi, sig, (tg_[:, 0], tg_[:, 1]) if tg_.ndim == 3 else tg_):
            sg = sg.clamp(1e-6, 1 - 1e-6)
            ce = ce + wk * -(tg * torch.log(sg) + (1 - tg) * torch.log(1 - sg))
        return (ce * mode_weights(gi, g)).sum(-1).mean()

    # Ordinal cost: the two warps of each group (ordinal_params of its central modes, identity for the cross-entropy)
    ordp = []
    for g in groups:
        mid = np.full(G - ns, -1.0) if mode_cost != 'ordinal' or g.get('mid') is None else \
            np.asarray(g['mid'], dtype=float)[ns:]
        m, a1, b1, a2, b2, has = ordinal_params(mid)
        ordp.append(dict(np=(m, a1, b1, a2, b2), has=has,
                         t=[torch.tensor(v, dtype=f64) for v in (m, a1, b1, a2, b2)],
                         has_t=torch.tensor(has.astype(float), dtype=f64)))

    def warp_t(x, m, sa, sb):
        """Warp of the ordinal cost (ordinal_consts, network.py) in torch, values only (differentiable)."""
        F = lambda y: sa * y + (sb - sa) * ORD_H * torch.nn.functional.softplus((y - m) / ORD_H)
        return (F(x) - F(torch.zeros_like(x))) / (F(torch.ones_like(x)) - F(torch.zeros_like(x)))

    def warped(gi, sig, tau):
        """[(sigma_k, tau_k, weight_k (Gt,))] of the network cost of group gi: the cross-entropy, (sigma, tau, 1); the
        ordinal cost, the two warps (the second one on the genes with a central mode only). tau numpy or torch, or for
        the ordinal cost a pair (u1, u2) of targets already warped (expected warped targets of soft targets)."""
        as_t = lambda x: x if torch.is_tensor(x) else torch.tensor(np.asarray(x, dtype=float), dtype=f64)
        if mode_cost != 'ordinal':
            return [(sig, as_t(tau), 1.0)]
        m, a1, b1, a2, b2 = ordp[gi]['t']
        t1, t2 = (as_t(tau[0]), as_t(tau[1])) if isinstance(tau, tuple) else \
            (warp_t(as_t(tau), m, a1, b1), warp_t(as_t(tau), m, a2, b2))
        return [(warp_t(sig, m, a1, b1), t1, 1.0), (warp_t(sig, m, a2, b2), t2, ordp[gi]['has_t'])]

    def ot_ce(a, sig, b, tau, gw, gi=0):
        """Entropic OT (B batches) between sigma (B, N, Gt) of weights a (B, N) and targets tau (B, M, Gt) of weights
        b (B, M), ground cost the weighted network cost (cross-entropy, or the ordinal cost: sum of the cross-entropies
        of the two warps of group gi, both linear in their targets; gw (Gt,), mode_weights); value <pi, C>, gradient
        pi in C and the dual potential f in the weights a (envelope of the entropic OT: the branching weights)."""
        C = 0.0
        for sg, tg, wk in warped(gi, sig, tau):
            w_ = gw * wk
            ls, l1s = torch.log(sg.clamp(1e-6, 1 - 1e-6)), torch.log1p(-sg.clamp(1e-6, 1 - 1e-6))
            C = C - (ls @ (w_ * tg).transpose(1, 2) + l1s @ (w_ * (1 - tg)).transpose(1, 2))  # (B, N, M)
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
        return (pi * C).sum(dim=(1, 2)) + (f * (a - a.detach())).sum(dim=1)

    # null_dev_weights, per group, on the positions tau of the hard modes of its observed cells ('hard': the basins, as
    # the OT of fit_network; 'targets' without), all times together, as the last stage of ot_gene_weight_schedule: the
    # cross-entropy / ordinal cost weighted by 1 / null deviance of the gene in that cost (null_deviance), the euclidean
    # one in units (k_hi - k_lo) max(std(tau), NULL_DEV_SD) (1 / (2 Var(tau)), same floor): a population of share p
    # missed costs the same whatever p. Normalised mRNA: its standard deviation in the group, floored at NULL_DEV_SD
    # scale_p (k_hi - k_lo)
    if null_dev_weights:
        w_dev, sd_k, sd_m = [], [], []
        for gi, g in enumerate(groups):
            span = np.maximum(g['k_hi'][ns:] - g['k_lo'][ns:], 1e-12)
            hard = g['hard'] if g.get('hard') is not None else g['targets']
            taus = np.clip((np.concatenate([a for a in hard if a is not None and len(a)])[:, ns:] - g['k_lo'][ns:])
                           / span, 0.0, 1.0)
            mid = np.where(ordp[gi]['has'], ordp[gi]['np'][0], -1.0)
            w_dev.append(1.0 / null_deviance(taus, cost_code(mode_cost), mid))
            sd_k.append(torch.tensor(span * np.maximum(taus.std(axis=0), NULL_DEV_SD), dtype=f64))
            if mode == 'sde2':
                mt = np.concatenate([m for m in g['M_targets'] if m is not None and len(m)])[:, ns:]
                sd_m.append(torch.tensor(np.maximum(mt.std(axis=0), NULL_DEV_SD * scale_p * span), dtype=f64))
            else:
                sd_m.append(None)

    def mode_weights(gi, g):
        """Gene weights (genes of the group's condition) of the cross-entropy / ordinal cost of the modes, on the scale
        of the euclidean one, sum_g omega_g 2 (sigma - tau)^2 (geomloss cost |x - y|^2 / 2 in the units of units(); the
        cross-entropy to second order at tau = 0.5): 1 / 4 as the euclidean cost on the positions tau; with
        null_dev_weights, 1 / (2 null deviance) as the euclidean 1 / (4 Var(tau))."""
        w = np.ones(G - ns) if g.get('gmask') is None else np.asarray(g['gmask'][ns:], dtype=float)
        return torch.tensor(w * w_dev[gi] / 2 if null_dev_weights else w / 4, dtype=f64)

    def windows_loss(p_sel, H, full=False, copies=None):
        """Mean loss of the windows starting at the observed times of rank p_sel of each group and spanning H of its
        observed intervals (all of them if fewer), all windows of a group in one batch: one batched Sinkhorn per
        interval (targets resampled to a common size; full: fixed draws of the targets). copies (C,) targets: every
        window repeated as C null copies (the copy c with the column copies[c] of the null model nctx, against the
        draws g['copy_draws'][c] when present); sum over the copies of their mean losses."""
        loss, n_terms = 0.0, 0
        C = 1 if copies is None else len(copies)
        for gi, g in enumerate(groups):
            ob = obs_t[gi]
            if w_term['init'] > 0 and len(ob):  # kon of the initial states (initial stimulus) against the initial modes
                loss = loss + w_term['init'] * init_term(gi, g, full, copies)
                n_terms += 1
            Hg = min(H, len(ob) - 1)
            p_ok = np.array([p for p in p_sel if p + Hg < len(ob)], dtype=int) if Hg >= 1 else np.zeros(0, int)
            if not len(p_ok):
                continue
            # Edge effect: each observed time weighted by the inverse of the number of windows of the horizon that
            # reach it (an end time is reached by one window, a middle one by Hg), so that every time counts equally
            cov = np.zeros(len(ob))
            for p in starts_of(Hg):
                if p + Hg < len(ob):
                    cov[p + 1:p + Hg + 1] += 1
            w_rank = np.where(cov > 0, cov[cov > 0].mean() / np.maximum(cov, 1), 0.0)
            w_time = dict(zip(ob, w_rank))
            a_ok, L = ob[p_ok], ob[p_ok + Hg] - ob[p_ok]  # starts and lengths (intervals of the time grid)
            W, n = len(a_ok), g['X'].shape[1]
            sub = nctx.get('sub') if full else None  # fixed subsample of the QUT (qut_cells)
            slots = np.stack([sub[gi][0] if sub is not None else
                              np.arange(n) if full or n <= batch_size else rng.choice(n, batch_size, replace=False)
                              for _ in range(W)])
            slots = np.repeat(slots, n_rep, axis=1)                       # (W, n_p)
            cidx = None
            if copies is not None:  # null copies of every window
                a_ok, L, slots = np.repeat(a_ok, C), np.repeat(L, C), np.repeat(slots, C, axis=0)
                cidx = np.tile(np.arange(C), len(p_ok))
                nctx['cols'] = np.asarray(copies)[cidx]
                W = len(a_ok)
            gv = (lambda w: g) if cidx is None or g.get('copy_draws') is None else \
                (lambda w: dict(g, **g['copy_draws'][cidx[w]]))
            P = torch.tensor(g['X'][a_ok[:, None], slots], dtype=f64)
            Mn = torch.tensor(g['M'][a_ok[:, None], slots], dtype=f64) if mode == 'sde2' else None
            # kon read on the mRNA (memory_filter): from the kon of the start states
            Kb = torch.tensor(g['K0'][a_ok[:, None], slots], dtype=f64) if memory_filter else None
            # Log weights of the starts (observed population: 1 / multiplicity of the real cell of each state)
            logw = torch.tensor(np.zeros(slots.shape) if g.get('w0') is None
                                else np.asarray(g['w0'])[a_ok[:, None], slots], dtype=f64)
            stim = torch.tensor(g['stim'], dtype=f64)
            Km = None  # kon of the start states matched with the trajectory states (nearest states)
            if g.get('nearest') is not None:
                with torch.no_grad():  # at the stimulus of their time (their observed modes); read on the mRNA
                    P0 = P.clone()
                    P0[..., :ns] = torch.tensor(g['stim_own'][a_ok], dtype=f64)[:, None, :]
                    Km = Kb.clone() if Kb is not None else kon_of(P0, gi, rates_of(a_ok, g.get('cond', 0))[2])
            for k in range(int(L.max())):
                if np.any(L <= k):  # finished windows leave the batch
                    keep = L > k
                    kt = torch.from_numpy(keep)
                    P, slots, logw, a_ok, L = P[kt], slots[keep], logw[kt], a_ok[keep], L[keep]
                    Mn = None if Mn is None else Mn[kt]
                    Kb = None if Kb is None else Kb[kt]
                    Km = None if Km is None else Km[kt]
                    W = len(a_ok)
                    if cidx is not None:
                        cidx = cidx[keep]
                        nctx['cols'] = np.asarray(copies)[cidx]
                tw = a_ok + k
                if bptt is not None and k > 0 and k % int(bptt) == 0:  # truncated backpropagation: states detached
                    P = P.detach()
                    Mn = None if Mn is None else Mn.detach()
                    Kb = None if Kb is None else Kb.detach()
                P = P.clone()
                P[..., :ns] = stim[tw][:, None, :]
                P, Mn, s, dlogw, P_obs, Kb = run_interval(P, Mn, gi, tw, slots, Kb, Km)
                P = P.clone()
                P[..., :ns] = stim[tw][:, None, :]
                P_obs = P_obs.clone()
                P_obs[..., :ns] = stim[tw][:, None, :]
                if Km is not None:  # kon at the end of the interval, as compared with its observed modes (own stimulus)
                    with torch.no_grad():
                        Km = Kb.clone() if Kb is not None else kon_read(P_obs, gi, s)
                logw = logw + dlogw
                obs = [w for w in range(W) if g['targets'][tw[w] + 1] is not None and len(g['targets'][tw[w] + 1])]
                if not obs:
                    continue
                if sub is not None:
                    M, sel = len(sub[gi][1][tw[obs[0]] + 1]), [sub[gi][1][tw[w] + 1] for w in obs]
                else:
                    M = len(g['targets'][tw[obs[0]] + 1]) if full else batch_size
                    sel = [rng.choice(len(g['targets'][tw[w] + 1]), M, replace=len(g['targets'][tw[w] + 1]) < M)
                           if not full else np.arange(M) for w in obs]
                Y = torch.tensor(np.stack([kon_targets(gv(w), tw[w] + 1, i) for w, i in zip(obs, sel)]),
                                 dtype=f64)
                wts = torch.softmax(logw[obs], dim=1)
                wv = torch.tensor([w_time[tw[w] + 1] for w in obs], dtype=f64)  # edge-effect weights
                u = torch.full((len(obs), M), 1.0 / M, dtype=f64)
                if Kb is not None:
                    kon_c = Kb[obs][..., ns:]
                else:
                    P_sel = P_obs[obs] if tens[gi]['lag_w'] is None else P_obs[:, obs]
                    kon_c = kon_read(P_sel, gi, s[obs])[..., ns:]
                if w_term['kon'] > 0 and mode_cost != 'euclidean':  # continuous sigma against the positions of the modes
                    lo, hi = tens[gi]['k_lo'][ns:], tens[gi]['k_hi'][ns:]
                    span = torch.clamp(hi - lo, min=1e-12)
                    sig = ((kon_c - lo) / span).clamp(0.0, 1.0)
                    tau = ((Y - lo) / span).clamp(0.0, 1.0)
                    if warped_targets(g, tw[obs[0]] + 1, sel[0]) is not None:  # soft targets: expected warped targets
                        TW = torch.tensor(np.stack([warped_targets(gv(w), tw[w] + 1, i) for w, i in zip(obs, sel)]),
                                          dtype=f64)
                        tau = (TW[:, :, 0], TW[:, :, 1])
                    loss = loss + w_term['kon'] * (wv * ot_ce(wts, sig, u, tau, mode_weights(gi, g), gi)).sum()
                elif w_term['kon'] > 0:
                    # Genes of the group's condition only (select_genes_per_condition): the others are not compared
                    gk = units(gi, g)
                    loss = loss + w_term['kon'] * (wv * sink(wts, kon_c / gk, u, Y / gk)).sum()
                fin = [j for j, w in enumerate(obs) if L[w] == k + 1]  # windows ending at this time
                if prot_weight > 0 and fin:  # proximity to the reference proteins at the end of the window
                    nX, wf = g['X'].shape[1], [obs[j] for j in fin]
                    sel_p = [rng.choice(nX, M, replace=nX < M) if not full else np.arange(nX) for _ in wf]
                    Yp = torch.tensor(np.stack([g['X'][tw[w] + 1][i][:, ns:] for w, i in zip(wf, sel_p)]), dtype=f64)
                    up = torch.full(Yp.shape[:2], 1.0 / Yp.shape[1], dtype=f64) if g.get('w0') is None else \
                        torch.softmax(torch.tensor(np.stack([np.asarray(g['w0'])[tw[w] + 1][i] for w, i in zip(wf, sel_p)]),
                                                   dtype=f64), dim=1)  # reference states as the observed population
                    gp = sdp if g.get('gmask') is None else sdp / torch.tensor(g['gmask'][ns:], dtype=f64).clamp(min=1e-12)
                    loss = loss + prot_weight * sink(wts[fin], P[wf][..., ns:] / gp, up, Yp / gp).sum()
                if mode == 'sde2' and w_term['mrna'] > 0:  # observed mRNA (same cells)
                    Ym = torch.tensor(np.stack([gv(w)['M_targets'][tw[w] + 1][i][:, ns:] for w, i in zip(obs, sel)]),
                                      dtype=f64)
                    gm_ = units(gi, g, mrna=True)
                    loss = loss + w_term['mrna'] * (wv * sink(wts, Mn[obs][..., ns:] / gm_, u, Ym / gm_)).sum()
                n_terms += float(wv.sum()) / C
        nctx['cols'] = None
        return loss / max(n_terms, 1)

    def penalty():
        th_r, _, ld_r, le_r, ls_r = ref
        p = 0.0
        if learn_theta and lam_theta > 0 and not free_theta:  # anchor to the starting network
            p = p + lam_theta * ((th - th_r) ** 2 * mask).sum() / ((th_r ** 2 * mask).sum() + 1e-12)
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
    pl = [] if prolif_params is None else [p for p in prolif_params if p.requires_grad]  # proliferation MLP
    H_max = max(1, n_obs - 1 if max_horizon is None else min(int(max_horizon), n_obs - 1))
    horizons = [1]  # geometric curriculum up to the whole time course
    while horizons[-1] < H_max:
        horizons.append(min(H_max, max(horizons[-1] + 1, int(np.ceil(horizons[-1] * horizon_growth)))))
    starts_of = lambda H: np.zeros(1, int) if first_start_only else np.arange(n_obs - H)  # ranks of the starts

    def adam(net, rates, prolif=False):
        """Adam on the learned network parameters (lr) and rates / scales (lr_rates); prolif: and on the parameters of
        the proliferation MLP (lr_prolif)."""
        groups_ = [{'params': [p for p in net if p.requires_grad], 'lr': lr},
                   {'params': [p for p in rates if p.requires_grad], 'lr': lr if lr_rates is None else lr_rates},
                   {'params': pl if prolif else [], 'lr': lr_prolif}]
        return torch.optim.Adam([pg for pg in groups_ if pg['params']])

    def train(opt, n_ep, prox=None, prolif=False):
        """Horizon curriculum (n_ep steps per horizon), proximal L1 step on theta after each step (prox); prolif: the
        proliferation MLP learned too, with its anchor (prolif_weight x prolif_anchor())."""
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
                if prolif and prolif_anchor is not None and prolif_weight > 0:
                    loss = loss + prolif_weight * prolif_anchor().to(f64)
                loss.backward()
                opt.step()
                if prox is not None:
                    prox(opt)
            if verb:
                print(f"[fit_free_running] horizon {H}/{H_max}: loss {float(loss):.4f} "
                      f"({time.time() - t_start:.0f} s)", flush=True)

    def null_draws(r):
        """Observed modes and mRNA of each group drawn from the current model, integrated from its initial states
        (all trajectories, resampled by their branching weights to the sizes of the observations): diagnostic."""
        out = []
        C = 1
        with torch.no_grad():
            for gi, g in enumerate(groups):
                ob = obs_t[gi]
                as_modes = lambda kon: kon.numpy()
                t_f, n = ob[0], g['X'].shape[1]
                sl = np.repeat(np.arange(n)[None], C, axis=0)
                rep = lambda a: torch.tensor(np.repeat(a[None], C, axis=0), dtype=f64)
                P = rep(g['X'][t_f])
                Mn = rep(g['M'][t_f]) if mode == 'sde2' else None
                Kb = rep(g['K0'][t_f]) if memory_filter else None
                logw = np.zeros((C, n)) if g.get('w0') is None else np.repeat(np.asarray(g['w0'])[t_f][None], C, axis=0)
                tg = [[None] * T for _ in range(C)]
                mt = [[None] * T for _ in range(C)] if mode == 'sde2' else None

                def record(t, kon, Mn_, logw):
                    m = len(g['targets'][t])
                    for c in range(C):
                        w = np.exp(logw[c] - logw[c].max())
                        i = r.choice(n, m, replace=True, p=w / w.sum())
                        k = np.zeros((m, G))
                        k[:, ns:] = kon[c][i]
                        tg[c][t] = k
                        if mt is not None:
                            mt[c][t] = Mn_[c][i].numpy()
                P0 = P.clone()
                P0[..., :ns] = torch.tensor(np.asarray(g['stim0'], dtype=float), dtype=f64)
                s0 = rates_of(np.full(C, min(t_f, T1 - 1)), g.get('cond', 0))[2]
                record(t_f, as_modes(kon_of(P0, gi, s0)[..., ns:]), Mn, logw)
                sl0 = init_tg[gi][0]
                k0 = as_modes(kon_of(P0[:, sl0], gi, s0)[..., ns:])
                lo, hi = g['k_lo'][ns:], g['k_hi'][ns:]
                tau0 = [(sl0, np.clip((k0[c] - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0)) for c in range(C)]
                stim = torch.tensor(g['stim'], dtype=f64)
                for t in range(t_f, ob[-1]):
                    tw = np.full(C, t)
                    P = P.clone()
                    P[..., :ns] = stim[tw][:, None, :]
                    P, Mn, s_, dlogw, P_obs, Kb = run_interval(P, Mn, gi, tw, sl, Kb)
                    logw = logw + dlogw.numpy()
                    if g['targets'][t + 1] is None or not len(g['targets'][t + 1]):
                        continue
                    P_obs = P_obs.clone()
                    P_obs[..., :ns] = stim[tw][:, None, :]
                    kon = Kb[..., ns:] if Kb is not None else kon_read(P_obs, gi, s_)[..., ns:]
                    record(t + 1, as_modes(kon), Mn, logw)
                draws = [dict(targets=tg[c], M_targets=None if mt is None else mt[c], tau0=tau0[c])
                         for c in range(C)]
                out.append(draws[0])
        return out

    def permuted(gi, g, js, r):
        """Observations of group gi with the modes and mRNA of each gene of js permuted across its
        observed cells, all times together (null network: time-invariant, independent genes), and their initial
        targets (tau0)."""
        ts = [t for t in range(T) if g['targets'][t] is not None and len(g['targets'][t])]
        cut = np.cumsum([len(g['targets'][t]) for t in ts])[:-1]
        perms = {j: r.permutation(int(sum(len(g['targets'][t]) for t in ts))) for j in js}
        out = {}
        for key in ('targets', 'M_targets', 'wtargets'):
            if g.get(key) is None:
                continue
            A = np.concatenate([g[key][t] for t in ts])
            for j, p in perms.items():
                A[:, ..., j] = A[p, ..., j]
            out[key] = [None] * T
            for t, part in zip(ts, np.split(A, cut)):
                out[key][t] = part
        out['tau0'] = init_targets(gi, dict(g, **out))
        return out

    def null_basal(b=None):
        """Basal of the null network in closed form, written in b (default ba): at theta 0 all the simulated sigma of a
        target are equal, and the cross-entropy is minimal at the mean position of its observed modes (all observed
        times of the samples of each row of the basal), sigma = that mean."""
        b = ba if b is None else b
        acc = {}
        for gi, g in enumerate(groups):
            lo, hi = g['k_lo'][ns:], g['k_hi'][ns:]
            pos = lambda k: np.clip((k - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0)
            for t in range(T):
                if g['targets'][t] is None or not len(g['targets'][t]):
                    continue
                tau = pos(g['targets'][t][:, ns:])
                a_ = acc.setdefault(g['idx'], [0.0, 0])
                a_[0], a_[1] = a_[0] + tau.sum(0), a_[1] + len(tau)
        s_ = torch.exp(ls.detach().clamp(ls_lo, ls_hi)).mean(dim=(0, 1)).numpy()[ns:]
        with torch.no_grad():
            for i, (tot, n) in acc.items():
                m = np.clip(tot / max(n, 1), 1e-3, 1 - 1e-3)
                b[i, ns:] = torch.tensor(np.log(m / (1 - m)) / s_, dtype=f64)

    def null_grad():
        """|dL/dtheta| (as th) at the null network (Z = 0 added to theta = 0) on the current observations."""
        (gz,) = torch.autograd.grad(windows_loss(starts_of(H_max), H_max, full=True), nctx['Z'])
        return np.abs(gz.numpy())

    def qut_subsample():
        """Fixed subsample of the QUT (qut_cells) in nctx['sub'] and the factor of its level, sqrt(qut_cells / n);
        nothing (factor 1) when every group has at most qut_cells cells per time."""
        counts = [len(g['targets'][t]) for g in groups for t in range(T)
                  if g['targets'][t] is not None and len(g['targets'][t])]
        n_med = float(np.median(counts)) if counts else 0.0
        if qut_cells is None or n_med <= qut_cells:
            return 1.0
        m, r = int(qut_cells), np.random.default_rng(seed + 7)
        sub = {}
        for gi, g in enumerate(groups):
            n = g['X'].shape[1]
            tg = {t: r.choice(len(g['targets'][t]), m, replace=len(g['targets'][t]) < m) for t in range(T)
                  if g['targets'][t] is not None and len(g['targets'][t])}
            sub[gi] = (np.sort(r.choice(n, min(m, n), replace=False)), tg)
        nctx['sub'] = sub
        return float(np.sqrt(m / n_med))

    def qut_lambda():
        """lam (G,) per target: qut_level-quantile of max_g |dL/dtheta_gj| at the null network (theta 0, current basal)
        on observations of each target permuted across its observed cells (fit_network's QUT)."""
        r = np.random.default_rng(seed + 3)
        saved = list(groups)
        cand = mask.detach().numpy() > 0
        cand[..., :ns] = False
        stats = np.zeros((int(qut_reps), G))
        for k in range(int(qut_reps)):
            for gi, g in enumerate(saved):
                groups[gi] = dict(g, **permuted(gi, g, range(ns, G), r))
            stats[k] = np.where(cand, null_grad(), 0.0).reshape(-1, G, G).max(axis=(0, 1))
        groups[:] = saved
        return np.quantile(stats, qut_level, axis=0)

    def adaptive_qut(w1):
        """lam (G,) of the adaptive lasso of weights 1 / w1 (w1 = |theta_1|): per target j, null model of j given the
        first pass (column j of theta at 0, basal of j that of the null network: closed form, then
        null_epochs_adaptive steps of the longest windows; the rest fixed), QUT of max_g w1_gj |dL/dtheta_gj| on the observations of j permuted; the null
        models of all targets simulated together (null copies of the windows, nctx)."""
        lam2 = np.zeros(G)
        w1m = np.asarray(w1).reshape(-1, G, G).max(axis=0)
        copies = np.array([j for j in range(ns, G) if np.any(w1m[:, j] > 0)], dtype=int)
        if not len(copies):
            return lam2
        D = torch.zeros((G, G), dtype=f64, requires_grad=True)  # null columns: theta 0, gradient of the statistic
        bn = ba.detach().clone()                                 # null basals (column j used by copy j only)
        null_basal(bn)
        bn.requires_grad_(True)
        nctx.update(D=D, b=bn, cols=None)
        opt = torch.optim.Adam([bn], lr=lr)
        for _ in range(int(null_epochs_adaptive)):
            opt.zero_grad()
            (bn.grad,) = torch.autograd.grad(windows_loss(starts_of(H_max), H_max, copies=copies), [bn])
            opt.step()
        r = np.random.default_rng(seed + 5)
        saved = list(groups)
        stats = np.zeros((int(qut_reps_adaptive), G))
        for k in range(int(qut_reps_adaptive)):
            for gi, g in enumerate(saved):
                groups[gi] = dict(g, copy_draws=[permuted(gi, g, [j], r) for j in copies])
            (gD,) = torch.autograd.grad(windows_loss(starts_of(H_max), H_max, full=True, copies=copies), [D])
            stats[k] = (w1m * np.abs(gD.numpy())).max(axis=0)
        groups[:] = saved
        nctx.clear()
        lam2[copies] = np.quantile(stats, qut_level, axis=0)[copies]
        return lam2

    def term_diag():
        """Diagnostic at the current model: gradient of each term (initial, modes, mRNA) on the stimulus edges and the
        basals, and mean modes / mRNA (units of units()) per observed time, drawn from the model against the
        observations."""
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
            uk, um = units(gi, g).numpy(), units(gi, g, mrna=True).numpy()
            for t in obs_t[gi]:
                obs_k = kon_targets(g, t, np.arange(len(g['targets'][t]))).mean(0) / uk
                line = f"[term_diag] group {gi} t={times[t]:g}: modes sim {p2(d['targets'][t][:, ns:].mean(0) / uk)} " \
                       f"obs {p2(obs_k)}"
                if mode == 'sde2':
                    line += f" | mRNA sim {p2(d['M_targets'][t][:, ns:].mean(0) / um)} " \
                            f"obs {p2(g['M_targets'][t][:, ns:].mean(0) / um)}"
                print(line, flush=True)

    lam_th = None
    if learn_theta and (qut_level is not None or lam_l1 > 0):
        if qut_level is not None:
            # Null network: theta 0, basal fitted (and the rates with free_theta; else start restored after the QUT)
            th0_, ba0_ = th.detach().clone(), ba.detach().clone()
            with torch.no_grad():
                th.zero_()
            th.requires_grad_(False)
            null_basal()
            if free_theta:  # start of the fit: basal and rates of the null network over the curriculum
                train(adam([ba], [ld, le, ls]), null_epochs)
            else:  # basal of the null network only (as qut_penalty of fit_network), longest windows
                opt_b = torch.optim.Adam([ba], lr=lr)
                for _ in range(int(null_epochs)):
                    opt_b.zero_grad()
                    (ba.grad,) = torch.autograd.grad(windows_loss(starts_of(H_max), H_max), [ba])
                    opt_b.step()
            nctx['Z'] = torch.zeros_like(th, requires_grad=True)
            f_sub = qut_subsample()
            lam_th = qut_lambda() * f_sub
            if verb:  # gradient of the observed data at the null network, against the threshold
                a = np.where(mask.detach().numpy() > 0, null_grad(), 0.0).reshape(-1, G, G).max(axis=0)
                a[:, :ns] = 0.0
            del nctx['Z']
            nctx.pop('sub', None)
            th.requires_grad_(True)
            if not free_theta:
                with torch.no_grad():
                    th.copy_(th0_)
                    ba.copy_(ba0_)
            if verb:
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
    if params or pl:
        train(adam([th, ba], [ld, le, ls], prolif=True), epochs,
              prox=prox_l1 if lam_t is not None and learn_theta else None, prolif=bool(pl))
        if adaptive and free_theta and qut_level is not None and learn_theta:
            # Second pass: adaptive lasso on the support of the first one, at its QUT given the first pass
            w1 = np.abs(signed_theta().detach().numpy())
            w1[..., :ns] = 0.0
            f_sub = qut_subsample()
            lam2 = adaptive_qut(w1) * f_sub
            nctx.pop('sub', None)
            sup = torch.tensor(w1 > 0, dtype=f64)
            with torch.no_grad():
                mask.copy_(sup)
                th.mul_(sup)
            lam_now[0] = torch.tensor(np.where(w1 > 0, lam2 / np.maximum(w1, 1e-12), 1e30), dtype=f64)
            hist['first'] = full_loss()
            if verb:
                print(f"[fit_free_running] adaptive lasso: {int((w1[..., ns:] > 0).sum())} edges of the first pass, "
                      f"QUT per target " + np.array2string(lam2[ns:], precision=4), flush=True)
            train(adam([th, ba], [ld, le, ls], prolif=True), epochs, prox=prox_l1, prolif=bool(pl))
    hist['after'] = full_loss()

    full_t = lambda p, lo, hi: np.broadcast_to(np.exp(np.clip(p.detach().numpy(), lo, hi)), (n_cond, T1, G)).copy()
    theta_new = signed_theta().detach().numpy()
    return (theta_new, ba.detach().numpy(), full_t(ld, -np.inf, np.inf), full_t(le, le_lo, le_hi),
            full_t(ls, ls_lo, ls_hi), hist)
