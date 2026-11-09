"""
Transition constraint of the optimal transport between cell types (transition_rates, TRANSk of perturbation_inference).

Population model of the types over an interval of length dt (row abundances m' = m G): G = Q - diag(Q 1) + diag(r),
Q[a, b] the transition rate a -> b over the interval (rates of transition_rates + stimulus effects), r[a] the net
growth rate of type a over the interval. Forward split P[a, b] = [exp(G dt)]_ab / sum_b' [exp(G dt)]_ab': share of
type b among the descendants of a cell of type a. Only the split is imposed, not the total mass of each source type,
which stays with the marginals (growth).

Plan pi_ij = u_i K_ij W[a(i), b(j)] v_j (K = exp(-C / reg), times mu nu^T in the unbalanced case, as POT): the
entropic plan with, besides the marginal terms, the penalty lambda sum_a S_a. KL(F_a || P_a) on the split F_a of the
block masses S_ab = sum_{i in a, j in b} pi_ij, lambda = reg f_T / (1 - f_T). Besides the updates of u and v (POT
sinkhorn / sinkhorn_unbalanced), each iteration sets the blocks by the proximal step of that penalty (scaling algorithm
of Chizat et al. 2018, Math Comp 87:2563): W_ab = (P_ab m_a / S0_ab)^f_T, S0 the block masses of the plan without W,
m_a = (sum_b S0_ab^(1 - f_T) P_ab^f_T)^(1 / (1 - f_T)). f_T = 1: KL projection on {S_ab = P_ab S_a.} (iterative Bregman
projections, Benamou et al. 2015, SIAM J Sci Comput 37:A1111), m_a = exp sum_b P_ab log(S0_ab / P_ab); f_T = 0: POT's
plan. Source types none of whose reachable types (P_ab > REACH) is in the target batch are left free (the model of the
types cannot tell where they go). Relaxed constraint (f_T < 1): the transitions the model forbids keep the probability
FLOOR_RELAXED (penalised by lambda log(F / FLOOR_RELAXED), w setting how much they may occur); exact split (f_T = 1):
forbidden, target types no source of the batch can reach being left free.
"""
import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp

FLOOR = 1e-12
FLOOR_RELAXED = 1e-3   # probability of the transitions the model forbids under the relaxed constraint
REACH = 1e-9   # P_ab above which type b is reachable from type a


def forward_split(Q, r, dt):
    """(K, K) forward split of the descendants of each type after dt: rows of exp(G dt), G = Q - diag(Q 1) + diag(r)
    (diagonal of Q ignored), normalised."""
    Q = np.array(Q, dtype=float)
    np.fill_diagonal(Q, 0.0)
    E = expm((Q - np.diag(Q.sum(axis=1)) + np.diag(np.asarray(r, dtype=float))) * dt)
    E = np.clip(E, 0.0, None)
    return E / np.maximum(E.sum(axis=1, keepdims=True), 1e-300)


def _targets(P, src_type, tgt_type, K, relaxed=False):
    """
    Constrained blocks of a batch: (split P of each source type restricted to the constrained target types,
    renormalised and floored (K, K), constrained source types (K,), constrained target types (K,)). A source type is
    constrained if present with a reachable target type in the batch. relaxed (f_T < 1): the transitions the model
    forbids (or makes rarer than FLOOR_RELAXED) keep the probability FLOOR_RELAXED, penalised but possible; exact
    split (f_T = 1): they stay forbidden, and a target type is constrained if fed by a source (a constrained source
    type reaching it, or a free one).
    """
    reach = np.asarray(P) > REACH
    tgt_ok = np.bincount(tgt_type, minlength=K) > 0
    src_in = np.bincount(src_type, minlength=K) > 0
    src_ok = src_in & (reach & tgt_ok[None, :]).any(axis=1)
    if not relaxed and not (src_in & ~src_ok).any():
        tgt_ok &= (reach & src_ok[:, None]).any(axis=0)
    Pm = np.where(tgt_ok[None, :], P, 0.0)
    Pm = Pm / np.maximum(Pm.sum(axis=1, keepdims=True), 1e-300)
    if relaxed:
        Pm = np.where(tgt_ok[None, :], np.maximum(Pm, FLOOR_RELAXED), 0.0)
        Pm = Pm / np.maximum(Pm.sum(axis=1, keepdims=True), 1e-300)
    return np.maximum(Pm, FLOOR), src_ok, tgt_ok


def _block_log_w(logS0, logP, src_ok, tgt_ok, f_T):
    """log W of the blocks: f_T (log P_ab + log m_a - log S0_ab) on the constrained blocks (proximal step of the split
    penalty, see the module), 0 elsewhere."""
    lw = np.zeros_like(logS0)
    b = tgt_ok
    for a in np.flatnonzero(src_ok):
        ls, lp = logS0[a, b], logP[a, b]
        if f_T >= 1.0 - 1e-6:   # KL projection (limit of m_a)
            lm = np.sum(np.exp(lp) * (ls - lp))
        else:
            lm = logsumexp((1.0 - f_T) * ls + f_T * lp) / (1.0 - f_T)
        lw[a, b] = f_T * (lp + lm - ls)
    return lw


def sinkhorn_blocks(mu, nu, C, reg, reg_m, src_type, tgt_type, P, f_T, n_iter=100000, tol=1e-9):
    """
    Entropic OT plan (balanced if reg_m is None, else unbalanced with KL marginals of weights reg_m and entropy
    relative to mu nu^T, as POT) with the blocks of types (src_type (n,), tgt_type (m,): indices < K) rescaled towards
    the forward split P (K, K) with exponent f_T. Scaling form, rows of the cost shifted by their minimum as
    NetworkModel._solve_ot (source masses compensating exactly); None on numerical failure (sinkhorn_blocks_log then).
    """
    K_types = P.shape[0]
    c_r = C.min(axis=1)
    if reg_m is not None and np.max(c_r) / reg_m[0] >= 500:
        return None
    Kr = np.exp(-(C - c_r[:, None]) / reg)
    if reg_m is None:
        a, b, f1, f2 = mu, nu, 1.0, 1.0
    else:
        a, b = mu * np.exp(-c_r / reg_m[0]), nu
        Kr = Kr * (mu[:, None] * nu[None, :])
        f1, f2 = reg_m[0] / (reg_m[0] + reg), reg_m[1] / (reg_m[1] + reg)
    Pm, src_ok, tgt_ok = _targets(P, src_type, tgt_type, K_types, relaxed=f_T < 1.0)
    logP = np.log(Pm)
    S1 = np.eye(K_types)[src_type]          # (n, K) indicator of the source type
    T1 = np.eye(K_types)[tgt_type]          # (m, K) indicator of the target type
    W = np.ones((K_types, K_types))
    u, v = np.ones(len(a)), np.ones(len(b))
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        Kv = Kr @ (v[:, None] * T1)          # (n, K): K v restricted to each target type
        for it in range(n_iter):
            u_prev, v_prev, W_prev = u, v, W
            u = (a / np.maximum((Kv * W[src_type]).sum(axis=1), 1e-300)) ** f1
            Ktu = Kr.T @ (u[:, None] * S1)       # (m, K): K^T u restricted to each source type
            v = (b / np.maximum((Ktu * W[:, tgt_type].T).sum(axis=1), 1e-300)) ** f2
            Kv = Kr @ (v[:, None] * T1)
            if f_T > 0:
                S0 = S1.T @ (u[:, None] * Kv)    # block masses without W
                W = np.exp(_block_log_w(np.log(np.maximum(S0, 1e-300)), logP, src_ok, tgt_ok, f_T))
            if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v)) and np.all(np.isfinite(W))):
                return None
            if it % 10 == 0:
                rel = lambda x, y: np.max(np.abs(x - y) / np.maximum(np.abs(x), 1e-300))
                if max(rel(u, u_prev), rel(v, v_prev), rel(W, W_prev)) < tol:
                    break
    plan = u[:, None] * Kr * W[src_type][:, tgt_type] * v[None, :]
    if not np.all(np.isfinite(plan)) or np.any(plan.sum(axis=1) <= 0):
        return None
    return plan


def sinkhorn_blocks_log(mu, nu, C, reg, reg_m, src_type, tgt_type, P, f_T, n_iter=100000, tol=1e-7):
    """Same plan as sinkhorn_blocks in the log domain (slower, never underflows): log potentials f, g, w."""
    K_types = P.shape[0]
    la, lb = np.log(np.maximum(mu, 1e-300)), np.log(np.maximum(nu, 1e-300))
    lK = -C / reg
    f1, f2 = 1.0, 1.0
    if reg_m is not None:
        lK = lK + la[:, None] + lb[None, :]
        f1, f2 = reg_m[0] / (reg_m[0] + reg), reg_m[1] / (reg_m[1] + reg)
    Pm, src_ok, tgt_ok = _targets(P, src_type, tgt_type, K_types, relaxed=f_T < 1.0)
    logP = np.log(Pm)
    f, g, w = np.zeros(len(mu)), np.zeros(len(nu)), np.zeros((K_types, K_types))
    for _ in range(n_iter):
        E = lK + w[src_type][:, tgt_type]
        f_new = f1 * (la - logsumexp(E + g[None, :], axis=1))
        g_new = f2 * (lb - logsumexp(E + f_new[:, None], axis=0))
        w_new = w
        if f_T > 0:
            L0 = f_new[:, None] + lK + g_new[None, :]   # plan without W
            logS0 = np.full((K_types, K_types), -np.inf)
            for a in np.flatnonzero(src_ok):
                ia = src_type == a
                for b in np.flatnonzero(tgt_ok):
                    logS0[a, b] = logsumexp(L0[np.ix_(ia, tgt_type == b)])
            w_new = _block_log_w(np.maximum(logS0, -700.0), logP, src_ok, tgt_ok, f_T)
        done = max(np.max(np.abs(f_new - f)), np.max(np.abs(g_new - g)), np.max(np.abs(w_new - w))) < tol
        f, g, w = f_new, g_new, w_new
        if done:
            break
    return np.exp(f[:, None] + lK + w[src_type][:, tgt_type] + g[None, :])


def block_masses(plan, src_type, tgt_type, K):
    """(K, K) mass of the plan in each block of types."""
    return np.eye(K)[src_type].T @ plan @ np.eye(K)[tgt_type]


def correct_transitions(flows, labels, names=None):
    """
    Transition rates of each sample consistent with the splits realised by its final plans (flows of
    NetworkModel._record_coupling: block masses, model of the interval): natural rates q >= 0 (off-diagonal) minimising
    sum over its intervals and source types of S_a. KL(F_a || P_a(q)), F the realised split, P(q) the forward split of
    q + the stimulus part of the interval (TRANSk, kept; raw effects, the sum clipped at 0 as in the model) with the
    net rates of the interval, as growth is corrected from
    the relaxed marginals. Returns {sample: dict(types, prescribed, corrected, intervals)}, rates {from: {to: h^-1}}.
    """
    from scipy.optimize import minimize
    K = len(labels)
    off = ~np.eye(K, dtype=bool)
    out = {}
    for s in sorted({f['sample'] for f in flows}):
        ivs = {}
        for f in (f for f in flows if f['sample'] == s):   # batches of an interval pooled
            key = (f['t_from'], f['t_to'])
            if key not in ivs:
                base = np.zeros((K, K)) if f['base'] is None else np.array(f['base'], dtype=float)
                stim = np.array(f['stim'], dtype=float) if 'stim' in f else np.array(f['Q'], dtype=float) - base
                ivs[key] = dict(masses=np.zeros((K, K)), stim=stim, base=base,
                                rates=np.asarray(f['rates'], dtype=float), dt=float(f['dt']), split=f['split'])
            ivs[key]['masses'] += f['masses']
        q0 = next(iter(ivs.values()))['base']

        def loss(x):
            q = np.zeros((K, K))
            q[off] = x
            L = 0.0
            for iv in ivs.values():
                P = forward_split(np.clip(q + iv['stim'], 0.0, None), iv['rates'], iv['dt'])
                S = iv['masses']
                F = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-300)
                pos = S > 0   # S_a. KL(F_a || P_a) = sum_b S_ab log(F_ab / P_ab)
                L += np.sum(S[pos] * np.log(F[pos] / np.maximum(P[pos], 1e-300)))
            return L

        res = minimize(loss, np.maximum(q0[off], 1e-6), method='L-BFGS-B', bounds=[(0.0, 1.0)] * int(off.sum()))
        q = np.zeros((K, K))
        q[off] = res.x
        name = names[int(s)] if names is not None and 0 <= int(s) < len(names) else str(s)
        as_dict = lambda M: {a: {b: float(M[i, j]) for j, b in enumerate(labels) if i != j} for i, a in enumerate(labels)}
        out[name] = dict(types=list(labels), prescribed=as_dict(q0), corrected=as_dict(q), intervals=[
            dict(t_from=k[0], t_to=k[1], model_split=iv['split'].tolist(),
                 realised_split=(iv['masses'] / np.maximum(iv['masses'].sum(axis=1, keepdims=True), 1e-300)).tolist(),
                 corrected_split=forward_split(np.clip(q + iv['stim'], 0.0, None), iv['rates'], iv['dt']).tolist(),
                 source_mass=iv['masses'].sum(axis=1).tolist())
            for k, iv in sorted(ivs.items())])
    return out


class TypeTransitions:
    """
    Transition rates of each sample over an interval, outside NetworkModel (classical OT): transition_rates of the
    sample (own, else default; a matrix missing a type of `labels` is ignored, as in NetworkModel) + TRANSk effects x
    the value of the inference stimuli at the end of the interval (schedule of the sample), aligned on `labels` (sorted
    types of the transition grouping). None for a sample without transition matrix nor effect.
    """

    def __init__(self, project, labels, times_sorted):
        from ..inputs import load_transition_rates, load_transition_effects
        self.project, self.labels, self.tu = project, list(labels), np.sort(np.asarray(times_sorted, dtype=float))
        self.rates = {k: M for k, M in ((k, self._aligned(df, k)) for k, df in
                                        (load_transition_rates(project) or {}).items() if df is not None)
                      if M is not None}
        eff = load_transition_effects(project) or {'default': {}, 'samples': {}}
        self.effects = eff

    def _aligned(self, df, name):
        df = df.rename(index=str, columns=str)
        missing = sorted(set(self.labels) - set(df.index) | set(self.labels) - set(df.columns))
        if missing:
            print(f"Warning: cell type(s) {missing} not found in the transition_rates of "
                  f"{'the default' if name == 'default' else 'sample ' + str(name)}: ignored")
            return None
        return np.clip(df.loc[self.labels, self.labels].to_numpy(dtype=float), 0.0, None)

    def matrix(self, sample, t_to):
        from ..stimulus_rates import cell_values
        Q = self.rates.get(str(sample), self.rates.get('default'))
        Q = None if Q is None else Q.copy()
        eff = {**self.effects['default'], **self.effects['samples'].get(str(sample), {})}
        if not eff:
            return Q
        K = len(self.labels)
        Q = np.zeros((K, K)) if Q is None else Q
        u = cell_values(self.project, [float(t_to)], [str(sample)], self.tu, max(eff))[0]
        low = {l.lower(): i for i, l in enumerate(self.labels)}   # types matched case-insensitively
        for k, entries in eff.items():
            for a, b, d in entries:
                i, j = low.get(str(a).lower()), low.get(str(b).lower())
                if i is not None and j is not None and i != j:
                    Q[i, j] += u[k - 1] * float(d)
        return np.clip(Q, 0.0, None)
