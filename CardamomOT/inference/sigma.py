"""
Regulation functions of the model, each with its own families (model learning), single source for the inference
and the simulators:
- sigma of the GRN: burst frequency kon = k_low + (k_high - k_low) sigma (gated by the chromatin in kon_last_gated),
  used by the network fit, trajectories, OT costs, alpha, filtering and the simulators (ODE, PDMP, explicit mRNA);
- sigma_c of the chromatin: opening / closing rates of the regions, k_open = kbar sigma_c, k_close = kbar (1 - sigma_c)
  (mirror), independent of the GRN sigma (chrom_open_pair, used by chrom_open_close of the simulators).

GRN: sigma_j = link_j(z_j), z_j = basal_j + sum_g theta_gj phi_j(P_g), link and phi given by the family of the target j.
spec (G, N_SPEC + G) (model_learning_grn; None = every gene logistic of a linear input, the default model):
    spec[j, SPEC_PHI]: transformation of the regulators of target j,
        PHI_IDENTITY (0): phi(P) = P,
        PHI_LOG (1): phi(P) = Pbar log(1 + P / kappa) / log(1 + Pbar / kappa), i.e. with the logistic link
                     sigma = A prod_g (1 + P_g / kappa_g)^(c_g theta_gj) / (1 + A prod_g (...)) (Hill type,
                     exponents c_g theta_gj, c_g = Pbar_g / log(1 + Pbar_g / kappa_g)); phi(0) = 0 and
                     phi(Pbar) = Pbar, so that theta is the change of the logit between 0 and Pbar in both families;
        PHI_HERBACH_APPROX (3): Herbach type, approximate form linear in theta (model_learning_herbach_form =
                     'approximate'): phi(P) = Pbar h(P / s) / h(Pbar / s), h(x) = x^m / (1 + x^m), i.e. the
                     contribution psi of Herbach et al. to first order in theta, theta times a Hill step of threshold s_g
                     (regulator) and coefficient m_j (target); phi(0) = 0, phi(Pbar) = Pbar;
        PHI_HERBACH (2): Herbach type, exact form ('exact'): the thermodynamic model of Herbach et al., each regulator
                     entering the logit through psi(t, x) = log((1 + e^t x^m) / (1 + x^m)), x = P_g / s_gj, instead
                     of theta_gj phi(P_g): saturating, 0 at P = 0, Hill transition of coefficient m_j around s_gj;
                     parametrised by its effect at the high mode, psi(t, Pbar / s) = theta_gj Pbar (units and
                     derivative at theta = 0 of the approximate form), t = herbach_theta(theta Pbar, m log(Pbar / s)),
                     theta_gj > -log(1 + (Pbar / s_gj)^m) / Pbar (repression bound, lifted by a lower threshold);
                     nonlinear in theta (network.HerbachArray); thresholds s_gj per edge (herbach_threshold_sd);
    spec[j, SPEC_LINK]: link of target j, LINK_LOGISTIC (0) only for now;
    spec[g, SPEC_KAPPA], spec[g, SPEC_PBAR]: scale kappa_g and protein level Pbar_g of the high mode of gene g as a
        regulator (PHI_LOG, PHI_HERBACH targets);
    spec[g, SPEC_THRESH]: threshold s_g of gene g as a regulator (PHI_HERBACH_APPROX, and prior centre of the
        thresholds of its edges in the exact form), halfway between its low and high protein levels;
    spec[j, SPEC_M]: Hill coefficient m_j of target j (Herbach types);
    spec[j, N_SPEC + g]: threshold s_gj of the edge g -> j of an exact Herbach target j (default s_g, learned in the
        network fit, see edge_thresholds).
The torch version (sigma_torch, free-running SDE fit) mirrors feature / logistic for autograd.
"""
import numpy as np
from numba import njit, prange

SPEC_PHI, SPEC_LINK, SPEC_KAPPA, SPEC_PBAR, SPEC_THRESH, SPEC_M = 0, 1, 2, 3, 4, 5
N_SPEC = 6
PHI_IDENTITY, PHI_LOG, PHI_HERBACH, PHI_HERBACH_APPROX = 0, 1, 2, 3
HERBACH_TYPES = (PHI_HERBACH, PHI_HERBACH_APPROX)  # both forms of the Herbach type
LINK_LOGISTIC = 0
PHI_CODES = {'identity': PHI_IDENTITY, 'logarithm': PHI_LOG, 'log': PHI_LOG, 'herbach': PHI_HERBACH_APPROX,
             'herbach_exact': PHI_HERBACH}
LINK_CODES = {'logistic': LINK_LOGISTIC}
PHI_NAMES = {PHI_IDENTITY: 'identity', PHI_LOG: 'logarithm', PHI_HERBACH: 'herbach_exact', PHI_HERBACH_APPROX: 'herbach'}
LINK_NAMES = {LINK_LOGISTIC: 'logistic'}


def herbach_code(form):
    """phi code of the Herbach type of form 'approximate' (linear in theta, default) or 'exact'."""
    key = str(form).strip().lower()
    if key not in ('approximate', 'exact'):
        raise ValueError(f"model_learning_herbach_form: 'approximate' or 'exact', not {form!r}")
    return PHI_HERBACH if key == 'exact' else PHI_HERBACH_APPROX


def is_herbach(phi):
    """True for both forms of the Herbach type."""
    return int(phi) in HERBACH_TYPES
START_M = 2.0  # Hill coefficient of the Herbach type when none is given


def family_name(fam):
    """Name of a candidate family (phi, link, m), e.g. 'herbach2/logistic'."""
    phi, link, m = fam
    return f"{PHI_NAMES[int(phi)]}{'%g' % m if is_herbach(phi) else ''}/{LINK_NAMES[int(link)]}"


def set_family(spec, rows, fam):
    """Family (phi, link, m) of the targets rows of spec, in place."""
    phi, link, m = fam
    spec[rows, SPEC_PHI], spec[rows, SPEC_LINK] = phi, link
    if is_herbach(phi):
        spec[rows, SPEC_M] = m


def family_index(spec, cands):
    """Index (G,) of the candidate family of each target of spec (0 if none matches)."""
    out = np.zeros(spec.shape[0], dtype=int)
    for f, (phi, link, m) in enumerate(cands):
        hit = (spec[:, SPEC_PHI] == phi) & (spec[:, SPEC_LINK] == link)
        if is_herbach(phi):
            hit &= np.isclose(spec[:, SPEC_M], m)
        out[hit] = f
    return out


SIGMA_SPEC_FILE = 'sigma_spec_grn.npy'  # in cardamomOT/: present only for a run with model_learning_grn


def load_sigma_spec_grn(cardamom_dir):
    """sigma_spec_grn of a run (None if absent: every gene logistic)."""
    import os
    path = os.path.join(cardamom_dir, SIGMA_SPEC_FILE)
    if not os.path.exists(path):
        return None
    spec = np.load(path)
    if spec.ndim != 2 or spec.shape[1] != N_SPEC + spec.shape[0]:
        raise ValueError(f"{path}: spec of shape {spec.shape}, expected (G, {N_SPEC} + G) (thresholds per edge): "
                         "rerun infer_network_structure")
    return spec


def save_sigma_spec_grn(cardamom_dir, spec):
    """Save spec (None: remove the file of an earlier run, i.e. logistic downstream)."""
    import os
    path = os.path.join(cardamom_dir, SIGMA_SPEC_FILE)
    if spec is not None:
        np.save(path, np.asarray(spec, dtype=float))
    elif os.path.exists(path):
        os.remove(path)


def phi_code(name):
    """Code of a phi family from its name (case-insensitive)."""
    key = str(name).lower()
    if key not in PHI_CODES:
        raise ValueError(f"unknown phi family {name!r} (known: {sorted(PHI_CODES)})")
    return PHI_CODES[key]


def link_code(name):
    """Code of a link family from its name (case-insensitive)."""
    key = str(name).lower()
    if key not in LINK_CODES:
        raise ValueError(f"unknown link family {name!r} (known: {sorted(LINK_CODES)})")
    return LINK_CODES[key]


def default_spec(G, kappa=0.1, pbar=1.0, phi=PHI_IDENTITY, link=LINK_LOGISTIC, thresh=None, m=START_M):
    """Spec of G genes, all of the family (phi, link), regulator scales kappa, high-mode levels pbar, thresholds thresh
    (None: pbar / 2) and Hill coefficients m (scalars or (G,)); the thresholds of the edges at those of their
    regulators."""
    spec = np.zeros((G, N_SPEC + G))
    spec[:, SPEC_PHI] = phi
    spec[:, SPEC_LINK] = link
    spec[:, SPEC_KAPPA] = kappa
    spec[:, SPEC_PBAR] = pbar
    spec[:, SPEC_THRESH] = np.asarray(pbar, dtype=float) / 2 if thresh is None else thresh
    spec[:, SPEC_M] = m
    reset_edge_thresholds(spec)
    return spec


def reset_edge_thresholds(spec, rows=slice(None)):
    """Thresholds of the edges of the targets rows at those of their regulators (spec[:, SPEC_THRESH]), in place."""
    spec[rows, N_SPEC:] = spec[:, SPEC_THRESH][None, :]


def edge_thresholds(spec):
    """View (G, G) of the thresholds of the edges, [target, regulator]."""
    return spec[:, N_SPEC:]


def uses_spec(spec):
    """True if spec has a gene outside the default family (None: no)."""
    if spec is None:
        return False
    spec = np.asarray(spec)
    return bool(np.any(spec[:, SPEC_PHI] != PHI_IDENTITY) or np.any(spec[:, SPEC_LINK] != LINK_LOGISTIC))


@njit(cache=True)
def logistic_pair(z):
    """(1 - sigmoid(z), sigmoid(z)), stable: exp of a negative argument only (no inf / inf for |z| > 709)."""
    e = np.exp(-abs(z))
    denom = e + 1.0
    if z >= 0:
        return e / denom, 1.0 / denom
    return 1.0 / denom, e / denom


@njit(cache=True)
def logistic(z):
    """sigmoid of each entry of z (1-D)."""
    out = np.empty(z.shape[0])
    for i in range(z.shape[0]):
        out[i] = logistic_pair(z[i])[1]
    return out


@njit(cache=True)
def link_pair(z, link):
    """(1 - sigma, sigma) of the input z for the link family link."""
    if link != LINK_LOGISTIC:
        raise ValueError("unknown link family of sigma")
    return logistic_pair(z)


@njit(cache=True)
def hill_ratio(x, m):
    """x^m / (1 + x^m) for x >= 0."""
    xm = max(x, 0.0) ** m
    return xm / (1.0 + xm)


@njit(cache=True)
def feature(p, phi, kappa, pbar, thresh, m):
    """phi(p) of a regulator of level p for the phi family phi (kappa, pbar, thresh: scale, high-mode level and
    threshold of the regulator; m: Hill coefficient of the target); the families linear in theta only (exact
    Herbach: herbach_psi)."""
    if phi == PHI_LOG:
        return pbar * np.log1p(max(p, 0.0) / kappa) / np.log1p(pbar / kappa)
    if phi == PHI_HERBACH_APPROX:
        return pbar * hill_ratio(p / thresh, m) / hill_ratio(pbar / thresh, m)
    return p


HERBACH_U_MIN = 1e-6  # floor of delta + log(1 + xbar^m) (exact Herbach): repression bound of the parametrisation


@njit(cache=True)
def herbach_theta(delta, lbar):
    """Fold-change exponent t of an exact Herbach edge of effect delta at the high mode (lbar = m log(Pbar / s)):
    psi(t, Pbar / s) = delta, i.e. t = log((1 + xbar^m) e^delta - 1) - lbar (floored at the repression bound)."""
    u = max(delta + np.logaddexp(0.0, lbar), HERBACH_U_MIN)
    return u + np.log(-np.expm1(-u)) - lbar


@njit(cache=True)
def herbach_psi(theta, p, thresh, pbar, m):
    """Contribution psi(t, x) = log((1 + e^t x^m) / (1 + x^m)) of a regulator of level p to the logit of a Herbach
    target (x = p / thresh, m its Hill coefficient), t = herbach_theta(theta pbar, .): theta pbar at p = pbar, 0 at
    p = 0."""
    if p <= 0.0:
        return 0.0
    lx = m * np.log(p / thresh)
    t = herbach_theta(theta * pbar, m * np.log(pbar / thresh))
    return np.logaddexp(0.0, t + lx) - np.logaddexp(0.0, lx)


@njit(cache=True)
def features(Y, phi, kappa, pbar, thresh, m):
    """phi of the regulator levels Y (N, K) for the phi family phi; kappa, pbar, thresh (K,) of the regulators, m the
    Hill coefficient of the target."""
    if phi == PHI_IDENTITY:
        return Y
    out = np.empty(Y.shape)
    for i in range(Y.shape[0]):
        for k in range(Y.shape[1]):
            out[i, k] = feature(Y[i, k], phi, kappa[k], pbar[k], thresh[k], m)
    return out


@njit(cache=True)
def edge_constants(theta_inter, spec):
    """Constants (G, Gm1) of the edges, independent of the cells: exact Herbach targets, e^t - 1 (t the fold-change
    exponent, herbach_theta) and log-threshold of each edge; approximate Herbach targets, theta Pbar / h(Pbar / s) and
    log-threshold of the regulator. c1 = 0 marks the absent edges (theta = 0, skipped by sigma_input; the network fits
    set the edges below seuil_min_network_intensity to 0)."""
    G, Gm1 = theta_inter.shape
    c1, c2 = np.zeros((G, Gm1)), np.zeros((G, Gm1))
    for j in range(Gm1):
        phi = spec[j, SPEC_PHI]
        for g in range(G):
            th = theta_inter[g, j]
            if th == 0.0:
                continue
            if phi == PHI_HERBACH:
                s = spec[j, N_SPEC + g]
                c1[g, j] = np.expm1(herbach_theta(th * spec[g, SPEC_PBAR], spec[j, SPEC_M] * np.log(spec[g, SPEC_PBAR] / s)))
                c2[g, j] = np.log(s)
            elif phi == PHI_HERBACH_APPROX:
                c1[g, j] = th * spec[g, SPEC_PBAR] / hill_ratio(spec[g, SPEC_PBAR] / spec[g, SPEC_THRESH], spec[j, SPEC_M])
                c2[g, j] = np.log(spec[g, SPEC_THRESH])
    return c1, c2


@njit(cache=True)
def sigma_input(theta_basal, theta_inter, prot, logp, j, spec, c1, c2):
    """Input z_j of sigma for target j at the protein state prot (G,), logp = log(prot) where positive, c1 / c2 the
    constants of the edges (edge_constants)."""
    z = theta_basal[j]
    if spec is not None:
        phi = spec[j, SPEC_PHI]
        if phi == PHI_HERBACH:  # psi(t, x) = log((1 + e^t x^m) / (1 + x^m)) = log1p((e^t - 1) h), h = x^m / (1 + x^m)
            m = spec[j, SPEC_M]
            for g in range(prot.shape[0]):
                if c1[g, j] != 0.0 and prot[g] > 0.0:
                    z += np.log1p(c1[g, j] / (1.0 + np.exp(-m * (logp[g] - c2[g, j]))))
            return z
        if phi == PHI_HERBACH_APPROX:  # theta Pbar h(x) / h(xbar), h(x) = x^m / (1 + x^m)
            m = spec[j, SPEC_M]
            for g in range(prot.shape[0]):
                if c1[g, j] != 0.0 and prot[g] > 0.0:
                    z += c1[g, j] / (1.0 + np.exp(-m * (logp[g] - c2[g, j])))
            return z
        if phi != PHI_IDENTITY:
            for g in range(prot.shape[0]):
                if theta_inter[g, j] != 0.0:
                    z += feature(prot[g], phi, spec[g, SPEC_KAPPA], spec[g, SPEC_PBAR], spec[g, SPEC_THRESH],
                                 spec[j, SPEC_M]) * theta_inter[g, j]
            return z
    for g in range(prot.shape[0]):
        if theta_inter[g, j] != 0.0:
            z += prot[g] * theta_inter[g, j]
    return z


@njit(cache=True)
def sigma_pairs(theta_basal, theta_inter, y_prot, spec=None):
    """(1 - sigma, sigma) (n_cells, Gm1, 2) per cell and gene (basal (Gm1,), inter (G, Gm1), y_prot (n_cells, G))."""
    n_cells = y_prot.shape[0]
    Gm1 = theta_basal.shape[0]
    G = y_prot.shape[1]
    result = np.empty((n_cells, Gm1, 2))
    if spec is None:
        c1, c2 = np.zeros((1, 1)), np.zeros((1, 1))
    else:
        c1, c2 = edge_constants(theta_inter, spec)
    logp = np.zeros(G)
    for i in range(n_cells):
        for g in range(G):
            logp[g] = np.log(y_prot[i, g]) if y_prot[i, g] > 0.0 else 0.0
        for j in range(Gm1):
            z = sigma_input(theta_basal, theta_inter, y_prot[i], logp, j, spec, c1, c2)
            if spec is None:
                s0, s1 = logistic_pair(z)
            else:
                s0, s1 = link_pair(z, spec[j, SPEC_LINK])
            result[i, j, 0] = s0
            result[i, j, 1] = s1
    return result


@njit(cache=True)
def k_high(kz, j):
    """Highest mode amplitude of gene j (kz (G, n_modes))."""
    k = kz[j, 0]
    for m in range(1, kz.shape[1]):
        if kz[j, m] > k:
            k = kz[j, m]
    return k


@njit(fastmath=True, parallel=True, cache=True)
def kon_matrix(y_prot, kz, theta_inter, theta_basal, spec=None):
    """kon (n_cells, Gm1) = k_low (1 - sigma) + k_high sigma, k_low the lowest mode and k_high the highest."""
    sigma = sigma_pairs(theta_basal, theta_inter, y_prot, spec)
    out = np.zeros(sigma.shape[:2])
    for i in prange(sigma.shape[0]):
        for j in prange(sigma.shape[1]):
            out[i, j] += kz[j, 0] * sigma[i, j, 0]
            out[i, j] += k_high(kz, j) * sigma[i, j, 1]
    return out


@njit(cache=True)
def kon_last(y_prot, kz, theta_inter, theta_basal, spec=None):
    """kon (Gm1,) of the last row of y_prot (simulators)."""
    result = sigma_pairs(theta_basal, theta_inter, y_prot[-1:], spec)
    res = np.zeros(theta_basal.shape[0])
    for j in range(theta_basal.shape[0]):
        res[j] += kz[j, 0] * result[0, j, 0]
        res[j] += k_high(kz, j) * result[0, j, 1]
    return res


@njit(cache=True)
def kon_last_gated(y_prot, kz, theta_inter, theta_basal, cbar, spec=None):
    """kon_last with the regulated part gated by cbar: k_low + (k_high - k_low) cbar sigma, written so that
    cbar = 1 gives kon_last bit for bit."""
    result = sigma_pairs(theta_basal, theta_inter, y_prot[-1:], spec)
    res = np.zeros(theta_basal.shape[0])
    for j in range(theta_basal.shape[0]):
        s1 = result[0, j, 1]
        res[j] += kz[j, 0] * (result[0, j, 0] + (1.0 - cbar[j]) * s1)
        res[j] += k_high(kz, j) * (cbar[j] * s1)
    return res


@njit(cache=True)
def kon_upper_bound(pmax, ks, inter, basal, ns=1, spec=None):
    """Sum over the genes of an upper bound of kon along the flow: positive interactions at pmax (phi and the link
    increasing)."""
    G = pmax.shape[0]
    tot = 0.0
    for j in range(ns, basal.shape[0]):
        phi = PHI_IDENTITY if spec is None else spec[j, SPEC_PHI]
        z = basal[j]
        for g in range(G):
            if inter[g, j] > 0:
                if phi == PHI_HERBACH:  # psi increasing in the regulator for theta > 0
                    z += herbach_psi(inter[g, j], pmax[g], 0.5 if spec is None else spec[j, N_SPEC + g],
                                     1.0 if spec is None else spec[g, SPEC_PBAR], START_M if spec is None else spec[j, SPEC_M])
                else:
                    z += feature(pmax[g], phi, 1.0 if spec is None else spec[g, SPEC_KAPPA],
                                 1.0 if spec is None else spec[g, SPEC_PBAR], 0.5 if spec is None else spec[g, SPEC_THRESH],
                                 START_M if spec is None else spec[j, SPEC_M]) * inter[g, j]
        sig = logistic_pair(z)[1] if spec is None else link_pair(z, spec[j, SPEC_LINK])[1]
        tot += (1 - sig) * ks[j, 0] + sig * k_high(ks, j) + 1e-10  # precision margin
    return tot


def sigma_torch(P, theta, basal, s, spec=None):
    """Torch mirror of sigma: sigma (W, n, G) of the states P (W, n, G), theta (G, G), basal (G,), theta scale
    s (W, G) of each window (logistic link)."""
    import torch
    if not uses_spec(spec):
        z = torch.einsum('wng,gj->wnj', P, theta) * s[:, None, :] + (basal * s)[:, None, :]
        return torch.sigmoid(z)
    spec = np.asarray(spec)
    if np.any(spec[:, SPEC_LINK] != LINK_LOGISTIC):
        raise ValueError("sigma_torch: logistic link only")
    fam = spec[:, SPEC_PHI]
    kappa = torch.tensor(spec[:, SPEC_KAPPA], dtype=P.dtype)
    pbar = torch.tensor(spec[:, SPEC_PBAR], dtype=P.dtype)
    col = lambda mask: torch.tensor(mask, dtype=P.dtype)  # targets of a family (columns of theta)
    z = torch.einsum('wng,gj->wnj', P, theta * col(fam == PHI_IDENTITY))  # (W, n, G)
    if np.any(fam == PHI_LOG):
        phi = pbar * torch.log1p(torch.relu(P) / kappa) / torch.log1p(pbar / kappa)
        z = z + torch.einsum('wng,gj->wnj', phi, theta * col(fam == PHI_LOG))
    if np.any(fam == PHI_HERBACH_APPROX):
        thresh = torch.tensor(spec[:, SPEC_THRESH], dtype=P.dtype)
        h = lambda x, m: x ** m / (1 + x ** m)
        for m in np.unique(spec[fam == PHI_HERBACH_APPROX, SPEC_M]):  # one feature map per Hill coefficient
            phi = pbar * h(torch.relu(P) / thresh, float(m)) / h(pbar / thresh, float(m))
            z = z + torch.einsum('wng,gj->wnj', phi, theta * col((fam == PHI_HERBACH_APPROX) & (spec[:, SPEC_M] == m)))
    z = z * s[:, None, :] + (basal * s)[:, None, :]
    if np.any(fam == PHI_HERBACH):
        # Exact Herbach contributions psi(s_j theta_gj, P_g / s_g) on the edges of the Herbach targets (support of
        # theta): the scale acts on theta, as inter * scale in the simulators
        F = torch.nn.functional.softplus
        herb = torch.tensor(fam == PHI_HERBACH)
        g_idx, j_idx = torch.nonzero((theta.detach() != 0) & herb[None, :], as_tuple=True)
        if len(g_idx):
            m_e = torch.tensor(spec[:, SPEC_M], dtype=P.dtype)[j_idx]
            thr = torch.tensor(spec[:, N_SPEC:], dtype=P.dtype)[j_idx, g_idx]  # threshold of each edge (E,)
            lx = torch.log(torch.clamp(P[..., g_idx] / thr, min=1e-12)) * m_e   # (W, n, E)
            lbar = m_e * torch.log(pbar[g_idx] / thr)
            # Fold-change exponent of the effect theta Pbar at the high mode (herbach_theta)
            u = torch.clamp(theta[g_idx, j_idx] * s[:, j_idx][:, None, :] * pbar[g_idx] + F(lbar), min=HERBACH_U_MIN)
            t = u + torch.log(-torch.expm1(-u)) - lbar
            psi = F(t + lx) - F(lx)
            z = z + torch.zeros_like(z).index_add(-1, j_idx, psi)
    return torch.sigmoid(z)


# --- Chromatin opening function sigma_c (independent of the GRN sigma) ---
# sigma_c_r = logistic(Zc_r), Zc_r = basal_c_r + sum_g inter_c[g, r] P_g. spec_c (R, N_SPEC_CHROM) (model_learning_chrom;
# None = every region logistic): spec_c[r, 0] family of region r (CHROM_LOGISTIC only for now).
CHROM_LOGISTIC = 0
N_SPEC_CHROM = 1


def default_spec_chrom(R):
    """Spec of R regions, all logistic."""
    return np.zeros((R, N_SPEC_CHROM))


@njit(cache=True)
def chrom_open_pair(P, inter_c, basal_c, r, spec_c=None):
    """(1 - sigma_c, sigma_c) of region r at the protein state P (G,): closing / opening fractions (mirror rates)."""
    if spec_c is not None and spec_c[r, 0] != CHROM_LOGISTIC:
        raise ValueError("unknown family of the chromatin opening function")
    z = basal_c[r]
    for g in range(P.shape[0]):
        z += P[g] * inter_c[g, r]
    return logistic_pair(z)
