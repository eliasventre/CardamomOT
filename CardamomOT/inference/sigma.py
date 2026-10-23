"""
Regulation functions of the model, each with its own families (model learning), single source for the inference
and the simulators:
- sigma of the GRN: burst frequency kon = k_low + (k_high - k_low) sigma (gated by the chromatin in kon_last_gated),
  used by the network fit, trajectories, OT costs, alpha, filtering and the simulators (ODE, PDMP, explicit mRNA);
- sigma_c of the chromatin: opening / closing rates of the regions, k_open = kbar sigma_c, k_close = kbar (1 - sigma_c)
  (mirror), independent of the GRN sigma (chrom_open_pair, used by chrom_open_close of the simulators).

GRN: sigma_j = link_j(z_j), z_j = basal_j + sum_g theta_gj phi_j(P_g), link and phi given by the family of the target j.
spec (G, N_SPEC) (model_learning_grn; None = every gene logistic of a linear input, the default model):
    spec[j, SPEC_PHI]: transformation of the regulators of target j,
        PHI_IDENTITY (0): phi(P) = P,
        PHI_LOG (1): phi(P) = Pbar log(1 + P / kappa) / log(1 + Pbar / kappa), i.e. with the logistic link
                     sigma = A prod_g (1 + P_g / kappa_g)^(c_g theta_gj) / (1 + A prod_g (...)) (Hill type,
                     exponents c_g theta_gj, c_g = Pbar_g / log(1 + Pbar_g / kappa_g)); phi(0) = 0 and
                     phi(Pbar) = Pbar, so that theta is the change of the logit between 0 and Pbar in both families;
    spec[j, SPEC_LINK]: link of target j, LINK_LOGISTIC (0) only for now;
    spec[g, SPEC_KAPPA], spec[g, SPEC_PBAR]: scale kappa_g and protein level Pbar_g of the high mode of gene g as a
        regulator (PHI_LOG targets only).
The torch version (sigma_torch, free-running SDE fit) mirrors feature / logistic for autograd.
"""
import numpy as np
from numba import njit, prange

SPEC_PHI, SPEC_LINK, SPEC_KAPPA, SPEC_PBAR = 0, 1, 2, 3
N_SPEC = 4
PHI_IDENTITY, PHI_LOG = 0, 1
LINK_LOGISTIC = 0
PHI_CODES = {'identity': PHI_IDENTITY, 'logarithm': PHI_LOG, 'log': PHI_LOG}
LINK_CODES = {'logistic': LINK_LOGISTIC}
PHI_NAMES = {PHI_IDENTITY: 'identity', PHI_LOG: 'logarithm'}
LINK_NAMES = {LINK_LOGISTIC: 'logistic'}


SIGMA_SPEC_FILE = 'sigma_spec_grn.npy'  # in cardamomOT/: present only for a run with model_learning_grn


def load_sigma_spec_grn(cardamom_dir):
    """sigma_spec_grn of a run (None if absent: every gene logistic)."""
    import os
    path = os.path.join(cardamom_dir, SIGMA_SPEC_FILE)
    return np.load(path) if os.path.exists(path) else None


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


def default_spec(G, kappa=0.1, pbar=1.0, phi=PHI_IDENTITY, link=LINK_LOGISTIC):
    """Spec of G genes, all of the family (phi, link), regulator scales kappa and high-mode levels pbar (scalars or
    (G,))."""
    spec = np.zeros((G, N_SPEC))
    spec[:, SPEC_PHI] = phi
    spec[:, SPEC_LINK] = link
    spec[:, SPEC_KAPPA] = kappa
    spec[:, SPEC_PBAR] = pbar
    return spec


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
def feature(p, phi, kappa, pbar):
    """phi(p) of a regulator of level p for the phi family phi (kappa, pbar: scale and high-mode level of the
    regulator)."""
    if phi == PHI_LOG:
        return pbar * np.log1p(max(p, 0.0) / kappa) / np.log1p(pbar / kappa)
    return p


@njit(cache=True)
def features(Y, phi, kappa, pbar):
    """phi of the regulator levels Y (N, K) for the phi family phi; kappa, pbar (K,) of the regulators."""
    if phi == PHI_IDENTITY:
        return Y
    out = np.empty(Y.shape)
    for i in range(Y.shape[0]):
        for k in range(Y.shape[1]):
            out[i, k] = feature(Y[i, k], phi, kappa[k], pbar[k])
    return out


@njit(cache=True)
def sigma_input(theta_basal, theta_inter, prot, j, spec=None):
    """Input z_j of sigma for target j at the protein state prot (G,)."""
    z = theta_basal[j]
    if spec is not None and spec[j, SPEC_PHI] != PHI_IDENTITY:
        phi = spec[j, SPEC_PHI]
        for g in range(prot.shape[0]):
            if theta_inter[g, j] != 0.0:
                z += feature(prot[g], phi, spec[g, SPEC_KAPPA], spec[g, SPEC_PBAR]) * theta_inter[g, j]
        return z
    for g in range(prot.shape[0]):
        z += prot[g] * theta_inter[g, j]
    return z


@njit(cache=True)
def sigma_pairs(theta_basal, theta_inter, y_prot, spec=None):
    """(1 - sigma, sigma) (n_cells, Gm1, 2) per cell and gene (basal (Gm1,), inter (G, Gm1), y_prot (n_cells, G))."""
    n_cells = y_prot.shape[0]
    Gm1 = theta_basal.shape[0]
    result = np.empty((n_cells, Gm1, 2))
    for i in range(n_cells):
        for j in range(Gm1):
            z = sigma_input(theta_basal, theta_inter, y_prot[i], j, spec)
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
                z += feature(pmax[g], phi, 1.0 if spec is None else spec[g, SPEC_KAPPA],
                             1.0 if spec is None else spec[g, SPEC_PBAR]) * inter[g, j]
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
    log_t = torch.tensor(spec[:, SPEC_PHI] == PHI_LOG, dtype=P.dtype)
    kappa = torch.tensor(spec[:, SPEC_KAPPA], dtype=P.dtype)
    pbar = torch.tensor(spec[:, SPEC_PBAR], dtype=P.dtype)
    phi = pbar * torch.log1p(torch.relu(P) / kappa) / torch.log1p(pbar / kappa)
    z = torch.einsum('wng,gj->wnj', P, theta * (1 - log_t)) + torch.einsum('wng,gj->wnj', phi, theta * log_t)
    return torch.sigmoid(z * s[:, None, :] + (basal * s)[:, None, :])


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
