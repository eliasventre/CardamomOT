"""
_for_figure7.py
===============
Pré-calcul des distributions de proportions de types cellulaires
pour chaque perturbation de figure7.py.

Pour chaque perturbation (KO/OV), N_SIMS simulations stochastiques indépendantes
sont effectuées (condition WT et condition perturbée).  Les vecteurs de proportions
de types cellulaires sont sauvegardés dans for_figure7_data/{perturb_id}.npz.

figure7.py charge ces résultats pour afficher la variance des effets de
perturbation (σ̄ par type cellulaire) à la place du test du chi2.

Usage :
    cd results_article/figures
    python _for_figure7.py [--n-sims 10]

Fichiers de sortie :
    for_figure7_data/{perturb_id}.npz
      props_wt     : (N_SIMS, n_celltypes)  proportions WT
      props_perturb: (N_SIMS, n_celltypes)  proportions perturbées
      cell_types   : liste de noms de types cellulaires (ordre = colonnes)
      stab_score   : float  — σ̄ = mean std par type cellulaire (condition perturbée)
"""

import sys
sys.path += ['./../../']

import os
import copy
import numpy as np
import anndata as ad
import scipy

from CardamomOT import NetworkModel as NetworkModel_beta
from CardamomOT import train_classifier, predict_cell_types

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_SIMS = 5    # nombre de simulations stochastiques par condition

LABEL = 'cell_type'

# Définition des 5 perturbations (identique à figure7.py)
PERTURBATIONS = [
    dict(
        path          = './../../experimental_datasets/Semrau',
        perturb_id    = 'KO_none_OV_Dnmt3a',
        KO            = [],
        OV            = ['Dnmt3a'],
        gene          = 'Dnmt3a',
        dataset_group = 'Semrau',
        split         = 'full',
    ),
    dict(
        path          = './../../experimental_datasets/Kameneva',
        perturb_id    = 'KO_CHGA_OV_none',
        KO            = ['CHGA'],
        OV            = [],
        gene          = 'CHGA',
        dataset_group = 'Kameneva',
        split         = 'full',
    ),
    dict(
        path          = './../../experimental_datasets/Kameneva',
        perturb_id    = 'KO_CHGA_OV_STMN2',
        KO            = ['CHGA'],
        OV            = ['STMN2'],
        gene          = 'STMN2',
        dataset_group = 'Kameneva',
        split         = 'full',
    ),
    dict(
        path          = './../../experimental_datasets/Schiebinger',
        perturb_id    = 'KO_none_OV_Zfp42',
        KO            = [],
        OV            = ['Zfp42'],
        gene          = 'Zfp42',
        dataset_group = 'Schiebinger',
        split         = 'train',
    ),
    dict(
        path          = './../../experimental_datasets/Schiebinger',
        perturb_id    = 'KO_none_OV_Obox6',
        KO            = [],
        OV            = ['Obox6'],
        gene          = 'Obox6',
        dataset_group = 'Schiebinger',
        split         = 'train',
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers : chargement du modèle
# ─────────────────────────────────────────────────────────────────────────────

def _load_stimulus_schedule(p, prefer_simul=True):
    candidates = (
        ['stimulus_schedule_simul.txt', 'stimulus_schedule.txt']
        if prefer_simul else ['stimulus_schedule.txt']
    )
    for fname in candidates:
        path = os.path.join(p, 'Data', fname)
        if os.path.exists(path):
            return np.loadtxt(path)
    return None


def _detect_n_stimuli(stim_sched):
    if stim_sched is None:
        return 1
    arr = np.asarray(stim_sched)
    return int(arr.shape[1]) if arr.ndim == 2 else 1


def load_model(p, split):
    """
    Charge le modèle CardamomOT depuis les fichiers de résultats de la pipeline.
    Même logique que simulate_network_KOV.py.
    """
    data_path = os.path.join(p, 'Data', f'data_{split}.h5ad')
    adata     = ad.read_h5ad(data_path)
    G         = adata.shape[1]

    stim_sched = _load_stimulus_schedule(p, prefer_simul=True)
    n_stimuli  = _detect_n_stimuli(stim_sched)

    model = NetworkModel_beta(G, n_stimuli=n_stimuli)

    cardamom_dir = os.path.join(p, 'cardamomOT')
    model.d_t       = np.load(os.path.join(cardamom_dir, 'degradations_temporal.npy'))
    model.inter_t   = np.load(os.path.join(cardamom_dir, 'inter_t_simul.npy'))
    model.inter     = np.load(os.path.join(cardamom_dir, 'inter_simul.npy'))

    # Corriger n_stimuli d'après inter_simul (source de vérité)
    n_stimuli_inter = model.inter.shape[0] - G
    if n_stimuli_inter != model.n_stimuli:
        model.n_stimuli = n_stimuli_inter

    model.a             = np.load(os.path.join(cardamom_dir, 'mixture_parameters.npy'))
    model.times_data    = np.load(os.path.join(cardamom_dir, 'data_times.npy'))
    model.kon_beta      = np.load(os.path.join(cardamom_dir, 'data_kon_beta.npy'))
    model.rna           = np.load(os.path.join(cardamom_dir, 'data_rna.npy'))
    model.proba_traj    = np.load(os.path.join(cardamom_dir, 'proba_traj.npy'))
    model.ratios        = np.load(os.path.join(cardamom_dir, 'ratios.npy'))
    model.n_networks    = np.load(os.path.join(cardamom_dir, 'n_networks.npy'))

    samples_path = os.path.join(cardamom_dir, 'data_samples.npy')
    if os.path.exists(samples_path):
        model.samples_data = np.load(samples_path)

    return model, adata, stim_sched


def build_clean_baseline(p, model):
    """
    Reconstruit le basal propre (non perturbé par les KOV d'entraînement).
    Même logique que simulate_network_KOV.py.
    """
    cardamom_dir        = os.path.join(p, 'cardamomOT')
    basal_simul_raw     = np.load(os.path.join(cardamom_dir, 'basal_simul.npy'))
    basal_t_simul_raw   = np.load(os.path.join(cardamom_dir, 'basal_t_simul.npy'))
    EPS = 1e-16

    if basal_simul_raw.ndim == 3:
        n_samp, G_tot_b, _ = basal_simul_raw.shape
        mask_path = os.path.join(cardamom_dir, 'basal_ref_mask.npy')

        basal_clean_3d = basal_simul_raw.copy()
        if os.path.exists(mask_path):
            basal_ref_mask = np.load(mask_path)
            for g in range(G_tot_b):
                perturbed = np.where( basal_ref_mask[:, g])[0]
                free      = np.where(~basal_ref_mask[:, g])[0]
                if len(perturbed) > 0:
                    clean_g = (basal_simul_raw[free, g, :].mean(axis=0)
                               if len(free) > 0
                               else basal_simul_raw[:, g, :].mean(axis=0))
                    basal_clean_3d[perturbed, g, :] = clean_g

        basal_clean = basal_clean_3d.mean(axis=0)

        if basal_t_simul_raw.ndim == 4:
            temporal_factor = basal_t_simul_raw / (basal_simul_raw[np.newaxis] + EPS)
            basal_t_clean   = basal_clean_3d[np.newaxis] * temporal_factor
        else:
            basal_mean_orig = basal_simul_raw.mean(axis=0)
            temporal_factor = basal_t_simul_raw / (basal_mean_orig[np.newaxis] + EPS)
            basal_t_clean   = basal_clean[np.newaxis] * temporal_factor

    else:
        basal_clean    = basal_simul_raw
        basal_clean_3d = None
        basal_t_clean  = basal_t_simul_raw

    return basal_clean, basal_clean_3d, basal_t_clean


def apply_perturbation(model_c, adata, cfg, ns):
    """
    Applique les KO/OV définis dans cfg au modèle (modifie basal_t in-place).
    """
    for gene in cfg['KO']:
        if gene not in adata.var_names:
            print(f"  Warning: gène KO '{gene}' absent du dataset")
            continue
        ind = ns + adata.var_names.get_loc(gene)
        if model_c.basal_t.ndim == 4:
            model_c.basal_t[:, :, ind] = -100 - np.sum(model_c.inter_t[-1, :, ind])
        else:
            model_c.basal_t[:, ind]    = -100 - np.sum(model_c.inter_t[-1, :, ind])
        print(f"  KO 100% : {gene} (ind={ind})")

    for gene in cfg['OV']:
        if gene not in adata.var_names:
            print(f"  Warning: gène OV '{gene}' absent du dataset")
            continue
        ind = ns + adata.var_names.get_loc(gene)
        if model_c.basal_t.ndim == 4:
            model_c.basal_t[:, :, ind] = 100 + np.sum(model_c.inter_t[-1, :, ind])
        else:
            model_c.basal_t[:, ind]    = 100 + np.sum(model_c.inter_t[-1, :, ind])
        print(f"  OV 100% : {gene} (ind={ind})")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers : simulation → AnnData → proportions
# ─────────────────────────────────────────────────────────────────────────────

def _get_sim_times(p, model):
    """Timepoints de simulation (depuis simulation_times.npy si dispo)."""
    st_path = os.path.join(p, 'cardamomOT', 'simulation_times.npy')
    if os.path.exists(st_path):
        sim_times_all = np.load(st_path)
        times = sorted(set(sim_times_all))
    else:
        times = sorted(set(model.times_data))
    return times


def _prot_to_adata(model_c, adata_ref, ns, sim_times):
    """
    Convertit model.kon_theta → AnnData RNA simulé (NB noise)
    en suivant la même logique que check_KOV_to_sim.py.
    """
    G     = adata_ref.shape[1]
    N_tot = model_c.kon_theta.shape[0]

    mixture_params = model_c.a   # shape (n_networks+1, G_tot)
    c  = mixture_params[-1, :]   # overdispersion (G_tot,)
    kz = mixture_params[:-1, :]  # burst sizes    (n_networks, G_tot)

    pi_zinb = getattr(model_c, '_pi_zinb', None)

    # NB paramétrique — même formule que check_KOV_to_sim.py :
    # np.max(kz, 0) = max sur axis=0 (sur les réseaux) → (G_tot,)
    lambda_ = np.max(kz, 0) * model_c.kon_theta             # (N_tot, G_tot)
    lambda_genes = lambda_[:, ns:]                           # (N_tot, G)
    c_genes = (c / (c + 1 + 1e-16))[ns:].reshape(G, 1)     # (G, 1)

    rna_sim = np.random.negative_binomial(
        np.maximum(lambda_genes.T, 1e-8),   # (G, N_tot)
        c_genes,                             # (G, 1)
    ).T.astype(float)                        # (N_tot, G)

    if pi_zinb is not None:
        mask = np.random.uniform(0, 1, rna_sim.shape) < pi_zinb.reshape(1, G)
        rna_sim[mask] = 0.0

    # Times pour l'AnnData
    N_per_t = N_tot // len(sim_times) if len(sim_times) > 0 else N_tot
    times_all = np.repeat(sim_times, N_per_t)
    if len(times_all) < N_tot:
        times_all = np.concatenate([times_all, np.full(N_tot - len(times_all), sim_times[-1])])

    adata_sim      = ad.AnnData(X=rna_sim)
    adata_sim.var  = adata_ref.var.copy()
    adata_sim.obs['time'] = times_all[:N_tot]
    return adata_sim


def simulate_n(model_base, adata_ref, ns, sim_times, n_sims,
               perturb_cfg=None, cardamom_dir=None):
    """
    Effectue n_sims simulations stochastiques (WT ou perturbée).
    Retourne liste de n_sims AnnData simulés.
    """
    adatas = []
    for k in range(n_sims):
        m = copy.deepcopy(model_base)
        # Ré-initialiser les conditions initiales
        if cardamom_dir is not None:
            fwd_prot = os.path.join(cardamom_dir, 'data_prot_forsimul.npy')
            fwd_kon  = os.path.join(cardamom_dir, 'data_kon_theta.npy')
            if os.path.exists(fwd_prot):
                m.prot      = np.load(fwd_prot)
            if os.path.exists(fwd_kon):
                m.kon_theta = np.load(fwd_kon)

        if perturb_cfg is not None:
            apply_perturbation(m, adata_ref, perturb_cfg, ns)

        stim_sched = getattr(model_base, '_stim_schedule', None)
        m.simulate_network(sim_times)

        adata_sim = _prot_to_adata(m, adata_ref, ns, sim_times)
        adatas.append(adata_sim)

    return adatas


def compute_proportions_from_adatas(adatas, clf, color_map):
    """
    Prédit les types cellulaires et calcule les proportions pour chaque AnnData.
    Retourne array (n_sims, n_celltypes).
    """
    cell_types = list(color_map.keys())
    props = []
    for adata in adatas:
        a_pred = predict_cell_types(adata.copy(), clf, label_key=LABEL)
        counts = {ct: (a_pred.obs[LABEL].astype(str) == ct).sum()
                  for ct in cell_types}
        total  = sum(counts.values()) or 1
        props.append([counts[ct] / total for ct in cell_types])
    return np.array(props)  # (n_sims, n_celltypes)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principale
# ─────────────────────────────────────────────────────────────────────────────

def prepare_wt_context(p, split, dataset_group, n_sims):
    """
    Charge le modèle, entraîne le classifier, simule le WT n_sims fois.
    Retourne un dict réutilisable par toutes les perturbations du même dataset.
    """
    print(f"\n══ Chargement dataset {dataset_group} ({p}, split={split}) ══")
    model, adata, stim_sched = load_model(p, split)
    ns           = model.n_stimuli
    cardamom_dir = os.path.join(p, 'cardamomOT')

    pi_zinb_path = os.path.join(cardamom_dir, 'pi_zinb.npy')
    model._pi_zinb = np.load(pi_zinb_path) if os.path.exists(pi_zinb_path) else None

    basal_clean, basal_clean_3d, basal_t_clean = build_clean_baseline(p, model)
    model.basal   = basal_clean_3d.copy() if basal_clean_3d is not None else basal_clean.copy()
    model.basal_t = basal_t_clean.copy()
    model.production_factor = None

    full_path  = os.path.join(p, 'Data', 'data_full.h5ad')
    train_path = os.path.join(p, 'Data', 'data_train.h5ad')
    adata_full = ad.read_h5ad(full_path if os.path.exists(full_path) else train_path)

    if dataset_group == 'Kameneva' and LABEL in adata_full.obs:
        adata_full.obs[LABEL] = adata_full.obs[LABEL].astype(str).str.capitalize()
    if dataset_group == 'Semrau' and LABEL in adata_full.obs:
        adata_full.obs[LABEL] = adata_full.obs[LABEL].astype(str).str.replace('_', ' ', regex=False)

    clf       = train_classifier(adata_full, label_key=LABEL)
    cats      = adata_full.obs[LABEL].astype(str).unique().tolist() if LABEL in adata_full.obs else []
    color_map = {cat: '' for cat in cats}

    sim_times = _get_sim_times(p, model)
    print(f"  Timepoints : {sim_times}")
    print(f"  Simulation WT ({n_sims} runs)…")
    adatas_wt = simulate_n(model, adata, ns, sim_times, n_sims,
                           perturb_cfg=None, cardamom_dir=cardamom_dir)
    props_wt  = compute_proportions_from_adatas(adatas_wt, clf, color_map)

    return dict(
        model        = model,
        adata        = adata,
        ns           = ns,
        cardamom_dir = cardamom_dir,
        clf          = clf,
        cats         = cats,
        color_map    = color_map,
        sim_times    = sim_times,
        adatas_wt    = adatas_wt,
        props_wt     = props_wt,
    )


def process_perturbation(cfg, ctx, n_sims):
    """
    Simule n_sims fois la condition perturbée en réutilisant le contexte WT déjà calculé.
    Calcule aussi props_single : pour chaque run WT, substitue la valeur moyenne
    du gène cible (sur les runs perturbés) et reclassifie — sans resimulation.
    Retourne dict avec props_wt, props_single, props_perturb, cell_types, stab_score.
    """
    print(f"\n── {cfg['perturb_id']} ({cfg['dataset_group']}) ──")
    print(f"  Simulation perturbée ({n_sims} runs) — KO:{cfg['KO']}  OV:{cfg['OV']}…")

    adatas_pert = simulate_n(
        ctx['model'], ctx['adata'], ctx['ns'], ctx['sim_times'], n_sims,
        perturb_cfg=cfg, cardamom_dir=ctx['cardamom_dir'],
    )
    props_pert = compute_proportions_from_adatas(adatas_pert, ctx['clf'], ctx['color_map'])
    stab_score = float(np.mean(props_pert.std(axis=0)))
    print(f"  σ̄ (stabilité perturbée) = {stab_score:.4f}")

    # Sim single : substitution du gène cible dans chaque run WT existant
    props_single = None
    gene     = cfg.get('gene')
    var_names = list(ctx['adata'].var_names)
    if gene and gene in var_names:
        idx_gene = var_names.index(gene)
        mean_pert_val = float(np.mean([
            a.X[:, idx_gene].toarray().mean() if hasattr(a.X, 'toarray')
            else a.X[:, idx_gene].mean()
            for a in adatas_pert
        ]))
        adatas_single = []
        for a_wt in ctx['adatas_wt']:
            a_s = a_wt.copy()
            if hasattr(a_s.X, 'toarray'):
                import scipy.sparse as _sp
                _X = a_s.X.toarray(); _X[:, idx_gene] = mean_pert_val
                a_s.X = _sp.csr_matrix(_X)
            else:
                a_s.X = a_s.X.copy(); a_s.X[:, idx_gene] = mean_pert_val
            adatas_single.append(a_s)
        props_single = compute_proportions_from_adatas(adatas_single, ctx['clf'], ctx['color_map'])
        print(f"  Sim single calculé ({n_sims} runs, gène '{gene}')")

    return dict(
        props_wt      = ctx['props_wt'],
        props_single  = props_single,
        props_perturb = props_pert,
        cell_types    = np.array(ctx['cats'], dtype=str),
        stab_score    = stab_score,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main(n_sims=N_SIMS, out_dir='for_figure7_data', perturbations=PERTURBATIONS):
    os.makedirs(out_dir, exist_ok=True)

    # Grouper les perturbations par (path, split) pour ne simuler le WT qu'une fois
    key_fn = lambda cfg: (cfg['path'], cfg['split'])
    groups = {}
    for cfg in perturbations:
        k = key_fn(cfg)
        groups.setdefault(k, []).append(cfg)

    for (p, split), cfgs in groups.items():
        # Vérifier si toutes les perturbations du groupe sont déjà calculées
        pending = [cfg for cfg in cfgs
                   if not os.path.exists(os.path.join(out_dir, f"{cfg['perturb_id']}.npz"))]
        for cfg in cfgs:
            if cfg not in pending:
                print(f"[{cfg['perturb_id']}] déjà calculé — ignoré (supprimer pour recalculer)")

        if not pending:
            continue

        # Charger le modèle et simuler le WT une seule fois pour ce groupe
        dataset_group = pending[0]['dataset_group']
        try:
            ctx = prepare_wt_context(p, split, dataset_group, n_sims)
        except Exception as e:
            print(f"  ERREUR chargement dataset ({p}, {split}) : {e}")
            import traceback; traceback.print_exc()
            continue

        for cfg in pending:
            perturb_id = cfg['perturb_id']
            out_path   = os.path.join(out_dir, f'{perturb_id}.npz')
            try:
                result = process_perturbation(cfg, ctx, n_sims)
                save_kw = dict(
                    props_wt      = result['props_wt'],
                    props_perturb = result['props_perturb'],
                    cell_types    = result['cell_types'],
                    stab_score    = np.array([result['stab_score']]),
                )
                if result['props_single'] is not None:
                    save_kw['props_single'] = result['props_single']
                np.savez(out_path, **save_kw)
                print(f"  → sauvegardé : {out_path}")
            except Exception as e:
                print(f"  ERREUR pour {perturb_id} : {e}")
                import traceback; traceback.print_exc()

    print("\n_for_figure7.py terminé.")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-sims', type=int, default=N_SIMS,
                    help='Nombre de simulations stochastiques par condition')
    ap.add_argument('--out-dir', type=str, default='for_figure7_data',
                    help='Dossier de sortie pour les .npz')
    args = ap.parse_args()
    main(n_sims=args.n_sims, out_dir=args.out_dir)
