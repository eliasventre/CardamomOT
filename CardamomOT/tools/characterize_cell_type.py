import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

SAMPLE_KEY = 'dataset_id'


def closest_sample(name, candidates):
    """Candidate whose name is the closest to `name`: longest common prefix (RMS2_ctrl -> RMS2_trt), then string similarity."""
    import difflib
    name = str(name)

    def score(c):
        c = str(c)
        k = 0
        while k < min(len(name), len(c)) and name[k] == c[k]:
            k += 1
        return (k, difflib.SequenceMatcher(None, name, c).ratio())
    return max(candidates, key=score)


class SampleClassifiers:
    """
    One classifier of cell_type per sample (dataset_id), trained on the cells of that sample: samples differ too
    much (cell line, batch) for a single classifier, even after integration. A sample without classifier of its own
    (held out of the inference, observed at a single type...) is classified by the one indicated in `assign`
    ({sample: sample whose classifier is used}, e.g. the reference_sample of perturbation_inference), else by the one
    with the closest name.
    """

    def __init__(self, clfs, sizes=None, assign=None):
        self.clfs = dict(clfs)
        self.assign = {str(k): str(v) for k, v in (assign or {}).items()}
        sizes = sizes or {k: 1 for k in self.clfs}
        self.default = max(self.clfs, key=lambda k: sizes.get(k, 0))   # largest sample: when the sample is unknown

    @property
    def classes_(self):
        return sorted({c for clf in self.clfs.values() for c in clf.classes_})

    def sample_for(self, sample):
        """Sample whose classifier is used for `sample`."""
        sample = str(sample)
        if sample in self.clfs:
            return sample
        ind = self.assign.get(sample)
        if ind in self.clfs:
            return ind
        return closest_sample(sample, list(self.clfs))

    def predict(self, adata, sample_key=SAMPLE_KEY, use=None, X=None):
        """Cell types of the cells of adata (matrix X, default adata.X), each with the classifier of its sample (use:
        classifier of this sample for all)."""
        X = adata.X if X is None else X
        if use is not None:
            return self.clfs[self.sample_for(use)].predict(X)
        if sample_key not in adata.obs:
            print(f"[cell types] Warning: no obs['{sample_key}']: the classifier of the largest sample ({self.default}) is used")
            return self.clfs[self.default].predict(X)
        samples = adata.obs[sample_key].astype(str).values
        out = np.empty(len(samples), dtype=object)
        for s_ in np.unique(samples):
            m = samples == s_
            out[m] = self.clfs[self.sample_for(s_)].predict(X[np.flatnonzero(m)])
        return out


METHODS = {'random_forest': 'random_forest', 'rf': 'random_forest', 'random forest': 'random_forest',
           'forest': 'random_forest', 'logistic': 'logistic', 'logistic_regression': 'logistic',
           'logistic regression': 'logistic', 'lr': 'logistic'}


def _log1p(X):
    X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X, dtype=float)
    return np.log1p(np.maximum(X, 0.0))


def representation(adata, cell_depth=False, kind=None):
    """
    Counts used to learn and assign the cell types (cell_depth_for_representation): with cell_depth, real cells
    divided by obs['depth_factor'] and model draws at the reference depth (layer 'reference_depth'), else the counts
    as they are. kind: 'observed' (real cells, e.g. data or reference trajectories) or 'model' (NB draws); None =
    'model' if the AnnData has a 'reference_depth' layer or uns['model_draws'], else 'observed'.
    """
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
    X = X.astype(float)
    if not cell_depth:
        return X
    if kind is None:
        kind = 'model' if ('reference_depth' in adata.layers or adata.uns.get('model_draws', False)) else 'observed'
    if kind == 'model':
        if 'reference_depth' in adata.layers:
            L = adata.layers['reference_depth']
            return np.asarray(L.toarray() if hasattr(L, 'toarray') else L, dtype=float)
        return X  # drawn without depth factor (run without use_depth_factor)
    if 'depth_factor' in adata.obs:
        return X / adata.obs['depth_factor'].to_numpy(dtype=float)[:, None]
    return X


LOGISTIC_CS = np.logspace(-3, 1, 9)  # inverse L2 penalties tried by the cross-validation


def make_classifier(method='random_forest', seed=0, n_estimators=100):
    """Unfitted classifier: random forest on the counts, or logistic regression on standardised log1p counts (relies
    less on fine co-expression, which models with independent genes given their state cannot reproduce), with
    balanced class weights (rare types count as much as the others) and its L2 penalty C chosen by 3-fold
    cross-validation on the training cells (balanced accuracy; every time pooled). Both take the counts of
    representation (divided by the depth factor or not)."""
    key = METHODS.get(str(method).strip().lower())
    if key is None:
        raise ValueError(f"classifier_method '{method}': use 'random_forest' or 'logistic'")
    if key == 'random_forest':
        return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    return make_pipeline(FunctionTransformer(_log1p), StandardScaler(),
                         LogisticRegressionCV(Cs=LOGISTIC_CS, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                                              class_weight='balanced', scoring='balanced_accuracy', max_iter=2000,
                                              n_jobs=-1))


def _chosen_c(clf):
    """Penalty C chosen by the cross-validation of a logistic pipeline, None otherwise."""
    last = clf[-1] if hasattr(clf, 'steps') else None
    return float(np.ravel(last.C_)[0]) if last is not None and hasattr(last, 'C_') else None


def classifier_method(p):
    """classifier_method of a project (Model_parameters sheet, else default of base.py)."""
    return classifier_options(p)['method']


def classifier_options(p):
    """Cell-type classifier options of a project for train_classifier (notebooks): dict(method = classifier_method,
    cell_depth = cell_depth_for_representation and a depth factor computed)."""
    from ..run_options import settings, StepOptions
    cfg = settings(StepOptions(p=str(p).rstrip('/') + '/'))
    return dict(method=cfg.classifier_method, cell_depth=bool(cfg.cell_depth_for_representation))


# 1. Entraînement du modèle
def train_classifier(adata, label_key='cell_type', sample_key=SAMPLE_KEY, min_cells=50, assign=None, n_estimators=100, seed=0,
                     method=None, cell_depth=False):
    """
    Classifier (method: 'random_forest' or 'logistic', see make_classifier; None = classifier_method of base.py, use
    classifier_options(project) for the value of a project) of obs[label_key] on the observed cells of
    adata, divided by their depth factor if cell_depth (representation; predict_cell_types then uses the model draws
    at the reference depth). With several samples (obs[sample_key]), one per sample (SampleClassifiers); a sample with
    fewer than min_cells cells or a single cell type has none and takes that of its `assign` or of the closest name.
    """
    if method is None:
        from ..model.base import NetworkModel
        method = NetworkModel(1).classifier_method
    X_all = representation(adata, cell_depth, 'observed')

    def fit(X, y, tag=''):
        clf = make_classifier(method, seed, n_estimators).fit(X, y)
        c = _chosen_c(clf)
        if c is not None:
            print(f"[cell types] {tag or 'all cells'}: logistic regression, C = {c:g} (cross-validated, balanced classes)")
        return clf

    y_all = adata.obs[label_key].astype(str).values
    if sample_key not in adata.obs or adata.obs[sample_key].astype(str).nunique() < 2:
        clf = fit(X_all, y_all)
        clf.cell_depth = bool(cell_depth)
        return clf
    samples = adata.obs[sample_key].astype(str).values
    clfs, sizes = {}, {}
    for s_ in np.unique(samples):
        idx = np.flatnonzero(samples == s_)
        if len(idx) < min_cells or len(np.unique(y_all[idx])) < 2:
            print(f"[cell types] Warning: sample {s_} has {len(idx)} cells and {len(np.unique(y_all[idx]))} cell type(s): "
                  f"no classifier of its own")
            continue
        clfs[s_] = fit(X_all[idx], y_all[idx], s_)
        sizes[s_] = len(idx)
    if not clfs:
        clf = fit(X_all, y_all)
        clf.cell_depth = bool(cell_depth)
        return clf
    print(f"[cell types] One {METHODS.get(str(method).lower(), method)} classifier per sample: {sorted(clfs)}")
    clf = SampleClassifiers(clfs, sizes, assign)
    clf.cell_depth = bool(cell_depth)
    return clf


# 2. Prédiction sur un nouvel AnnData
def predict_cell_types(adata_new, clf, label_key='cell_type', sample_key=SAMPLE_KEY, use=None, kind=None):
    """Cell types of adata_new (obs[label_key]), in the representation of the classifier (kind: 'observed' or 'model',
    see representation); with per-sample classifiers each cell takes that of its sample (use: of this sample)."""
    X = representation(adata_new, getattr(clf, 'cell_depth', False), kind)
    preds = clf.predict(adata_new, sample_key=sample_key, use=use, X=X) if isinstance(clf, SampleClassifiers) \
        else clf.predict(X)

    adata_new.obs[label_key] = pd.Categorical(preds, ordered=True)

    return adata_new

# 3. Création du plot de proportions
def plot_cell_type_proportions(adatas, labels, label_key='cell_type', colors=None):
    proportions = []

    for adata, label in zip(adatas, labels):
        counts = adata.obs[label_key].value_counts(normalize=True) * 100
        df = pd.DataFrame(counts).T
        df.index = [label]
        proportions.append(df)

    prop_df = pd.concat(proportions).fillna(0)

    # Plot
    ax = prop_df.plot(kind='bar', stacked=True, figsize=(8, 6))
    plt.ylabel('Percentage')
    plt.xlabel('Sample')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    print(prop_df)

    return prop_df


def check_cell_types_full(clf, p, stim=1.0, prior=1.0, label_key='cell_type'):
    """Predict cell types on all pipeline outputs and plot proportions.

    Runs the classifier on RNA trajectories, NB mixture, network modes and
    simulation, writes predictions back to the h5ad files, then plots stacked
    bar proportions.

    Parameters
    ----------
    clf : sklearn classifier
        Trained classifier (e.g. from :func:`train_classifier`).
    p : str
        Path to the project directory (trailing slash included).
    stim : float
        Stimulus value used during inference.
    prior : float
        Prior value used during inference.
    label_key : str
        ``obs`` key to store predictions in.
    """
    import anndata as ad
    import matplotlib.pyplot as plt
    s, q = stim, prior
    adata_train = ad.read_h5ad(p + f'cardamomOT/adata_rna_traj_stim{s}_prior{q}.h5ad')
    adata_beta = ad.read_h5ad(p + f'cardamomOT/adata_beta_stim{s}_prior{q}.h5ad')
    adata_sim_train = ad.read_h5ad(p + f'cardamomOT/adata_sim_stim{s}_prior{q}.h5ad')
    adata_theta_train = ad.read_h5ad(p + f'cardamomOT/adata_theta_stim{s}_prior{q}.h5ad')

    # Reference trajectories = real cells; the others = model draws (representation of the classifier)
    adata_train = predict_cell_types(adata_train, clf, label_key=label_key, kind='observed')
    adata_beta = predict_cell_types(adata_beta, clf, label_key=label_key, kind='model')
    adata_sim_train = predict_cell_types(adata_sim_train, clf, label_key=label_key, kind='model')
    adata_theta_train = predict_cell_types(adata_theta_train, clf, label_key=label_key, kind='model')

    adata_train.write(p + f'cardamomOT/adata_rna_traj_stim{s}_prior{q}.h5ad')
    adata_beta.write(p + f'cardamomOT/adata_beta_stim{s}_prior{q}.h5ad')
    adata_sim_train.write(p + f'cardamomOT/adata_sim_stim{s}_prior{q}.h5ad')
    adata_theta_train.write(p + f'cardamomOT/adata_theta_stim{s}_prior{q}.h5ad')

    cmap_cat = plt.get_cmap('Dark2')
    cats = adata_train.obs[label_key].astype(str).unique().tolist()
    colors = [cmap_cat(i % 8) for i in range(len(cats))]

    plot_cell_type_proportions(
        adatas=[adata_train, adata_beta, adata_theta_train, adata_sim_train],
        labels=["data", "NB mixture", "modes", "sim"],
        label_key=label_key,
        colors=colors,
    )


def check_cell_types_mixture(clf, p, adata_full, stim=1.0, prior=1.0, label_key='cell_type'):
    """Predict cell types on the NB mixture and compare to observed data.

    Parameters
    ----------
    clf : sklearn classifier
        Trained classifier.
    p : str
        Path to the project directory (trailing slash included).
    adata_full : AnnData
        Observed data AnnData (used as the reference proportion).
    stim : float
        Stimulus value (unused here, kept for API consistency).
    prior : float
        Prior value (unused here, kept for API consistency).
    label_key : str
        ``obs`` key to store/read predictions in.
    """
    import anndata as ad
    adata_train = ad.read_h5ad(p + 'cardamomOT/adata_beta.h5ad')
    # Files written before obs['dataset_id'] was saved: one NB draw per cell of the data, in the same order
    if SAMPLE_KEY not in adata_train.obs and SAMPLE_KEY in adata_full.obs:
        if adata_train.n_obs == adata_full.n_obs and np.allclose(adata_train.obs['time'].astype(float).values,
                                                                 adata_full.obs['time'].astype(float).values):
            adata_train.obs[SAMPLE_KEY] = adata_full.obs[SAMPLE_KEY].values
        else:
            print(f"[cell types] Warning: adata_beta has no obs['{SAMPLE_KEY}'] and does not match adata_full: "
                  "rerun check_mixture_to_data.py")
    adata_train = predict_cell_types(adata_train, clf, label_key=label_key, kind='model')
    adata_train.write(p + 'cardamomOT/adata_beta.h5ad')

    plot_cell_type_proportions(
        adatas=[adata_full, adata_train],
        labels=["train", "mixture train"],
        label_key=label_key,
    )

# 4. Cell-type identity kept by a model (fair baseline for the proportions)
def _identity_classifiers(seed=0):
    """Both classifiers of make_classifier, by display name."""
    return {'random forest': lambda: make_classifier('random_forest', seed),
            'logistic regression': lambda: make_classifier('logistic', seed)}


def _permute_within(X, keys, rng):
    """Counts of each gene permuted among the cells sharing its key (keys: (N, G) labels)."""
    P = X.copy()
    for j in range(X.shape[1]):
        k = keys[:, j]
        for v in np.unique(k):
            i = np.flatnonzero(k == v)
            P[i, j] = X[rng.permutation(i), j]
    return P


def identity_diagnostic(adata, draws, modes=None, label_key='cell_type', sample_key=SAMPLE_KEY, n_folds=3,
                        min_cells=50, seed=0):
    """
    Cell-type identity kept by a model giving one draw per real cell (draws: (N, G) aligned with adata, e.g. the NB
    mixture of check_mixture_to_data). Cross-fitting within each sample (pooled for a sample with fewer than
    min_cells cells or a single type): classifiers trained on the real cells of the other folds predict the held-out
    cells as
      - 'data (cross-validated)': the real cells: the baseline, limit of the classifier on these genes;
      - 'data permuted within modes' (modes: (N, G) mode of each cell and gene): counts of each gene permuted among
        the held-out cells of the same sample, time and mode: the best a model with these modes and independent
        genes given them can do;
      - 'model draws': the draws.
    Returns a DataFrame (classifier, version, cell_type, share, recall, n_cells): share of the cells predicted as
    cell_type, and recall = fraction of the cells of that true type predicted as it.
    """
    from sklearn.model_selection import KFold
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
    X = X.astype(float)
    D = np.asarray(draws, dtype=float)
    y = adata.obs[label_key].astype(str).values
    smp = adata.obs[sample_key].astype(str).values if sample_key in adata.obs else np.full(len(y), '0')
    times = adata.obs['time'].astype(str).values if 'time' in adata.obs else np.full(len(y), '0')
    rng = np.random.default_rng(seed)
    versions = ['data (cross-validated)'] + (['data permuted within modes'] if modes is not None else []) + ['model draws']
    units = []
    pooled = []
    for s_ in np.unique(smp):
        idx = np.flatnonzero(smp == s_)
        (units if len(idx) >= min_cells and len(np.unique(y[idx])) > 1 else pooled).append(idx)
    if pooled:
        units.append(np.concatenate(pooled))
    preds = {(c, v): np.empty(len(y), dtype=object) for c in _identity_classifiers() for v in versions}
    for idx in units:
        for tr, te in KFold(min(n_folds, len(idx)), shuffle=True, random_state=seed).split(idx):
            tr, te = idx[tr], idx[te]
            ver = {'data (cross-validated)': X[te], 'model draws': D[te]}
            if modes is not None:
                keys = np.char.add(np.char.add(smp[te, None].astype('U'), times[te, None].astype('U')),
                                   np.asarray(modes)[te].astype('U'))
                ver['data permuted within modes'] = _permute_within(X[te], keys, rng)
            for c, make in _identity_classifiers(seed).items():
                if len(np.unique(y[tr])) < 2:
                    for v in versions:
                        preds[(c, v)][te] = y[tr][0]
                    continue
                clf = make().fit(X[tr], y[tr])
                for v in versions:
                    preds[(c, v)][te] = clf.predict(ver[v])
    rows = []
    for ct in sorted(np.unique(y)):
        true = y == ct
        rows.append(dict(classifier='', version='observed', cell_type=ct, share=float(true.mean()), recall=np.nan,
                         n_cells=int(true.sum())))
        for (c, v), p in preds.items():
            rows.append(dict(classifier=c, version=v, cell_type=ct, share=float(np.mean(p == ct)),
                             recall=float(np.mean(p[true] == ct)), n_cells=int(true.sum())))
    return pd.DataFrame(rows)
