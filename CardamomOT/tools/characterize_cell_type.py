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
    with the closest name. cols ({sample: column indices}, optional): genes each classifier reads (the genes of the
    sample's condition with select_genes_per_condition).
    """

    def __init__(self, clfs, sizes=None, assign=None, cols=None):
        self.clfs = dict(clfs)
        self.cols = dict(cols or {})
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

        def run(key, rows):
            Xr = X[rows] if rows is not None else X
            return self.clfs[key].predict(Xr[:, self.cols[key]] if key in self.cols else Xr)

        if use is not None:
            return run(self.sample_for(use), None)
        if sample_key not in adata.obs:
            print(f"[cell types] Warning: no obs['{sample_key}']: the classifier of the largest sample ({self.default}) is used")
            return run(self.default, None)
        samples = adata.obs[sample_key].astype(str).values
        out = np.empty(len(samples), dtype=object)
        for s_ in np.unique(samples):
            m = samples == s_
            out[m] = run(self.sample_for(s_), np.flatnonzero(m))
        return out


METHODS = {'random_forest': 'random_forest', 'rf': 'random_forest', 'random forest': 'random_forest',
           'forest': 'random_forest', 'logistic': 'logistic', 'logistic_regression': 'logistic',
           'logistic regression': 'logistic', 'lr': 'logistic'}


def _log1p(X):
    X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X, dtype=float)
    return np.log1p(np.maximum(X, 0.0))


def representation(adata, cell_depth=False):
    """
    Counts used to learn and assign the cell types (cell_depth_for_representation): with cell_depth, counts divided by
    obs['depth_factor'] (depth of the real cell, or of the cell a model draw mimics: data and draws then undergo the
    same transformation, hence the same noise), else the counts as they are. Without obs['depth_factor'] (draws of a
    run without depth factors, at the reference depth) the counts are returned unchanged.
    """
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
    X = X.astype(float)
    if cell_depth and 'depth_factor' in adata.obs:
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
    from ..inputs import depth_factor_used
    cfg = settings(StepOptions(p=str(p).rstrip('/') + '/'))
    # Counts / depth factor only when the run modelled the depth (data and model draws transformed alike)
    return dict(method=cfg.classifier_method,
                cell_depth=bool(cfg.cell_depth_for_representation) and depth_factor_used(p))


# 1. Entraînement du modèle
def train_classifier(adata, label_key='cell_type', sample_key=SAMPLE_KEY, min_cells=50, assign=None, n_estimators=100, seed=0,
                     method=None, cell_depth=False):
    """
    Classifier (method: 'random_forest' or 'logistic', see make_classifier; None = classifier_method of base.py, use
    classifier_options(project) for the value of a project) of obs[label_key] on the observed cells of
    adata, divided by their depth factor if cell_depth (representation; predict_cell_types then divides the model draws
    by the depth factor of the cells they mimic). With several samples (obs[sample_key]), one per sample (SampleClassifiers); a sample with
    fewer than min_cells cells or a single cell type has none and takes that of its `assign` or of the closest name.
    """
    if method is None:
        from ..model.base import NetworkModel
        method = NetworkModel(1).classifier_method
    X_all = representation(adata, cell_depth)

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
    # Genes of each sample's condition (select_genes_per_condition): each classifier reads only them
    from ..condition_genes import row_mask
    gm = row_mask(adata, samples)
    clfs, sizes, cols = {}, {}, {}
    for s_ in np.unique(samples):
        idx = np.flatnonzero(samples == s_)
        if len(idx) < min_cells or len(np.unique(y_all[idx])) < 2:
            print(f"[cell types] Warning: sample {s_} has {len(idx)} cells and {len(np.unique(y_all[idx]))} cell type(s): "
                  f"no classifier of its own")
            continue
        if gm is not None:
            cols[s_] = np.flatnonzero(gm[idx[0]])
        clfs[s_] = fit(X_all[idx][:, cols[s_]] if s_ in cols else X_all[idx], y_all[idx], s_)
        sizes[s_] = len(idx)
    if not clfs:
        clf = fit(X_all, y_all)
        clf.cell_depth = bool(cell_depth)
        return clf
    print(f"[cell types] One {METHODS.get(str(method).lower(), method)} classifier per sample: {sorted(clfs)}")
    clf = SampleClassifiers(clfs, sizes, assign, cols)
    clf.cell_depth = bool(cell_depth)
    return clf


# 2. Prédiction sur un nouvel AnnData
def predict_cell_types(adata_new, clf, label_key='cell_type', sample_key=SAMPLE_KEY, use=None):
    """Cell types of adata_new (obs[label_key]), in the representation of the classifier (see representation); with
    per-sample classifiers each cell takes that of its sample (use: of this sample)."""
    X = representation(adata_new, getattr(clf, 'cell_depth', False))
    preds = clf.predict(adata_new, sample_key=sample_key, use=use, X=X) if isinstance(clf, SampleClassifiers) \
        else clf.predict(X)

    adata_new.obs[label_key] = pd.Categorical(preds, ordered=True)

    return adata_new

# Model stages compared with the data, as check_sim_to_data writes them (cardamomOT/adata_<file>_<tag>.h5ad)
STAGE_FILES = (('Reference', 'rna_traj'), ('NB mixture', 'beta'), ('Network', 'theta'), ('Simulation', 'sim'))


def predicted_key(label_key='cell_type'):
    """obs key of the out-of-fold predicted labels of the observed cells."""
    return f'{label_key}_predicted'


def attach_samples(A, samples_idx, names):
    """
    obs['dataset_id'] of a stage when absent: trajectory states (one row per entry of data_samples.npy) or simulated
    cells (the first slots of data_samples.npy tiled over the simulated times). The cell types are then predicted with
    the classifier of the sample of each cell.
    """
    if A is None or SAMPLE_KEY in A.obs or samples_idx is None or not len(names) or 'time' not in A.obs:
        return A
    n, S = A.n_obs, len(samples_idx)
    t = A.obs['time'].to_numpy(dtype=float)
    N = int(np.sum(t == t.min())) if len(t) else 0
    if n == S:
        idx = samples_idx
    elif N and n % N == 0 and N <= S:
        idx = np.tile(samples_idx[:N], n // N)
    else:
        return A
    A.obs[SAMPLE_KEY] = pd.Categorical(np.asarray(names)[np.minimum(idx, len(names) - 1)])
    return A


def read_model_stages(p, tag, names):
    """
    Model stages of a run (STAGE_FILES, tag 'stim<s>_prior<q>'): obs['dataset_id'] attached when absent (names: sorted
    samples of the data), virtual trajectory states (obs['observed'] False: time not observed for their sample) left
    out. {stage: AnnData, None if the file is missing}; the same stages for the report and the notebooks.
    """
    import os
    import anndata as ad
    cdir = os.path.join(p, 'cardamomOT')
    samples_path = os.path.join(cdir, 'data_samples.npy')
    samples_traj = np.load(samples_path).astype(int) if os.path.exists(samples_path) else None
    out = {}
    for stage, name in STAGE_FILES:
        path = os.path.join(cdir, f'adata_{name}_{tag}.h5ad')
        A = ad.read_h5ad(path) if os.path.exists(path) else None
        if A is not None:
            attach_samples(A, samples_traj, names)  # one row per state (before the virtual states are left out)
            if 'observed' in A.obs and not A.obs['observed'].astype(bool).all():
                A = A[A.obs['observed'].astype(bool).values].copy()
        out[stage] = A
    return out


def predict_out_of_fold(adata, label_key='cell_type', sample_key=SAMPLE_KEY, n_folds=3, seed=0, **kw):
    """
    Cell types of the observed cells of adata predicted by classifiers trained without them (n_folds folds stratified
    by sample and label; train_classifier options kw): the labels the classifier gives to cells it has not seen, the
    reference of the model stages it classifies (the annotated labels differ from them by its errors).
    """
    import warnings
    from sklearn.model_selection import StratifiedKFold
    y = adata.obs[label_key].astype(str).values
    strata = y if sample_key not in adata.obs else adata.obs[sample_key].astype(str).values + '|' + y
    pred = np.empty(adata.n_obs, dtype=object)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The least populated class')
        folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), strata))
    for tr, te in folds:
        clf = train_classifier(adata[tr], label_key=label_key, sample_key=sample_key, seed=seed, **kw)
        part = adata[te].copy()
        predict_cell_types(part, clf, label_key='_oof', sample_key=sample_key)
        pred[te] = part.obs['_oof'].astype(str).values
    return pred


def reference_labels(ref, data_labels):
    """Labels of the Reference stage when it holds the counts of the real cells behind the states (obs['real_idx'],
    uns['real_counts'] from check_sim_to_data): those of these cells in data_labels (e.g. out-of-fold); else None."""
    if ref is None or 'real_idx' not in ref.obs or not bool(ref.uns.get('real_counts', False)):
        return None
    idx = ref.obs['real_idx'].to_numpy(dtype=int)
    if idx.min() < 0 or idx.max() >= len(data_labels):
        return None
    return np.asarray(data_labels)[idx]


def project_classifier(p, label_key='cell_type'):
    """
    Observed cells of a run (Data/data_<split>.h5ad) and the cell-type classifier of the report on them (per sample,
    classifier_method and representation of the project, assign = reference samples of the held-out ones), with its
    options (for predict_out_of_fold). Returns (adata, clf, options).
    """
    import os
    import anndata as ad
    from ..run_options import settings, StepOptions
    from ..inputs import removed_samples
    split = settings(StepOptions(p=str(p).rstrip('/') + '/')).split
    adata = ad.read_h5ad(os.path.join(p, 'Data', f'data_{split}.h5ad'))
    opts = dict(classifier_options(p), assign=removed_samples(p)[1])
    return adata, train_classifier(adata, label_key=label_key, **opts), opts


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


def check_cell_types_full(p, stim=1.0, prior=1.0, label_key='cell_type'):
    """
    Cell-type proportions of the data and of the model stages of a run, as in the report: same stage files
    (read_model_stages), same classifier (project_classifier), predictions in memory (nothing written). Bars: Data
    (annotated labels), Data (predicted) (out-of-fold predictions, the reference of the classified stages), Reference
    (out-of-fold label of the real cell behind each state), NB mixture, Network, Simulation. Returns the table (%).

    Parameters
    ----------
    p : str
        Path to the project directory.
    stim, prior : float
        Stimulus and prior values of the run (tag of the files).
    label_key : str
        Annotated cell types (obs).
    """
    adata, clf, opts = project_classifier(p, label_key)
    names = sorted(adata.obs[SAMPLE_KEY].astype(str).unique()) if SAMPLE_KEY in adata.obs else []
    stages = read_model_stages(p, f'stim{stim}_prior{prior}', names)
    missing = [k for k, A in stages.items() if A is None]
    if missing:
        raise FileNotFoundError(f"Missing check_sim_to_data outputs for {missing} (tag stim{stim}_prior{prior})")
    pred = predict_out_of_fold(adata, label_key=label_key, **opts)
    data_pred = adata.copy()
    data_pred.obs[label_key] = pd.Categorical(pred)
    for stage, A in stages.items():
        predict_cell_types(A, clf, label_key=label_key)
    ref = reference_labels(stages['Reference'], pred)
    if ref is not None:
        stages['Reference'].obs[label_key] = pd.Categorical(ref)
    return plot_cell_type_proportions(adatas=[adata, data_pred] + list(stages.values()),
                                      labels=['Data', 'Data (predicted)'] + list(stages), label_key=label_key)


def check_cell_types_mixture(p, label_key='cell_type'):
    """
    Cell types of the NB-mixture check (cardamomOT/adata_beta.h5ad of check_mixture_to_data: one draw per real cell)
    with the classifier of the report (project_classifier), against the annotated and the out-of-fold predicted labels
    of the data; nothing written. Returns the table (%).

    Parameters
    ----------
    p : str
        Path to the project directory.
    label_key : str
        Annotated cell types (obs).
    """
    import os
    import anndata as ad
    adata, clf, opts = project_classifier(p, label_key)
    draws = ad.read_h5ad(os.path.join(p, 'cardamomOT', 'adata_beta.h5ad'))
    # Files without obs['dataset_id']: one NB draw per cell of the data, in the same order
    if SAMPLE_KEY not in draws.obs and SAMPLE_KEY in adata.obs:
        if draws.n_obs == adata.n_obs and np.allclose(draws.obs['time'].astype(float).values,
                                                      adata.obs['time'].astype(float).values):
            draws.obs[SAMPLE_KEY] = adata.obs[SAMPLE_KEY].values
        else:
            print(f"[cell types] Warning: adata_beta has no obs['{SAMPLE_KEY}'] and does not match the data: "
                  "rerun check_mixture_to_data.py")
    predict_cell_types(draws, clf, label_key=label_key)
    data_pred = adata.copy()
    data_pred.obs[label_key] = pd.Categorical(predict_out_of_fold(adata, label_key=label_key, **opts))
    return plot_cell_type_proportions(adatas=[adata, data_pred, draws],
                                      labels=['Data', 'Data (predicted)', 'NB mixture (draws)'], label_key=label_key)

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
    from ..condition_genes import row_mask
    gm = row_mask(adata, smp)  # genes of each cell's condition (select_genes_per_condition)
    for idx in units:
        cols = slice(None) if gm is None else np.flatnonzero(gm[idx].all(axis=0))  # genes of the unit's samples
        Xu, Du = X[:, cols], D[:, cols]
        for tr, te in KFold(min(n_folds, len(idx)), shuffle=True, random_state=seed).split(idx):
            tr, te = idx[tr], idx[te]
            ver = {'data (cross-validated)': Xu[te], 'model draws': Du[te]}
            if modes is not None:
                keys = np.char.add(np.char.add(smp[te, None].astype('U'), times[te, None].astype('U')),
                                   np.asarray(modes)[te][:, cols].astype('U'))
                ver['data permuted within modes'] = _permute_within(Xu[te], keys, rng)
            for c, make in _identity_classifiers(seed).items():
                if len(np.unique(y[tr])) < 2:
                    for v in versions:
                        preds[(c, v)][te] = y[tr][0]
                    continue
                clf = make().fit(Xu[tr], y[tr])
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
