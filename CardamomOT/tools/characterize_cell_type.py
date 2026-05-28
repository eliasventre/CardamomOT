import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Entraînement du modèle
def train_classifier(adata, label_key='cell_type'):
    X = adata.X
    y = adata.obs[label_key]

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)

    return clf

# 2. Prédiction sur un nouvel AnnData
def predict_cell_types(adata_new, clf, label_key='cell_type'):
    preds = clf.predict(adata_new.X)
    
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

    adata_train = predict_cell_types(adata_train, clf, label_key=label_key)
    adata_beta = predict_cell_types(adata_beta, clf, label_key=label_key)
    adata_sim_train = predict_cell_types(adata_sim_train, clf, label_key=label_key)
    adata_theta_train = predict_cell_types(adata_theta_train, clf, label_key=label_key)

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
    adata_train = predict_cell_types(adata_train, clf, label_key=label_key)
    adata_train.write(p + 'cardamomOT/adata_beta.h5ad')

    plot_cell_type_proportions(
        adatas=[adata_full, adata_train],
        labels=["train", "mixture train"],
        label_key=label_key,
    )