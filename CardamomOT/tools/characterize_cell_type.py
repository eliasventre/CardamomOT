import scanpy as sc
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