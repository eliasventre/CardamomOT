import sys; sys.path += ['../']
import numpy as np
import pandas as pd
import anndata as ad
import getopt

verb = 1

def main(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            inputfile = arg

    p = '{}/'.format(inputfile)  # Name of the file where are the data

    fname_panel = p+'Data/panel_real.txt'
    fname_genes = p+'Data/panel_genes.txt'
    # 🔹 Lecture de la matrice : on lit tout d'abord brut
    raw_matrix = np.loadtxt(fname_panel, delimiter='\t').astype(np.int64)

    # 🔹 Extraction des temps (première ligne, à partir de la deuxième colonne)
    time = raw_matrix[0, 1:]

    # 🔹 Extraction des données d'expression (à partir de la 2e ligne, 2e colonne)
    data_rna = raw_matrix[2:, 1:].T  # on transpose pour avoir (cellules x gènes)

    # 🔹 Lecture des noms de gènes
    genes_df = pd.read_csv(fname_genes, sep='\t', header=None, names=['gene_id', 'gene_name'])

    # Vérification de la correspondance
    assert data_rna.shape[1] == genes_df.shape[0]-1, "Incohérence entre nombre de gènes et noms"

    # 🔹 Création de l'objet AnnData
    adata = ad.AnnData(X=data_rna)

    # ➕ Ajout des noms de gènes
    adata.var_names = genes_df['gene_name'].values[1:].astype(str)
    print(adata.var_names)

    # ➕ Ajout des temps dans adata.obs
    adata.obs['time'] = time.astype(int)

    # 💾 Sauvegarde
    adata.write(p+'Data/data.h5ad')

    print(adata)

    # 🔹 Rechargement et extraction de la matrice Numpy
    adata = ad.read_h5ad(p+'Data/data.h5ad')

    # ⬇️ Extraction de la matrice de comptes (X) et ajout de la ligne des temps
    data_rna_extracted = adata.X.T

    # 🚀 Ajouter la ligne des temps au début de la matrice
    data_rna_with_time = np.vstack([adata.obs['time'].values, data_rna_extracted]).T

    data_rna = np.loadtxt(fname_panel, delimiter='\t').astype(np.int64)[1:, 1:].T
    data_rna[:, 0] = np.loadtxt(fname_panel, delimiter='\t').astype(np.int64)[0, 1:]

    print(data_rna.shape, data_rna_with_time.shape)

    print(np.linalg.norm(data_rna_with_time - data_rna))

if __name__ == "__main__":
   main(sys.argv[1:])

