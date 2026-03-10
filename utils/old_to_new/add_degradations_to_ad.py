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

    fname_deg = p+'Rates/degradation_rates.txt'
    # 🔹 Lecture de la matrice : on lit tout d'abord brut
    degradations = np.loadtxt(fname_deg, delimiter='\t')

    # 🔹 Chargement de l'ad
    adata = ad.read_h5ad(p+'Data/data.h5ad')

    # 🚀 Ajouter la ligne des temps au début de la matrice
    adata.var['d0'] = degradations[1:, 0]
    adata.var['d1'] = degradations[1:, 1]

    # 💾 Sauvegarde
    adata.write(p+'Data/data.h5ad')

if __name__ == "__main__":
   main(sys.argv[1:])

