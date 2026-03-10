import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import ks_2samp as ks
import scipy.stats as stats
import os



def negative_binomial_pmf(x, k, c):
        return stats.nbinom.pmf(x, k, c/(c+1))


def compute_mixture_weights(data_reference, kz, c):
    """Calcule les poids de chaque composante pour chaque cellule et chaque gène."""
    weights = np.zeros((data_reference.shape[1], kz.shape[0], kz.shape[1]))
    
    for g in range(kz.shape[1]):  # Pour chaque gène
        for i in range(data_reference.shape[1]):  # Pour chaque cellule
            likelihoods = np.array([negative_binomial_pmf(data_reference[g, i], k, c[g]) for k in kz[:, g]])
            weights[i, :, g] = likelihoods / (np.sum(likelihoods) + 1e-16)  # Normalisation
    
    return weights


def plot_data_pmf_temporal(data_reference, kz, c, data_simulated, t_real, t_netw, names, 
                           file_from, file_to, complement):
    """Affiche les histogrammes et superpose le mélange de binomiales négatives."""
    
    rat = 5
    nb_by_pages = 10
    nb_genes = len(names)
    list_genes = np.arange(nb_genes)+1
    nb_pages = int(nb_genes / nb_by_pages) + 1

    weights = compute_mixture_weights(data_reference, kz, c)
    
    # Construire le chemin du dossier
    output_dir = os.path.join('.', file_from, file_to)
    os.makedirs(output_dir, exist_ok=True)
    # Construire le chemin du fichier PDF
    output_path = os.path.join(output_dir, f'Marginals_temporal_pmf_{complement}.pdf')
    # Utiliser PdfPages
    with PdfPages(output_path) as pdf:
        for i in range(nb_pages):
            fig, ax = plt.subplots(len(t_netw), min(nb_by_pages, nb_genes),
                                   figsize=(min(nb_by_pages, nb_genes) * rat, len(t_netw) * rat))
            
            if nb_genes - i*nb_by_pages < nb_by_pages and nb_by_pages < nb_genes:
                for j in range(nb_genes - i*nb_by_pages, nb_by_pages):
                    for cnt_t in range(len(t_real)):
                        ax[cnt_t, j].set_axis_off()
            
            for cnt_g, g in enumerate(list_genes[i*nb_by_pages:min((i+1)*nb_by_pages, nb_genes)]):
                n_max = max(np.quantile(data_reference[g, :], 1), np.quantile(data_simulated[g, :], 1)) + 1
                n_bins = 25
                x_vals = np.arange(0, n_max)
                
                for cnt_t, time in enumerate(t_real):
                    data_tmp_reference = data_reference[g, data_reference[0, :] == t_real[cnt_t]]
                    weights_tmp = weights[data_reference[0, :] == t_real[cnt_t], :, g]
                    
                    # Histogrammes des données
                    ax[cnt_t, cnt_g].hist(data_tmp_reference, density=True, bins=np.linspace(0, n_max, n_bins),
                                           color='grey', histtype='bar', alpha=0.7)
                    
                    # Mélange pondéré des binomiales négatives
                    mixture_pmf = np.zeros_like(x_vals, dtype=float)
                    for j in range(kz.shape[0]):
                        mixture_pmf += weights_tmp[:, j].mean() * negative_binomial_pmf(x_vals, kz[j, g], c[g])
                    
                    ax[cnt_t, cnt_g].plot(x_vals, mixture_pmf, color='red', linewidth=3, label='Mixture Model')
                    
                    if time == t_netw[-1]:
                        ax[-1, cnt_g].set_xlabel('mRNA (copies per cell)', fontsize=20)
                    if time == t_netw[0]:
                        ax[cnt_t, cnt_g].set_title(names[g-1], fontweight="bold", fontsize=30)
                    
                    ax[cnt_t, cnt_g].legend(labels=['Model (t = {}h)'.format(int(t_real[cnt_t])),
                                                     'Data (t = {}h)'.format(int(t_real[cnt_t])),
                                                     'Mixture Model'])
            
            pdf.savefig(fig)
            plt.close()



def plot_data_pmf_total(weights, data_reference, kz, c, data_simulated, names, 
                        file_from, file_to, complement):
    """Affiche les histogrammes et superpose le mélange de binomiales négatives."""
    
    rat = 5
    nb_by_pages = 10
    nb_genes = len(names)
    list_genes = np.arange(nb_genes)+1
    nb_pages = int(nb_genes / nb_by_pages) + 1
    
    # Construire le chemin du dossier
    output_dir = os.path.join('.', file_from, file_to)
    os.makedirs(output_dir, exist_ok=True)

    # Construire le chemin du fichier PDF
    output_path = os.path.join(output_dir, f'Marginals_total_pmf_{complement}.pdf')
    # Utiliser PdfPages
    with PdfPages(output_path) as pdf:
        for i in range(nb_pages):
            fig, ax = plt.subplots(1, min(nb_by_pages, nb_genes),
                                   figsize=(min(nb_by_pages, nb_genes) * rat, rat))
            
            if nb_genes - i*nb_by_pages < nb_by_pages and nb_by_pages < nb_genes:
                for j in range(nb_genes - i*nb_by_pages, nb_by_pages):
                    ax[j].set_axis_off()
            
            for cnt_g, g in enumerate(list_genes[i*nb_by_pages:min((i+1)*nb_by_pages, nb_genes)]):
                n_max = max(np.quantile(data_reference[g, :], 1), np.quantile(data_simulated[g, :], 1)) + 1
                n_bins = 100
                x_vals = np.arange(0, n_max)
                
                    
                # Histogrammes des données
                ax[cnt_g].hist(data_reference[g, :], density=True, bins=np.linspace(0, n_max, n_bins),
                                        color='grey', histtype='bar', alpha=0.7)
                
                # Mélange pondéré des binomiales négatives
                mixture_pmf = np.zeros_like(x_vals, dtype=float)
                for j in range(kz.shape[0]):
                    mixture_pmf += weights[j, g] * negative_binomial_pmf(x_vals, kz[j, g], c[g])
                
                ax[cnt_g].plot(x_vals, mixture_pmf, color='red', linewidth=3, label='Mixture Model')
                
                ax[cnt_g].set_xlabel('mRNA (copies per cell)', fontsize=20)
                ax[cnt_g].set_title(names[g-1], fontweight="bold", fontsize=30)
                
                ax[cnt_g].legend(labels=['Data','Mixture Model'])
            
            pdf.savefig(fig)
            plt.close()



def plot_data_distrib(data_reference, data_simulated, t_data, t_sim, names, file_from, file_to, complement):

    times = np.unique(sorted(list(t_data) + list(t_sim)))
    rat = 5
    nb_by_pages = 10
    nb_genes = len(names)
    list_genes = np.arange(nb_genes)+1
    nb_pages = int(nb_genes / nb_by_pages) + 1

    # Construire le chemin du dossier
    output_dir = os.path.join('.', file_from, file_to)
    os.makedirs(output_dir, exist_ok=True)

    # Construire le chemin du fichier PDF
    output_path = os.path.join(output_dir, f'Marginals_temporal_{complement}.pdf')
    # Utiliser PdfPages
    with PdfPages(output_path) as pdf:
        for i in range(nb_pages):
            fig, ax = plt.subplots(len(times), min(nb_by_pages, nb_genes),
                                   figsize=(min(nb_by_pages, nb_genes) * rat, len(times) * rat))
            if nb_genes - i*nb_by_pages < nb_by_pages and nb_by_pages < nb_genes:
                for j in range(nb_genes - i*nb_by_pages, nb_by_pages):
                    for cnt_t, time in enumerate(times):
                        ax[cnt_t, j].set_axis_off()
            for cnt_g, g in enumerate(list_genes[i*nb_by_pages:min((i+1)*nb_by_pages, nb_genes)]):
                n_max = max(np.quantile(data_reference[g, :], 1), np.quantile(data_simulated[g, :], 1)) + 1
                n_bins = 25
                for cnt_t, time in enumerate(times):
                    data_tmp_simulated = data_simulated[g, data_simulated[0, :] == times[cnt_t]]
                    data_tmp_reference = data_reference[g, data_reference[0, :] == times[cnt_t]]
                    if time == times[-1]: ax[-1, cnt_g].set_xlabel('mRNA (copies per cell)', fontsize=20)
                    if time == times[0]: ax[cnt_t, cnt_g].set_title(names[g-1], fontweight="bold", fontsize=30)
                    ax[cnt_t, cnt_g].hist(data_tmp_reference, density=True, bins=np.linspace(0, n_max, n_bins),
                                        color='grey', histtype='bar', alpha=0.7)
                    ax[cnt_t, cnt_g].hist(data_tmp_simulated, density=True, bins=np.linspace(0, n_max, n_bins),
                                                  ec='red', histtype=u'step', alpha=1, linewidth=2)
                    ax[cnt_t, cnt_g].legend(labels=['Model (t = {}h)'.format(int(times[cnt_t])),
                                                  'Reference (t = {}h)'.format(int(times[cnt_t]))])
            pdf.savefig(fig)
            plt.close()


def compare_marginals(data_real, data_netw, t_real, t_netw, genes, file_from, file_to, complement):


    T = len(t_real)
    G = len(genes)-1

    pval_netw = np.ones((T, G))
    for cnt_t in range(T):
        data_tmp_real = data_real[:,data_real[0,:] == t_real[cnt_t]]
        data_tmp_netw = data_netw[:,data_netw[0,:] == t_netw[cnt_t]]
        for cnt_g in range(1,G+1):
            stat_tmp = ks(data_tmp_real[cnt_g, :], data_tmp_netw[cnt_g, :])
            pval_netw[cnt_t, cnt_g-1] = stat_tmp[1]

    # Figure
    fig = plt.figure(figsize=(G,T+1))
    grid = gs.GridSpec(6, 4, wspace=0, hspace=0,
        width_ratios=[0.09,1.48,0.32,1],
        height_ratios=[0.49,0.2,0.031,0.85,0.22,0.516])
    panelA = grid[0,:]
    # Panel settings
    opt = {'xy': (0,1), 'xycoords': 'axes fraction', 'fontsize': 10,
        'textcoords': 'offset points', 'annotation_clip': False}

    # Color settings
    colors = ['#d73027','#f46d43','#fdae61','#fee08b','#ffffbf',
        '#d9ef8b','#a6d96a','#66bd63','#1a9850']

    # A. KS test p-values
    axA = plt.subplot(panelA)
    axA.annotate('A', xytext=(-14,6), fontweight='bold', **opt)
    axA.annotate('KS test p-values', xytext=(0,6), **opt)
    # axA.set_title('KS test p-values', fontsize=10)
    cmap = LinearSegmentedColormap.from_list('pvalue', colors)
    norm = Normalize(vmin=0, vmax=0.1)
    # Plot the heatmap
    im = axA.imshow(pval_netw, cmap=cmap, norm=norm)
    axA.set_aspect('equal','box')
    axA.set_xlim(-0.5,G-0.5)
    axA.set_ylim(T-0.5,-0.5)
    # Create colorbar
    divider = make_axes_locatable(axA)
    cax = divider.append_axes('right', '1.5%', pad='2%')
    cbar = axA.figure.colorbar(im, cax=cax, extend='max')
    pticks = np.array([0,1,3,5,7,9])
    cbar.set_ticks(pticks/100 + 0.0007)
    cbar.ax.set_yticklabels([0]+[f'{p}%' for p in pticks[1:]], fontsize=6)
    cbar.ax.spines[:].set_visible(False)
    cbar.ax.tick_params(axis='y',direction='out', length=1.5, pad=1.5)
    axA.set_xticks(np.arange(G))
    axA.set_yticks(np.arange(T))
    axA.set_xticklabels(genes[1:], rotation=45, ha='right', rotation_mode='anchor',
        fontsize=3)
    axA.set_yticklabels([f'{int(t)}h' for t in t_real], fontsize=6.5)
    axA.spines[:].set_visible(False)
    axA.set_xticks(np.arange(G+1)-0.5, minor=True)
    axA.set_yticks(np.arange(T+1)-0.5, minor=True)
    axA.grid(which='minor', color='w', linestyle='-', linewidth=1)
    axA.tick_params(which='minor', bottom=False, left=False)
    axA.tick_params(which='major', bottom=False, left=False)
    axA.tick_params(axis='x',direction='out', pad=-0.1)
    axA.tick_params(axis='y',direction='out', pad=-0.1)

    # Export the figure
    # Chemin du dossier cible
    output_dir = os.path.join('.', file_from, file_to)

    # Créer le dossier s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Chemin complet du fichier PDF
    output_path = os.path.join(output_dir, f'Comparison_{complement}.pdf')

    # Sauvegarde de la figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
