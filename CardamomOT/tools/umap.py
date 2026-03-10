from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
import os


def configure(ax, xlim, ylim):
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('UMAP1', fontsize=7, weight='bold')
    ax.set_ylabel('UMAP2', fontsize=7, weight='bold')
    ax.tick_params(axis='both', labelsize=7)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_data_umap_toref(data_ref_base, data_sim_base, times, file_from, file_to, complement, logscale=True):
    data_ref, data_sim = data_ref_base.copy(), data_sim_base.copy()
    if logscale:
        data_ref[1:, :], data_sim[1:, :] = np.log(1 + data_ref[1:, :]), np.log(1 + data_sim[1:, :])
    # Compute the UMAP projection
    reducer = UMAP(random_state=42, min_dist=.7)
    proj = reducer.fit(data_ref[1:,:].T)
    x_ref = proj.transform(data_ref[1:,:].T)
    x_sim = proj.transform(data_sim[1:,:].T)

    # Figure
    fig = plt.figure(figsize=(10, 4))
    grid = gs.GridSpec(2, 2, height_ratios=[1, 0.05], wspace=0.3, hspace=0.35)
    ax0 = plt.subplot(grid[0, 0])
    ax1 = plt.subplot(grid[0, 1])
    ax3 = plt.subplot(grid[1, :])

    # Panel settings
    opt = {'xy': (0, 1), 'xycoords': 'axes fraction', 'fontsize': 10,
           'textcoords': 'offset points', 'annotation_clip': False}

    # Timepoint colors
    T = len(times)
    cmap = [plt.get_cmap('viridis', T)(i) for i in range(T)]
    colors_ref = [cmap[np.argwhere(times==t)[0,0]] for t in data_ref[0, :]]
    colors_sim = [cmap[np.argwhere(times==t)[0,0]] for t in data_sim[0, :]]

    all_x = np.vstack([x_ref, x_sim])
    xlim = (all_x[:, 0].min(), all_x[:, 0].max())
    ylim = (all_x[:, 1].min(), all_x[:, 1].max())

    # A. Original data
    configure(ax0, xlim, ylim)
    title = 'Reference'
    ax0.annotate('A', xytext=(-11, 6), fontweight='bold', **opt)
    ax0.annotate(title, xytext=(3, 6), **opt)
    ax0.scatter(x_ref[:, 0], x_ref[:, 1], c=colors_ref, s=1, alpha=1)

    # B. Inferred network beta
    configure(ax1, xlim, ylim)
    title = 'Sampling'
    ax1.annotate('B', xytext=(-11, 6), fontweight='bold', **opt)
    ax1.annotate(title, xytext=(3, 6), **opt)
    ax1.scatter(x_sim[:, 0], x_sim[:, 1], c=colors_sim, s=1, alpha=1)
    ax1.set(xlim=ax0.get_xlim(), ylim=ax0.get_ylim())


    # Legend panel
    labels = [f'{int(times[k])}h' for k in range(T)]
    lines = [Line2D([0], [0], color=cmap[k], lw=5) for k in range(T)]
    ax3.legend(lines, labels, ncol=min(T,10), frameon=False, borderaxespad=0,
               loc='lower right', handlelength=1, fontsize=8.5)
    ax3.text(-0.02, 2.1, 'Timepoints:', transform=ax3.transAxes, fontsize=8.5, )
    ax3.axis('off')

    # Export the figure
    # Construire le chemin du dossier
    output_dir = os.path.join('.', file_from, file_to)

    # Créer le dossier s’il n’existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Construire le chemin du fichier
    output_path = os.path.join(output_dir, f'UMAP_{complement}.pdf')

    # Sauvegarder la figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)


def plot_data_umap_altogether(data_real_base, data_ref_base, data_beta_base, 
                              data_theta_base, data_sim_base, times_data, times_simul, file_from, file_to, complement, logscale=True):

    data_real, data_ref, data_beta, \
    data_theta, data_sim = data_real_base.copy(), data_ref_base.copy(), data_beta_base.copy(), \
                        data_theta_base.copy(), data_sim_base.copy()
    ncells_subset = 5000

    data_real, data_ref, data_beta, data_theta, data_sim = subset_cells(data_real, ncells_subset), \
    subset_cells(data_ref, ncells_subset), subset_cells(data_beta, ncells_subset), \
    subset_cells(data_theta, ncells_subset), subset_cells(data_sim, ncells_subset)
    
    if logscale:
        data_real[1:, :], data_ref[1:, :], data_beta[1:, :], data_theta[1:, :], data_sim[1:, :] = \
            np.log(1 + data_real[1:, :]), np.log(1 + data_ref[1:, :]), np.log(1 + data_beta[1:, :]),  \
            np.log(1 + data_theta[1:, :]), np.log(1 + data_sim[1:, :])
        
    # Compute the UMAP projection
    reducer = UMAP(random_state=42, min_dist=.7)
    proj = reducer.fit(data_real[1:,:].T)
    x_ref = proj.transform(data_ref[1:,:].T)
    x_sim = proj.transform(data_sim[1:,:].T)
    x_beta = proj.transform(data_beta[1:,:].T)
    x_theta = proj.transform(data_theta[1:,:].T)

    # Figure
    fig = plt.figure(figsize=(20, 4))
    grid = gs.GridSpec(2, 4, height_ratios=[1, 0.05], wspace=0.3)
    ax0 = plt.subplot(grid[0, 0])
    ax1 = plt.subplot(grid[0, 1])
    ax2 = plt.subplot(grid[0, 2])
    ax3 = plt.subplot(grid[0, 3])
    ax4 = plt.subplot(grid[1, :])

    # Panel settings
    opt = {'xy': (0, 1), 'xycoords': 'axes fraction', 'fontsize': 10,
           'textcoords': 'offset points', 'annotation_clip': False}

    # Timepoint colors
    times = np.unique(sorted(list(times_data) + list(times_simul)))
    times.sort()
    T = len(times)
    cmap = [plt.get_cmap('viridis', T)(i) for i in range(T)]
    colors_ref = [cmap[np.argwhere(times==t)[0,0]] for t in data_ref[0, :]]
    colors_sim = [cmap[np.argwhere(times==t)[0,0]] for t in data_sim[0, :]]

    all_x = np.vstack([x_ref, x_beta, x_theta, x_sim])
    xlim = (all_x[:, 0].min(), all_x[:, 0].max())
    ylim = (all_x[:, 1].min(), all_x[:, 1].max())

    # A. Original data
    configure(ax0, xlim, ylim)
    title = 'Reference'
    ax0.annotate('A', xytext=(-11, 6), fontweight='bold', **opt)
    ax0.annotate(title, xytext=(3, 6), **opt)
    ax0.scatter(x_ref[:, 0], x_ref[:, 1], c=colors_ref, s=2, alpha=1)

    # B. Inferred network beta
    configure(ax1, xlim, ylim)
    title = 'Sampling beta'
    ax1.annotate('B', xytext=(-11, 6), fontweight='bold', **opt)
    ax1.annotate(title, xytext=(3, 6), **opt)
    ax1.scatter(x_beta[:, 0], x_beta[:, 1], c=colors_ref, s=2, alpha=1)
    ax1.set(xlim=ax0.get_xlim(), ylim=ax0.get_ylim())

    # C. Inferred network theta
    configure(ax2, xlim, ylim)
    title = 'Sampling theta'
    ax2.annotate('C', xytext=(-11, 6), fontweight='bold', **opt)
    ax2.annotate(title, xytext=(3, 6), **opt)
    ax2.scatter(x_theta[:, 0], x_theta[:, 1], c=colors_ref, s=2, alpha=1)
    ax2.set(xlim=ax0.get_xlim(), ylim=ax0.get_ylim())

    # d. Inferred network sim
    configure(ax3, xlim, ylim)
    title = 'Simulations'
    ax3.annotate('d', xytext=(-11, 6), fontweight='bold', **opt)
    ax3.annotate(title, xytext=(3, 6), **opt)
    ax3.scatter(x_sim[:, 0], x_sim[:, 1], c=colors_sim, s=2, alpha=1)
    ax3.set(xlim=ax0.get_xlim(), ylim=ax0.get_ylim())


    # Legend panel
    labels = [f'{int(times[k])}h' for k in range(T)]
    lines = [Line2D([0], [0], color=cmap[k], lw=5) for k in range(T)]
    ax4.legend(lines, labels, ncols=min(T,15), frameon=False, borderaxespad=0,
               loc='lower right', handlelength=1, fontsize=8.5)
    ax4.text(.5, .8, 'Timepoints:', transform=ax4.transAxes, fontsize=8.5)
    ax4.axis('off')

    # Export the figure
    # Construire le chemin du dossier
    output_dir = os.path.join('.', file_from, file_to)

    # Créer le dossier s’il n’existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Construire le chemin du fichier
    output_path = os.path.join(output_dir, f'UMAP_{complement}.pdf')

    # Sauvegarder la figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)


def subset_cells(dataset, ncells):
    if dataset.shape[1]>ncells:
        random_indices = np.random.choice(dataset.shape[1], size=ncells, replace=False)
        dataset = dataset[:,random_indices]
    return dataset

