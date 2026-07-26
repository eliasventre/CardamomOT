"""
Network analysis and visualisation utilities for inferred GRNs.
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc


def analyse_reseau(matrix, gene_names=None, 
                   top_regulateurs=10, top_cibles=10, width_scale=5):
    """Plot per-regulator subgraphs for the top regulators in the GRN.

    Parameters
    ----------
    matrix : np.ndarray or pd.DataFrame
        G×G interaction matrix (positive = activation, negative = inhibition).
    gene_names : list of str, optional
        Gene names matching matrix rows/columns.
    top_regulateurs : int
        Number of top regulators to show.
    top_cibles : int
        Number of top targets per regulator.
    width_scale : float
        Multiplier for edge width (proportional to |weight|).
    """
    matrix = matrix.copy()
    matrix -= np.diag(np.diag(matrix))

    if isinstance(matrix, np.ndarray):
        matrix = pd.DataFrame(matrix)
    n = matrix.shape[0]

    if gene_names is None or len(gene_names) != n:
        gene_names = [f"G{i}" for i in range(n)]
    matrix.index = matrix.columns = gene_names

    score_sortant = matrix.abs().sum(axis=1)
    top_regs = score_sortant.nlargest(top_regulateurs).index

    max_intensity = matrix.abs().max().max()
    if max_intensity == 0:
        max_intensity = 1

    subgraphs = []
    for reg in top_regs:
        G = nx.DiGraph()
        series = matrix.loc[reg]
        top_idx = series.abs().nlargest(top_cibles).index
        top_targets = series.loc[top_idx]
        for cible, intensite in top_targets.items():
            if intensite != 0:
                G.add_edge(reg, cible, weight=float(intensite))
        subgraphs.append(G)

    ncols = 5
    nrows = int(np.ceil(len(subgraphs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3 * nrows))
    axes = axes.flatten()

    for i, (G, reg) in enumerate(zip(subgraphs, top_regs)):
        ax = axes[i]
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightblue", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

        edges_pos = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if d['weight'] > 0]
        edges_neg = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if d['weight'] < 0]

        widths_pos = [abs(w) / max_intensity * width_scale for (_, _, w) in edges_pos]
        widths_neg = [abs(w) / max_intensity * width_scale for (_, _, w) in edges_neg]

        if edges_pos:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for (u, v, _) in edges_pos],
                                   edge_color="green", width=widths_pos, arrows=True, ax=ax)
        if edges_neg:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for (u, v, _) in edges_neg],
                                   edge_color="red", width=widths_neg, arrows=True, ax=ax)

        ax.set_title(f"{reg} (régulateur top #{i+1})", fontsize=10)
        ax.axis("off")

    for j in range(len(subgraphs), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def reseau_top_regulateurs(matrix, gene_names=None,
                            name="Réseau réduit top régulateurs",
                            top_regulators=10, top_targets_per_reg=8,
                            leaf_threshold=0.0, width_scale=5):
    """Draw a reduced GRN showing only top regulators and their top targets.

    Parameters
    ----------
    matrix : np.ndarray or pd.DataFrame
        G×G interaction matrix.
    gene_names : list of str, optional
        Gene names matching matrix rows/columns.
    name : str
        Figure title.
    top_regulators : int
        Number of top regulators (by total outgoing strength).
    top_targets_per_reg : int
        Maximum targets per regulator.
    leaf_threshold : float
        Minimum absolute weight (as fraction of max) to include a target.
    width_scale : float
        Multiplier for edge width.

    Returns
    -------
    nx.DiGraph
        The reduced graph.
    """
    matrix = matrix.copy()
    matrix -= np.diag(np.diag(matrix))

    if isinstance(matrix, np.ndarray):
        matrix = pd.DataFrame(matrix)
    n = matrix.shape[0]

    if gene_names is None or len(gene_names) != n:
        gene_names = [f"G{i}" for i in range(n)]
    matrix.index = matrix.columns = gene_names

    score_sortant = matrix.abs().sum(axis=1)
    top_regs = list(score_sortant.nlargest(top_regulators).index)

    max_intensity = matrix.abs().max().max()
    th_abs = leaf_threshold * max_intensity

    G = nx.DiGraph()
    for reg in top_regs:
        series = matrix.loc[reg].abs().sort_values(ascending=False)
        targets = series.iloc[:top_targets_per_reg]
        targets = targets[targets >= th_abs]
        for cible in targets.index:
            w = matrix.loc[reg, cible]
            if w != 0:
                G.add_edge(reg, cible, weight=float(w))

    plt.figure(figsize=(14, 10))
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    node_colors = ["skyblue" if n in top_regs else "lightgray" for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900)
    nx.draw_networkx_labels(G, pos, font_size=9)

    edges_pos = [(u, v, d["weight"]) for u, v, d in G.edges(data=True) if d["weight"] > 0]
    edges_neg = [(u, v, d["weight"]) for u, v, d in G.edges(data=True) if d["weight"] < 0]
    wp = [abs(w) / max_intensity * width_scale for (_, _, w) in edges_pos]
    wn = [abs(w) / max_intensity * width_scale for (_, _, w) in edges_neg]

    if edges_pos:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for (u, v, _) in edges_pos],
                               edge_color="green", width=wp, arrows=True)
    if edges_neg:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for (u, v, _) in edges_neg],
                               edge_color="red", width=wn, arrows=True)

    plt.title(f"{name} — top regulators with outgoing edges only", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return G


def plot_network(p, seuil=0.3, net_toplot='inter_simul', net_index=0, train="full", ns=1):
    """Visualise the inferred GRN for a project directory.

    Loads ``net_toplot.npy`` and plots the top regulators using
    :func:`analyse_reseau` and :func:`reseau_top_regulateurs`.

    Parameters
    ----------
    p : str
        Path to the project directory (trailing slash included).
    seuil : float
        Edge-weight threshold passed as ``leaf_threshold`` to
        :func:`reseau_top_regulateurs` (fraction of max weight).
    net_index : int
        Index along the third axis of ``net_toplot.npy`` to visualise
        (default 0).
    train : str
        Data split used to recover gene names (``"full"`` or ``"train"``).
    ns : int
        Number of leading rows/columns to skip in ``net_toplot.npy``
        (default 1, which drops the basal/stimulus node).
    """
    adata = sc.read_h5ad(f'{p}Data/data_{train}.h5ad')
    grn_mat = np.load(f'{p}cardamomOT/{net_toplot}.npy')[ns:, ns:]
    genes_init = list(adata.var_names)
    if grn_mat.ndim < 3: 
        grn_slice = grn_mat
    else:
        grn_slice = grn_mat[:, :, net_index]

    analyse_reseau(grn_slice, genes_init, top_regulateurs=1000)
    reseau_top_regulateurs(
        grn_slice,
        gene_names=genes_init,
        top_regulators=10,
        leaf_threshold=seuil,
        width_scale=5,
    )

    print(net_toplot)

    import seaborn as sns

    # Sommes par gène
    row_sums = grn_slice.sum(axis=1)  # sortant (out)
    col_sums = grn_slice.sum(axis=0)  # entrant (in)

    # DataFrame long
    df = pd.DataFrame({
        "value": np.concatenate([row_sums, col_sums]),
        "Type": (["Row_sums"] * len(row_sums)) + (["Col_sums"] * len(col_sums))
    })

    plt.figure(figsize=(5, 5))

    sns.violinplot(
        data=df,
        x="Type",
        y="value",
        inner=None,          # pas de box interne
        color="lightgrey"    # contour gris clair comme base
    )

    sns.stripplot(
        data=df,
        x="Type",
        y="value",
        hue="Type",
        dodge=False,
        palette={
            "Row_sums": "deepskyblue",   # bleu ciel
            "Col_sums": "violet"         # violet
        },
        size=4,
        alpha=0.8
    )

    plt.xlabel("")
    plt.ylabel("Sum of GRN weights")

    plt.legend(title="Type")
    plt.tight_layout()
    plt.show()