"""
Network analysis and visualisation utilities for inferred GRNs.
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc


def add_edges_from_matrix(network, inter_real, threshold=0.0, reg_max=5):
    """Add interactions to a Neko Network object from an interaction matrix.

    Parameters
    ----------
    network : neko.core.network.Network
        Network object with ``.nodes`` and ``.add_edge()`` attributes.
    inter_real : np.ndarray or pd.DataFrame
        Square G×G matrix of interaction weights.
    threshold : float
        Minimum absolute weight to create an edge.
    reg_max : int
        Maximum number of outgoing edges kept per regulator.
    """
    if isinstance(inter_real, pd.DataFrame):
        inter_real = inter_real.values

    genes = list(network.nodes["Genesymbol"])
    G = len(genes)
    assert inter_real.shape == (G, G), (
        "inter_real must be square and match the number of genes."
    )

    for j in range(G):
        out_strength = np.abs(inter_real[:, j])
        top_indices = np.argsort(out_strength)[::-1][:reg_max]
        for i in range(G):
            val = inter_real[i, j]
            if abs(val) > threshold and i in top_indices:
                interaction_type = "stimulation" if val > 0 else "inhibition"
                edge_df = pd.DataFrame([{
                    "source": genes[i],
                    "target": genes[j],
                    "type": interaction_type,
                    "references": "inferred_from_matrix",
                }])
                network.add_edge(edge_df)

    network.edges['Effect'] = network.edges['Type']


def analyse_reseau(matrix, gene_names=None, name="Réseau",
                   top_regulateurs=10, top_cibles=10, width_scale=5):
    """Plot per-regulator subgraphs for the top regulators in the GRN.

    Parameters
    ----------
    matrix : np.ndarray or pd.DataFrame
        G×G interaction matrix (positive = activation, negative = inhibition).
    gene_names : list of str, optional
        Gene names matching matrix rows/columns.
    name : str
        Title prefix for the figure.
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


def plot_networks(p, stim=1.0, prior=1.0, seuil=0.3, reg_max=5, train="full"):
    """Visualise the inferred GRN for a project directory.

    Loads ``inter_simul.npy`` and plots the top regulators using
    :func:`analyse_reseau` and :func:`reseau_top_regulateurs`.

    Parameters
    ----------
    p : str
        Path to the project directory (trailing slash included).
    stim : float
        Stimulus value (currently unused, kept for API consistency).
    prior : float
        Prior value (currently unused, kept for API consistency).
    seuil : float
        Edge-weight threshold for display.
    reg_max : int
        Maximum number of outgoing edges per regulator.
    train : str
        Data split used to recover gene names (``"full"`` or ``"train"``).
    """
    adata = sc.read_h5ad(f'{p}Data/data_{train}.h5ad')
    grn_mat = np.load(f'{p}cardamomOT/inter_simul.npy')[1:, 1:]
    genes_init = list(adata.var_names)

    analyse_reseau(grn_mat[:, :, 0], genes_init, top_regulateurs=1000)
    reseau_top_regulateurs(
        grn_mat[:, :, 0],
        gene_names=genes_init,
        name="Reduced network",
        top_regulators=10,
        leaf_threshold=0.01,
        width_scale=5,
    )
