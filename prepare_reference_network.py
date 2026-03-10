import sys; sys.path += ['../']
import numpy as np
from CardamomOT import NetworkModel as NetworkModel_beta
import anndata as ad
import pandas as pd
import getopt
import scipy.sparse
from neko.core.network import Network
from neko._visual.visualize_network import NetworkVisualizer
import omnipath as op
import os

verb = 1

"""
Script utility to populate `ref_network` edges from `neko_network` according to the rules
you specified.

Main function:
    build_edges_from_neko(neko_network, ref_network, max_path_length=6)

Behavior summary (implemented exactly as requested):
 - For each ordered pair A->B of nodes present in ref_network:
   * If a direct edge A->B exists in neko_network, it is copied as-is into ref_network.
   * Otherwise, find all simple paths A->...->B in neko_network such that **no intermediate
     node of the path belongs to ref_network**. For each such path we compute the cumulative
     effect by multiplying the signs of intermediate edges (stimulation=+1, inhibition=-1, other=>=+1).
     If one or more valid paths exist, an aggregated propagated edge A->B is created and added to
     ref_network via `ref_network.add_edge(edge_df)`.
 - If multiple distinct paths produce the same signed effect, their references are aggregated
   together into the created edge's `references` field.
 - A numpy matrix (GxG) is returned where rows/cols follow the order of ref_nodes list. Values:
     * +1 => net activation (stimulation or other types like 'bimodal')
     * -1 => net inhibition
     *  0 => no edge (neither direct nor propagated)

Assumptions about the `neko` objects (kept permissive):
 - `neko_network` and `ref_network` each expose a pandas DataFrame of nodes as either
   `neko_network.nodes` or `neko_network.nodes_df` (with a column 'Uniprot' whenever available) or
   `list(neko_network.nodes)` for simple lists. The code normalizes these possibilities.
 - `neko_network` exposes its edges as a pandas DataFrame `neko_network.edges` (or `.edges_df`) with
   columns containing the source/target identifiers. Column name matching is permissive and case-insensitive.
 - `ref_network` implements a method `add_edge(edge_df)` (as you described). The function will build
   a pandas DataFrame with the expected columns and call `ref_network.add_edge(...)`.

Note: the function will not write any file; it mutates `ref_network` by calling `add_edge` as requested
and returns the adjacency matrix plus a mapping of node->index.

"""

import pandas as pd
import networkx as nx
import numpy as np
import re
from collections import defaultdict


def _get_nodes_list(neko):
    """Return a list of node identifiers (prefer Uniprot if available).
    Handles several possible shapes for `neko.nodes`.
    """
    # try attributes
    if hasattr(neko, 'nodes_df') and isinstance(neko.nodes_df, pd.DataFrame):
        df = neko.nodes_df
    elif hasattr(neko, 'nodes') and isinstance(neko.nodes, pd.DataFrame):
        df = neko.nodes
    elif hasattr(neko, 'nodes') and isinstance(neko.nodes, (list, tuple)):
        return list(neko.nodes)
    else:
        # last resort: try to access attribute `nodes` as iterable
        try:
            return list(neko.nodes)
        except Exception:
            raise ValueError('Cannot read nodes from neko object; expected nodes_df/nodes DataFrame or list')

    # prefer 'Uniprot' column if present, else first column, else index
    if 'Uniprot' in df.columns:
        return df['Uniprot'].astype(str).tolist()
    elif 'uniprot' in [c.lower() for c in df.columns]:
        # find actual column name
        col = [c for c in df.columns if c.lower() == 'uniprot'][0]
        return df[col].astype(str).tolist()
    elif 'Genesymbol' in df.columns:
        return df['Genesymbol'].astype(str).tolist()
    elif df.shape[1] >= 1:
        return df.iloc[:,0].astype(str).tolist()
    else:
        return df.index.astype(str).tolist()


def _get_edges_df(neko):
    """Return edges DataFrame from neko object with columns normalized to lowercase.
    We try attributes `edges` or `edges_df`.
    """
    if hasattr(neko, 'edges') and isinstance(neko.edges, pd.DataFrame):
        df = neko.edges.copy()
    elif hasattr(neko, 'edges_df') and isinstance(neko.edges_df, pd.DataFrame):
        df = neko.edges_df.copy()
    else:
        # try attribute 'interactions' or fall back to attribute access
        if hasattr(neko, 'interactions') and isinstance(neko.interactions, pd.DataFrame):
            df = neko.interactions.copy()
        else:
            raise ValueError('Cannot read edges from neko object; expected edges DataFrame on neko.edges or neko.edges_df')

    # lowercase column names for convenience
    df.columns = [c.lower() for c in df.columns]
    # try to ensure source/target columns exist
    if 'source' not in df.columns or 'target' not in df.columns:
        # try common alternatives
        possible_src = [c for c in df.columns if 'source' in c or c in ('from','node1','a')]
        possible_tgt = [c for c in df.columns if 'target' in c or c in ('to','node2','b')]
        if possible_src and possible_tgt:
            df = df.rename(columns={possible_src[0]:'source', possible_tgt[0]:'target'})
        else:
            raise ValueError('Edges DataFrame does not have source/target columns')

    # ensure references/effect/type columns exist (if not, create empty)
    for c in ('effect','references','type'):
        if c not in df.columns:
            df[c] = None

    return df


def _effect_sign(effect_str):
    """Return +1 for activation-like effects and -1 for inhibition-like.
    We treat anything containing 'inhibit' as -1, everything else as +1 (including 'bimodal').
    If effect_str is None/NaN, default to +1.
    """
    if effect_str is None or (isinstance(effect_str, float) and np.isnan(effect_str)):
        return 1
    s = str(effect_str).lower()
    if 'inhibit' in s:
        return -1
    # explicit inhibition words
    if re.search(r"\b(repression|repress|negative|downregulat|antagonis)", s):
        return -1
    return 1


def _gather_references(edge_row):
    """Split references string into a cleaned list of tokens (split by ; or ,)"""
    refs = edge_row if edge_row is not None else ''
    if isinstance(refs, (list, tuple)):
        tokens = []
        for r in refs:
            if pd.isna(r):
                continue
            tokens.extend(re.split(r'[;,]\s*', str(r)))
    else:
        tokens = re.split(r'[;,]\s*', str(refs)) if refs not in (None, 'nan', 'NaN') else []
    tokens = [t.strip() for t in tokens if t and str(t).lower() not in ('nan','none')]
    return tokens


def build_edges_from_neko(neko_network, ref_network, max_path_length=10, verbose=False):
    """
    Version corrigée et optimisée.
    adj[i,j] = mean_sign * (1/(longueur_en_arêtes)) where longueur_en_arêtes = len(shortest_path)-1
    """

    import numpy as np
    import pandas as pd
    import networkx as nx

    ref_nodes = _get_nodes_list(ref_network)
    edges_df = _get_edges_df(neko_network)

    if verbose:
        print(f"[INFO] Ref nodes: {len(ref_nodes)}, edges: {len(edges_df)}")

    # build directed graph with signs stored
    G = nx.DiGraph()
    for _, r in edges_df.iterrows():
        src, tgt = str(r['source']), str(r['target'])
        sign = _effect_sign(r.get('effect') or r.get('type'))
        refs = _gather_references(r.get('references'))
        if G.has_edge(src, tgt):
            G.edges[src, tgt]['signs'].append(sign)
            G.edges[src, tgt]['refs'].extend(refs)
        else:
            G.add_edge(src, tgt, signs=[sign], refs=list(refs))

    n = len(ref_nodes)
    adj = np.zeros((n, n), dtype=float)
    node_index = {node: idx for idx, node in enumerate(ref_nodes)}

    direct_edges = set((str(r['source']), str(r['target'])) for _, r in edges_df.iterrows())

    for i, A in enumerate(ref_nodes):
        if not G.has_node(A):
            continue

        # distances (nombre d'arêtes) depuis A jusqu'aux autres noeuds, limitées par cutoff
        try:
            lengths = nx.single_source_shortest_path_length(G, A, cutoff=max_path_length)
        except nx.NodeNotFound:
            continue

        # parcourir toutes les cibles B atteignables depuis A
        for B, dist_nodes in lengths.items():
            if A == B:
                continue
            # dist_nodes is number of edges (shortest_path_length returns #edges)
            dist = dist_nodes
            if dist > max_path_length:
                continue
            if B not in node_index:
                continue

            # 1) si arête directe dans l'input, recopier les arêtes originales (comme avant)
            if (A, B) in direct_edges:
                matches = edges_df[(edges_df['source'].astype(str) == A) & (edges_df['target'].astype(str) == B)]
                for _, m in matches.iterrows():
                    edge_payload = pd.DataFrame([{
                        'source': A,
                        'target': B,
                        'type': m.get('type') or 'original',
                        'effect': m.get('effect') or m.get('type'),
                        'references': m.get('references')
                    }])
                    ref_network.add_edge(edge_payload)
                    s = _effect_sign(edge_payload.loc[0, 'effect'])
                    adj[i, node_index[B]] = float(s) * 1.0  # direct => weight 1
                continue

            # 2) Obtenir les plus courts chemins (networkx renvoie uniquement les chemins de longueur minimale)
            # NB: all_shortest_paths n'accepte pas 'cutoff' dans certaines versions, donc on l'appelle sans cutoff.
            try:
                shortest_paths_gen = nx.all_shortest_paths(G, source=A, target=B)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            path_signs = []
            any_path_kept = False

            for path in shortest_paths_gen:
                # la longueur en arêtes doit correspondre à dist ; sinon on ignore (habituellement ils sont égaux)
                if (len(path) - 1) != dist:
                    continue
                intermediates = path[1:-1]
                # exclure si le chemin passe par un noeud de ref_nodes (intermédiaire)
                if any(node in ref_nodes for node in intermediates):
                    if verbose:
                        print(f"Skipping path {path} because passes through ref node(s)")
                    continue

                # calcul du produit des signes le long du chemin
                hop_signs = []
                valid = True
                for k in range(len(path) - 1):
                    u, v = path[k], path[k+1]
                    if not G.has_edge(u, v):
                        valid = False
                        break
                    signs = G.edges[u, v].get('signs', [1])
                    # produit des signes parallèles sur la même arête
                    net_sign = 1
                    for ss in signs:
                        net_sign *= int(ss)
                    hop_signs.append(net_sign)
                if not valid:
                    continue
                path_signs.append(np.prod(hop_signs))
                any_path_kept = True

            if not any_path_kept:
                continue

            # moyenne des signes des plus courts chemins, puis signe final
            mean_val = np.mean(path_signs)
            # si mean_val est très proche de 0 -> on prend 0, sinon signe
            if abs(mean_val) < 1e-12:
                mean_sign = 1
            else:
                mean_sign = float(np.sign(mean_val))

            weight = 1.0 # / (float(dist) - 1.0)  # dist = nombre d'arêtes
            adj[i, node_index[B]] = mean_sign * weight

            # ajouter edge dans ref_network (type/effect simplifiés)
            if mean_sign == 0.0:
                eff_str = 'neutral'   # ou None si tu préfères
            else:
                eff_str = 'stimulation' if mean_sign > 0 else 'inhibition'

            edge_payload = pd.DataFrame([{
                'source': A,
                'target': B,
                'type': eff_str,
                'effect': eff_str,
                'references': None
            }])
            if verbose:
                print(f"Add weighted edge {A}->{B}: mean_sign={mean_sign}, dist={dist}, value={mean_sign*weight:.4f}")
            ref_network.add_edge(edge_payload)

    # alignement final
    ref_network.edges['Effect'] = ref_network.edges['Type']

    return adj



def subset_adj_by_genes(adj_matrix, node_index_list, gene_list, min_adj = 0):
    """
    Return a reordered adjacency matrix restricted to a given list of genes.


    Args:
    adj_matrix (np.ndarray): square adjacency matrix (values -1, 0, 1)
    node_index_list (dict): mapping {node_id -> index} for rows/columns of adj_matrix
    gene_list (list of str): list of gene identifiers (order to follow)


    Returns:
    sub_adj (np.ndarray): adjacency matrix of size len(gene_list) x len(gene_list)
    - reordered following the input gene_list order
    - missing genes (not in node_index_list) get rows of +1 (potential regulation from all)
    - diagonal entries all set to +1
    """
    n = len(gene_list)
    sub_adj = min_adj * np.ones((n, n), dtype=float) # start with +1 everywhere per spec
    sub_adj_real = np.zeros((n, n), dtype=float) # start with +1 everywhere per spec


    # Fill where both genes exist in node_index_list
    for i, g1 in enumerate(gene_list):
        for j, g2 in enumerate(gene_list):
            if g1 in node_index_list and g2 in node_index_list:
                i0 = node_index_list.index(g1)
                j0 = node_index_list.index(g2)
                if adj_matrix[i0, j0] > min_adj:
                    sub_adj[i, j] = adj_matrix[i0, j0]
                    sub_adj_real[i, j] = np.sign(adj_matrix[i0, j0])
            elif g1 not in node_index_list and g2 in node_index_list:
                sub_adj[i, j] = min_adj
                sub_adj_real[i, j] = 0
            else: # Genes not contained can interact between each other 
                sub_adj[i, j] = 1
                

    # # ensure diagonal and stimulus = 1
    np.fill_diagonal(sub_adj, 1)
    sub_adj[0, :] = min_adj
    return sub_adj, sub_adj_real



def main(argv):
    inputfile = ''
    depth = ''
    try:
        opts, args = getopt.getopt(argv, "hi:d:", ["input=", "depth="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
        if opt in ("-d", "--depth"):
            depth = int('{}'.format(arg))


    p = '{}/'.format(inputfile)  # Name of the file where are the data

    data_path = os.path.join(p, 'Data', 'data_full.h5ad')
    if os.path.exists(data_path):
        adata = ad.read_h5ad(data_path)
    else:
        raise FileNotFoundError(
            "There is no data available. Create a subfolder 'Data' in your main folder "
            "and put inside a count table named 'data.h5ad'."
        )

    genes_list_init = list(adata.var_names[:])
    genes_list_final = ['Stimulus'] + genes_list_init

    if depth > 0:

        ### Initial reference = full 1
        ref_network_mat = np.ones((len(genes_list_init)+1, len(genes_list_init)+1))

        ### Built theta_ref

        new_net1 = Network(genes_list_init, resources = 'omnipath')
        new_net1.complete_connection(maxlen=depth, algorithm="bfs", only_signed=True, connect_with_bias=False, consensus=True)
        ref_network = Network(genes_list_init, resources = 'omnipath')
        nodes_index_list = list(ref_network.nodes['Genesymbol'])
        adj_matrix = build_edges_from_neko(new_net1, ref_network, max_path_length=depth, verbose=False)
        visualizer = NetworkVisualizer(ref_network, color_by='effect', noi=True)
        visualizer.render()

        genes_list_final = ['Stimulus'] + genes_list_init
        ref_network_mat, ref_network_real = subset_adj_by_genes(adj_matrix, nodes_index_list, genes_list_final)
        print('prior network', ref_network_mat)
        
        # Save the reference matrix
        print('Network density and size: ', 'real = ', np.sum(np.abs(ref_network_real)), 
                                            'for inference =', np.sum(np.abs(ref_network_mat)), 
                                            'size', ref_network_real.shape)
        
        df = pd.DataFrame(ref_network_mat, 
                        index=genes_list_final, 
                        columns=genes_list_final)

    else:
        path_ref = os.path.join(p, 'cardamom', 'ref_network.csv')
        if os.path.exists(path_ref):
            df = pd.read_csv(path_ref, index_col=0)
        else:
            df = pd.DataFrame(np.ones((len(genes_list_final), len(genes_list_final))), 
                        index=genes_list_final, 
                        columns=genes_list_final)

    df.to_csv(p+"cardamom/ref_network.csv")


if __name__ == "__main__":
   main(sys.argv[1:])

