import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation


def compute_max_variation_times(inter_t):
    T, G, _ = inter_t.shape
    variations = np.zeros((T, G, G))
    variations[0] = np.abs(inter_t[0])
    for t in range(1, T):
        variations[t] = np.abs(inter_t[t] - inter_t[t - 1])
    max_var_times = np.argmax(variations, axis=0)  # shape (G, G)
    for t in range(0, T):
        inter_t[t] *= (max_var_times == t)
    return inter_t


def filter_edges(matrix, ref, abs_thresh=1, rel_thresh=0.1):
    matrix = matrix.copy()
    col_sums = np.sum(np.abs(ref), axis=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if abs(matrix[i, j]) < abs_thresh or abs(matrix[i, j]) < rel_thresh * col_sums[j]:
                matrix[i, j] = 0
    return matrix


def enforce_min_distance(pos, min_dist=0.1):
    nodes = list(pos.keys())
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i >= j:
                continue
            dist = np.linalg.norm(pos[node1] - pos[node2])
            if dist < min_dist:
                direction = pos[node1] - pos[node2]
                direction /= np.linalg.norm(direction)
                pos[node1] += direction * (min_dist - dist) / 2
                pos[node2] -= direction * (min_dist - dist) / 2
    return pos


def interactive_edit_positions(G, pos, labels, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    nodes = list(G.nodes())

    scatter = ax.scatter(
        [pos[n][0] for n in nodes],
        [pos[n][1] for n in nodes],
        s=400, color='lightgray', alpha=0.5, picker=True
    )
    texts = {
        n: ax.text(pos[n][0], pos[n][1], labels[n], fontsize=8, ha='center', va='center') for n in nodes
    }
    lines = {}
    for u, v in G.edges():
        lines[(u, v)] = ax.annotate('', xy=pos[v], xytext=pos[u], arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5))

    dragged_node = [None]

    def on_pick(event):
        ind = event.ind[0]
        dragged_node[0] = nodes[ind]

    def on_motion(event):
        if dragged_node[0] is None or event.xdata is None or event.ydata is None:
            return
        n = dragged_node[0]
        pos[n] = np.array([event.xdata, event.ydata])
        scatter.set_offsets([pos[k] for k in nodes])
        texts[n].set_position((event.xdata, event.ydata))
        for (u, v), line in lines.items():
            line.remove()
        for (u, v) in G.edges():
            lines[(u, v)] = ax.annotate('', xy=pos[v], xytext=pos[u], arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5))
        fig.canvas.draw()

    def on_release(event):
        dragged_node[0] = None

    def on_key(event):
        if event.key == 's':
            np.save(save_path, pos)
            print(f"✅ Positions saved to {save_path}")
            plt.close(fig)

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.title("Drag nodes and press 's' to save")
    plt.axis('off')
    plt.ioff()  # Désactive l'interactivité (empêche le popup de sauvegarde)
    plt.show()
    plt.close("all")


def animate_dynamic_grns(pos, inter_t, gene_names, timepoints, output_path, G):
    fig, ax = plt.subplots(figsize=(8, 8))
    Gr = nx.DiGraph()
    Gr.add_nodes_from(range(G))
    prev_edges = set()

    def update(i):
        ax.clear()
        matrix = inter_t[i]
        edges_present = {(a, b) for a in range(G) for b in range(G) if matrix[a, b] != 0}
        edge_colors = {}
        edge_alpha = {}
        for edge in edges_present:
            if edge in prev_edges:
                edge_colors[edge] = "gray"
                edge_alpha[edge] = 0.3
            else:
                edge_colors[edge] = "green" if matrix[edge] > 0 else "red"
                edge_alpha[edge] = 1.0

        draw_edges = [e for e in edges_present if matrix[e] != 0]

        nx.draw_networkx_nodes(Gr, pos, ax=ax, node_size=10*np.sqrt(G), node_color="lightgray", alpha=0.5)
        nx.draw_networkx_labels(Gr, pos, labels={i: gene_names[i] for i in range(G)}, ax=ax, font_size=6)
        nx.draw_networkx_edges(
            Gr, pos, edgelist=draw_edges,
            edge_color=[edge_colors[e] for e in draw_edges],
            width=.3, alpha=[edge_alpha[e] for e in draw_edges],
            ax=ax, arrows=True, arrowsize=15, connectionstyle='arc3,rad=0.05')
        ax.set_title(f"Time = {24*timepoints[i]}h", fontsize=16)
        ax.axis('off')
        prev_edges.clear()
        prev_edges.update(edges_present)

    ani = animation.FuncAnimation(fig, update, frames=len(inter_t), interval=1500, repeat=False)
    ani.save(output_path, writer='ffmpeg', fps=1)
    plt.close(fig)
    print("🎞️ MP4 animation saved to:", output_path)