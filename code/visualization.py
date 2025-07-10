import argparse
import os
import ast
import pandas as pd
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def save_checkpoint(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def calculate_louvain_communities(G, topic):
    from networkx.algorithms import community
    path = f"checkpoints_{topic}/louvain_communities.pkl"
    checkpoint = load_checkpoint(path)
    if checkpoint is not None:
        return checkpoint
    comms = list(community.louvain_communities(G))
    partition = {}
    for i, comm in enumerate(comms):
        for node in comm:
            partition[node] = i
    save_checkpoint(partition, path)
    return partition

def create_graph_from_csv(df, topic):
    graph_path = f"checkpoints_{topic}/graph.pkl"
    G = load_checkpoint(graph_path)
    winners = set()
    for _, paper in df[df["is_prize_winning"] == "YES"].iterrows():
        try:
            authors = ast.literal_eval(paper["authors"])
        except Exception:
            continue
        for author in authors:
            name = author.get("display_name")
            if paper["laureate_name"].split(",")[0].title() in name:
                winners.add(name)
    if G is not None:
        return G, winners
    G = nx.Graph()
    for _, paper in df.iterrows():
        try:
            authors = ast.literal_eval(paper["authors"])
        except Exception:
            continue
        if not isinstance(authors, list) or len(authors) == 0 or len(authors) > 10:
            continue
        for author in authors:
            name = author.get("display_name")
            if not name:
                continue
            color = "red" if name in winners else "blue"
            G.add_node(name, color=color, alpha=0.8)
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                name_i = authors[i].get("display_name")
                name_j = authors[j].get("display_name")
                if name_i and name_j and name_i != name_j:
                    G.add_edge(name_i, name_j)
    save_checkpoint(G, graph_path)
    return G, winners

def build_metagraph(G, partition, winners):
    comm_to_nodes = defaultdict(list)
    for node, comm_id in partition.items():
        comm_to_nodes[comm_id].append(node)
    comm_reps = {}
    comm_sizes = {}
    for comm_id, nodes in comm_to_nodes.items():
        subg = G.subgraph(nodes)
        rep = max(subg.degree, key=lambda x: x[1])[0]
        comm_reps[comm_id] = rep
        comm_sizes[comm_id] = len(nodes)
    M = nx.Graph()
    for comm_id, rep in comm_reps.items():
        M.add_node(comm_id,
                   label=rep,
                   is_winner=(rep in winners),
                   size=comm_sizes[comm_id])
    added = set()
    for n1, n2 in G.edges():
        c1 = partition[n1]
        c2 = partition[n2]
        if c1 != c2:
            edge = tuple(sorted((c1, c2)))
            if edge not in added:
                M.add_edge(*edge)
                added.add(edge)
    return M, comm_reps, comm_sizes

def repel_close_nodes(pos, min_dist=0.35, passes=12):
    """
    Moves nodes apart if they are closer than min_dist.
    Args:
        pos: dict of node positions {node: (x, y)}
        min_dist: minimal allowed distance between nodes (in layout space)
        passes: number of repulsion sweeps (increase for better separation)
    Returns:
        new pos dictionary with updated positions
    """
    import numpy as np
    node_list = list(pos.keys())
    for _ in range(passes):
        moved = False
        for i in range(len(node_list)):
            xi, yi = pos[node_list[i]]
            for j in range(i+1, len(node_list)):
                xj, yj = pos[node_list[j]]
                dx, dy = xi - xj, yi - yj
                dist = np.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    # Nudge nodes apart
                    if dist < 1e-6:
                        # Prevent division by zero, random direction
                        angle = np.random.rand() * 2 * np.pi
                        dx, dy = np.cos(angle), np.sin(angle)
                        dist = 1e-3
                    else:
                        angle = np.arctan2(dy, dx)
                    shift = (min_dist - dist) / 2
                    xi_new = xi + np.cos(angle) * shift
                    yi_new = yi + np.sin(angle) * shift
                    xj_new = xj - np.cos(angle) * shift
                    yj_new = yj - np.sin(angle) * shift
                    pos[node_list[i]] = (xi_new, yi_new)
                    pos[node_list[j]] = (xj_new, yj_new)
                    moved = True
        if not moved:
            break
    return pos

def plot_metagraph(M, comm_reps, comm_sizes, winners, topic):
    import numpy as np
    os.makedirs("images", exist_ok=True)

    # Find unlinked communities (degree 0)
    unlinked_nodes = [n for n in M.nodes if M.degree(n) == 0]
    n_unlinked = len(unlinked_nodes)

    # Remove unlinked nodes for the plot
    M_plot = M.copy()
    M_plot.remove_nodes_from(unlinked_nodes)

    # If graph is now empty, skip plotting
    if len(M_plot.nodes) == 0:
        print("No linked communities to plot.")
        return

    # Layout: Use a strong spring_layout for spacing, then repel
    n_nodes = len(M_plot.nodes)
    """
    increase k_val to spread nodes more widely based on the number of nodes.
    This helps prevent overlap and makes the graph more readable. 
    2 is enough for physics, other two need more, like > 10 
    """
    k_val = 15.0 * (1 + np.log1p(n_nodes)/5)
    pos = nx.spring_layout(M_plot, seed=42, k=k_val, iterations=300)

    # Repel nodes that are too close
    pos = repel_close_nodes(pos, min_dist=0.33, passes=16)

    # Prepare node properties
    node_colors = []
    node_labels = {}
    node_sizes = []
    for comm_id in M_plot.nodes:
        rep = comm_reps[comm_id]
        size = comm_sizes[comm_id]
        node_labels[comm_id] = f"{rep} ({size})"
        node_sizes.append(220 + 10*size)
        node_colors.append('yellow' if rep in winners else 'royalblue')

    # Dramatically increase figure size for more space
    N = n_nodes
    figw = min(64, max(18, 0.7 + 0.19*N))
    figh = min(48, max(12, 0.48 + 0.13*N))
    plt.figure(figsize=(figw, figh))

    nx.draw_networkx_edges(M_plot, pos, alpha=0.32, width=1.1, connectionstyle='arc3,rad=0.13')
    nx.draw_networkx_nodes(M_plot, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black', linewidths=1.2)

    # Draw labels
    for comm_id, (x, y) in pos.items():
        label = node_labels[comm_id]
        plt.text(x, y, label, fontsize=12, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.15'))

    plt.title(
        f"Meta-graph: Communities as Nodes ({topic})\n"
        f"Yellow = Nobel Winner | Label: Representative (Community size)\n"
        f"Removed {n_unlinked} unlinked communities from visualization",
        fontsize=18
    )
    plt.axis('off')
    out_path = f"images/{topic}_metagraph.png"
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path} (removed {n_unlinked} unlinked communities)")

def main():
    parser = argparse.ArgumentParser(description="Meta-graph visualization for Nobel Prize networks.")
    parser.add_argument('--input', type=str, required=True, help="Input CSV file")
    parser.add_argument('--topic', type=str, required=True, help="Topic string for checkpoints/images")
    args = parser.parse_args()
    assert os.path.exists(args.input), "Input file does not exist"
    df = pd.read_csv(args.input)
    G, winners = create_graph_from_csv(df, args.topic)
    partition = calculate_louvain_communities(G, args.topic)
    M, comm_reps, comm_sizes = build_metagraph(G, partition, winners)
    plot_metagraph(M, comm_reps, comm_sizes, winners, args.topic)

if __name__ == "__main__":
    main()