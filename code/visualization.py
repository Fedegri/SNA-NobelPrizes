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

def plot_metagraph(M, comm_reps, comm_sizes, winners, topic):
    import numpy as np
    os.makedirs("images", exist_ok=True)

    # 1. Identify components
    components = list(nx.connected_components(M))
    components.sort(key=len, reverse=True)
    main_component = components[0]
    side_components = components[1:]

    # 2. Layout main component with higher k for more spacing
    n_main = len(main_component)
    k_val = 1.8 * (1 + np.log1p(n_main)/4)   # Scale k based on main size
    M_main = M.subgraph(main_component)
    pos_main = nx.spring_layout(M_main, seed=42, k=k_val, iterations=150)

    # 3. Spread any "too close" nodes further apart
    pos_main_arr = np.array(list(pos_main.values()))
    node_list = list(pos_main.keys())
    min_dist = 0.13 + 0.36 / np.sqrt(n_main)   # Minimal allowed distance
    for i, (xi, yi) in enumerate(pos_main_arr):
        for j, (xj, yj) in enumerate(pos_main_arr):
            if i >= j: continue
            dx, dy = xi - xj, yi - yj
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                # Slightly nudge nodes apart
                angle = np.arctan2(dy, dx)
                shift = (min_dist - dist) / 2
                pos_main[node_list[i]] = (xi + np.cos(angle)*shift, yi + np.sin(angle)*shift)
                pos_main[node_list[j]] = (xj - np.cos(angle)*shift, yj - np.sin(angle)*shift)

    # 4. Layout side components in a vertical stack on the right
    side_nodes = []
    for comp in side_components:
        side_nodes.extend(comp)
    pos_side = {}
    if side_nodes:
        side_nodes = sorted(side_nodes, key=lambda n: comm_sizes[n], reverse=True)
        n_side = len(side_nodes)
        x_base = max(x for x, y in pos_main.values()) + 0.6
        y_base = np.linspace(-1, 1, n_side)
        for i, node in enumerate(side_nodes):
            pos_side[node] = (x_base, y_base[i])

    pos = {**pos_main, **pos_side}

    # 5. Prepare node properties
    node_colors = []
    node_labels = {}
    node_sizes = []
    for comm_id in M.nodes:
        rep = comm_reps[comm_id]
        size = comm_sizes[comm_id]
        node_labels[comm_id] = f"{rep} ({size})"
        node_sizes.append(220 + 10*size)
        node_colors.append('yellow' if rep in winners else 'royalblue')

    # 6. Dynamically scale figure size
    N = len(M.nodes)
    figw = min(36, max(14, 0.5 + 0.16*N))
    figh = min(25, max(10, 0.37 + 0.12*N))
    plt.figure(figsize=(figw, figh))
    nx.draw_networkx_edges(M, pos, alpha=0.34, width=1.1, connectionstyle='arc3,rad=0.13')
    nx.draw_networkx_nodes(M, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black', linewidths=1.2)

    # 7. Draw labels
    for comm_id, (x, y) in pos.items():
        label = node_labels[comm_id]
        if comm_id in side_nodes:
            plt.text(x + 0.09, y, label, fontsize=10, ha='left', va='center',
                     bbox=dict(facecolor='white', alpha=0.83, edgecolor='none', boxstyle='round,pad=0.13'))
        else:
            plt.text(x, y, label, fontsize=10, ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.83, edgecolor='none', boxstyle='round,pad=0.13'))

    plt.title(f"Meta-graph: Communities as Nodes ({topic})\nYellow = Nobel Winner | Label: Representative (Community size)", fontsize=15)
    plt.axis('off')
    out_path = f"images/{topic}_metagraph.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

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