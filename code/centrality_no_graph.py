import os
import pandas as pd
import ast
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from networkx.algorithms import community

def save_checkpoint(data, path):
    """Save data to a checkpoint file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(path):
    """Load data from a checkpoint file"""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"Checkpoint loaded: {path}")
        return data
    return None

def calculate_degree_centrality(G, topic):
    """Calculate degree centrality with checkpointing"""
    degree_centrality_path = f"checkpoints_{topic}/degree_centrality.pkl"
    checkpoint = load_checkpoint(degree_centrality_path)
    if checkpoint is not None:
        return checkpoint

    print("Calculating degree centrality...")
    degree_centrality = nx.degree_centrality(G)
    save_checkpoint(degree_centrality, degree_centrality_path)
    return degree_centrality

def calculate_eigenvector_centrality(G, topic):
    """Calculate eigenvector centrality with checkpointing"""
    eigenvector_centrality_path = f"checkpoints_{topic}/eigenvector_centrality.pkl"
    checkpoint = load_checkpoint(eigenvector_centrality_path)
    if checkpoint is not None:
        return checkpoint

    print("Calculating eigenvector centrality...")
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=2000, tol=1e-4)

    save_checkpoint(eigenvector_centrality, eigenvector_centrality_path)
    return eigenvector_centrality

def calculate_closeness_centrality(G, topic):
    """Calculate closeness centrality with checkpointing"""
    closeness_centrality_path = f"checkpoints_{topic}/closeness_centrality.pkl"
    checkpoint = load_checkpoint(closeness_centrality_path)
    if checkpoint is not None:
        return checkpoint

    print("Calculating closeness centrality...")
    closeness_centrality = nx.closeness_centrality(G)
    save_checkpoint(closeness_centrality, closeness_centrality_path)
    return closeness_centrality

def calculate_betweenness_centrality(G, topic):
    """Calculate betweenness centrality with checkpointing"""
    betweenness_centrality_path = f"checkpoints_{topic}/betweenness_centrality.pkl"
    checkpoint = load_checkpoint(betweenness_centrality_path)
    if checkpoint is not None:
        return checkpoint

    print("Calculating betweenness centrality...")
    betweenness_centrality = nx.betweenness_centrality(G)
    save_checkpoint(betweenness_centrality, betweenness_centrality_path)
    return betweenness_centrality

def calculate_pagerank(G, topic):
    """Calculate PageRank with checkpointing"""
    pagerank_path = f"checkpoints_{topic}/pagerank.pkl"
    checkpoint = load_checkpoint(pagerank_path)
    if checkpoint is not None:
        return checkpoint

    print("Calculating PageRank...")
    pagerank = nx.pagerank(G)
    save_checkpoint(pagerank, pagerank_path)
    return pagerank

def calculate_louvain_communities(G, topic):
    """Calculate Louvain communities with checkpointing"""
    louvain_communities_path = f"checkpoints_{topic}/louvain_communities.pkl"
    checkpoint = load_checkpoint(louvain_communities_path)
    if checkpoint is not None:
        return checkpoint

    print("Calculating Louvain communities using NetworkX...")
    communities_generator = community.louvain_communities(G)
    communities_list = list(communities_generator)

    # Convert to node-to-community-id dictionary
    partition = {}
    for i, comm in enumerate(communities_list):
        for node in comm:
            partition[node] = i

    save_checkpoint(partition, louvain_communities_path)
    return partition

def calculate_density(G):
    """Calculate the density of the graph"""
    return nx.density(G)

def find_cliques(G):
    """Find cliques in the graph. Returns list of cliques."""
    cliques = list(nx.find_cliques(G))
    return cliques
def draw_all_cliques(G, cliques, topic, min_size=4, max_plots=20):
    """
    Draw the largest or most relevant cliques as subgraphs in a single image and save it.
    Only cliques of at least min_size will be shown, up to max_plots cliques.
    """
    # Filter to only cliques with size >= min_size
    filtered_cliques = [c for c in cliques if len(c) >= min_size]
    # Limit to max_plots cliques for display
    filtered_cliques = filtered_cliques[:max_plots]
    n = len(filtered_cliques)
    if n == 0:
        print("No cliques of minimum size to plot.")
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, clique in zip(axes, filtered_cliques):
        H = G.subgraph(clique)
        nx.draw(H, ax=ax, with_labels=True, node_color='skyblue', edge_color='gray')
        ax.set_title(f"Clique size: {len(clique)}")

    # Hide any unused subplots
    for ax in axes[len(filtered_cliques):]:
        ax.axis('off')

    os.makedirs("images", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"images/{topic}_cliques.png")
    plt.close(fig)
    print(f"All cliques image saved to images/{topic}_cliques.png")

def save_extra_metrics(G, topic):
    """Compute and save extra network metrics: density and cliques"""
    metrics = {}
    metrics["density"] = calculate_density(G)
    cliques = find_cliques(G)
    metrics["num_cliques"] = len(cliques)
    if cliques:
        metrics["max_clique_size"] = max(len(clique) for clique in cliques)
        # Save the largest cliques (all of largest size)
        max_size = metrics["max_clique_size"]
        metrics["largest_cliques"] = [clique for clique in cliques if len(clique) == max_size]
    else:
        metrics["max_clique_size"] = 0
        metrics["largest_cliques"] = []

    # Save metrics to a text file
    extra_metrics_path = f"data/{topic}_extra_network_metrics.txt"
    with open(extra_metrics_path, "w") as f:
        f.write(f"Density: {metrics['density']:.6f}\n")
        f.write(f"Number of cliques: {metrics['num_cliques']}\n")
        f.write(f"Largest clique size: {metrics['max_clique_size']}\n")
        f.write("Largest cliques:\n")
        for clique in metrics["largest_cliques"]:
            f.write(f"  {clique}\n")
    print(f"Extra network metrics saved to {extra_metrics_path}")

    # Optionally, also save a CSV listing all cliques (one row per clique, as comma-separated names):
    cliques_csv_path = f"data/{topic}_all_cliques.csv"
    pd.DataFrame({'clique': [', '.join(cl) for cl in cliques]}).to_csv(cliques_csv_path, index=False)
    print(f"All cliques saved to {cliques_csv_path}")

    # Draw the cliques as graphs
    draw_all_cliques(G, cliques, topic)

    return metrics

def calculate_all_centrality_metrics(G, topic):
    """Calculate all centrality metrics and community detection"""
    print("Calculating centrality metrics...")

    metrics = {}
    metrics['degree_centrality'] = calculate_degree_centrality(G, topic)
    metrics['eigenvector_centrality'] = calculate_eigenvector_centrality(G, topic)
    metrics['closeness_centrality'] = calculate_closeness_centrality(G, topic)
    metrics['betweenness_centrality'] = calculate_betweenness_centrality(G, topic)
    metrics['pagerank'] = calculate_pagerank(G, topic)
    metrics['louvain_communities'] = calculate_louvain_communities(G, topic)

    return metrics

def apply_metrics_to_graph(metrics, winners, topic):
    """Apply metrics to graph and return DataFrame"""
    # Create list to store all rows
    rows = []

    for winner in winners:
        row = {
            "name": winner,
            "degree_centrality": metrics["degree_centrality"].get(winner, None),
            "eigenvector_centrality": metrics["eigenvector_centrality"].get(winner, None),
            "closeness_centrality": metrics["closeness_centrality"].get(winner, None),
            "betweenness_centrality": metrics["betweenness_centrality"].get(winner, None),
            "pagerank": metrics["pagerank"].get(winner, None),
            "louvain_communities": metrics["louvain_communities"].get(winner, None),
        }
        rows.append(row)

    # Create DataFrame from list of rows (avoids concatenation warning)
    df = pd.DataFrame(rows)

    # Save to CSV with topic in filename
    df.to_csv(f"data/authors_{topic}_centrality_metrics.csv", index=False)

    return df

def get_top_nodes_by_metric(metric_dict, top_n=10):
    """Get top N nodes by metric value"""
    return sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

def build_centrality_analysis_text(metrics, winners):
    """Return detailed centrality analysis results as a text string"""
    output = []
    output.append("\n=== CENTRALITY ANALYSIS RESULTS ===\n")

    for metric_name, metric_values in metrics.items():
        if metric_name == 'louvain_communities':
            output.append(f"\n--- LOUVAIN COMMUNITIES ---")
            num_communities = len(set(metric_values.values()))
            output.append(f"Number of communities found: {num_communities}")

            # Distribution of winners across communities
            winner_community_counts = {}
            for winner in winners:
                if winner in metric_values:
                    community_id = metric_values[winner]
                    winner_community_counts[community_id] = winner_community_counts.get(community_id, 0) + 1

            if winner_community_counts:
                output.append("\nDistribution of Winners in Communities:")
                for community_id, count in sorted(winner_community_counts.items()):
                    output.append(f"  Community {community_id}: {count} winners")
            else:
                output.append("No winners found in the identified communities.")
            continue

        output.append(f"\n--- {metric_name.upper().replace('_', ' ')} ---")
        top_nodes = get_top_nodes_by_metric(metric_values, 15)

        output.append("Top 15 nodes:")
        for i, (node, value) in enumerate(top_nodes, 1):
            winner_status = "🏆 WINNER" if node in winners else ""
            output.append(f"{i:2d}. {node:<40} {value:.6f} {winner_status}")

        # Count winners in top positions
        winners_in_top10 = sum(1 for node, _ in top_nodes[:10] if node in winners)
        winners_in_top15 = sum(1 for node, _ in top_nodes if node in winners)
        output.append(f"Winners in top 10: {winners_in_top10}/10")
        output.append(f"Winners in top 15: {winners_in_top15}/15")

    return "\n".join(output)

def create_graph(df: pd.DataFrame, title: str, topic: str):
    """Create and analyze the collaboration network graph, saving analysis as text"""
    graph_path = f"checkpoints_{topic}/graph.pkl"

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Try to load existing graph
    G = load_checkpoint(graph_path)

    winners = set()
    for _, paper in df[df["is_prize_winning"] == "YES"].iterrows():
        try:
            authors = ast.literal_eval(paper["authors"])
        except (ValueError, SyntaxError):
            continue
        for author in authors:
            name = author.get("display_name")
            if paper["laureate_name"].split(",")[0].title() in name:
                winners.add(name)

    if G is not None:
        print("Graph loaded from checkpoint")
    else:
        print("Creating new graph...")
        G = nx.Graph()

        papers_filtered = df

        for _, paper in papers_filtered.iterrows():
            try:
                authors = ast.literal_eval(paper["authors"])
            except (ValueError, SyntaxError):
                continue

            if not isinstance(authors, list) or len(authors) == 0 or len(authors) <= 0 or len(authors) > 10:
                continue

            # Add nodes
            for author in authors:
                name = author.get("display_name")
                if not name:
                    continue

                if name in winners:
                    G.add_node(name, color="red", alpha=0.8)
                else:
                    G.add_node(name, color="blue", alpha=0.8)

            # Add edges between co-authors
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    name_i = authors[i].get("display_name")
                    name_j = authors[j].get("display_name")
                    if name_i and name_j and name_i != name_j:
                        G.add_edge(name_i, name_j)

        save_checkpoint(G, graph_path)

    # Calculate centrality metrics
    metrics = calculate_all_centrality_metrics(G, topic)

    # Apply metrics and get DataFrame (now properly returns the DataFrame)
    df_metrics = apply_metrics_to_graph(metrics, winners, topic)

    # Save analysis as text instead of printing or drawing
    analysis_text = build_centrality_analysis_text(metrics, winners)
    analysis_path = f"data/{topic}_centrality_analysis.txt"
    with open(analysis_path, "w") as f:
        f.write(analysis_text)
    print(f"Centrality analysis saved to {analysis_path}")

    # Save extra metrics: density and cliques
    save_extra_metrics(G, topic)

    return df_metrics