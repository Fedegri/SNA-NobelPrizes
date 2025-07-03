import os
import pandas as pd
import ast
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from networkx.algorithms import community
from pprint import pprint

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

def print_centrality_analysis(metrics, winners):
    """Print detailed centrality analysis results"""
    print("\n=== CENTRALITY ANALYSIS RESULTS ===\n")
    
    for metric_name, metric_values in metrics.items():
        if metric_name == 'louvain_communities':
            print(f"\n--- LOUVAIN COMMUNITIES ---")
            num_communities = len(set(metric_values.values()))
            print(f"Number of communities found: {num_communities}")
            
            # Distribution of winners across communities
            winner_community_counts = {}
            for winner in winners:
                if winner in metric_values:
                    community_id = metric_values[winner]
                    winner_community_counts[community_id] = winner_community_counts.get(community_id, 0) + 1
            
            if winner_community_counts:
                print("\nDistribution of Winners in Communities:")
                for community_id, count in sorted(winner_community_counts.items()): 
                    print(f"  Community {community_id}: {count} winners")
            else:
                print("No winners found in the identified communities.")
            continue 
        
        print(f"\n--- {metric_name.upper().replace('_', ' ')} ---")
        top_nodes = get_top_nodes_by_metric(metric_values, 15)
        
        print("Top 15 nodes:")
        for i, (node, value) in enumerate(top_nodes, 1):
            winner_status = "🏆 WINNER" if node in winners else ""
            print(f"{i:2d}. {node:<40} {value:.6f} {winner_status}")
        
        # Count winners in top positions
        winners_in_top10 = sum(1 for node, _ in top_nodes[:10] if node in winners)
        winners_in_top15 = sum(1 for node, _ in top_nodes if node in winners)
        print(f"Winners in top 10: {winners_in_top10}/10")
        print(f"Winners in top 15: {winners_in_top15}/15")

def draw_graph(G, title, partition=None):
    """Draw the network graph with optional community coloring"""
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.5, seed=42)

    # Determine node colors
    node_colors = []
    if partition:
        num_communities = len(set(partition.values()))
        if num_communities <= 20:
            cmap = plt.cm.get_cmap('tab20', num_communities)
        else:
            cmap = plt.cm.get_cmap('viridis', num_communities)

        unique_communities = sorted(list(set(partition.values())))
        community_to_color_idx = {comm_id: i for i, comm_id in enumerate(unique_communities)}

        for node in G.nodes():
            if node in partition:
                community_id = partition[node]
                node_colors.append(cmap(community_to_color_idx[community_id]))
            elif G.nodes[node].get("color") == "red":
                node_colors.append("red")
            else:
                node_colors.append("lightgray")
    else:
        for node in G.nodes():
            if G.nodes[node].get("color") == "red":
                node_colors.append("red")
            else:
                node_colors.append("blue")

    alphas = [G.nodes[node].get("alpha", 0.8) for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=alphas, node_size=50)
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    nx.draw_networkx_labels(G, pos, font_size=2)

    plt.title(title, fontsize=20)
    plt.axis("off")
    plt.tight_layout()

def create_graph(df: pd.DataFrame, title: str, topic: str):
    """Create and analyze the collaboration network graph"""
    graph_path = f"checkpoints_{topic}/graph.pkl"
    
    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)
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

        # Filter papers from 2000 onwards
        # papers_filtered = df[df["year"] >= 2000]
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
    
    # Print analysis results
    print_centrality_analysis(metrics, winners)
    
    # Visualizations
    louvain_partition = metrics.get('louvain_communities', None)
    
    if louvain_partition:
        draw_graph(G, title + " (Louvain Communities)", louvain_partition)
        plt.savefig(f"images/{topic}_communities.png", dpi=300)
        print(f"Graph with communities saved to images/{topic}_communities.png - nodes: {len(G.nodes())}, edges: {len(G.edges())}")

    # Original graph
    draw_graph(G, title + " (All Nodes)")
    plt.savefig(f"images/{topic}_all_nodes.png", dpi=300)
    print(f"Graph saved to images/{topic}_all_nodes.png - nodes: {len(G.nodes())}, edges: {len(G.edges())}")
    
    # Graph with winners removed
    G_copy = G.copy() 
    ns_to_remove = [i for i in G_copy.nodes() if G_copy.nodes[i].get("color") == "red"]
    for n in ns_to_remove:
        if n in G_copy:
            G_copy.remove_node(n)

    draw_graph(G_copy, title + " (Winners Removed)")
    plt.savefig(f"images/{topic}_winners_removed.png", dpi=300)
    print(f"Graph with winners removed saved to images/{topic}_winners_removed.png - nodes: {len(G_copy.nodes())}, edges: {len(G_copy.edges())}")
    
    return df_metrics