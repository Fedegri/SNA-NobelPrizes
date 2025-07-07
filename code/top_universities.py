import pandas as pd
import networkx as nx
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys


def main(topic):
    df = pd.read_csv(f"import/success_{topic}.csv")
    institutions_list = []
    for _, paper in df[df["is_prize_winning"] == "YES"].iterrows():
        try:
            authors = ast.literal_eval(paper["authors"])
        except (ValueError, SyntaxError):
            continue
        for author in authors:
            name = author.get("display_name")
            if paper["laureate_name"].split(",")[0].title() in name:
                affiliations = author.get("affiliations", [])
                # Extract only display_names of institutions
                institutions_list.append(
                    [
                        aff.get("display_name")
                        for aff in affiliations
                        if aff.get("display_name")
                    ]
                )

    # Build graph
    G = nx.DiGraph()

    for institutions in institutions_list:
        for i in range(0, len(institutions) - 1):
            G.add_edge(institutions[i], institutions[i + 1])

    # Get all strongly connected components (as sets of nodes)
    sccs = list(nx.strongly_connected_components(G))

    # Find the largest one
    largest_scc = max(sccs, key=len)

    # Create a subgraph with only that component
    G_largest_scc = G.subgraph(largest_scc).copy()

    G = G_largest_scc

    # Degree computation
    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    top_10_nodes = {node for node, _ in top_nodes[:10]}

    # Labels only for top 10
    labels = {node: node for node in G.nodes() if node in top_10_nodes}

    # Top 3 for medal colors
    top_1 = top_nodes[0][0]
    top_2 = top_nodes[1][0]
    top_3 = top_nodes[2][0]

    # Node colors
    node_colors = []
    for node in G.nodes():
        if node == top_1:
            node_colors.append("gold")
        elif node == top_2:
            node_colors.append("silver")
        elif node == top_3:
            node_colors.append("#cd7f32")  # bronze
        else:
            node_colors.append("skyblue")

    # Node sizes
    node_sizes = [degrees[node] * 100 for node in G.nodes()]

    # Plot
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.7, seed=24)

    # Draw nodes and edges
    nx.draw(
        G,
        pos,
        labels=labels,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color="gray",
        font_size=8,
        with_labels=False,
    )

    # Draw top 10 labels inside nodes
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=9,
        font_color="black",
        verticalalignment="center",
        horizontalalignment="center",
    )

    # Create legend handles
    legend_handles = []
    for node, deg in top_nodes[:10]:
        patch = mpatches.Patch(color="skyblue", label=f"{node} (deg: {deg})")
        legend_handles.append(patch)

    # Override medal colors for top 3
    legend_handles[0] = mpatches.Patch(
        color="gold", label=f"{top_1} (deg: {degrees[top_1]})"
    )
    legend_handles[1] = mpatches.Patch(
        color="silver", label=f"{top_2} (deg: {degrees[top_2]})"
    )
    legend_handles[2] = mpatches.Patch(
        color="#cd7f32", label=f"{top_3} (deg: {degrees[top_3]})"
    )

    # Add legend
    plt.legend(
        handles=legend_handles,
        title=f"Top 10 Nodes by Degree for {topic.title()}",
        loc="upper right",
        fontsize=9,
        title_fontsize=10,
    )

    plt.title("Graph with Node Sizes Proportional to Degree", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"top_universities_{topic}.png")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: You must provide a TOPIC argument!")
        print("Usage: python top_universities.py <TOPIC>")
        print("Available TOPICs: physics, medicine, chemistry")
        sys.exit(1)

    topic = sys.argv[1].lower()
    main(topic)
