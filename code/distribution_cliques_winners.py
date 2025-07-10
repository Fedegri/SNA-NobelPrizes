import ast
import sys
from collections import Counter

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def build_graph_for_winners(df: pd.DataFrame, winners: set) -> nx.Graph:
    G = nx.Graph()
    for _, row in df.iterrows():
        authors = [a["display_name"] for a in row["authors"]]
        # Keep only authors who are winners
        filtered_authors = [a for a in authors if a in winners]
        for i in range(len(filtered_authors)):
            for j in range(i + 1, len(filtered_authors)):
                G.add_edge(filtered_authors[i], filtered_authors[j])
    return G


def count_cliques_by_size(G: nx.Graph, sizes=range(2, 13)) -> Counter:
    """
    Count maximal cliques by their size.
    Only sizes in `sizes` are retained (default 2-12).
    """
    counts = Counter()
    for clique in nx.find_cliques(G):
        k = len(clique)
        if k in sizes:
            counts[k] += 1

    for s in sizes:
        counts.setdefault(s, 0)
    return counts


def plot_counts(counts: Counter, topic: str):
    sizes = sorted(counts)
    values = [counts[s] for s in sizes]

    plt.figure(figsize=(10, 6))
    plt.bar(sizes, values)
    plt.xticks(sizes)
    plt.xlabel("Clique size")
    plt.ylabel("Number of maximal cliques")
    plt.title(f"Distribution of Maximal Cliques (Size 2-12) – {topic.title()} (Winners Only)")
    plt.tight_layout()
    plt.savefig(f"distribution_cliques_winners_{topic}.png")
    plt.close()


def main(topic: str):
    df = pd.read_csv(f"import/success_{topic}.csv")

    df["authors"] = df["authors"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    winners = set()

    for _, paper in df[df["is_prize_winning"] == "YES"].iterrows():
        try:
            authors = paper["authors"]
        except (ValueError, SyntaxError):
            continue
        lname = paper["laureate_name"]
        for author in authors:
            name = author.get("display_name")
            if lname.split(",")[0].strip().title() in name:
                winners.add(name)

    print(f"Found {len(winners)} winners")

    # Build graph restricted to winners only
    G = build_graph_for_winners(df, winners)

    counts = count_cliques_by_size(G)
    plot_counts(counts, topic)

    print(f"Clique‐size distribution for winners in {topic.title()}:")
    for size in sorted(counts):
        print(f"  size {size}: {counts[size]} cliques")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: You must provide a TOPIC argument!")
        print("Usage: python cliques_sizes_winners.py <TOPIC>")
        print("Available TOPICs: physics, medicine, chemistry")
        sys.exit(1)

    topic = sys.argv[1].lower()
    main(topic)
