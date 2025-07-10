import ast
import sys
from collections import Counter

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def build_full_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in df.iterrows():
        authors = [a["display_name"] for a in row["authors"]]
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                G.add_edge(authors[i], authors[j])
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
    plt.title(f"Distribution of Maximal Cliques (Size 2-12) – {topic.title()}")
    plt.tight_layout()
    plt.savefig(f"distribution_cliques_{topic}.png")
    plt.close()


def main(topic: str):
    df = pd.read_csv(f"import/success_{topic}.csv")
    df["authors"] = df["authors"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    G = build_full_graph(df)
    counts = count_cliques_by_size(G)
    plot_counts(counts, topic)

    print(f"Clique‐size distribution for {topic.title()}:")
    for size in sorted(counts):
        print(f"  size {size}: {counts[size]} cliques")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: You must provide a TOPIC argument!")
        print("Usage: python cliques_sizes_global.py <TOPIC>")
        print("Available TOPICs: physics, medicine, chemistry")
        sys.exit(1)

    topic = sys.argv[1].lower()
    main(topic)
