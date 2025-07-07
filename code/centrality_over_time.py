import ast
import pandas as pd
import ast
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import sys


def main(topic):
    df = pd.read_csv(f"import/success_{topic}.csv")

    df["authors"] = df["authors"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    winners = set()
    laureate_map = dict()

    for _, paper in df[df["is_prize_winning"] == "YES"].iterrows():
        try:
            authors = paper["authors"]
        except (ValueError, SyntaxError):
            continue
        lname = paper["laureate_name"]
        prize_year = paper["prize_year"]
        for author in authors:
            name = author.get("display_name")
            if lname.split(",")[0].strip().title() in name:
                winners.add(name)
                laureate_map[name] = prize_year

    G_before = nx.Graph()
    G_after = nx.Graph()

    paper_counts = defaultdict(lambda: {"before": 0, "after": 0})
    coauthors = defaultdict(lambda: {"before": set(), "after": set()})

    for _, row in df.iterrows():
        year = row["year"]
        authors = [author["display_name"] for author in row["authors"]]

        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                a1, a2 = authors[i], authors[j]
                for author in (a1, a2):
                    if author in winners:
                        prize_year = laureate_map[author]
                        period = (
                            "before"
                            if year < prize_year
                            else "after"
                            if year > prize_year
                            else None
                        )
                        if period:
                            paper_counts[author][period] += 1
                            coauthors[author][period].add(a2 if author == a1 else a1)
                # Add edges to graphs
                if year < laureate_map.get(
                    a1, float("inf")
                ) and year < laureate_map.get(a2, float("inf")):
                    G_before.add_edge(a1, a2)
                elif year > laureate_map.get(a1, 0) and year > laureate_map.get(a2, 0):
                    G_after.add_edge(a1, a2)

    centrality_before = nx.degree_centrality(G_before)
    centrality_after = nx.degree_centrality(G_after)

    centrality_diff = []
    for winner in winners:
        before_c = centrality_before.get(winner, 0)
        after_c = centrality_after.get(winner, 0)
        diff = after_c - before_c
        centrality_diff.append((winner, diff))

    # Sort by difference
    centrality_diff_sorted = sorted(centrality_diff, key=lambda x: x[1])

    # Get bottom 10 and top 10
    bottom_10 = centrality_diff_sorted[:10]
    top_10 = centrality_diff_sorted[-10:]

    separator = ("...", 0)

    plot_data = bottom_10 + [separator] + top_10
    names, diffs = zip(*plot_data)

    # Color: red for decrease, green for increase, gray for "..."
    colors = [
        "gray" if name == "..." else ("red" if diff < 0 else "green")
        for name, diff in plot_data
    ]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(names, diffs, color=colors)
    plt.axvline(0, color="black", linestyle="--")
    plt.title(
        f"Top 10 and Bottom 10 Changes in Degree Centrality for {topic.title()} (After - Before)"
    )
    plt.xlabel("Δ Centrality")
    plt.ylabel("Laureate")

    for i, (bar, value) in enumerate(zip(bars, diffs)):
        if i == 10: continue # Skip the "..."

        if bar.get_width() > 0:
            plt.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}",
                va="center",
                ha="left",
                fontsize=9,
            )
        else:
            plt.text(
                bar.get_width() - 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}",
                va="center",
                ha="right",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(f"centrality_over_time_{topic}.png")

    rows = []
    for winner, _ in reversed(bottom_10 + top_10):
        prize_year = laureate_map[winner]
        before_c = centrality_before.get(winner, 0)
        after_c = centrality_after.get(winner, 0)
        papers_before = paper_counts[winner]["before"]
        papers_after = paper_counts[winner]["after"]
        coauths_before = len(coauthors[winner]["before"])
        coauths_after = len(coauthors[winner]["after"])

        rows.append(
            {
                "Name": winner,
                "Prize Year": prize_year,
                "Centrality Before": round(before_c, 4),
                "Centrality After": round(after_c, 4),
                "Centrality Δ": round(after_c - before_c, 4),
                "Papers Before": papers_before,
                "Papers After": papers_after,
                "Coauthors Before": coauths_before,
                "Coauthors After": coauths_after,
            }
        )

    df_summary = pd.DataFrame(rows)

    df_summary = df_summary.sort_values(by="Centrality Δ", ascending=False)

    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: You must provide a TOPIC argument!")
        print("Usage: python centrality_over_time.py <TOPIC>")
        print("Available TOPICs: physics, medicine, chemistry")
        sys.exit(1)

    topic = sys.argv[1].lower()
    main(topic)
