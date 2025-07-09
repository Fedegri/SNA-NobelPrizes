import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
import sys

class DetailedAuthorAnalysis:
    def __init__(self, csv_file, topic):
        self.df = pd.read_csv(csv_file)
        self.topic = topic
        self.author_counts = []
        self.distribution_data = None

    def extract_author_count(self):
        author_counts = []
        reasons_with_counts = []

        for _, reason in enumerate(self.df["reason"]):
            if pd.notna(reason) and "Too many authors" in str(reason):
                match = re.search(r"\((\d+)\)", str(reason))
                if match:
                    count = int(match.group(1))
                    author_counts.append(count)
                    reasons_with_counts.append(str(reason))
                else:
                    author_counts.append(None)
                    reasons_with_counts.append(None)
            else:
                author_counts.append(None)
                reasons_with_counts.append(None)

        self.df["author_count"] = author_counts
        self.df["reason_with_count"] = reasons_with_counts

        valid_data = self.df[self.df["author_count"].notna()].copy()
        self.author_counts = valid_data["author_count"].tolist()

        print(f"Found {len(self.author_counts)} articles with author count information")
        return valid_data

    def analyze_distribution(self):
        if not self.author_counts:
            print("No data available. Run extract_author_count() first")
            return None

        count_frequency = Counter(self.author_counts)
        sorted_counts = sorted(count_frequency.items())

        self.distribution_data = pd.DataFrame(
            {
                "num_authors": [x[0] for x in sorted_counts],
                "num_papers": [x[1] for x in sorted_counts],
            }
        )

        total_papers = sum(count_frequency.values())
        self.distribution_data["percentage"] = (
            self.distribution_data["num_papers"] / total_papers
        ) * 100
        self.distribution_data["cumulative_percentage"] = self.distribution_data[
            "percentage"
        ].cumsum()

        stats_data = {
            "total_papers": len(self.author_counts),
            "unique_author_counts": len(count_frequency),
            "min_authors": min(self.author_counts),
            "max_authors": max(self.author_counts),
            "mean_authors": np.mean(self.author_counts),
            "median_authors": np.median(self.author_counts),
            "std_authors": np.std(self.author_counts),
            "q25": np.percentile(self.author_counts, 25),
            "q75": np.percentile(self.author_counts, 75),
            "mode_authors": count_frequency.most_common(1)[0][0],
            "mode_frequency": count_frequency.most_common(1)[0][1],
        }

        return self.distribution_data, stats_data

    def create_detailed_visualizations(self):
        if self.distribution_data is None:
            print("No data available. Run analyze_distribution() first")
            return

        plt.style.use("default")
        fig = plt.figure(figsize=(16, 12))

        plt.subplot(2, 3, 1)
        bars = plt.bar(
            self.distribution_data["num_authors"],
            self.distribution_data["num_papers"],
            color="skyblue",
            alpha=0.8,
            edgecolor="navy",
            linewidth=0.5,
        )
        plt.title("Full Distribution: Number of Authors per Article", fontsize=12, fontweight="bold")
        plt.xlabel("Number of Authors")
        plt.ylabel("Number of Articles")
        plt.grid(axis="y", alpha=0.3)
        for i, (bar, row) in enumerate(zip(bars, self.distribution_data.itertuples())):
            if i < 20:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{int(row.num_papers)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.subplot(2, 3, 2)
        top_20 = self.distribution_data.head(20)
        bars = plt.bar(
            range(len(top_20)),
            top_20["num_papers"],
            color="lightcoral",
            alpha=0.8,
            edgecolor="darkred",
            linewidth=0.5,
        )
        plt.title("Top 20: Most Frequent Author Counts", fontsize=12, fontweight="bold")
        plt.xlabel("Number of Authors")
        plt.ylabel("Number of Articles")
        plt.xticks(range(len(top_20)), top_20["num_authors"], rotation=45)
        plt.grid(axis="y", alpha=0.3)
        for bar, row in zip(bars, top_20.itertuples()):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{int(row.num_papers)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.subplot(2, 3, 3)
        plt.plot(
            self.distribution_data["num_authors"],
            self.distribution_data["cumulative_percentage"],
            "o-",
            color="green",
            linewidth=2,
            markersize=4,
        )
        plt.title("Cumulative Distribution (%)", fontsize=12, fontweight="bold")
        plt.xlabel("Number of Authors")
        plt.ylabel("Cumulative Percentage")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=50, color="red", linestyle="--", alpha=0.7, label="50%")
        plt.axhline(y=95, color="orange", linestyle="--", alpha=0.7, label="95%")
        plt.legend()

        plt.subplot(2, 3, 4)
        plt.hist(
            self.author_counts,
            bins=50,
            density=True,
            alpha=0.7,
            color="purple",
            edgecolor="black",
        )
        plt.title("Histogram (Density)", fontsize=12, fontweight="bold")
        plt.xlabel("Number of Authors")
        plt.ylabel("Density")
        plt.grid(axis="y", alpha=0.3)

        plt.subplot(2, 3, 5)
        plt.boxplot(
            self.author_counts,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
        )
        plt.title("Box Plot - Outlier Identification", fontsize=12, fontweight="bold")
        plt.ylabel("Number of Authors")
        plt.grid(axis="y", alpha=0.3)

        plt.subplot(2, 3, 6)
        plt.bar(
            self.distribution_data["num_authors"],
            self.distribution_data["num_papers"],
            color="orange",
            alpha=0.8,
            edgecolor="darkorange",
            linewidth=0.5,
        )
        plt.title("Distribution (Logarithmic Scale)", fontsize=12, fontweight="bold")
        plt.xlabel("Number of Authors")
        plt.ylabel("Number of Articles (log scale)")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f"analyze_skipped_{self.topic}.png", dpi=300)

    def create_summary_table(self):
        """
        Creates a detailed summary table
        """
        if self.distribution_data is None:
            print("No data available. Run analyze_distribution() first")
            return

        print("\n" + "=" * 80)
        print("DETAILED TABLE: AUTHOR COUNT DISTRIBUTION")
        print("=" * 80)

        # Show the first 30 records
        print("\nFIRST 30 RECORDS:")
        print("-" * 60)
        print(
            f"{'N° Authors':<10} {'N° Papers':<12} {'Percentage':<12} {'Cum. %':<10}"
        )
        print("-" * 60)

        for _, row in self.distribution_data.head(30).iterrows():
            print(
                f"{int(row['num_authors']):<10} {int(row['num_papers']):<12} "
                f"{row['percentage']:<12.2f} {row['cumulative_percentage']:<10.1f}"
            )

        # Show records with high frequencies
        print(f"\nTOP 10 MOST FREQUENT:")
        print("-" * 60)
        top_10 = self.distribution_data.nlargest(10, "num_papers")
        for _, row in top_10.iterrows():
            print(
                f"{int(row['num_authors']):<10} {int(row['num_papers']):<12} "
                f"{row['percentage']:<12.2f} {row['cumulative_percentage']:<10.1f}"
            )

        # Show records with very high author counts
        print(f"\nARTICLES WITH MANY AUTHORS (>100):")
        print("-" * 60)
        high_authors = self.distribution_data[
            self.distribution_data["num_authors"] > 100
        ]
        if len(high_authors) > 0:
            for _, row in high_authors.iterrows():
                print(
                    f"{int(row['num_authors']):<10} {int(row['num_papers']):<12} "
                    f"{row['percentage']:<12.2f} {row['cumulative_percentage']:<10.1f}"
                )
        else:
            print("No articles with more than 100 authors found")

    def generate_detailed_statistics(self):
        """
        Generates detailed statistics
        """
        if not self.author_counts:
            print("No data available. Run extract_author_count() first")
            return

        _, stats_data = self.analyze_distribution()

        print("\n" + "=" * 60)
        print("DETAILED DESCRIPTIVE STATISTICS")
        print("=" * 60)

        print(f"Total articles analyzed: {stats_data['total_papers']}")
        print(f"Unique values (different n° authors): {stats_data['unique_author_counts']}")
        print(f"Minimum authors: {stats_data['min_authors']}")
        print(f"Maximum authors: {stats_data['max_authors']}")
        print(f"Mean authors: {stats_data['mean_authors']:.2f}")
        print(f"Median authors: {stats_data['median_authors']:.1f}")
        print(f"Standard deviation: {stats_data['std_authors']:.2f}")
        print(f"First quartile (Q1): {stats_data['q25']:.1f}")
        print(f"Third quartile (Q3): {stats_data['q75']:.1f}")
        print(f"Mode (most frequent): {stats_data['mode_authors']} authors")
        print(f"Mode frequency: {stats_data['mode_frequency']} articles")

        # Additional percentiles
        percentiles = [90, 95, 99]
        print(f"\nPERCENTILES:")
        for p in percentiles:
            value = np.percentile(self.author_counts, p)
            print(f"  {p}° percentile: {value:.1f} authors")

        # Distribution by range
        print(f"\nDISTRIBUTION BY RANGE:")
        ranges = [
            (1, 5, "1-5 authors"),
            (6, 10, "6-10 authors"),
            (11, 20, "11-20 authors"),
            (21, 50, "21-50 authors"),
            (51, 100, "51-100 authors"),
            (101, 1000, "101-1000 authors"),
            (1001, float("inf"), ">1000 authors"),
        ]

        for min_val, max_val, label in ranges:
            count = sum(1 for x in self.author_counts if min_val <= x <= max_val)
            percentage = (count / len(self.author_counts)) * 100
            print(f"  {label}: {count} articles ({percentage:.1f}%)")

    def find_extreme_cases(self):
        """
        Finds extreme cases (many authors or few authors)
        """
        if self.df["author_count"].isna().all():
            print("No data available. Run extract_author_count() first")
            return

        valid_data = self.df[self.df["author_count"].notna()]

        print("\n" + "=" * 80)
        print("EXTREME CASES ANALYSIS")
        print("=" * 80)

        # Articles with more authors
        top_10_authors = valid_data.nlargest(10, "author_count")
        print(f"\nTOP 10 ARTICLES WITH MOST AUTHORS:")
        print("-" * 80)
        for _, row in top_10_authors.iterrows():
            print(
                f"Authors: {int(row['author_count']):<4} | Laureate: {str(row['laureate_name'])[:30]:<30} | "
                f"Year: {row.get('prize_year', 'N/A')}"
            )

        # Distribution by laureate
        print(f"\nLAUREATES WITH HIGH-COLLABORATION ARTICLES (>50 authors):")
        print("-" * 80)
        high_collab = valid_data[valid_data["author_count"] > 50]
        if len(high_collab) > 0:
            laureate_high_collab = (
                high_collab.groupby("laureate_name")
                .agg({"author_count": ["count", "mean", "max"]})
                .round(1)
            )
            laureate_high_collab.columns = ["N_articles", "Mean_authors", "Max_authors"]
            print(laureate_high_collab.sort_values("N_articles", ascending=False))
        else:
            print("No articles with more than 50 authors found")

    def run_complete_analysis(self):
        """
        Runs the complete analysis
        """
        print("STARTING DETAILED AUTHOR COUNT ANALYSIS")
        print("=" * 60)

        # 1. Data extraction
        self.extract_author_count()

        # 2. Distribution analysis
        self.analyze_distribution()

        # 3. Detailed statistics
        self.generate_detailed_statistics()

        # 4. Summary table
        self.create_summary_table()

        # 5. Extreme cases
        self.find_extreme_cases()

        # 6. Visualizations
        print(f"\nGenerating visualizations...")
        self.create_detailed_visualizations()

        print(f"\nANALYSIS COMPLETE!")


def main(topic):
    analyzer = DetailedAuthorAnalysis(f"import/skipped_{topic}.csv", topic)

    analyzer.run_complete_analysis()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: You must provide a TOPIC argument!")
        print("Usage: python top_universities.py <TOPIC>")
        print("Available TOPICs: physics, medicine, chemistry")
        sys.exit(1)

    topic = sys.argv[1].lower()
    main(topic)
