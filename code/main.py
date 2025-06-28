import os
import requests
import zipfile
import pandas as pd
from diophila import OpenAlex
import ast
import networkx as nx
import matplotlib.pyplot as plt

# Dataset path
PERSISTENT_ID = "doi:10.7910/DVN/6NJ5RN"
SERVER_URL = "https://dataverse.harvard.edu"
API_URL = f"{SERVER_URL}/api/access/dataset/:persistentId/?persistentId={PERSISTENT_ID}"

SUCCESS_PATH = "success_chemistry_42k.csv"

# Dataset folder path
DRIVE_FOLDER = "SNA"
ZIP_FILE_PATH = os.path.join(DRIVE_FOLDER, "dataset.zip")
EXTRACT_FOLDER = os.path.join(DRIVE_FOLDER, "csvs")

# CSVs paths for each topic
physics = EXTRACT_FOLDER + "/Physics publication record.csv"
medicine = EXTRACT_FOLDER + "/Medicine publication record.csv"
chemistry = EXTRACT_FOLDER + "/Chemistry publication record.csv"

# OpenAlex API path
OPENALEX_BASE = "https://api.openalex.org/works"
openalex = OpenAlex()


# Check if the required path already exists and it's not empty
def folder_exists_and_not_empty(path):
    return os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0


# Download from Harvard Dataverse
def download_dataset(api_url, output_file):
    print("Downloading dataset...")
    response = requests.get(api_url, allow_redirects=True)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Downloaded to {output_file}")
    else:
        raise Exception(
            f"Failed to download dataset. Status code: {response.status_code}"
        )


# Unzip method applied on the
def unzip_file(zip_file, extract_to):
    print("Unzipping file...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to folder: {extract_to}")


# Extract the
def extract_dataframe_by_topic(csv_file):
    df = pd.read_csv(csv_file)
    # df = df[df["Prize year"] == 2016]
    # df = df[df["Laureate ID"].isin(df["Laureate ID"].unique()[:5])]

    successful_papers = []
    failed_dois = []

    # Get all the papers with data from OpenAlex API
    for i in range(len(df)):
        print(f"Processing paper {i + 1}/{len(df)}")

        current_doi = df.iloc[i]["DOI"]

        data = {
            "laureate_id": df.iloc[i]["Laureate ID"],
            "laureate_name": df.iloc[i]["Laureate name"],
            "prize_year": df.iloc[i]["Prize year"],
            "title": df.iloc[i]["Title"],
            "journal": df.iloc[i]["Journal"],
            "affiliation": df.iloc[i]["Affiliation"],
            "is_prize_winning": df.iloc[i]["Is prize-winning paper"],
        }

        try:
            paper = openalex.get_single_work("https://doi.org/" + current_doi, "doi")

            # Se arriviamo qui, la chiamata API è riuscita
            authors = [author_info["author"] for author_info in paper["authorships"]]
            doi = paper["doi"]
            year = paper["publication_year"]

            # Aggiungi anche le informazioni originali dal DataFrame
            data["authors"] = authors
            data["doi"] = doi
            data["year"] = year

            successful_papers.append(data)

        except Exception as e:
            # Salva i dettagli dell'errore insieme al DOI
            data["doi"] = current_doi
            data["error_message"] = str(e)
            failed_dois.append(data)
            print(f"Error with DOI {current_doi}: {e}")

        if i % 1000 == 0:
            # Salva i risultati in file CSV separati
            if successful_papers:
                successful_df = pd.DataFrame(successful_papers)
                successful_df.to_csv(SUCCESS_PATH, index=False)
                print("File '{SUCCESS_PATH}' salvato con successo")

            if failed_dois:
                failed_df = pd.DataFrame(failed_dois)
                failed_df.to_csv("failed_papers.csv", index=False)
                print("File 'failed_papers.csv' salvato con successo")

            print(f"Successful papers: {len(successful_papers)}")
            print(f"Failed papers: {len(failed_dois)}")

    # Salva i risultati in file CSV separati
    if successful_papers:
        successful_df = pd.DataFrame(successful_papers)
        successful_df.to_csv("successful_papers.csv", index=False)
        print("File 'successful_papers.csv' salvato con successo")

    if failed_dois:
        failed_df = pd.DataFrame(failed_dois)
        failed_df.to_csv("failed_papers.csv", index=False)
        print("File 'failed_papers.csv' salvato con successo")

    return successful_papers


def create_graph(df: pd.DataFrame, title: str):
    G = nx.Graph()
    author_paper_count = {}

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

    papers = df
    papers = papers[papers["year"] == 2000]
    # papers = papers[papers["laureate_name"] == "stoddart, j"]
    # papers = papers[papers["prize_year"] == 2016]
    # papers = papers[papers["laureate_id"].isin(papers["laureate_id"].unique()[:5])]

    for _, paper in papers.iterrows():
        try:
            authors = ast.literal_eval(paper["authors"])
        except (ValueError, SyntaxError):
            continue

        if not isinstance(authors, list) or len(authors) == 0:
            continue

        if len(authors) > 5:
            continue

        laureate_name = paper["laureate_name"]
        # Add nodes and count papers
        for author in authors:
            name = author.get("display_name")
            if not name:
                continue

            if name in winners:
                G.add_node(name, color="red", alpha=0.8)
            # if paper["is_prize_winning"] == "YES":
            #     print(paper["year"])
            #     if laureate_name in name:
            #         G.add_node(name, color="red", alpha=0.8)
            else:
                G.add_node(name)
            author_paper_count[name] = author_paper_count.get(name, 0) + 1

        # Add edges between co-authors
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                name_i = authors[i].get("display_name")
                name_j = authors[j].get("display_name")
                if name_i and name_j:
                    G.add_edge(name_i, name_j)

    # Node sizes based on paper count
    # node_sizes = [300 + 200 * author_paper_count.get(node, 1) for node in G.nodes()]

    # Assign node colors: highlight prize winners
    node_colors = [G.nodes[node].get("color", "gray") for node in G.nodes()]
    alphas = [G.nodes[node].get("alpha", 0.1) for node in G.nodes()]
    # winners = [i.split(",")[0].title() for i in PHYSICS_WINNERS]
    # for node in G.nodes():
    #     if laureate_name in node:
    #         node_colors.append("gold")
    #         print(node)
    #     else:
    #         node_colors.append("skyblue")

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.5, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=alphas)
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    nx.draw_networkx_labels(G, pos, font_size=4)

    plt.title(title, fontsize=20)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig("coauthorship_graph.png", dpi=300)


def main():
    # Check if CSV folder exists and is not empty
    if folder_exists_and_not_empty(EXTRACT_FOLDER):
        print(f"Dataset already exists in '{EXTRACT_FOLDER}'. Skipping download.")
    else:
        # Create output directory
        os.makedirs(EXTRACT_FOLDER, exist_ok=True)
        # Download and extract
        download_dataset(API_URL, ZIP_FILE_PATH)
        unzip_file(ZIP_FILE_PATH, EXTRACT_FOLDER)

    # Chemistry
    # Check if the successful_papers_5_authors.csv already exists
    if os.path.exists(SUCCESS_PATH):
        chemicals = pd.read_csv(SUCCESS_PATH)
    else:
        chemicals = extract_dataframe_by_topic(chemistry)
        chemicals = pd.read_csv(SUCCESS_PATH)

    create_graph(chemicals, "Chemistry Nobel Prize Winners")


if __name__ == "__main__":
    main()
