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

# Dataset folder path
DRIVE_FOLDER = "SNA"
ZIP_FILE_PATH = os.path.join(DRIVE_FOLDER, "dataset.zip")
EXTRACT_FOLDER = os.path.join(DRIVE_FOLDER, "csvs")

# CSVs paths for each topic
physics = EXTRACT_FOLDER + "/Physics publication record.csv"
medicine = EXTRACT_FOLDER + "/Medicine publication record.csv"
chemistry = EXTRACT_FOLDER + "/Chemistry publication record.csv"


############################
#    SET THESE VALUES      #
############################
SUCCESS_PATH = "import/success_physics.csv"
FAILS_PATH = "import/failed_physics.csv"
START_FROM = 0
WHAT_TO_SEARCH = physics
############################
############################
############################
############################

# OpenAlex API path
OPENALEX_BASE = "https://api.openalex.org/works"
openalex = OpenAlex()


def draw_graph(G, title):
    node_colors = []
    for node in G.nodes():
        if G.nodes[node].get("color") == "red":
            node_colors.append("red")
        else:
            node_colors.append("blue")

    alphas = [G.nodes[node].get("alpha", 0.8) for node in G.nodes()]

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.5, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=alphas)
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    nx.draw_networkx_labels(G, pos, font_size=2)

    plt.title(title, fontsize=20)
    plt.axis("off")
    plt.tight_layout()


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
    df = pd.read_csv(csv_file, on_bad_lines="skip", encoding_errors="ignore")
    # df = df[df["Prize year"] == 2016]
    # df = df[df["Laureate ID"].isin(df["Laureate ID"].unique()[:5])]

    successful_papers = []
    failed_dois = []

    vauthors = {}

    # Get all the papers with data from OpenAlex API
    for i in range(START_FROM, len(df)):
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
            if len(paper["authorships"]) > 5:
                continue

            # Se arriviamo qui, la chiamata API è riuscita
            authors = []
            for author_info in paper["authorships"]:
                author_oa = vauthors.get(author_info["author"]["id"])

                if author_oa:
                    author_data = author_oa
                else:
                    author_oa = openalex.get_single_author(author_info["author"]["id"])

                    author_data = {
                        "id": author_oa["id"],
                        "display_name": author_oa["display_name"],
                        "affiliations": [
                            {
                                "id": i["institution"]["id"],
                                "display_name": i["institution"]["display_name"],
                                "country_code": i["institution"]["country_code"],
                                "years": i["years"],
                            }
                            for i in author_oa["affiliations"]
                        ],
                    }

                    vauthors[author_info["author"]["id"]] = author_data

                authors.append(author_data)

            locations = [
                {"id": i["source"]["id"], "display_name": i["source"]["display_name"]}
                for i in paper["locations"]
            ]

            doi = paper["doi"]
            year = paper["publication_year"]

            # Aggiungi anche le informazioni originali dal DataFrame
            data["locations_count"] = paper["locations_count"]
            data["locations"] = locations
            data["authors"] = authors
            data["doi"] = doi
            data["year"] = year

            successful_papers.append(data)

        except Exception as e:
            # Salva i dettagli dell'errore insieme al DOI
            data["locations_count"] = 0
            data["locations"] = []
            data["doi"] = current_doi
            data["error_message"] = str(e)
            failed_dois.append(data)
            print(f"Error with DOI {current_doi}: {e}")

        if i % 1000 == 0:
            # Salva i risultati in file CSV separati
            if successful_papers:
                successful_df = pd.DataFrame(successful_papers)
                successful_df.to_csv(SUCCESS_PATH, index=False)
                print(f"File '{SUCCESS_PATH}' salvato con successo")

            if failed_dois:
                failed_df = pd.DataFrame(failed_dois)
                failed_df.to_csv(FAILS_PATH, index=False)
                print(f"File '{FAILS_PATH}' salvato con successo")

            print(f"Successful papers: {len(successful_papers)}")
            print(f"Failed papers: {len(failed_dois)}")

    # Salva i risultati in file CSV separati
    if successful_papers:
        successful_df = pd.DataFrame(successful_papers)
        successful_df.to_csv(SUCCESS_PATH, index=False)
        print(f"File '{SUCCESS_PATH}' salvato con successo")

    if failed_dois:
        failed_df = pd.DataFrame(failed_dois)
        failed_df.to_csv(FAILS_PATH, index=False)
        print(f"File '{FAILS_PATH}' salvato con successo")

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
    # papers = papers[papers["laureate_name"] == "stoddart, j"]
    papers = papers[papers["year"] >= 2000]
    # papers = papers[papers["is_prize_winning"] == "YES"]
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

    draw_graph(G, title)

    plt.savefig("co1.png", dpi=300)
    # print(len(G.edges()))
    # ns = [i for i in G.nodes() if G.nodes[i].get("color") == "red"]
    # for n in ns:
    #     try:
    #         # G.remove_nodes_from(list(G.neighbors(n)))
    #         G.remove_node(n)
    #     except:
    #         pass
    # # top_node, top_degree = max(dict(G.degree()).items(), key=lambda x: x[1])
    # print(len(G.edges()))
    # # print(f"Most connected node: {top_node} with {top_degree} connections")
    # #
    # #
    # # G.remove_nodes_from(list(G.neighbors(top_node)))
    # # G.remove_node(top_node)
    #
    # draw_graph(G, title)
    #
    # plt.savefig(f"co2.png", dpi=300)


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
        data = pd.read_csv(SUCCESS_PATH)
    else:
        data = extract_dataframe_by_topic(WHAT_TO_SEARCH)
        data = pd.read_csv(SUCCESS_PATH)

    create_graph(data, "Chemistry Nobel Prize Winners")


if __name__ == "__main__":
    main()
