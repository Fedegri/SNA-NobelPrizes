import os
import sys
import requests
import zipfile
import pandas as pd
from diophila import OpenAlex
from centrality import create_graph
from pprint import pprint

# Dataset path
PERSISTENT_ID = "doi:10.7910/DVN/6NJ5RN"
SERVER_URL = "https://dataverse.harvard.edu"
API_URL = f"{SERVER_URL}/api/access/dataset/:persistentId/?persistentId={PERSISTENT_ID}"

# Dataset folder path
DRIVE_FOLDER = "SNA"
ZIP_FILE_PATH = os.path.join(DRIVE_FOLDER, "dataset.zip")
EXTRACT_FOLDER = os.path.join(DRIVE_FOLDER, "csvs")

# CSVs paths for each TOPIC
physics = EXTRACT_FOLDER + "/Physics publication record.csv"
medicine = EXTRACT_FOLDER + "/Medicine publication record.csv"
chemistry = EXTRACT_FOLDER + "/Chemistry publication record.csv"

# OpenAlex API
openalex = OpenAlex()

def folder_exists_and_not_empty(path):
    """Check if folder exists and is not empty"""
    return os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0

def download_dataset(api_url, output_file):
    """Download dataset from Harvard Dataverse"""
    print("Downloading dataset...")
    response = requests.get(api_url, allow_redirects=True)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Downloaded to {output_file}")
    else:
        raise Exception(f"Failed to download dataset. Status code: {response.status_code}")

def unzip_file(zip_file, extract_to):
    """Unzip the downloaded file"""
    print("Unzipping file...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to folder: {extract_to}")

def extract_dataframe_by_topic(csv_file, topic):
    """Extract and process papers from the CSV file"""
    df = pd.read_csv(csv_file, on_bad_lines="skip", encoding_errors="ignore")
    successful_papers = []
    failed_dois = []

    success_path = f"import/success_{topic}.csv"
    fails_path = f"import/failed_{topic}.csv"

    vauthors = {}  # Cache for author information

    for i in range(650, len(df)):
        print(f"Processing paper {i + 1}/{len(df)}")

        current_doi = df.iloc[i]["DOI"]
        
        # Check if DOI is missing or empty
        if pd.isna(current_doi) or current_doi == "" or str(current_doi).strip() == "":
            print(f"Skipping paper {i + 1} - DOI is missing or empty")
            data = {
                "laureate_id": df.iloc[i]["Laureate ID"],
                "laureate_name": df.iloc[i]["Laureate name"],
                "prize_year": df.iloc[i]["Prize year"],
                "title": df.iloc[i]["Title"],
                "journal": df.iloc[i]["Journal"],
                "affiliation": df.iloc[i]["Affiliation"],
                "is_prize_winning": df.iloc[i]["Is prize-winning paper"],
                "locations_count": 0,
                "locations": [],
                "doi": current_doi,
                "error_message": "DOI missing or empty"
            }
            failed_dois.append(data)
            continue

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
            current_doi_str = str(current_doi).strip()
            paper = openalex.get_single_work("https://doi.org/" + current_doi_str, "doi")
            
            # Skip papers with too many authors
            if len(paper["authorships"]) > 5:
                continue

            # Process authors with caching
            authors = []
            for author_info in paper["authorships"]:
                author_oa = vauthors.get(author_info["author"]["id"])

                if author_oa:
                    author_data = author_oa
                else:
                    author_oa = openalex.get_single_author(author_info["author"]["id"])

                    # Handle missing affiliations key
                    affiliations = author_oa.get("affiliations", [])
                    if affiliations is None:
                        affiliations = []

                    author_data = {
                        "id": author_oa["id"],
                        "display_name": author_oa["display_name"],
                        "affiliations": [
                            {
                                "id": i.get("institution", {}).get("id", ""),
                                "display_name": i.get("institution", {}).get("display_name", ""),
                                "country_code": i.get("institution", {}).get("country_code", ""),
                                "years": i.get("years", []),
                            }
                            for i in affiliations
                            if i.get("institution") is not None
                        ],
                    }

                    vauthors[author_info["author"]["id"]] = author_data

                authors.append(author_data)

            # Fixed locations processing with None checks
            locations = []
            for location in paper["locations"]:
                if location and location.get("source") is not None:
                    locations.append({
                        "id": location["source"]["id"],
                        "display_name": location["source"]["display_name"]
                    })

            doi = paper["doi"]
            year = paper["publication_year"]

            # Add processed information
            data["locations_count"] = paper["locations_count"]
            data["locations"] = locations
            data["authors"] = authors
            data["doi"] = doi
            data["year"] = year

            successful_papers.append(data)

        except Exception as e:
            data["locations_count"] = 0
            data["locations"] = []
            data["doi"] = current_doi
            data["error_message"] = str(e)
            failed_dois.append(data)
            print(f"Error with DOI {current_doi}: {e}")

        # Save checkpoints every 1000 papers
        if i % 1000 == 0:
            if successful_papers:
                successful_df = pd.DataFrame(successful_papers)
                successful_df.to_csv(success_path, index=False)
                print(f"File '{success_path}' saved successfully")

            if failed_dois:
                failed_df = pd.DataFrame(failed_dois)
                failed_df.to_csv(fails_path, index=False)
                print(f"File '{fails_path}' saved successfully")

            print(f"Successful papers: {len(successful_papers)}")
            print(f"Failed papers: {len(failed_dois)}")

    # Final save
    if successful_papers:
        successful_df = pd.DataFrame(successful_papers)
        successful_df.to_csv(success_path, index=False)
        print(f"File '{success_path}' saved successfully")

    if failed_dois:
        failed_df = pd.DataFrame(failed_dois)
        failed_df.to_csv(fails_path, index=False)
        print(f"File '{fails_path}' saved successfully")

    return successful_papers

def main():
    """Main function to orchestrate the analysis"""
    # Get TOPIC from command line arguments
    if len(sys.argv) < 2:
        print("Error: You must provide a TOPIC argument!")
        print("Usage: python script.py <TOPIC>")
        print("Available TOPICs: physics, medicine, chemistry")
        return
    
    topic = sys.argv[1].lower()
    
    # Validate TOPIC argument
    if topic not in ["physics", "medicine", "chemistry"]:
        print(f"Error: Invalid TOPIC '{topic}'")
        print("Available TOPICs: physics, medicine, chemistry")
        return
    
    # Set TOPIC-specific paths
    topic_files = {
        "physics": physics,
        "medicine": medicine,
        "chemistry": chemistry
    }
    
    what_to_search = topic_files[topic]
    success_path = f"import/success_{topic}.csv"
    checkpoints_folder = f"checkpoints_{topic}"
    
    print(f"Processing TOPIC: {topic}")
    
    # Create necessary directories
    os.makedirs(DRIVE_FOLDER, exist_ok=True)
    os.makedirs(checkpoints_folder, exist_ok=True)
    os.makedirs("import", exist_ok=True)

    # Check if dataset already exists
    if folder_exists_and_not_empty(EXTRACT_FOLDER):
        print(f"Dataset already exists in '{EXTRACT_FOLDER}'. Skipping download.")
    else:
        os.makedirs(EXTRACT_FOLDER, exist_ok=True)
        download_dataset(API_URL, ZIP_FILE_PATH)
        unzip_file(ZIP_FILE_PATH, EXTRACT_FOLDER)

    # Load or extract data
    if os.path.exists(success_path):
        print(f"Loading data from '{success_path}'...")
        data = pd.read_csv(success_path)
    else:
        print(f"'{success_path}' not found. Extracting data for {topic} TOPIC...")
        data = extract_dataframe_by_topic(what_to_search, topic)
        data = pd.read_csv(success_path)

    # Create and analyze graph
    create_graph(data, f"{topic.title()} Nobel Prize Winners Collaboration Network", topic)

if __name__ == "__main__":
    main()