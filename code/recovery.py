import os
import sys
import pandas as pd
from diophila import OpenAlex
from pprint import pprint
import re

# OpenAlex API
openalex = OpenAlex()

def extract_author_count_from_reason(reason):
    """Extract the number of authors from the reason string"""
    # Pattern to match "Too many authors (X)"
    match = re.search(r'Too many authors \((\d+)\)', reason)
    if match:
        return int(match.group(1))
    return None

def recover_papers_by_topic(topic, max_authors):
    """Recover papers that were skipped but have acceptable number of authors"""
    
    # File paths
    skipped_path = f"import/skipped_{topic}.csv"
    recovery_success_path = f"import/recovery_success_{topic}.csv"
    recovery_failed_path = f"import/recovery_failed_{topic}.csv"
    recovery_skipped_path = f"import/recovery_skipped_{topic}.csv"
    
    # Check if skipped file exists
    if not os.path.exists(skipped_path):
        print(f"Error: File '{skipped_path}' not found!")
        return
    
    # Load skipped papers
    print(f"Loading skipped papers from '{skipped_path}'...")
    skipped_df = pd.read_csv(skipped_path)
    print(f"Found {len(skipped_df)} skipped papers")
    
    # Initialize result DataFrames
    recovery_successful = pd.DataFrame(columns=[
        "laureate_id", "laureate_name", "prize_year", "title", "journal", 
        "affiliation", "is_prize_winning", "locations_count", "locations", 
        "authors", "doi", "year"
    ])
    recovery_failed = pd.DataFrame(columns=[
        "laureate_id", "laureate_name", "prize_year", "title", "journal", 
        "affiliation", "is_prize_winning", "locations_count", "locations", 
        "doi", "error_message"
    ])
    recovery_skipped = pd.DataFrame(columns=[
        "laureate_id", "laureate_name", "prize_year", "title", "journal", 
        "affiliation", "is_prize_winning", "doi", "reason"
    ])
    
    # Cache for author information
    vauthors = {}
    
    # Counters
    processed_count = 0
    success_count = 0
    failed_count = 0
    still_skipped_count = 0
    recoverable_count = 0
    
    # Filter papers that can potentially be recovered
    for i in range(len(skipped_df)):
        row = skipped_df.iloc[i]
        reason = row["reason"]
        
        # Extract author count from reason
        author_count = extract_author_count_from_reason(reason)
        
        if author_count is not None and author_count <= max_authors:
            recoverable_count += 1
        elif author_count is not None and author_count > max_authors:
            # Still skip this paper
            still_skipped_data = {
                "laureate_id": row["laureate_id"],
                "laureate_name": row["laureate_name"],
                "prize_year": row["prize_year"],
                "title": row["title"],
                "journal": row["journal"],
                "affiliation": row["affiliation"],
                "is_prize_winning": row["is_prize_winning"],
                "doi": row["doi"],
                "reason": reason
            }
            recovery_skipped = pd.concat([recovery_skipped, pd.DataFrame([still_skipped_data])], ignore_index=True)
            still_skipped_count += 1
            
            # Save checkpoint every 10 skipped papers during initial filtering
            if still_skipped_count % 10 == 0:
                recovery_skipped.to_csv(recovery_skipped_path, index=False)
                print(f"Checkpoint Initial Skipped: {still_skipped_count} papers salvati")
    
    print(f"Papers that can be recovered (≤{max_authors} authors): {recoverable_count}")
    print(f"Papers still skipped (>{max_authors} authors): {still_skipped_count}")
    
    # Process recoverable papers
    for i in range(len(skipped_df)):
        row = skipped_df.iloc[i]
        reason = row["reason"]
        
        # Extract author count from reason
        author_count = extract_author_count_from_reason(reason)
        
        # Skip if still too many authors
        if author_count is None or author_count > max_authors:
            continue
            
        processed_count += 1
        print(f"Processing recoverable paper {processed_count}/{recoverable_count}")
        
        current_doi = row["doi"]
        
        # Base data
        base_data = {
            "laureate_id": row["laureate_id"],
            "laureate_name": row["laureate_name"],
            "prize_year": row["prize_year"],
            "title": row["title"],
            "journal": row["journal"],
            "affiliation": row["affiliation"],
            "is_prize_winning": row["is_prize_winning"],
        }
        
        # Check if DOI is missing or empty
        if pd.isna(current_doi) or current_doi == "" or str(current_doi).strip() == "":
            print(f"Skipping paper {processed_count} - DOI is missing or empty")
            failed_data = base_data.copy()
            failed_data.update({
                "locations_count": 0,
                "locations": [],
                "doi": current_doi,
                "error_message": "DOI missing or empty"
            })
            recovery_failed = pd.concat([recovery_failed, pd.DataFrame([failed_data])], ignore_index=True)
            failed_count += 1
            
            # Save checkpoint every 10 failed papers
            if failed_count % 10 == 0:
                recovery_failed.to_csv(recovery_failed_path, index=False)
                print(f"Checkpoint Failed: {failed_count} papers salvati")
            continue
        
        try:
            current_doi_str = str(current_doi).strip()
            paper = openalex.get_single_work("https://doi.org/" + current_doi_str, "doi")
            
            # Double-check author count from actual API response
            actual_author_count = len(paper["authorships"])
            if actual_author_count > max_authors:
                print(f"Skipping paper {processed_count} - Still too many authors ({actual_author_count})")
                skipped_data = base_data.copy()
                skipped_data.update({
                    "doi": current_doi,
                    "reason": f"Too many authors ({actual_author_count})"
                })
                recovery_skipped = pd.concat([recovery_skipped, pd.DataFrame([skipped_data])], ignore_index=True)
                still_skipped_count += 1
                
                # Save checkpoint every 10 skipped papers
                if still_skipped_count % 10 == 0:
                    recovery_skipped.to_csv(recovery_skipped_path, index=False)
                    print(f"Checkpoint Skipped: {still_skipped_count} papers salvati")
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
            
            # Process locations with None checks
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
            success_data = base_data.copy()
            success_data.update({
                "locations_count": paper["locations_count"],
                "locations": locations,
                "authors": authors,
                "doi": doi,
                "year": year
            })
            
            recovery_successful = pd.concat([recovery_successful, pd.DataFrame([success_data])], ignore_index=True)
            success_count += 1
            
            # Save checkpoint every 10 successful papers
            if success_count % 10 == 0:
                recovery_successful.to_csv(recovery_success_path, index=False)
                print(f"Checkpoint Success: {success_count} papers salvati")
                
        except Exception as e:
            print(f"Error with DOI {current_doi}: {e}")
            failed_data = base_data.copy()
            failed_data.update({
                "locations_count": 0,
                "locations": [],
                "doi": current_doi,
                "error_message": str(e)
            })
            recovery_failed = pd.concat([recovery_failed, pd.DataFrame([failed_data])], ignore_index=True)
            failed_count += 1
            
            # Save checkpoint every 10 failed papers
            if failed_count % 10 == 0:
                recovery_failed.to_csv(recovery_failed_path, index=False)
                print(f"Checkpoint Failed: {failed_count} papers salvati")
            continue
    
    # Final save of all files
    if len(recovery_successful) > 0:
        recovery_successful.to_csv(recovery_success_path, index=False)
        print(f"File '{recovery_success_path}' saved successfully with {len(recovery_successful)} papers")
    
    if len(recovery_failed) > 0:
        recovery_failed.to_csv(recovery_failed_path, index=False)
        print(f"File '{recovery_failed_path}' saved successfully with {len(recovery_failed)} papers")
    
    if len(recovery_skipped) > 0:
        recovery_skipped.to_csv(recovery_skipped_path, index=False)
        print(f"File '{recovery_skipped_path}' saved successfully with {len(recovery_skipped)} papers")
    
    # Final statistics
    print(f"\n=== STATISTICHE RECOVERY ===")
    print(f"Paper processati per recovery: {processed_count}")
    print(f"Paper recuperati con successo: {success_count}")
    print(f"Paper ancora skippati (>{max_authors} autori): {still_skipped_count}")
    print(f"Paper falliti durante recovery: {failed_count}")
    print(f"Totale dovrebbe essere: {success_count + still_skipped_count + failed_count}")

def main():
    """Main function for recovery script"""
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Error: You must provide TOPIC and MAX_AUTHORS arguments!")
        print("Usage: python recovery.py <TOPIC> <MAX_AUTHORS>")
        print("Available TOPICs: physics, medicine, chemistry")
        print("Example: python recovery.py physics 10")
        return
    
    topic = sys.argv[1].lower()
    
    try:
        max_authors = int(sys.argv[2])
    except ValueError:
        print("Error: MAX_AUTHORS must be a valid integer!")
        return
    
    # Validate TOPIC argument
    if topic not in ["physics", "medicine", "chemistry"]:
        print(f"Error: Invalid TOPIC '{topic}'")
        print("Available TOPICs: physics, medicine, chemistry")
        return
    
    # Validate MAX_AUTHORS argument
    if max_authors < 1:
        print("Error: MAX_AUTHORS must be greater than 0!")
        return
    
    print(f"Starting recovery for TOPIC: {topic}")
    print(f"Maximum authors allowed: {max_authors}")
    
    # Create necessary directories
    os.makedirs("import", exist_ok=True)
    
    # Run recovery
    recover_papers_by_topic(topic, max_authors)

if __name__ == "__main__":
    main()