import concurrent.futures
import requests
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("download_log.log"), logging.StreamHandler()],
)

# Directory to save the files
download_directory = ".temp/data/raw"

# Create the directory if it doesn't exist
os.makedirs(download_directory, exist_ok=True)

# List of files to download
files_to_download = {
    "psrdadosabertos2006a2015.xlsx": "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/97f29a77-4e7e-44bf-99b3-a2d75911b6bf/download/psrdadosabertos2006a2015csv.csv",
    "psrdadosabertos2016a2021.xlsx": "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/54e04a6b-15b3-4bda-a330-b8e805deabe4/download/psrdadosabertos2016a2021csv.csv",
    "psrdadosabertos2022a2023.xlsx": "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/8fd05876-7679-44c6-ae9a-50e07c765a25/download/psrdadosabertos2022a2023csv.csv",
    "psrdadosabertos2024.xlsx": "https://dados.agricultura.gov.br/dataset/baefdc68-9bad-4204-83e8-f2888b79ab48/resource/ac7e4351-974f-4958-9294-627c5cbf289a/download/psrdadosabertos2024csv.csv",
}


# Function to download a single file with logging
def download_file(url):
    local_filename = url.split("/")[-1]
    local_filepath = os.path.join(download_directory, local_filename)
    logging.info(f"Starting download: {local_filename}")

    try:
        # Stream the download to avoid loading the entire file into memory
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logging.info(f"Successfully downloaded: {local_filename}")
        return local_filename
    except Exception as e:
        logging.error(f"Failed to download {local_filename}. Error: {e}")
        return None


# Function to download all files concurrently with logging
def download_all_files_concurrently():
    logging.info("Starting concurrent downloads.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit download tasks to the thread pool
        futures = [
            executor.submit(download_file, url) for url in files_to_download.values()
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    logging.info(f"Completed: {result}")
                else:
                    logging.warning(f"Download failed for one of the files.")
            except Exception as e:
                logging.error(f"Error in download task: {e}")

    logging.info("All downloads complete.")


# Run the concurrent download
if __name__ == "__main__":
    download_all_files_concurrently()
