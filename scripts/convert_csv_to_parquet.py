import os
import pandas as pd
import logging
import re
import unicodedata


# Function to remove accents/diacritics
def remove_accents(col_name):
    nfkd_form = unicodedata.normalize("NFKD", col_name)
    return "".join([char for char in nfkd_form if not unicodedata.combining(char)])


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("conversion_csv_parquet.log"),
        logging.StreamHandler(),
    ],
)


# Define a function to apply the regex
def clean_numeric_values(value):
    if isinstance(value, str):  # Ensure the value is a string
        # Check for 'nan' as a string and handle it
        if value.lower() == "nan":
            return ""
        return re.sub(r"[^0-9.,]", "", value)
    return value  # If not a string, return the value as-is (e.g., NaN)


class PSRDataManager:
    PSR_FILES = [
        "psrdadosabertos2006a2015csv.csv",
        "psrdadosabertos2016a2021csv.csv",
        "psrdadosabertos2022a2023csv.csv",
        "psrdadosabertos2024csv.csv",
    ]

    def __init__(
        self, raw_data_dir=".temp/data/raw", processed_data_dir=".temp/data/processed"
    ):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(
            self.processed_data_dir, exist_ok=True
        )  # Ensure processed data directory exists

    def read_and_concatenate(self, sample=None):
        """
        Read and concatenate PSR files.

        Parameters:
        - sample (int or None): The number of rows to sample from each file.
                                If None, all rows are read. If an integer,
                                that number of rows will be sampled from each file.
        """
        # Construct full file paths based on PSR_FILES
        files = [
            os.path.join(self.raw_data_dir, filename) for filename in self.PSR_FILES
        ]

        # Check if the files exist
        for file in files:
            logging.info(file)
            if not os.path.exists(file):
                raise FileNotFoundError(f"{file} does not exist in the directory")

        # Read and concatenate all files without losing any columns
        dataframes = []
        for file in files:
            # If sample is provided, read only the specified number of rows
            df = pd.read_csv(
                file,
                nrows=sample,
                decimal=",",
                sep=";",
                encoding="latin-1",
                low_memory=False,
            )
            df = self.tidy_up_formatting(df)
            dataframes.append(df)

        # Concatenate all DataFrames, ignore index to keep a continuous index
        full_data = pd.concat(dataframes, ignore_index=True)
        return full_data

    def tidy_up_formatting(self, data):
        df = data.copy()

        # Convert column names to lowercase
        df.columns = df.columns.str.lower()

        dt_cols = df.filter(regex="dt").columns.str.lower().to_list()
        for d in dt_cols:
            df[d] = pd.to_datetime(df[d], format="%d/%m/%Y", errors="coerce")

        # Replace commas with dots specifically in the 'valor_indenização' column if it exists
        if "valor_indenização" in df.columns:
            df["valor_indenização"] = (
                df["valor_indenização"].astype(str).apply(clean_numeric_values)
            )
            df["valor_indenização"] = (
                df["valor_indenização"].astype(str).str.replace(",", ".", regex=False)
            )
            # df["valor_indenização"] = df["valor_indenização"].str.replace('nan', '')
            df["valor_indenização"] = pd.to_numeric(df["valor_indenização"])
        else:
            df["valor_indenização"] = 0
            logging.info("'valor_indenização' column is missing from the data")

        # Replace commas with dots in all other object-type columns, in case other columns are affected
        for col in df.select_dtypes(include=["object"]).columns:
        # Avoid re-processing 'valor_indenização'
            if col  in ['nr_grau_lat', 'nr_min_lat', 'nr_seg_lat', 'nr_grau_long', 'nr_min_long', 'nr_seg_long', 'nr_area_total', 'nr_animal', 'nr_produtividade_estimada', 'nr_produtividade_segurada', 'niveldecobertura'] or 'vl' in col:
                df[col] = (
                    df[col]
                    .str.replace(",", ".", regex=False)
                    .str.replace("-", "", regex=False)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in df.columns:
            if "cd" in col:  # Avoid re-processing 'valor_indenização'
                df[col] = df[col].astype(str)

        df.columns = [remove_accents(col) for col in df.columns]
        return df

    def save_as_parquet(self, df):
        # Define the output parquet file path
        output_file = os.path.join(self.processed_data_dir, "psr_full_history")
        output_file += ".parquet"

        # Save the DataFrame as a parquet file
        df.to_parquet(output_file)
        logging.info(f"Data successfully saved to {output_file}")

    def process_psr_data(self, sample=None):
        """
        Orchestrates the PSR data processing pipeline.

        Parameters:
        - sample (int or Noe): The number of rows to sample from each file.
                                If None, the entire files are processed.
                                Otherwise, only the specified number of rows are used.
        """
        # Step 1: Read and concatenate all raw files
        df = self.read_and_concatenate(sample=sample)

        # Step 3: Save the cleaned DataFrame as parquet
        self.save_as_parquet(df)


# Instantiate the PSRDataManager and process the data
if __name__ == "__main__":
    # Example: process with a sample of 1000 rows
    psr_manager = PSRDataManager()
    psr_manager.process_psr_data(sample=None)
