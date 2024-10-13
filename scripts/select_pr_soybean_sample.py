import pandas as pd
import os
import logging

os.makedirs(".temp/logs/", exist_ok=True)

# Set up logging
logging.basicConfig(
    filename=".temp/logs/psr_sampled_features.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define paths for input and output data
PRD_SAMP_FEAT_DATA = ".temp/data/processed/psr_full_history.parquet"
OUTPUT_PARQUET = ".temp/data/processed/psr_soja_pr_sample.parquet"


def load_parquet_data(filepath):
    """
    Load data from a parquet file.

    Parameters:
    filepath (str): The path to the parquet file to be loaded.

    Returns:
    pd.DataFrame: Loaded dataframe.
    """
    if not os.path.exists(filepath):
        logging.error(f"The file {filepath} does not exist.")
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    logging.info(f"Loading data from {filepath}")
    return pd.read_parquet(filepath)


def filter_data(data):
    """
    Filter the dataframe based on specific conditions.

    Filters for 'sg_uf_propriedade' == 'PR' and 'nm_cultura_global' == 'Soja'.

    Parameters:
    data (pd.DataFrame): The input dataframe to filter.

    Returns:
    pd.DataFrame: The filtered dataframe.
    """
    logging.info(
        "Filtering data for 'sg_uf_propriedade == PR' and 'nm_cultura_global == Soja'"
    )
    return data.query("sg_uf_propriedade == 'PR' and nm_cultura_global == 'Soja'")


def save_parquet_data(data, output_path):
    """
    Save the filtered data to a parquet file.

    Parameters:
    data (pd.DataFrame): The dataframe to be saved.
    output_path (str): The path where the parquet file will be saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_parquet(output_path)
    logging.info(f"Filtered data saved to {output_path}")


def main():
    """
    Main function to execute the data loading, filtering, and saving process.
    """
    try:
        # Load the parquet data
        data = load_parquet_data(PRD_SAMP_FEAT_DATA)

        # Filter the data
        filtered_data = filter_data(data)

        # Save the filtered data
        save_parquet_data(filtered_data, OUTPUT_PARQUET)

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
