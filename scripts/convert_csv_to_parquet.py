import os
import pandas as pd

class PSRDataManager:
    PSR_FILES = [
        'psrdadosabertos2006a2015.csv',
        'psrdadosabertos2016a2021.csv',
        'psrdadosabertos2022a2023.csv',
        'psrdadosabertos2024.csv'
    ]

    def __init__(self, raw_data_dir='.temp/data/raw', processed_data_dir='.temp/data/processed'):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)  # Ensure processed data directory exists

    def read_and_concatenate(self, sample=None):
        """
        Read and concatenate PSR files.
        
        Parameters:
        - sample (int or None): The number of rows to sample from each file. 
                                If None, all rows are read. If an integer, 
                                that number of rows will be sampled from each file.
        """
        # Construct full file paths based on PSR_FILES
        files = [os.path.join(self.raw_data_dir, filename) for filename in self.PSR_FILES]

        # Check if the files exist
        for file in files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"{file} does not exist in the directory")

        # Read and concatenate all files without losing any columns
        dataframes = []
        for file in files:
            # If sample is provided, read only the specified number of rows
            df = pd.read_csv(file, nrows=sample, decimal=",", sep=';', encoding='latin-1', low_memory=False)
            dataframes.append(df)

        # Concatenate all DataFrames, ignore index to keep a continuous index
        full_data = pd.concat(dataframes, ignore_index=True)
        return full_data

    def tidy_up_formatting(self, df):
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()

       # Replace commas with dots specifically in the 'valor_indenização' column if it exists
        if "valor_indenização" in df.columns:
            df["valor_indenização"] = df["valor_indenização"].astype(str).str.replace(",", ".", regex=False)
            df["valor_indenização"] = df["valor_indenização"].str.replace(r'\D', '', regex=True)
            df["valor_indenização"] = pd.to_numeric(df["valor_indenização"].str.replace(r'\D', '', regex=True))
        else:
            raise KeyError("'valor_indenização' column is missing from the data")

        # Replace commas with dots in all other object-type columns, in case other columns are affected
        for col in df.select_dtypes(include=['object']).columns:
            if col != "valor_indenização":  # Avoid re-processing 'valor_indenização'
                df[col] = df[col].str.replace(r'\D', '', regex=True).str.replace(",", ".", regex=False)

        return df

    def save_as_parquet(self, df):
        # Define the output parquet file path
        output_file = os.path.join(self.processed_data_dir, 'psr_full_history')
        output_file += ".parquet"
        
        # Save the DataFrame as a parquet file
        df.to_parquet(output_file)
        print(f"Data successfully saved to {output_file}")

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

        # Step 2: Tidy up formatting
        df_cleaned = self.tidy_up_formatting(df)

        # Step 3: Save the cleaned DataFrame as parquet
        self.save_as_parquet(df_cleaned)


# Instantiate the PSRDataManager and process the data
if __name__ == "__main__":
    # Example: process with a sample of 1000 rows
    psr_manager = PSRDataManager()
    psr_manager.process_psr_data(sample=None)