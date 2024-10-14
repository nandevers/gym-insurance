import json
import pandas as pd
import logging
from f_utils import CumulativeCountEncoder, CumulativeEncoder

# File paths
PRD_SAMPLE_DATA = ".temp/data/processed/psr_soja_pr_sample.parquet"
PRD_FEAT_DATA = ".temp/data/processed/psr_features.parquet"
PRD_SAMP_FEAT_DATA = ".temp/data/processed/psr_sampled_features.parquet"

# Setup logging
logging.basicConfig(
    filename="data_processing.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_missing_data(df, name):
    """Logs the amount and percentage of missing data for each column in the DataFrame."""
    missing_data = df.isnull().sum()
    missing_data_percent = (df.isnull().sum() / len(df)) * 100
    logging.info(f"Missing data report for {name}:")
    for col, missing, pct in zip(df.columns, missing_data, missing_data_percent):
        if missing > 0:
            logging.info(f"{col}: {missing} missing values ({pct:.2f}%)")

if __name__ == "__main__":
    # Load column dictionary
    logging.info("Loading column dictionary.")
    with open("./scripts/cols_dict.json") as f:
        cols_dict = json.load(f)

    # Load sample data
    logging.info(f"Loading data from {PRD_SAMPLE_DATA}.")
    data = pd.read_parquet(PRD_SAMPLE_DATA).query("valor_indenizacao>0")
    
    # Log missing data for initial load
    log_missing_data(data, "Initial Data")
    
    tdata = []

    # Cumulative count encoding
    logging.info("Applying CumulativeCountEncoder.")
    data["dt_proposta"] = pd.to_datetime(data["dt_proposta"], dayfirst=True)
    transformer = CumulativeCountEncoder(
        group_by_cols=["nr_documento_segurado"], 
        sort_col=["dt_proposta", "id_proposta"]
    )
    transformer.fit(data)
    data_transformed = transformer.transform(data)
    data_transformed.rename(
        columns={"nr_previous_['dt_proposta', 'id_proposta']": "nr_previous_proposals"},
        inplace=True,
    )
    logging.info(f"Data shape after cumulative count encoding: {data_transformed.shape}")
    
    # Cumulative operation encoding for each column in value_columns
    for c in cols_dict["value_columns"]:
        logging.info(f"Processing column: {c} with CumulativeEncoder.")
        encoder = CumulativeEncoder(
            operation=["max", "min", "sum"],
            col=c,
            group_by_cols=["nr_documento_segurado"],
            sort_col=["dt_proposta", "id_proposta"],
        )
        encoder.fit(data)
        td = encoder.transform(data)
        tdata.append(td)
    
    # Concatenate all cumulative encoded data
    logging.info("Concatenating cumulative operation encoded data.")
    tdata = pd.concat(tdata, axis=1)
    tdata = tdata.loc[:, ~tdata.columns.duplicated()]  # Remove duplicated columns
    
    # Log missing data after encoding
    log_missing_data(tdata, "Cumulative Encoded Data")

    # Join transformed data
    logging.info("Joining cumulative count encoded data and cumulative operation encoded data.")
    data = pd.concat([data_transformed, tdata], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    
    # Log missing data after joining
    log_missing_data(data, "Joined Data")
    
    # Feature engineering
    logging.info("Applying feature engineering on valor_indenizacao.")
    data["first_digit"] = data["valor_indenizacao"].astype(str).str[0]
    data["second_digit"] = data["valor_indenizacao"].astype(str).str[1]
    data["third_digit"] = data["valor_indenizacao"].astype(str).str[2]

    # Save processed features to Parquet
    logging.info(f"Saving processed data to {PRD_FEAT_DATA}.")
    data.to_parquet(PRD_FEAT_DATA, index=False)

    # Prepare columns to write
    write_cols = [
        "nr_documento_segurado",
        "nr_proposta",
        "id_proposta",
        "dt_proposta",
        "evento_preponderante",
        "valor_indenizacao",
    ] + data.columns.to_list()
    write_cols = list(set(write_cols))

    # Log final columns and save sampled features data
    logging.info(f"Writing selected columns to {PRD_SAMP_FEAT_DATA}.")
    data[write_cols].to_parquet(PRD_SAMP_FEAT_DATA, index=False)

    # Final missing data check
    log_missing_data(data[write_cols], "Final Saved Data")

    logging.info("Data processing completed successfully.")
