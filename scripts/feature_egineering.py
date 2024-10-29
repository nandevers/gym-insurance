import logging
import argparse
import numpy as np
import pandas as pd
from category_encoders import BaseNEncoder, CountEncoder
from feature_engine.imputation import MeanMedianImputer
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler, normalize)
import os

# Constants
CUTOFF_YEAR = 2023
PRD_SAMP_FEAT_DATA = ".temp/data/processed/psr_sampled_features.parquet"
OUTPUT_DIR = ".temp/data/processed/"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Column definitions
ID_COLS = [
    "nr_documento_segurado",
    "nr_proposta",
    "id_proposta",
    "dt_proposta",
    "evento_preponderante",
]
STD_COLS = [
    "valor_indenizacao",
    "nr_previous_proposals",
    "max_nr_area_total",
    "min_nr_area_total",
    "sum_nr_area_total",
    "max_niveldecobertura",
    "min_niveldecobertura",
    "sum_niveldecobertura",
    "max_valor_indenizacao",
    "min_valor_indenizacao",
    "sum_valor_indenizacao",
    "max_vl_subvencao_federal",
    "min_vl_subvencao_federal",
    "sum_vl_subvencao_federal",
    "max_pe_taxa",
    "min_pe_taxa",
    "sum_pe_taxa",
    "max_vl_premio_liquido",
    "min_vl_premio_liquido",
    "sum_vl_premio_liquido",
    "max_nr_produtividade_segurada",
    "min_nr_produtividade_segurada",
    "sum_nr_produtividade_segurada",
    "max_vl_limite_garantia",
    "min_vl_limite_garantia",
    "sum_vl_limite_garantia",
]


# Functions for preprocessing
def handle_missing_values(data, method):
    logger.info(f"Handling missing values using {method} method.")
    data["evento_preponderante"].fillna("outro", inplace=True)
    mmi = MeanMedianImputer(imputation_method=method)
    mmi.fit(data.loc[data.dt_proposta.dt.year <= CUTOFF_YEAR, STD_COLS[1:]])
    data.loc[:, STD_COLS[1:]] = mmi.transform(data.loc[:, STD_COLS[1:]])
    return data


def one_hot_encode(data, feature):
    logger.info(f"One hot encoding feature: {feature}")
    enc = OneHotEncoder(handle_unknown="infrequent_if_exist")
    ep_train = data.loc[data.dt_proposta.dt.year <= CUTOFF_YEAR, feature]
    ep = data[feature]
    enc.fit(np.reshape(ep_train.values, (-1, 1)))
    ep_matrix = enc.transform(np.reshape(ep.values, (-1, 1))).toarray()
    ep_matrix = pd.DataFrame(ep_matrix)
    ep_cols = [f"{feature}_{c}" for c in ep_matrix.columns]
    ep_matrix.columns = ep_cols
    data = pd.concat([data, ep_matrix], axis=1)
    return data, ep_cols


def label_encode(data, feature):
    logger.info(f"Label encoding feature: {feature}")
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    return data, [feature]


def standard_scaler(data):
    logger.info("Standard scaling data.")
    std_new_cols = [f"std_{c}" for c in STD_COLS]
    scaler = StandardScaler()
    scaler.fit(data.loc[data.dt_proposta.dt.year <= CUTOFF_YEAR, STD_COLS])
    std_data = scaler.transform(data.loc[:, STD_COLS])
    std_data = pd.DataFrame(std_data, columns=std_new_cols)
    data = pd.concat([data, std_data], axis=1)
    return data, std_new_cols


def quantile_transformer(data, n_quantiles=100, output_distribution="uniform", random_state=None):
    logger.info(f"Applying quantile transformer with {n_quantiles} quantiles and {output_distribution} distribution.")
    qt_new_cols = [f"qt_{c}" for c in STD_COLS]
    transformer = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        random_state=random_state,
    )
    transformer.fit(data.loc[data.dt_proposta.dt.year <= CUTOFF_YEAR, STD_COLS])
    qt_data = transformer.transform(data.loc[:, STD_COLS])
    qt_data = pd.DataFrame(qt_data, columns=qt_new_cols)
    data = pd.concat([data, qt_data], axis=1)
    return data, qt_new_cols


def save_data(data, cols_to_write, output_dir, cutoff_year):
    logger.info(f"Saving processed data to {output_dir}.")
    train_path = os.path.join(output_dir, "psr_train_set.parquet")
    test_path = os.path.join(output_dir, "psr_test_set.parquet")
    full_data_path = os.path.join(output_dir, "psr_train_and_test_set.parquet")

    data.loc[data.dt_proposta.dt.year <= cutoff_year, cols_to_write].to_parquet(train_path, index=False)
    data.loc[data.dt_proposta.dt.year > cutoff_year, cols_to_write].to_parquet(test_path, index=False)
    data.loc[:, cols_to_write].to_parquet(full_data_path, index=False)
    logger.info("Data saved successfully.")


def main(input_path, output_dir, method):
    logger.info(f"Loading data from {input_path}.")
    data = pd.read_parquet(input_path)
    data["dt_proposta"] = pd.to_datetime(data["dt_proposta"])
    data = data.query("valor_indenizacao > 0")
    logger.info(f"Data loaded with shape {data.shape}")

    data = handle_missing_values(data, method)

    # Encoding features
    data, ep_cols = one_hot_encode(data, "evento_preponderante")
    data, fd_cols = one_hot_encode(data, "first_digit")
    data, sd_cols = one_hot_encode(data, "second_digit")
    data, td_cols = one_hot_encode(data, "third_digit")

    # Quantile transformation
    data, std_cols = quantile_transformer(data)

    # Add sample column for train/test split
    data["sample"] = 'train'
    data.loc[data.dt_proposta.dt.year > CUTOFF_YEAR, 'sample'] = 'test'

    cols_to_write = ID_COLS + std_cols + ep_cols + fd_cols + sd_cols + td_cols + ["valor_indenizacao", "sample"]
    save_data(data, cols_to_write, output_dir, CUTOFF_YEAR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and save dataset with transformations.")
    parser.add_argument("--input_path", type=str, default=PRD_SAMP_FEAT_DATA, help="Path to the input data file.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory where output data will be saved.")
    parser.add_argument("--method", type=str, default="median", help="Method for missing value imputation.")
    
    args = parser.parse_args()

    main(args.input_path, args.output_dir, args.method)
