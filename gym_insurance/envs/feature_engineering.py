import numpy as np
import pandas as pd
from category_encoders import BaseNEncoder, CountEncoder
from feature_engine.imputation import MeanMedianImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    normalize,
)

PRD_SAMP_FEAT_DATA = "data/processed/psr_sampled_features.csv"
data = pd.read_csv(PRD_SAMP_FEAT_DATA)
data["dt_proposta"] = pd.to_datetime(data["dt_proposta"])
CUTOFF_YEAR = 2020
print(data.shape)

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


def handle_missing_values(data, method):
    data["evento_preponderante"].fillna("outro", inplace=True)
    mmi = MeanMedianImputer(imputation_method=method)
    mmi.fit(data.loc[data.dt_proposta.dt.year <= CUTOFF_YEAR, STD_COLS[1:]])
    data.loc[:, STD_COLS[1:]] = mmi.transform(data.loc[:, STD_COLS[1:]])
    return data


def one_hot_encode(data, feature):
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
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    return data, [feature]


def standard_scaler(data):
    std_new_cols = [f"std_{c}" for c in STD_COLS]
    scaler = StandardScaler()
    scaler.fit(data.loc[data.dt_proposta.dt.year <= CUTOFF_YEAR, STD_COLS])
    std_data = scaler.transform(data.loc[:, STD_COLS])
    std_data = pd.DataFrame(std_data, columns=std_new_cols)
    data = pd.concat([data, std_data], axis=1)
    return data, std_new_cols


def robust_scaler(data):
    rs_new_cols = [f"rs_{c}" for c in STD_COLS]
    scaler = RobustScaler()
    scaler.fit(data.loc[data.dt_proposta.dt.year <= CUTOFF_YEAR, STD_COLS])
    rs_data = scaler.transform(data.loc[:, STD_COLS])
    rs_data = pd.DataFrame(rs_data, columns=rs_new_cols)
    data = pd.concat([data, rs_data], axis=1)
    return data, rs_new_cols


def minmax_scaler(data, feature_range=(0, 1)):
    mm_new_cols = [f"mm_{c}" for c in STD_COLS]
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(data.loc[data.dt_proposta.dt.year <= CUTOFF_YEAR, STD_COLS])
    mm_data = scaler.transform(data.loc[:, STD_COLS])
    mm_data = pd.DataFrame(mm_data, columns=mm_new_cols)
    data = pd.concat([data, mm_data], axis=1)
    return data, mm_new_cols


def normalize_data(data, norm="l2"):
    norm_new_cols = [f"norm_{c}" for c in STD_COLS]
    norm_data = normalize(data[STD_COLS], norm=norm)
    norm_data = pd.DataFrame(norm_data, columns=norm_new_cols)
    data = pd.concat([data, norm_data], axis=1)
    return data, norm_new_cols


def quantile_transformer(
    data, n_quantiles=100, output_distribution="uniform", random_state=None
):
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


def power_transformer(data):
    pt_new_cols = [f"pt_{c}" for c in PT_COLS]
    scaler = PowerTransformer()
    scaler.fit(data.loc[data.dt_proposta.dt.year <= CUTOFF_YEAR, PT_COLS])
    pt_data = scaler.transform(data.loc[:, PT_COLS])
    pt_data = pd.DataFrame(pt_data, columns=pt_new_cols)
    data = pd.concat([data, pt_data], axis=1)
    return data, pt_new_cols


def base_n_encode(data, feature, base=2):
    encoder = BaseNEncoder(base=base)
    encoded = encoder.fit_transform(data[feature])
    columns = [f"{feature}_encoded_{i}" for i in range(encoded.shape[1])]
    encoded.columns = columns
    return pd.concat([data, encoded], axis=1), columns


def count_encode(data, feature):
    ce = CountEncoder()
    data = ce.fit_transform(data[feature])
    data.rename(columns={feature: f"{feature}_count"}, inplace=True)
    return pd.concat([data], axis=1), [f"{feature}_count"]


data = handle_missing_values(data, "median")
data, ep_cols = one_hot_encode(data, "evento_preponderante")
data, fd_cols = one_hot_encode(data, "first_digit")
data, sd_cols = one_hot_encode(data, "second_digit")
data, td_cols = one_hot_encode(data, "third_digit")
data, std_cols = quantile_transformer(data)

cols_to_write = (
    ID_COLS + std_cols + ep_cols + fd_cols + sd_cols + td_cols + ["valor_indenizacao"]
)
data.loc[data.dt_proposta.dt.year <= CUTOFF_YEAR, cols_to_write].to_csv(
    "data/processed/psr_train_set.csv", index=False
)

data.loc[data.dt_proposta.dt.year > CUTOFF_YEAR, cols_to_write].to_csv(
    f"data/processed/psr_test_set.csv", index=False
)

data.loc[:, cols_to_write].to_csv(
    f"data/processed/psr_train_and_test_set.csv", index=False
)
