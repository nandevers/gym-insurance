import json

import pandas as pd
from f_utils import CumulativeCountEncoder, CumulativeEncoder

PRD_SAMPLE_DATA = ".temp/data/processed/psr_soja_pr_sample.parquet"
PRD_FEAT_DATA = ".temp/data/processed/psr_features.parquet"
PRD_SAMP_FEAT_DATA = ".temp/data/processed/psr_sampled_features.parquet"

if __name__ == "__main__":
    # load column dict
    with open("./scripts/cols_dict.json") as f:
        cols_dict = json.load(f)

    # load data
    data = pd.read_parquet(PRD_SAMPLE_DATA)
    tdata = []

    # cumulative count encoding
    data["dt_proposta"] = pd.to_datetime(data.dt_proposta, dayfirst=True)
    transformer = CumulativeCountEncoder(
        group_by_cols=["nr_documento_segurado"], sort_col=["dt_proposta", "id_proposta"]
    )
    transformer.fit(data)
    data_transformed = transformer.transform(data)
    data_transformed.rename(
        columns={"nr_previous_['dt_proposta', 'id_proposta']": "nr_previous_proposals"},
        inplace=True,
    )
    print(data_transformed.shape)

    # cumulative operation encoding
    for c in cols_dict["value_columns"]:
        print(f"Processing column: {c}")
        encoder = CumulativeEncoder(
            operation=["max", "min", "sum"],
            col=c,
            group_by_cols=["nr_documento_segurado"],
            sort_col=["dt_proposta", "id_proposta"],
        )
        encoder.fit(data)
        td = encoder.transform(data)
        tdata.append(td)
    tdata = pd.concat(tdata, axis=1)
    tdata = tdata.loc[:, ~tdata.columns.duplicated()]
    data = pd.concat([data_transformed, tdata], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    del tdata
    data["first_digit"] = data["valor_indenizacao"].astype(str).str[0]
    data["second_digit"] = data["valor_indenizacao"].astype(str).str[1]
    data["third_digit"] = data["valor_indenizacao"].astype(str).str[2]
    # save to csv
    # TODO: change this to parquet
    data.to_parquet(PRD_FEAT_DATA, index=False)

    write_cols = [
        "nr_documento_segurado",
        "nr_proposta",
        "id_proposta",
        "dt_proposta",
        "evento_preponderante",
        "valor_indenizacao",
    ] + data.columns.to_list()
    write_cols = list(set(write_cols))

    data[write_cols].to_parquet(PRD_SAMP_FEAT_DATA, index=False)
