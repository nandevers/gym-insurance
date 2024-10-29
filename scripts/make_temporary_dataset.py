import os

import numpy as np
import pandas as pd

np.random.seed(42)

import numpy as np

np.random.seed(42)  # For reproducibility

data = pd.DataFrame(
    {
        "claim_id": np.arange(1, 101),
        "amount": np.random.uniform(500, 5000, 100).round(2),
        "fraud": np.random.choice([0, 1], 100, p=[0.8, 0.2]),
        "customer_age": np.random.randint(18, 70, 100),
        "claim_type": np.random.choice(
            ["Theft", "Accident", "Natural Disaster", "Other"], 100
        ),
        "days_to_report": np.random.randint(1, 30, 100),
        "previous_claims": np.random.randint(0, 5, 100),
        "feature_1": np.random.uniform(0, 1, 100).round(2),
        "feature_2": np.random.uniform(0, 1, 100).round(2),
        "feature_3": np.random.uniform(0, 1, 100).round(2),
        "claim_date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
    }
)


TEMP_PATH = ".temp/"
filename = "insurance_claims.parquet"

if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)

data.to_parquet(os.path.join(TEMP_PATH, filename))
