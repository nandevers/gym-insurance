import pandas as pd
from gym_insurance.envs.insurenv import InsurEnv

data = pd.DataFrame(
    {
        "index_column": [0, 1, 2],
        "value_column": [100, 200, 300],
        "state_column1": [1, 2, 3],
        "state_column2": [4, 5, 6],
    }
)
env = InsurEnv(
    data, "index_column", "value_column", ["state_column1", "state_column2"], 1000
)

observartions = env.reset()


env.step(action=0)
