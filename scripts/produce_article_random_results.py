import os
import pandas as pd
import gymnasium as gym
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import gym_insurance
from constants import BUDGET_LIST, DATAPATH

def get_state_columns(data):
    return data.columns[5:-1]


def load_data():
    return pd.read_csv(DATAPATH / "psr_train_and_test_set.csv").query(
        "valor_indenizacao==valor_indenizacao"
    )


# Updated experiment function
def run_experiment(budgets, num_episodes):
    results_folder = "experiment_results"
    os.makedirs(results_folder, exist_ok=True)

    for budget in budgets:
        index_column = "id_proposta"
        value_column = "valor_indenizacao"
        data = load_data()
        state_columns = get_state_columns(data)

        # Initialize the InsurEnv environment
        env = gym.make(
            "InsurEnv-v0",
            data=data,
            index_column=index_column,
            state_columns=state_columns,
            value_column=value_column,
            budget=budget,
        )

        env.reset(seed=1234)

        results_df = []
        # Random Agent Interaction with the Environment
        for episode in range(num_episodes):
            done = False
            obs = env.reset()
            while not done:
                action = env.action_space.sample()  # Random action
                obs, reward, done, _, _ = env.step(action)  # Take the action

            # Collect results from the environment
            results_df.append(env.results.copy(deep=True))
            results_df[-1]["episodes"] = episode

        # Save results to CSV
        results_df = pd.concat(results_df, axis=0)
        csv_filename = f"{results_folder}/results_{budget}.paquet"
        #results_df.to_parquet(csv_filename, index=False)
    print(f"Experiment completed. Results are saved in {results_folder}")


# Example budgets for running the experiment
run_experiment(BUDGET_LIST, num_episodes=500)
