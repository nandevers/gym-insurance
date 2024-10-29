
import gym_insurance.envs.utils as ut
# Imports at the beginning of the file
import os
import shutil

import pandas as pd

from gym_insurance.constants import BUDGET_LIST, DATAPATH
from gym_insurance.envs.insurenv2 import InsurEnv

# Constants
RESULTS_FOLDER = ".temp/data/experiment_results"
SOURCE_DIR = ".temp/data/experiment_results/insurenv"
DESTINATION_DIR = ".temp/data/experiment_results/random/insurenv/"
NUM_EPISODES = 142  # Number of episodes to run in the experiment
MAX_STEPS = 3_000


# Check if the directory exists
if os.path.exists(SOURCE_DIR):
    # Remove the entire directory and its contents
    shutil.rmtree(SOURCE_DIR)
    print(f"Directory '{SOURCE_DIR}' and its contents have been removed.")
else:
    print(f"Directory '{SOURCE_DIR}' does not exist.")
    
    

# Function to extract specific state columns from the dataset
def get_state_columns(data):
    """
    Extract state columns from the given dataset.
    
    :param data: DataFrame containing the data
    :return: List of column names that represent the state (features)
    """
    return data.columns[5:-1]

# Function to load and filter the dataset
def load_data():
    """
    Load and filter the dataset for running experiments. 
    Only selects rows where the 'valor_indenizacao' is greater than 100.
    
    :return: Filtered DataFrame
    """
    return pd.read_parquet(DATAPATH / "processed/psr_train_and_test_set.parquet").query(
        "valor_indenizacao > 100"
    )

# Function to run the experiment with random agent interaction
def run_experiment(budgets, num_episodes):
    """
    Run the experiment by interacting with the insurance environment using random actions.
    
    :param budgets: List of budget values to test
    :param num_episodes: Number of episodes to run for each budget
    """
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    for budget in budgets:
        index_column = "id_proposta"
        value_column = "valor_indenizacao"
        
        # Load the dataset and extract the state columns
        data = load_data().query(f"{value_column}<={budget}")
        state_columns = get_state_columns(data)

        # Initialize the insurance environment (InsurEnv)
        env = InsurEnv(
            data=data,
            index_column=index_column,
            state_columns=state_columns,
            value_column=value_column,
            budget=budget,
            max_steps=MAX_STEPS,
        )

        # Set random seed and reset the environment
        env.reset(seed=1234)

        results_df = []

        # Run the experiment for the specified number of episodes
        for episode in range(num_episodes):
            done = False
            obs = env.reset()
            while not done:
                action = env.action_space.sample()  # Sample a random action
                obs, reward, done, _ = env.step(action)  # Take action, get reward, check if done

            # Store the results for the current episode
            results_df.append(env.results.copy(deep=True))
            results_df[-1]["episodes"] = episode

        # Save results for the current budget
        results_df = pd.concat(results_df, axis=0)
        filename = f"{RESULTS_FOLDER}/random/results/results_{budget}.parquet"
        results_df.to_parquet(filename)

    print(f"Experiment completed. Results are saved in {RESULTS_FOLDER}")

# Function to move parquet files from one directory to another
def move_parquet_files(source_dir, destination_dir):
    """
    Move all parquet files from source_dir (including subdirectories) to destination_dir.
    
    :param source_dir: The directory to move files from.
    :param destination_dir: The directory to move files to.
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".parquet"):
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_dir, file)
                print(f"Moving file: {source_file_path} to {destination_file_path}")
                shutil.move(source_file_path, destination_file_path)

def main():
    """
    Main function to run the experiment and move files after completion.
    """
    # Run the experiment with the provided budget list and number of episodes
    run_experiment(BUDGET_LIST, num_episodes=NUM_EPISODES)
    for budget in BUDGET_LIST:
        # Move the parquet files from the experiment results folder
        ut.move_parquet_files(SOURCE_DIR, DESTINATION_DIR, budget=budget)
    
if __name__ == "__main__":
    main()
