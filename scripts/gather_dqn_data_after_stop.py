
import gym_insurance.envs.utils as ut
# Imports at the beginning of the file
import os
import shutil

import pandas as pd

from gym_insurance.constants import BUDGET_LIST, DATAPATH
from gym_insurance.envs.insurenv2 import InsurEnv

# Constants
RESULTS_FOLDER = ".temp/data/experiment_results"
SOURCE_DIR = "C:/Users/nande/Repos/gym-insurance/.temp/data/experiment_results/insurenv"
DESTINATION_DIR = ".temp/data/experiment_results/dqn/insurenv/"
NUM_EPISODES = 116  # Number of episodes to run in the experiment
MAX_STEPS = 3_000


def main():
    """
    Main function to run the experiment and move files after completion.
    """
    # Run the experiment with the provided budget list and number of episodes

    for budget in BUDGET_LIST:
        # Move the parquet files from the experiment results folder
        ut.move_parquet_files(SOURCE_DIR, DESTINATION_DIR, budget=budget)
    
if __name__ == "__main__":
    main()
