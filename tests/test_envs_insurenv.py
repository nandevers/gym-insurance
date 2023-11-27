import unittest
import gymnasium as gym

import gym_insurance
import pandas as pd
from definitions import TEMP_PATH, FILENAME, logger
import warnings
import os
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)


class TestInsurEnv(unittest.TestCase):
    def setUp(self, budget = 1000):
        file_path = os.path.join(TEMP_PATH, FILENAME)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        data = pd.read_parquet(file_path).reset_index(drop=True)
        logger.info("Data columns: %s", data.columns.tolist())

        index_column = "claim_id"
        value_column = "amount"
        state_columns = [
            "customer_age",
            "days_to_report",
            "previous_claims",
            "feature_1",
            "feature_2",
            "feature_3",
        ]

        self.env = gym.make(
            "InsurEnv-v0",
            data=data,
            index_column=index_column,
            state_columns=state_columns,
            value_column=value_column,
            budget=budget,
        )

    def test_initialization(self):
        self.assertIsNotNone(self.env)

    def test_reset_behavior(self):
        initial_state = self.env.reset()
        self.assertIsNotNone(initial_state)

    def test_step_method_behavior(self):
        initial_state, _  = self.env.reset()
        next_state, reward, done, truncated, _ = self.env.step(1)  # Example action
        self.assertTrue(np.array_equal(initial_state, next_state))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_reward_calculation(self):
        self.env.reset()
        action = 1  # Example action
        reward = self.env.reward(action)
        self.assertIsInstance(reward, float)

    def test_observation_space_initialization(self):
        self.assertIsNotNone(self.env.observation_space)

    def test_action_space_initialization(self):
        self.assertIsNotNone(self.env.action_space)

    def test_internal_state_updates(self):
        self.env.reset()
        self.env.step(1)  # Example action
        self.assertTrue(hasattr(self.env, 'steps'))

    def test_benford_dist_method(self):
        self.env.reset()
        wc, woc = self.env.benford_dist()
        self.assertIsInstance(wc, float)
        self.assertIsInstance(woc, float)


    # Add more test methods as needed

if __name__ == '__main__':
    unittest.main()