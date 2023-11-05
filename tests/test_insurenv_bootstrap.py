import pandas as pd
import numpy as np
import unittest
from gym_insurance.envs.insurenv import InsurEnv


class TestInsurEnv(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset
        data = pd.DataFrame(
            {
                "index_column": range(100),
                "value_column": np.random.rand(100) * 100,
                "feature_1": np.random.rand(100),
                "feature_2": np.random.rand(100),
            }
        )

        # Initialize the InsurEnv environment
        self.env = InsurEnv(
            data=data,
            index_column="index_column",
            value_column="value_column",
            state_columns=["feature_1", "feature_2"],
            budget=100000,
        )

        # Run the environment with a fixed set of actions
        np.random.seed(42)  # for reproducibility
        self.num_steps = 100
        self.actions = np.random.choice([0, 1], size=self.num_steps)
        self.approved = 0
        self.rejected = 0

        for action in self.actions:
            _, _, _, _ = self.env.step(action)
            if action == 1:
                self.approved += 1
            else:
                self.rejected += 1

    def tearDown(self):
        # Any necessary cleanup after each test
        pass

    def test_action_distribution(self):
        # Check if the number of approved and rejected cases are roughly 50% each
        self.assertAlmostEqual(self.approved / self.num_steps, 0.5, delta=0.1)
        self.assertAlmostEqual(self.rejected / self.num_steps, 0.5, delta=0.1)

    def test_results(self):
        results_df = self.env.results

        # Check if the results DataFrame has the correct columns
        expected_columns = [
            "decision",
            "rewards",
            "budget",
            "initial_budget",
            "wc",
            "woc",
            "steps",
            "episodes",
        ]
        self.assertTrue(
            all(column in results_df.columns for column in expected_columns)
        )

        # Check if the length of the results DataFrame is correct
        self.assertEqual(len(results_df), self.num_steps)

        # Check if the 'decision' column is correctly populated
        self.assertTrue(
            all(
                results_df["decision"].iloc[i] == self.actions[i]
                for i in range(self.num_steps)
            )
        )

        # Check if the 'steps' column is correctly populated
        self.assertTrue(
            all(results_df["steps"].iloc[i] == i + 1 for i in range(self.num_steps))
        )

        # Add any other assertions needed to verify the results are correct


if __name__ == "__main__":
    unittest.main()
