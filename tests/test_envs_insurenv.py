import unittest
import pandas as pd
import numpy as np
from gym_insurance.envs.insurenv import InsurEnv  # Adjust the import based on your project structure

class TestInsurEnv(unittest.TestCase):

    def setUp(self):
        # Assuming a DataFrame structure based on the discussions
        data = pd.DataFrame({
            'index_column': [0, 1, 2],
            'value_column': [100, 200, 300],
            'state_column1': [1, 2, 3],
            'state_column2': [4, 5, 6],
        })
        self.env = InsurEnv(data, 'index_column', 'value_column', ['state_column1', 'state_column2'], 1000)

    def test_init(self):
        self.assertIsInstance(self.env._data, pd.DataFrame)
        self.assertEqual(self.env._value_column, 'value_column')
        self.assertEqual(self.env._state_columns, ('state_column1', 'state_column2'))
        self.assertEqual(self.env._budget.budget, 1000)

    def test_reset(self):
        observation_space = self.env._reset()
        self.assertEqual(len(observation_space), len(self.env._state_columns) + 4)
        self.assertEqual(self.env._budget.spent, 0)
        self.assertEqual(self.env._episodes, 0)  # Assuming _episodes gets reset to 0 in _reset()

    def test_shuffle_data(self):
        pre_shuffle_data = self.env._data.copy()
        self.env._shuffle_data()
        post_shuffle_data = self.env._data
        self.assertFalse(pre_shuffle_data.equals(post_shuffle_data))

    def test_state(self):
        # Assuming some setup to get to a state where budget.spent != 0 and budget.pct_budget <= 1
        # ...
        state = self.env.state
        expected_length = len(self.env._state_columns) + 4
        self.assertEqual(len(state), expected_length)

if __name__ == "__main__":
    unittest.main()
