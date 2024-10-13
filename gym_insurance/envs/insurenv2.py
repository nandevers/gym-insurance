import os
import numpy as np
import warnings
import pandas as pd
import logging
from typing import List
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

warnings.filterwarnings("ignore", category=RuntimeWarning)

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

configure_logging()
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class InsurEnv(py_environment.PyEnvironment):
    def __init__(self, data: pd.DataFrame, index_column: str, value_column: str, state_columns: List[str], budget: float) -> None:
        """
        Initializes the Insurance Environment.

        Args:
            data (pd.DataFrame): The data to be used in the environment.
            index_column (str): The column name to be used as the index.
            value_column (str): The column name for values.
            state_columns (List[str]): A list of column names to be used for states.
            budget (float): The initial budget for the environment.
        """
        super().__init__()

        self._validate_inputs(data, index_column, value_column, state_columns)
        data.set_index(index_column, inplace=True)

        self._data = data
        self._index_column = index_column
        self._value_column = value_column
        self._state_columns = state_columns
        self.initial_budget = budget

        self._reset()
        
    def _validate_inputs(self, data: pd.DataFrame, index_column: str, value_column: str, state_columns: List[str]) -> None:
        """Validates the input parameters provided to the environment."""
        if index_column not in data.columns or value_column not in data.columns:
            raise ValueError("Invalid column names provided.")
        if not all(column in data.columns for column in state_columns):
            raise ValueError("Invalid state_columns provided.")
        if data.empty:
            raise ValueError("Data cannot be empty.")
    
    def action_spec(self):
        """Returns the action specification for the environment."""
        return array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')

    def observation_spec(self):
        """Returns the observation specification for the environment."""
        return array_spec.BoundedArraySpec(shape=(len(self._state_columns) + 4,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name='observation')

    def _reset(self):
        """Resets the environment to its initial state."""
        self.steps = 0
        self.done = False
        self._budget = Wallet(self.initial_budget, self.sampled_idx)
        self._budget.reset()
        self.initial_state = self._get_initial_state()
        return ts.restart(self.initial_state)

    def reset(self):
        """Resets the environment for a new episode."""
        return self._reset()

    def _get_initial_state(self):
        """Returns the initial state of the environment."""
        initial_state = np.zeros(self.observation_spec().shape)
        return initial_state
    
    def _step(self, action):
        """Takes a step in the environment."""
        if self.done:
            return self.reset()
        
        reward = self._reward(action)

        self.steps += 1
        observation = self.state
        
        if self.steps >= len(self.sampled_idx):
            self.done = True
            return ts.termination(observation, reward)
        
        return ts.transition(observation, reward)

    def step(self, action: int):
        """Performs the action and returns the new state."""
        return self._step(action)

    def _reward(self, action):
        """Calculates the reward for a given action."""
        r = self.current_value
        wc, woc = self.benford_dist()
        reward = 0

        if action == 1 and wc < woc:
            reward = woc / wc * r
        elif action == 1 and wc > woc:
            reward = -wc / woc * r
        elif action == 0 and wc < woc:
            reward = -woc / wc * r
        elif action == 0 and wc > woc:
            reward = wc / woc * r

        if reward > 0:
            return reward - 1
        elif reward < 0:
            return reward + 1
        else:
            return reward

    def render(self, mode="human") -> None:
        """Renders the environment's state."""
        if mode != "human":
            raise ValueError("Invalid mode for rendering. Only 'human' mode is supported.")
        
        print(f"Steps: {self.steps}, Approved: {self.approved}, Budget: {self._budget.available}")

    def benford_dist(self):
        """Calculates the Benford distance for the current case."""
        cases = self._budget.get_approved_cases()
        sample = self._data.loc[cases, self._value_column].values
        woc = Benford(3).distance(sample)

        cases.append(self.current_case)
        sample = self._data.loc[cases, self._value_column].values
        wc = Benford(3).distance(sample)
        return wc, woc

    @property
    def state(self):
        """Returns the current state of the environment."""
        zero_state = np.zeros(len(self._state_columns) + 4)

        if self._budget._spent == 0:
            return zero_state
        elif self._budget.budget_spent_percentage <= 1:
            approved_cases = [
                case for case, decision in self._budget.decisions.items() if decision == 1
            ]
            approved_ratio = len(approved_cases) / self.steps if self.steps else 0
            current_case_values = self.case[self._state_columns].values.flatten()
            current_case_values = np.array(current_case_values, dtype=np.float32)
            additional_state_info = [
                self._budget.budget_spent_percentage, approved_ratio, self.wc, self.woc,
            ]
            return np.concatenate([current_case_values, additional_state_info], dtype=np.float32)
        else:
            return zero_state

