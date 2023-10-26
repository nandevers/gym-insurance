import logging
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from gym import Env, spaces
from gym_insurance.envs.benford import Benford
from gym_insurance.envs.utils import Wallet

# Set the seed
np.random.seed(42)

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


from gym import Env, spaces
import pandas as pd
from gym_insurance.envs.utils import Wallet
from typing import List

class InsurEnv(Env):
    """A custom insurance environment for reinforcement learning."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        index_column: str,
        value_column: str,
        state_columns: List[str],
        budget: int,
        shuffle: bool = False,
    ) -> None:
        """
        Initializes the InsurEnv object with the given parameters.
        
        Parameters:
        - data (pd.DataFrame): The data to be used in the environment.
        - index_column (str): The column name to be used as the index.
        - value_column (str): The column name for values.
        - state_columns (List[str]): A list of column names to be used for states.
        - budget (int): The initial budget.
        - shuffle (bool, optional): Whether to shuffle the data on initialization. Defaults to False.
        """
        super().__init__()
        self._validate_inputs(data, index_column, value_column, state_columns)
        self._data = data.set_index(index_column)
        self._value_column = value_column
        self._state_columns = tuple(state_columns)
        self._budget = Wallet(budget)
        self._episodes = 0
        self.action_space = spaces.Discrete(2)
        if shuffle:
            self._shuffle_data()

        self.states, self.rewards, self.wcs, self.wocs = [], [], [], []
    
    @staticmethod
    def _validate_inputs(data: pd.DataFrame, index_column: str, value_column: str, state_columns: List[str]) -> None:
        if index_column not in data.columns or value_column not in data.columns:
            raise ValueError("Invalid column names provided.")
        if not all(column in data.columns for column in state_columns):
            raise ValueError("Invalid state_columns provided.")
        if data.empty:
            raise ValueError("Data cannot be empty.")

    def _shuffle_data(self):
        # Assuming shuffling data rows is the intention
        self._data = self._data.sample(frac=1).reset_index(drop=True)
        self.sampled_idx = self._data.index.to_list()

    def _reset(self):
        self.steps = self.approved = 0
        self.done = False
        self.max_value = self._data[self._value_column].max()
        self._shuffle_data()
        self._budget.reset()
        self.observation_space = np.zeros(len(self._state_columns) + 4)
        self.initial_state = self._data.loc[
            self.sampled_idx[0], self._state_columns
        ].values
        return self.observation_space

    def reset(self):
        return self._reset()

    @property
    def state(self) -> np.array:
        """
        Computes the state representation based on the current budget and case data.

        Returns:
            np.array: The state representation.
        """
        zero_state = np.zeros(len(self._state_columns) + 4)
        
        if self._budget.spent == 0:
            return zero_state
        elif self._budget.pct_budget <= 1:
            approved_cases = [
                case
                for case, decision in self._budget.decisions.items()
                if decision == 1
            ]
            approved_ratio = len(approved_cases) / self.steps if self.steps else 0
            current_case_values = self.case[self._state_columns].values.flatten()
            additional_state_info = [self._budget.pct_budget, approved_ratio, self.wc, self.woc]
            return np.concatenate([current_case_values, additional_state_info])
        else:
            return zero_state



    @property
    def case(self):
        case_data = self._data.loc[self.sampled_idx[self.steps]]
        if len(case_data.shape) > 1:
            logger.debug("More than one case presented at a time.")
            raise KeyError
        logger.info(f"Step: {self.steps} -- Case {case_data}")
        return case_data

    def _update_steps(self):
        self.steps += 1

    def _update_benford(self, action):
        if self.steps == 0:
            self.wc = 0.1
            self.woc = 1.10 * self.wc
        elif self.approved == 0:
            self.wc = self.woc / 2
        else:
            self.wc = self.benford_dist(action)
            self.woc = self.benford_dist(action, previous=True)

        if np.isnan(self.wc) or np.isnan(self.woc):
            logger.debug("Impossible to calculate reward")
            import pdb

            pdb.set_trace()

        self.wcs.append(self.wc)
        self.wocs.append(self.woc)

    def step(self, action):
        reward = 0
        self._update_benford(action)
        self._update_steps()

        if self.steps >= len(self.sampled_idx):
            logger.info("More steps than data, impossible to continue.")
            return np.zeros(len(self._state_columns) + 4), reward, True, {}

        self._budget.current_budget[self.current_case] = self._budget.balance
        indemnification_value = self.case[self._value_column].values[0]
        observation = self.state

        if indemnification_value is not None:
            if action == 1 and not self.done:
                try:
                    self._budget.withdraw(indemnification_value)
                    self._budget.decisions[self.current_case] = 1
                    reward = self._reward(action)
                except ValueError:
                    if self._budget.pct_budget <= self._data[self._value_column].min():
                        logger.error("Not enough budget")
                    if self.states:
                        observation = self.states[-1]
                    self._budget.decisions[self.current_case] = 0
                    self.done = True
                    self.episodes += 1
            elif action == 0 and not self.done:
                self._budget.decisions[self.current_case] = 0
                if self._budget.pct_budget < 1:
                    reward = self._reward(action)
                else:
                    self.done = True
                    self.episodes += 1

        self.rewards.append(reward)
        self.states.append(self.state)
        return observation, reward, self.done, {}

    def benford_dist(self, decision, previous=False):
        sample = self._sample(decision=decision, previous=previous)
        return Benford(3).distance(sample[self._value_column].values)

    def _sample(self, decision=1, previous=False):
        approved_cases = [
            case for case, dec in self._budget.decisions.items() if dec == 1
        ]
        if decision != 1:
            approved_cases.append(self.current_case)
        if previous:
            return self._data.loc[approved_cases[:-1]]
        return self._data.loc[approved_cases]

    def _reward(self, action):
        if self.steps == 0:
            return 0
        rev = 1  
        if action == 1 and self.wc < self.woc:
            return self.woc / self.wc * rev
        elif action == 1 and self.wc >= self.woc:
            return -self.wc / self.woc * rev
        elif action == 0 and self.wc < self.woc:
            return -self.woc / self.wc * rev
        elif action == 0 and self.wc >= self.woc:
            return self.wc / self.woc * rev

    @property
    def results(self):
        results_data = pd.DataFrame.from_dict(
            self._budget.decisions, orient="index", columns=["decision"]
        )
        results_data["rewards"] = self.rewards
        results_data["budget"] = list(self._budget.current_budget.values())
        results_data["initial_budget"] = self._budget.budget
        results_data["wc"] = self.wcs
        results_data["woc"] = self.wocs
        results_data["steps"] = range(1, self.steps + 1)
        results_data["episodes"] = self.episodes
        return results_data.join(self._data)
