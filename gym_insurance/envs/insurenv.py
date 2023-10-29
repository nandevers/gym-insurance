import numpy as np
import pandas as pd
import logging
from typing import List, Tuple
from gym import Env
from gym_insurance.envs.benford import Benford
from gym_insurance.envs.utils import Wallet  # Assuming Budget is defined elsewhere or imported from a module


def configure_logging():
    logging.basicConfig(
        level=logging.ERROR,
        handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

configure_logging()
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class InsurEnv(Env):
    def __init__(
        self, 
        data: pd.DataFrame, 
        index_column: str, 
        value_column: str, 
        state_columns: List[str], 
        budget: float, 
        episodes: int = 10
    ) -> None:
        """
        Initializes the Insurance Environment.

        Args:
            data (pd.DataFrame): The data to be used in the environment.
            index_column (str): The column name to be used as the index.
            value_column (str): The column name for values.
            state_columns (List[str]): A list of column names to be used for states.
            budget (float): The initial budget for the environment.
            episodes (int, optional): The number of episodes. Defaults to 10.
        """
        super(InsurEnv, self).__init__()
        
        self._validate_inputs(data, index_column, value_column, state_columns)
        
        data.set_index(index_column, inplace=True)
        self._data = data
        self._index_column = index_column
        self._value_column = value_column
        self._state_columns = state_columns
        self._budget = Wallet(budget)
        self.episodes = episodes
        self.wcs = []
        self.wocs = []
        self.rewards = []
        self.states = []
        self._reset()
        

    def _validate_inputs(
        self,
        data: pd.DataFrame,
        index_column: str,
        value_column: str,
        state_columns: List[str],
    ) -> None:
        """
        Validates the input parameters provided to the environment.

        Args:
            data (pd.DataFrame): The data to be used in the environment.
            index_column (str): The column name to be used as the index.
            value_column (str): The column name for values.
            state_columns (List[str]): A list of column names to be used for states.

        Raises:
            ValueError: If any of the provided column names are not found in the data.
            ValueError: If the provided data is empty.
        """
        if index_column not in data.columns or value_column not in data.columns:
            raise ValueError("Invalid column names provided.")
        if not all(column in data.columns for column in state_columns):
            raise ValueError("Invalid state_columns provided.")
        if data.empty:
            raise ValueError("Data cannot be empty.")
        
    
    def _reset(self) -> None:
        """
        Resets the environment to the initial state for a new episode.

        This method initializes various attributes of the environment, preparing it for a new episode.
        It also shuffles the data if necessary and resets the budget and other counters.
        """
        self.steps = 0
        self.approved = 0
        self.done = False
        self.wc = 0  # Assuming wc is an attribute that needs to be reset, replace with actual initialization as necessary
        self.woc = 0  # Assuming woc is an attribute that needs to be reset, replace with actual initialization as necessary

        # Reset or initialize other necessary attributes here
        
        self._shuffle_data()  # Shuffle the data if necessary
        self._budget.reset()  # Reset the budget
        self._budget.decisions = self.sampled_idx
        # Add any additional initialization or resetting of attributes as necessary
        self.current_case = self.sampled_idx[self.steps]
        
        # Initialize or reset the observation space if necessary
        self.observation_space = np.zeros(len(self._state_columns), dtype=np.float32)
        self.initial_state = self._get_initial_state()

        return self.observation_space

    def _get_initial_state(self):
        initial_state = np.zeros(self.observation_space.shape)
        return initial_state

    def _shuffle_data(self) -> None:
        """
        Shuffles the data used in the environment.
        
        This method shuffles the rows of the data DataFrame and updates the sampled index list.
        """
        # Ensure there is data to shuffle
        if self._data.empty:
            raise ValueError("Cannot shuffle empty data.")

        self._data = (self._data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True))
        self.sampled_idx = self._data.index.to_list()



    @property
    def case(self):
        self._budget.current_case = self.sampled_idx[self.steps]
        case_data  = self._data.loc[self._budget.current_case]
        if len(case_data.shape) > 1:
            logger.debug("More than one case presented at a time.")
            raise KeyError
        logger.info(f"Step: {self.steps} -- \n Case: {case_data}")
        return case_data
    
    @property
    def state(self) -> np.array:
        """
        Computes the state representation based on the current budget and case data.

        Returns:
            np.array: The state representation.
        """
        zero_state = np.zeros(len(self._state_columns) + 4)

        if self._budget._spent == 0:
            return zero_state
        elif self._budget.budget_spent_percentage <= 1:
            approved_cases = [
                case
                for case, decision in self._budget.decisions.items()
                if decision == 1
            ]
            approved_ratio = len(approved_cases) / self.steps if self.steps else 0
            current_case_values = self.case[self._state_columns].values.flatten()
            additional_state_info = [
                self._budget.budget_spent_percentage,
                approved_ratio,
                self.wc,
                self.woc,
            ]
            return np.concatenate([current_case_values, additional_state_info])
        else:
            return zero_state




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
        self.wcs.append(self.wc)
        self.wocs.append(self.woc)

    def _sample(self, decision: int, previous: bool = False) -> pd.DataFrame:
        """
        Samples data based on the given decision and conditions.
        
        Args:
            decision (int): The decision based on which to sample the data.
            previous (bool, optional): If True, sample from previous cases. Defaults to False.
        
        Returns:
            pd.DataFrame: The sampled data.
        """
        if decision == 0:
            sample = self._data.loc[self.sampled_idx]
        elif previous:
            sample = self._data.loc[self.sampled_idx[self.steps - self.approved :]]
        else:
            sample = self._data
        
        return sample

    def reward(self, action):
        return self._reward(action)
    

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:       
        return self._step(action)

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
        

    def _step(self, action):
        self._update_benford(action)
        #FIXME: why use this werid looking local variable at all?
        #TODO: Why is reward zero here
        reward = 0

        if self.steps >= len(self.sampled_idx):
            logger.info("More steps than data, impossible to continue.")
            return np.zeros(len(self._state_columns) + 4), reward, True, {}

        self._budget.current_budget[self.current_case] = self._budget.balance
        benford_value = self.case[self._value_column]
        observation = self.state

        if benford_value is not None:
            if action == 1 and not self.done:
                try:
                    self._budget.withdraw(benford_value)
                    self._budget.decisions[self.current_case] = 1
                    reward = self._reward(action)
                except ValueError:
                    if self._budget.budget_spent_percentage <= self._data[self._value_column].min():
                        logger.error("Not enough budget")
                    if self.states:
                        observation = self.states[-1]
                    self._budget.decisions[self.current_case] = 0
                    self.done = True
                    self.episodes += 1
            elif action == 0 and not self.done:
                self._budget.decisions[self.current_case] = 0
                if self._budget.budget_spent_percentage < 1:
                    reward = self._reward(action)
                else:
                    self.done = True
                    self.episodes += 1

        self.rewards.append(reward)
        self.states.append(self.state)
        self._update_steps()
        return observation, reward, self.done, {}




    def _get_next_state(self, action):
        next_state = self.state.copy()
        next_state[-1] = action  # Example: set the last element of the state to the current action
        return next_state
    
    
    def render(self, mode='human') -> None:
        """
        Renders the current state of the environment.

        Args:
            mode (str, optional): The mode in which to render the environment. Defaults to 'human'.
        """
        if mode != 'human':
            raise ValueError("Invalid mode for rendering. Currently, only 'human' mode is supported.")
        
        print(f"Steps: {self.steps}, Approved: {self.approved}, Budget: {self._budget.available}, Max Value: {self.max_value}")

    def benford_dist(self, decision: int, previous: bool = False) -> float:
        """
        Calculates the Benford distance for a given decision.

        Args:
            decision (int): The decision for which to calculate the Benford distance.
            previous (bool, optional): If True, use previous cases for the calculation. Defaults to False.

        Returns:
            float: The calculated Benford distance.
        """
        sample = self._sample(decision=decision, previous=previous)
        return Benford(3).distance(sample[self._value_column].values)


    @property
    def results(self) -> pd.DataFrame:
        """
        Retrieves the results of the environment's interactions.

        Returns:
            pd.DataFrame: A DataFrame containing the results data.
        """
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





