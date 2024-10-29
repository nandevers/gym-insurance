import os
import numpy as np
import warnings
import pandas as pd
import logging
from typing import List, Tuple
from gym import Env, spaces
from gym_insurance.envs.benford import Benford
from gym_insurance.envs.wallet import Wallet

# Suppress specific runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configure logging
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


class InsurEnv(Env):
    def __init__(
        self,
        data: pd.DataFrame,
        index_column: str,
        value_column: str,
        state_columns: List[str],
        budget: float,
        max_steps: int = 100,  # Added max_steps parameter with a default value
    ) -> None:
        """
        Initializes the Insurance Environment.

        Args:
            data (pd.DataFrame): The data to be used in the environment.
            index_column (str): The column name to be used as the index.
            value_column (str): The column name for values.
            state_columns (List[str]): A list of column names to be used for states.
            budget (float): The initial budget for the environment.
            max_steps (int, optional): The maximum number of steps per episode. Defaults to 100.
        """
        super().__init__()

        self._validate_inputs(data, index_column, value_column, state_columns)

        data.set_index(index_column, inplace=True)
        self._data = data
        self._index_column = index_column
        self._value_column = value_column
        self._state_columns = state_columns
        self.initial_budget = budget
        self.max_steps = max_steps  # Store the maximum number of steps
        self.episodes = 0
        self.episode_data = {}
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

    def reset(self, seed=None, options=None):
        if self.episodes > 0:
            self.episode_data[str(self.episodes)] = self.results.copy(deep=True)
            os.makedirs(f".temp/data/experiment_results/insurenv/{self.episodes}", exist_ok=True)
            df = pd.concat(self.episode_data).reset_index(level=0)
            df.rename(columns = {'level_0': 'episodes'}, inplace=True)
            df = df.reset_index(drop=True)
            df.to_parquet(
                f".temp/data/experiment_results/insurenv/{self.episodes}/results_{self.initial_budget}.parquet"
            )
        if seed is not None:
            super().reset(seed=seed)
        self.episodes += 1
        return self._reset().astype(np.float32), {}

    def _reset(self) -> None:
        """
        Resets the environment to the initial state for a new episode.

        This method initializes various attributes of the environment, preparing it for a new episode.
        It also shuffles the data if necessary and resets the budget and other counters.
        """
        self.wcs = []
        self.wocs = []
        self.rewards = []
        self.states = []
        self.action_space = spaces.Discrete(2)
        self.steps = 0  # Reset step counter at the start of each episode
        self.approved = 0
        self.wc = 0
        self.woc = 0
        self.truncated = False
        self.done = False

        self._shuffle_data()  # Shuffle the data if necessary
        self._budget = Wallet(self.initial_budget, self.sampled_idx)
        self._budget.reset()  # Reset the budget
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self._state_columns) + 4,),
            dtype=np.float32,
        )
        self.initial_state = self._get_initial_state()
        return self.initial_state
   
   
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict, bool]:
        logger.info(f"Starting step {self.steps}")
        reward = 0

        # Check if case or current_value is NaN
        if self.case is None or np.isnan(self.current_value):
            logger.warning(f"Encountered NaN case or value at step {self.steps}, ending episode.")
            self.done = True
            return self.state, reward, self.done, {}, self.truncated
        
        # Check if maximum steps have been reached
        if self.steps >= len(self.sampled_idx) or self.steps >= self.max_steps:
            logger.info("Maximum steps reached or more steps than data, ending episode.")
            self.done = True
            return np.zeros(len(self._state_columns) + 4), reward, True, False, {}

        observation = self.state
        if self.current_value is not None:
            if action == 1 and not self.done:  # Approve action
                try:
                    # Use the decision method to handle approval and budget logic
                    self._budget.decision(case=self.current_case, decision=1, value=self.current_value)
                    reward = self._reward(action)
                except ValueError:
                    if self._budget.budget_spent_percentage * self.initial_budget <= self._data[self._value_column].min():
                        logger.error("Not enough budget")
                    if self.states:
                        observation = self.states[-1]
                    self._budget.decision(case=self.current_case, decision=0, value=self.current_value)
                    self.truncated = False
                    self.done = True
            elif action == 0 and not self.done:  # Reject action
                self._budget.decision(case=self.current_case, decision=0, value=self.current_value)
                if self._budget.budget_spent_percentage <= 1:
                    reward = self._reward(action)
                else:
                    self.done = True
                    self.truncated = False

        self.rewards.append(reward)
        self.states.append(self.state)

        self._update_steps()

        if self.steps >= self.max_steps:
            self.done = True

        return observation.astype(np.float32), reward, self.done, {}
 

    def _update_steps(self):
        """
        Updates the step count and logs the current status.
        """
        self.steps += 1
        if np.isnan(self.wc) or np.isnan(self.woc):
            logger.debug("Impossible to calculate reward")
        self.wcs.append(self.wc)
        self.wocs.append(self.woc)

    def render(self, mode="human") -> None:
        """
        Renders the current state of the environment.

        Args:
            mode (str, optional): The mode in which to render the environment. Defaults to 'human'.
        """
        if mode != "human":
            raise ValueError(
                "Invalid mode for rendering. Currently, only 'human' mode is supported."
            )

        print(
            f"Steps: {self.steps}, Approved: {self.approved}, Budget: {self._budget.balance}, Max Value: {self.max_value}"
        )

    def _get_initial_state(self):
        """
        Returns the initial state for the environment.
        """
        initial_state = np.zeros(self.observation_space.shape)
        return initial_state

    def _shuffle_data(self) -> None:
        """
        Shuffles the data used in the environment.
        """
        if self._data.empty:
            raise ValueError("Cannot shuffle empty data.")

        self._data = self._data.sample(frac=1, random_state=RANDOM_SEED).reset_index(
            drop=True
        )
        self.sampled_idx = self._data.index.to_list()

    def reward(self, action):
        return self._reward(action)

    def _reward(self, action):
        self.wc, self.woc = self.benford_dist()
        

        # Handle early steps (steps <= 1)
        if self.steps < 1:
            return self.wc if action == 1 else 1 

        # Handle cases where wc or woc is zero
        if self.wc == 0 and self.woc == 0:
            return 0  # Neutral outcome since both are zero
        elif self.wc == 0:  # If wc is zero, handle based on woc
            return 1 if action == 0 else -1
        elif self.woc == 0:  # If woc is zero, handle based on wc
            if action == 0: 
                return 1 
            elif self.wc != 0:
                return self.wc 
            else:
                raise ValueError("Check approved cases")

        # Handle case where wc equals woc
        if self.wc == self.woc:
            return 0  # Neutral reward if wc equals woc

        # Calculate reward based on the action and wc/woc comparison
        if action == 1:  # Approve action
            reward = self.woc / self.wc if self.wc < self.woc else -self.wc / self.woc
        else:  # Reject action (action == 0)
            reward = -self.woc / self.wc if self.wc < self.woc else self.wc / self.woc

        # Return the computed reward directly
        return reward


    def benford_dist(self) -> float:
        """
        Calculates the Benford distance for a given decision.

        Returns:
            float: The calculated Benford distance.
        """
        cases = self._budget.get_approved_cases()[:-1]
        sample = self._data.loc[cases, self._value_column].values
        woc, wc = (0, 0)
        if len(sample)>1:
            woc = Benford(3).distance(sample)
            
        cases = self._budget.get_approved_cases()
        sample = self._data.loc[cases, self._value_column].values
        if len(sample)>0:
            wc = Benford(3).distance(sample)
        return wc, woc

    @property
    def current_case(self):
        return self._budget.current_case

    @property
    def current_value(self):
        return self.case[self._value_column]

    @property
    def case(self):
        if self.steps >= len(self.sampled_idx):
            logger.error("Step index exceeds the available data cases.")
            raise IndexError("Invalid step index: No more cases to process.")
        
        self._budget.current_case = self.sampled_idx[self.steps]
        
        try:
            case_data = self._data.loc[self._budget.current_case]
            if len(case_data.shape) > 1:
                logger.debug("More than one case presented at a time.")
                raise KeyError("Multiple cases found for a single index.")
            logger.info(f"Step: {self.steps}, Case: {self._budget.current_case}")
            return case_data
        except KeyError as e:
            logger.error(f"Invalid case data for step {self.steps}: {e}")
            return None


    @property
    def state(self) -> np.array:
        """
        Computes the state representation based on the current budget and case data.

        Returns:
            np.array: The state representation.
        """
        zero_state = np.zeros(len(self._state_columns) + 4)

        if self.steps == 0:
            return zero_state
        elif self._budget.budget_spent_percentage <= 1:
            approved_cases = self._budget.get_approved_cases()
            approved_ratio = len(approved_cases) / self.steps if self.steps else 0
            current_case_values = self.case[self._state_columns].values.flatten()
            current_case_values = np.array(current_case_values, dtype=np.float32)
            additional_state_info = [
                self._budget.budget_spent_percentage,
                approved_ratio,
                self.wc,
                self.woc,
            ]
            return np.concatenate(
                [current_case_values, additional_state_info], dtype=np.float32
            )
        else:
            return zero_state

    @property
    def results(self) -> pd.DataFrame:
        """
        Retrieves the results of the environment's interactions.

        Returns:
            pd.DataFrame: A DataFrame containing the results data.
        """
        results_data = self._budget.get_decisions_dataframe() 
        results_data["rewards"] = self.rewards
        results_data["cum_rewards"] = results_data["rewards"].cumsum()
        results_data["initial_budget"] = self.initial_budget
        results_data["wc"] = self.wcs
        results_data["woc"] = self.wocs
        results_data["steps"] = list(range(0, self.steps))
        output_data = results_data.join(self._data)
        return output_data

    def save_as_parquet(self, data, episode):
        data.to_parquet(
            f".temp/data/experiment_results/insurenv/{episode}/{self.initial_budget}.parquet"
        )
        return data
