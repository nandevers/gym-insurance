import logging
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from gym import Env, spaces
from gym_insurance.envs.benford import Benford
from gym_insurance.envs.utils import Wallet

# Set the seed
np.random.seed(42)


# Create a custom logger
logger = logging.getLogger(__name__)

# Set level of logger
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('app.log')

# Set level of handlers
console_handler.setLevel(logging.WARNING)
file_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
file_handler.setFormatter(file_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


class InsurEnv(Env):
    def __init__(
        self,
        data: pd.DataFrame,
        index_column: str,
        value_column: str,
        state_columns: list,
        budget: int,
    ) -> None:
        super().__init__()
        self.index_column = index_column
        self.value_column = value_column
        self.state_columns = state_columns
        self.data = data.set_index(self.index_column)
        self.budget = Wallet(budget)
        self.episodes = 0

        # Action space should have either approve, reject or do nothing
        self.action_space = spaces.Discrete(2)
        self._shuffle_data()

    def _shuffle_data(self):
        self.data = shuffle(self.data, random_state=1234)
        idx = self.data.index.to_list().copy()
        np.random.shuffle(idx)
        self.sampled_idx = idx

    def _zero_array(self):
        return np.zeros((1, len(self.state_columns) + 4))

    def _reset(self):
        self.steps = 0
        self.approved = 0
        self.states = []
        self.rewards = []
        self.wcs = []
        self.wocs=[]
        self.done = False
        self.max = self.data[self.value_column].max()
        self._shuffle_data()

        # write functions to make the following less ugly
        self.current_case = self.sampled_idx[self.steps]
        self.budget = Wallet(self.budget.budget)
        self.observation_space = np.zeros(len(self.state_columns) + 4)
        self.initial_state = self.data.loc[
            self.sampled_idx[0], self.state_columns
        ].values
        return self.observation_space

    def reset(self):
        return self._reset()

    @property
    def state(self):
        cols = self.state_columns
        _values = list(*self.case[cols].values)
        if self.budget.spent == 0:
            return np.array(_values + [0.0, 0.0, 0.0, 0.0])
        elif self.budget.pct_budget <= 1:
            idx = [k for k, value in self.budget.decisions.items() if value == 1]
            self.approved = len(idx)
            return np.array(
                _values
                + [
                    self.budget.pct_budget,
                    self.approved / self.steps,
                    self.wc,
                    self.woc,
                ]
            )
        else:
            return self._zero_array()

    @property
    def case(self):
        self.current_case = self.sampled_idx[self.steps]
        _case = self.data.filter(items=[self.current_case], axis=0)
        if _case.shape[0] > 1:
            logger.debug("More than one case presented at a time.")
            raise KeyError 
        else:
            logger.info(f"Step: {self.step} -- Case {_case}")
            return _case

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
            done = True
            return self._zero_array(), reward, done, {}

        self.budget.current_budget[self.current_case] = self.budget.balance
        vl = self.case["valor_indenizacao"].values[0]
        observation = self.state

        if vl is not None:
            if action == 1 and self.done is False:
                try:
                    self.budget.withdraw(vl)
                    self.budget.decisions[self.current_case] = 1
                    reward = self._reward(action, wc=self.wc, woc=self.woc)
#                    self.wcs.append(self.wc)
#                    self.wocs.append(self.woc)
                
                except ValueError:
                    if self.budget.pct_budget <= self.data[self.value_column].min():
                        logger.error("Not enough budget")

                    if len(self.states) > 1:
                        observation = self.states[-1]
                    
                    self.budget.decisions[self.current_case] = 0
                    self.done = True
                    self.episodes +=1

            elif action == 0 and self.done is False:
                self.budget.decisions[self.current_case] = 0
                if self.budget.pct_budget < 1:
                    reward = self._reward(action, wc=self.wc, woc=self.woc)
 #                   self.wcs.append(self.wc)
 #                   self.wocs.append(self.woc)
                else:
                    self.done = True
                    self.episodes +=1

        self.rewards.append(reward)
        self.approved_pct = sum(self.budget.decisions.values()) / self.steps
        self.move_state()

        return observation, reward, self.done, {}

    def move_state(self):
        self.states.append(self.state)

    def _sample(self, index=0, decision=1):
        idx = [k for k, value in self.budget.decisions.items() if value == 1]
        if decision != 1:
            idx.append(self.current_case)
            return self.data.loc[idx, :]
        elif index != 0:
            return self.data.loc[idx[0:-index], :]
        else:
            return self.data.loc[idx, :]

    def benford_dist(self, decision, previous=False):
        vl = self._sample(decision=decision)[self.value_column].values
        if previous:
            vl = self._sample(index=1)[self.value_column].values
        return Benford(3).distance(vl)

    def _reward(self, action, wc, woc):
        if self.steps == 0:
            return 0
        rev = 1  # (self.case["valor_indenizacao"].values[0]/self.max)
        if action == 1 and wc < woc:
            return woc / wc * rev
        elif action == 1 and wc >= woc:
            return -wc / woc * rev
        elif action == 0 and wc < woc:
            return -woc / wc * rev
        elif action == 0 and wc >= woc:
            return wc / woc * rev

    @property
    def results(self):
        results_data = pd.DataFrame(self.budget.decisions, index=[0]).T
        results_data.columns = ["decision"]
        
        

        results_data["rewards"] = self.rewards
        results_data["budget"] = self.budget.current_budget.values()
        results_data["initial_budget"] = self.budget.budget
        
        try:
            results_data["wc"] = self.wcs
            results_data["woc"] = self.wocs
        except ValueError:
            import pdb; pdb.set_trace()

        results_data['steps'] =range(1, self.steps+1)
        results_data['episodes'] = self.episodes
        return results_data.join(self.data)
