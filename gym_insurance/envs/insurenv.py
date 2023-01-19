import time
import numpy as np
import pandas as pd
from gym import Env, spaces
from gym_insurance.envs.benford import Benford
from gym_insurance.envs.utils import Wallet, ModifiedTensorBoard

MODEL_NAME = "br_crop_insurance"


class InsurEnv(Env):
    def __init__(
        self,
        data: pd.DataFrame,
        value_column: str,
        state_columns: list,
        budget: int,
    ) -> None:
        super().__init__()
        self.data = data
        self.value_column = value_column
        self.state_columns = state_columns
        self.budget = Wallet(budget)
        self.episodes = 0

        # Action space should have either approve, reject or do nothing
        self.action_space = spaces.Discrete(3)

        # Shuffle data
        self.sampled_idx = list(self.data.sample(frac=1).index)

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs/{MODEL_NAME}-{int(time.time())}"
        )
        self.ACTION_SPACE_SIZE = 3

    def _zero_array(self):
        return np.zeros(len(self.state_columns) + 2)

    def reset(self):
        return self._reset()

    def _reset(self):
        self.steps = 0
        self.current_case = self.sampled_idx[self.steps]
        self.states = []
        self.rewards = []
        self.done = False
        self.budget = Wallet(self.budget.budget)
        # All attributes which define the space as well as the metrics pct_budget and percenteage approved
        self.observation_space = np.zeros(len(self.state_columns) + 2)
        # Pick initial state
        self.initial_state = self.data.loc[
            self.sampled_idx[0], self.state_columns
        ].values
        return self.observation_space

    @property
    def state(self):
        cols = self.state_columns
        _values = list(*self.case[cols].values)
        if self.budget.spent == 0:
            return np.array(_values + [0.0, 0.0])
        elif self.budget.pct_budget <= 1:
            self.approved = sum(self.budget.decisions.values())
            return np.array(
                _values + [self.budget.pct_budget, self.approved / self.steps]
            )
        else:
            return self._zero_array()

    @property
    def case(self):
        self.current_case = self.sampled_idx[self.steps]
        _case = self.data.filter(items=[self.current_case], axis=0)
        if _case.shape[0] > 1:
            raise ValueError
        else:
            return _case

    def step(self, action):
        self.steps += 1
        vl = self.case["valor_indenização"].values[0]
        observation = self.state
        if vl is not None:
            if action == 1 and self.done is False:
                try:
                    self.budget.decisions[self.current_case] = 1
                    self.budget.withdraw(vl)
                    reward = self._reward(action)
                    self.done = False

                except ValueError:
                    observation = self.states[-1]
                    self.done = True
                    self.episodes += 1

                    self.tensorboard.update_stats(
                        reward=sum(self.rewards),
                        min_reward=min(self.rewards),
                        approved_pct=self.approved_pct,
                        budget_pct=self.budget.pct_budget,
                        step=self.episodes,
                    )
                    reward = 0
            else:
                self.budget.decisions[self.current_case] = 0
                if self.budget.pct_budget <= 1:
                    reward = self._reward(action)
                    self.done = False
                else:
                    self.done = True
                    self.episodes += 1
                    reward = 0
        else:
            if action == 3:
                reward = 100
            else:
                reward = -100

        self.move_state()
        self.rewards.append(reward)
        self.approved_pct = sum(self.budget.decisions.values()) / self.steps
        info = {}

        return observation, reward, self.done, info

    def move_state(self):
        self.states.append(self.state)

    def _sample(self, index=0):
        idx = list(self.budget.decisions.keys())
        if index != 0 and len(idx) > 1:
            idx = list(self.budget.decisions.keys())[0:-index]
            return self.data.loc[idx, :]
        return self.data.loc[idx, :]

    def benford_dist(self, previous=False):
        vl = self._sample()[self.value_column].values
        if previous:
            vl = self._sample(index=1)[self.value_column].values
        return Benford().distance(vl)

    def _reward(self, action):
        if self.steps == 0:
            return 0
        wc = self.benford_dist()
        woc = self.benford_dist(previous=True)
        return wc if wc <= woc else -wc if action == 0 else wc


    @property
    def results(self):
        return self.budget.decisions
