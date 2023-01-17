import time

import numpy as np
import pandas as pd
import tensorflow as tf
from bf2 import benfordslaw as bfl
from gym import Env, spaces
from keras.callbacks import TensorBoard


MODEL_NAME = "insurance_dqn"

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


class Wallet:
    def __init__(self, budget) -> None:
        self.budget = budget
        self.pct_budget = 0
        self.decisions = {}
        self.spent = 0

    def withdraw(self, value):
        assert value > 0
        self.spent += value
        if self.pct_budget <= 1:
            self.pct_budget = self.spent / self.budget
        else:
            raise ValueError

    @property
    def balance(self):
        return self.budget - self.spent


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
                    self.done = False
                    reward = self._reward(action)

                except ValueError:
                    observation = self.states[-1]
                    self.done = True
                    self.episodes += 1
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
        if wc <= woc:
            if action == 0:
                return wc
            else:
                return -wc
        else:
            if action == 0:
                return -wc
            else:
                return wc
