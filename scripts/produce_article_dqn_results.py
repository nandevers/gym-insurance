import torch.nn as nn
from constants import DATAPATH, MODELSPATH, logger
import pandas as pd
import gymnasium as gym
import gym_insurance
from stable_baselines3 import DQN
from stable_baselines3 import DQN

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import MlpPolicy
from constants import DATAPATH, BUDGET_LIST

from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

# Define early stopping callback
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=-100, verbose=1)
# Adjust exploration settings
exploration_initial_eps = .7  # Initial value of epsilon
exploration_final_eps = 0.01  # Final value of epsilon
exploration_fraction = 0.1  # Fraction of the total training steps over which the exploration rate is reduced


def get_state_columns(data):
    return data.columns[5:-1]


def load_data():
    return pd.read_csv(DATAPATH / "psr_train_and_test_set.csv").query(
        "valor_indenizacao==valor_indenizacao"
    )


## Main class for customized neural network
class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomNetwork, self).__init__(observation_space, features_dim)

        self.network = nn.Sequential(
            nn.Linear(observation_space.shape[0], 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(16, 4),
            nn.BatchNorm1d(4),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(4, 2),
            nn.BatchNorm1d(2),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(2, features_dim)
        )

    def forward(self, observations):
        return self.network(observations)


# Create a custom policy
class CustomDQNPolicy(MlpPolicy):
    def _build_mlp_extractor(self):
        return CustomNetwork(self.observation_space, features_dim=self.features_dim)


if __name__ == "__main__":
    MODELSPATH.mkdir(parents=True, exist_ok=True)
    index_column = "id_proposta"
    value_column = "valor_indenizacao"

    for i, b in enumerate([10_000_000]):
        logger.info(f"Training DQN Agent for {b}")
        data = load_data()
        state_columns = get_state_columns(data)
        env = gym.make(
            "InsurEnv-v0",
            data=data,
            index_column=index_column,
            state_columns=state_columns,
            value_column=value_column,
            budget=b,
        )

        eval_callback = EvalCallback(
            env,
            callback_on_new_best=stop_callback,
            eval_freq=1000,
            best_model_save_path=MODELSPATH,
            verbose=1,
        )
        # Create the DQN agent with your custom policy
        model = DQN(
            CustomDQNPolicy,
            env,
            verbose=1,
            buffer_size=5000,
            target_update_interval=100,
            batch_size=1000,
            learning_rate=0.5,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            exploration_fraction=exploration_fraction,
        )

        # Train the model
        model.learn(total_timesteps=100_000, progress_bar=True)
        model.save(MODELSPATH / f"dqn/dqn_insurenv_{b}")
