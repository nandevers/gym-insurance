import torch.nn as nn


from constants import DATAPATH, MODELSPATH, logger
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import pandas as pd
import gymnasium as gym
import gym_insurance
import time
from stable_baselines3 import DQN
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import MlpPolicy
from constants import DATAPATH, BUDGET_LIST

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

# Define early stopping callback
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1.5, verbose=1)
# Adjust exploration settings
exploration_initial_eps = 0.7  # Initial value of epsilon
exploration_final_eps = 0.01  # Final value of epsilon
exploration_fraction = 0.2  # Fraction of the total training steps over which the exploration rate is reduced



def get_state_columns(data):
    return data.columns[5:-1]


def load_data():
    return pd.read_csv(DATAPATH / "psr_train_and_test_set.csv").query(
        "valor_indenizacao==valor_indenizacao"
    )

class EpisodeDataCollectorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeDataCollectorCallback, self).__init__(verbose)
        self.episode_validation = []
        self.episode_count = 1
        self.save_path = DATAPATH
        self.terminate_collector = []
        self.steps = 0

    def _on_step(self) -> bool:
        self.steps +=1
        df = self.training_env.get_attr("results")[0].copy(deep=True)
        if df.empty: 
            return True

        df["episodes"] = self.episode_count
        self.episode_validation.append(df)
        self.current_df = self.episode_validation[-1]
        # Check if episode has ended
        if any(self.locals["dones"]):
            budget = self.episode_validation[-1]["initial_budget"]
            self.episode_validation[-1].to_parquet(DATAPATH / f'dqn_training_results_{budget}.parquet')
            self.terminate_collector.append(self.episode_validation[-1])
            self.episode_count += 1
        return True

    def _on_training_end(self):
        if len(self.episode_validation) > 0:
            if self.episode_validation[-1]["episodes"].values[0] == 1:
                self.episode_validation[-1].to_parquet(self.save_path / f"dqn_results_{self.episode_count}.parquet")
            try:
                # Concatenate all dataframes
                all_data = pd.concat(self.terminate_collector, axis=0)
                # Save to Parquet
                all_data.to_parquet(
                    self.save_path / f"all_dqn_results_{self.episode_count}.parquet"
                )
            except ValueError:
                logger.error("No data produced. This is very wrong.")






class CollectEpisodeStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CollectEpisodeStatsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.mean_rewards = []
        self.exploration_rates = []
        self.losses = []
        self.learning_rates = []
        self.cumulative_reward = 0
        self.max_rewards = []
        self.min_rewards = []
        self.episode_success_rates = []

    def _on_step(self):
        info = self.locals.get("infos", [{}])[-1]
        done = self.locals.get("dones", [False])[-1]

        if done:
            episode_reward = sum(self.locals["rewards"])
            episode_length = len(self.locals["rewards"])
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.mean_rewards.append(episode_reward / episode_length)
            self.cumulative_reward += episode_reward
            self.max_rewards.append(max(self.locals["rewards"]))
            self.min_rewards.append(min(self.locals["rewards"]))

            # Example: Assuming 'success' is a boolean indicating episode success in info dict
            self.episode_success_rates.append(float(info.get("success", False)))

            # Add exploration rate and losses if available
            self.exploration_rates.append(self.model.exploration_rate)
            self.losses.append(self.locals.get("loss", 0))

            # Assuming model has attribute 'learning_rate'
            self.learning_rates.append(self.model.learning_rate)

        return True



# Instantiate the callback
data_collection_callback = EpisodeDataCollectorCallback()
episode_stats_callback = CollectEpisodeStatsCallback()


## Main class for customized neural network
class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomNetwork, self).__init__(observation_space, features_dim)

        # Define the layers of the network
        self.network = nn.Sequential(
            nn.Linear(observation_space.shape[0], 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.5),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(p=0.5),
            nn.Linear(16, 4),
            nn.BatchNorm1d(4),
            nn.Dropout(p=0.5),
            nn.Linear(4, 2),
            nn.BatchNorm1d(2),
            nn.Dropout(p=0.5),
            nn.Linear(2, features_dim),
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
    

    for i, b in enumerate(BUDGET_LIST):
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
            eval_freq=100,
            best_model_save_path=MODELSPATH,
            verbose=1,
        )
        # Create the DQN agent with your custom policy
        model = DQN(
            CustomDQNPolicy,
            env,
            verbose=1,
            buffer_size=300, 
            target_update_interval=10, 
            batch_size=50, 
            learning_rate=0.005, 
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            exploration_fraction=exploration_fraction,
        )

   

        # Train the model
        model.learn(
            total_timesteps=1000,
            progress_bar=True,
            callback=[eval_callback, episode_stats_callback, data_collection_callback],
        )
        model.save(MODELSPATH / f"dqn/dqn_insurenv_{b}")



        start_time = time.time()
        vec_env = model.get_env()
        obs = vec_env.reset()

        num_episodes = 10
        episode_count = 0

        while episode_count < num_episodes:            
            episode_validation = []
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
            if dones[-1]:
                df = [env.results for env in vec_env.envs]
                df = pd.concat(df, axis = 0)
                df["episodes"] = episode_count
                obs = vec_env.reset()
                episode_validation.append(df)
                episode_count += 1

        # Stop the timer
        end_time = time.time()

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        print(f"Elapsed time for 10 episodes: {elapsed_time} seconds")

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed Time: {elapsed_time:.2f} seconds")

        pd.concat(episode_validation, axis=0).to_parquet(
            DATAPATH / f"dqn_results_{b}.parquet"
        )
