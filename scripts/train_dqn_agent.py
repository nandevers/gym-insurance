import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from gym_insurance.constants import DATAPATH, MODELSPATH, BUDGET_LIST
import pandas as pd
from typing import List
from gym import Env
from gym_insurance.envs.insurenv import InsurEnv  # Assuming InsurEnv is defined in insur_env.py
import gym
import logging
from datetime import datetime
from pathlib import Path

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# TensorFlow optimization settings for better CPU usage
tf.config.threading.set_intra_op_parallelism_threads(8)  # Tune the number of threads
tf.config.threading.set_inter_op_parallelism_threads(4)  # Increase threads for parallel ops
tf.config.set_soft_device_placement(True)  # Ensure TensorFlow dynamically places ops on CPU/GPU

# Hyperparameters
EPISODES = 50
BATCH_SIZE = 64  # Adjusted for smaller batches to better match CPU capacity
MAX_MEMORY_SIZE = 1000  # Increased memory size for experience replay
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.001  # Lower learning rate for more stable convergence on CPU
EPSILON = 1.0  # Exploration rate
EPSILON_DECAY = 0.995  # Exploration decay rate
MIN_EPSILON = 0.01  # Minimum exploration rate
LOG_DIR = ".temp/logs"  # TensorBoard log directory

# Setup logging
logging.basicConfig(
    filename='dqn_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# TensorBoard writer
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_writer = tf.summary.create_file_writer(f"{LOG_DIR}/{current_time}")

index_column = "id_proposta"
value_column = "valor_indenizacao"

# Helper function to load data
def load_data():
    try:
        data_path = Path(DATAPATH) / "processed/psr_train_and_test_set.parquet"
        data = pd.read_parquet(data_path).query("valor_indenizacao > 100")
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# Define DQN Agent
class DQNAgent:
    def __init__(self, env: InsurEnv):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=MAX_MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.min_epsilon = MIN_EPSILON
        self.learning_rate = LEARNING_RATE
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.losses = []  # Track training losses

    def build_model(self):
        model = Sequential([
            layers.Dense(128, input_dim=self.state_size, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(self.action_size, activation="linear"),
        ])
        # Use Adam optimizer without mixed precision for CPU training
        opt = Adam(learning_rate=self.learning_rate)
        model.compile(loss="mse", optimizer=opt)
        return model

    def update_target_model(self):
        """Update the target model with the weights of the main model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size, episode, step):
        """Replay experiences and update the network. Log loss after each step."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        losses = []

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t[0])

            # Perform one step of training
            history = self.model.fit(np.expand_dims(state, axis=0), target, epochs=1, verbose=0)
            loss = history.history['loss'][0]
            losses.append(loss)

        mean_loss = np.mean(losses)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        # Log loss once per step, not multiple times per batch
        self.log_loss_to_tensorboard(episode, step, mean_loss)

        return mean_loss

    def log_loss_to_tensorboard(self, episode, step, loss):
        """Log the loss to TensorBoard with a unique step."""
        global_step = episode * 1000 + step  # Use global step for uniqueness
        with tensorboard_writer.as_default():
            tf.summary.scalar('Loss', loss, step=global_step)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, filepath):
        self.model.save_weights(filepath)

def train_dqn(env: InsurEnv):
    agent = DQNAgent(env)
    episode_rewards = []
    full_loss_data = []

    try:
        for e in range(EPISODES):
            state, _ = env.reset()
            # Track cumulative reward per episode
            cumulative_reward = 0  # Add this inside the main training loop, after state reset
            done = False
            total_reward = 0
            step = 0

            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                with tensorboard_writer.as_default():
                    tf.summary.scalar('Total Reward', total_reward, step=e)
                    tf.summary.scalar('Epsilon', agent.epsilon, step=e)

                # Replay and train the agent, log loss at each step
                loss = agent.replay(BATCH_SIZE, e, step)
                if loss is not None:
                    full_loss_data.append({"episode": e, "step": step, "loss": loss})

                step += 1

            logging.info(f"Episode: {e + 1}/{EPISODES}, Reward: {total_reward}, Epsilon: {agent.epsilon}")
            episode_rewards.append(total_reward)

            # Log episode-level metrics to TensorBoard
                


            # Update the target model after each episode
            agent.update_target_model()

            if e % 10 == 0:  # Save the model more frequently
                filepath = f".temp/data/experiment_results/dqn_weights_{e}.weights.h5"
                agent.save(filepath)

    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

    return episode_rewards, full_loss_data

def get_state_columns(data):
    return data.columns[5:-1]

# Example usage with gym.make()
def run_dqn_with_insurenv():
    try:
        data = load_data()
        budget = 50_000
        state_columns = get_state_columns(data)

        # Create the environment
        env = gym.make(
            "InsurEnv-v0",
            data=data,
            index_column=index_column,
            state_columns=state_columns,
            value_column=value_column,
            budget=budget,
        )
        logging.info("Environment created successfully.")
        return train_dqn(env)

    except Exception as e:
        logging.error(f"Error in setting up the environment or training: {e}")
        raise

if __name__ == "__main__":
    os.makedirs(".temp/data/experiment_results", exist_ok=True)
    episode_rewards, loss = run_dqn_with_insurenv()
    loss_df = pd.DataFrame(loss)  # Convert loss data to a DataFrame
    loss_df.to_parquet(f".temp/data/experiment_results/loss_dqn.parquet", index=False)
    logging.info("Training completed and results saved.")
