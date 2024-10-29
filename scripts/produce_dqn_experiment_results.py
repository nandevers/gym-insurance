import logging
import os
import random
from collections import deque
from datetime import datetime

import gym
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from gym_insurance.constants import DATAPATH
from gym_insurance.envs.insurenv import \
    InsurEnv  # Assuming InsurEnv is defined in insur_env.py

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Parameters
EPISODES = 50
BATCH_SIZE = 10
MAX_MEMORY_SIZE = 2000
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
LOG_DIR = "./logs"  # TensorBoard log directory

# Setup logging
logging.basicConfig(
    filename="dqn_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# TensorBoard writer
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_writer = tf.summary.create_file_writer(f"{LOG_DIR}/{current_time}")

index_column = "id_proposta"
value_column = "valor_indenizacao"


def load_data():
    try:
        data = pd.read_parquet(
            DATAPATH / "processed/psr_train_and_test_set.parquet"
        ).query("valor_indenizacao>0")
        logging.info("Data loaded successfully.")
        return data
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

        # For tracking results
        self.losses = []
        self.action_counts = np.zeros(self.action_size)

    def build_model(self):
        model = Sequential()
        model.add(layers.Dense(64, input_dim=self.state_size, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            action = np.argmax(q_values[0])
        self.action_counts[action] += 1  # Track action distribution
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return (
                None,
                None,
            )  # Return None for both loss and grad_norm when not enough memory

        minibatch = random.sample(self.memory, batch_size)
        losses = []
        gradient_norms = []

        mse_loss = tf.keras.losses.MeanSquaredError()  # Use MeanSquaredError class

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(
                    np.expand_dims(next_state, axis=0), verbose=0
                )
                target[0][action] = reward + self.gamma * np.amax(t[0])

            history = self.model.fit(
                np.expand_dims(state, axis=0), target, epochs=1, verbose=0
            )
            loss = history.history["loss"][0]
            losses.append(loss)

            # Capture gradient norms
            with tf.GradientTape() as tape:
                q_values = self.model(np.expand_dims(state, axis=0))
                loss_value = mse_loss(
                    target, q_values
                )  # Correctly call MeanSquaredError as a function
            gradients = tape.gradient(loss_value, self.model.trainable_variables)
            grad_norm = np.sum(
                [tf.norm(grad).numpy() for grad in gradients if grad is not None]
            )
            gradient_norms.append(grad_norm)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        return np.mean(losses), np.mean(gradient_norms)

    #    def replay(self, batch_size):
    #        if len(self.memory) < batch_size:
    #            return None, None  # Return None for both loss and grad_norm when not enough memory
    #
    #        minibatch = random.sample(self.memory, batch_size)
    #        losses = []
    #        gradient_norms = []
    #
    #        for state, action, reward, next_state, done in minibatch:
    #            target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
    #            if done:
    #                target[0][action] = reward
    #            else:
    #                t = self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)
    #                target[0][action] = reward + self.gamma * np.amax(t[0])
    #
    #            history = self.model.fit(np.expand_dims(state, axis=0), target, epochs=1, verbose=0)
    #            loss = history.history['loss'][0]
    #            losses.append(loss)
    #
    #            # Capture gradient norms
    #            with tf.GradientTape() as tape:
    #                q_values = self.model(np.expand_dims(state, axis=0))
    #                loss_value = tf.keras.losses.mean_squared_error(target, q_values)
    #            gradients = tape.gradient(loss_value, self.model.trainable_variables)
    #            grad_norm = np.sum([tf.norm(grad).numpy() for grad in gradients if grad is not None])
    #            gradient_norms.append(grad_norm)
    #
    #        if self.epsilon > self.min_epsilon:
    #            self.epsilon *= self.epsilon_decay
    #
    #        return np.mean(losses), np.mean(gradient_norms)
    #

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
            done = False
            total_reward = 0
            step = 0

            agent.action_counts = np.zeros(
                agent.action_size
            )  # Reset action counts for this episode

            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # Replay and train the agent, log the loss during each step
                loss, grad_norm = agent.replay(BATCH_SIZE)

                if (
                    loss is not None and grad_norm is not None
                ):  # Only append if replay occurred
                    full_loss_data.append(
                        {
                            "episode": e,
                            "step": step,
                            "loss": loss,
                            "grad_norm": grad_norm,
                        }
                    )

                step += 1

            # Log total reward for the episode after the episode ends
            logging.info(
                f"Episode: {e + 1}/{EPISODES}, Reward: {total_reward}, Epsilon: {agent.epsilon}"
            )
            episode_rewards.append(total_reward)

            # Log multiple metrics to TensorBoard
            with tensorboard_writer.as_default():
                tf.summary.scalar("Total Reward", total_reward, step=e)
                if loss is not None:
                    tf.summary.scalar("Loss", loss, step=e)
                    tf.summary.scalar("Gradient Norm", grad_norm, step=e)
                tf.summary.scalar("Epsilon", agent.epsilon, step=e)
                tf.summary.histogram("Action Distribution", agent.action_counts, step=e)

            # Update target model after every episode
            agent.update_target_model()

            if e % 50 == 0:
                filepath = f".temp/data/experiment_results/dqn_weights_{e}.weights.h5"
                agent.save(filepath)

    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

    return episode_rewards, full_loss_data


def get_state_columns(data):
    return data.columns[5:-1]


# Example Usage with gym.make()
def run_dqn_with_insurenv():
    try:
        data = load_data()
        budget = 10_000_000
        state_columns = get_state_columns(data)
        index_column = "id_proposta"
        value_column = "valor_indenizacao"

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
    loss_df = pd.DataFrame(loss)  # Convert the loss data to a DataFrame
    loss_df.to_parquet(".temp/data/experiment_results/loss_dqn.parquet", index=False)
    logging.info("Training completed and results saved.")
