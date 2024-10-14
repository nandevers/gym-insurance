import logging
import os
import random
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from typing import Any, Dict, List, Optional, Tuple

from gym_insurance.constants import DATAPATH, BUDGET_LIST
from gym_insurance.envs.insurenv import InsurEnv # Assuming InsurEnv is defined in insur_env.py

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# TensorFlow optimization settings for better CPU usage
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.set_soft_device_placement(True)

# Hyperparameters
EPISODES = 10
BATCH_SIZE = 200
MAX_MEMORY_SIZE = 30_000
GAMMA = 0.99
LEARNING_RATE = 0.01
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
LOG_DIR = ".temp/logs"

# Setup logging
logger = logging.getLogger('DQN_Training')
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('dqn_training.log')

console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

# Create formatters and add to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def create_tensorboard_writer(log_dir: str) -> tf.summary.SummaryWriter:
    """Create a TensorBoard writer with a unique subdirectory."""
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = f"{log_dir}/{current_time}"
    os.makedirs(run_log_dir, exist_ok=True)
    return tf.summary.create_file_writer(run_log_dir), run_log_dir


index_column = "id_proposta"
value_column = "valor_indenizacao"


def load_data() -> pd.DataFrame:
    """Load and preprocess data for the environment."""
    try:
        data_path = Path(DATAPATH) / "processed/psr_train_and_test_set.parquet"
        data = pd.read_parquet(data_path).query("valor_indenizacao > 10_000")
        logger.info("Data loaded successfully from %s.", data_path)
        logger.debug("Data shape: %s", data.shape)
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise
    except Exception as e:
        logger.exception("Error loading data: %s", e)
        raise


def get_state_columns(data: pd.DataFrame) -> List[str]:
    """Extract the list of state columns from the data."""
    state_columns = data.columns[5:-1].tolist()
    logger.debug("State columns extracted: %s", state_columns)
    return state_columns


class DQNAgent:
    def __init__(self, env: InsurEnv, tensorboard_writer: tf.summary.SummaryWriter, tensorboard_log_dir: str) -> None:
        """Initialize the DQNAgent with the environment."""
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=MAX_MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.min_epsilon = MIN_EPSILON
        self.learning_rate_schedule = ExponentialDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=10,
            decay_rate=0.96,
            staircase=True
        )
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.losses = []  # Track training losses
        self.tensorboard_writer = tensorboard_writer

        # Setup the Keras TensorBoard callback
        self.tensorboard_callback = TensorBoard(
            log_dir=tensorboard_log_dir, histogram_freq=1, write_graph=True, write_images=False
        )
        logger.debug("DQNAgent initialized with state_size=%d, action_size=%d", self.state_size, self.action_size)


    def build_model(self) -> tf.keras.Model:
        """Build the Q-network model."""
        model = Sequential([
            tf.keras.layers.Dense(512, input_dim=self.state_size, kernel_regularizer=l2(0.01)),
            tf.keras.layers.LeakyReLU(alpha=0.01),

            tf.keras.layers.Dense(256, kernel_regularizer=l2(0.01)),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.BatchNormalization(),  # Reduced BatchNorm usage

            tf.keras.layers.Dense(256, kernel_regularizer=l2(0.01)),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            
            tf.keras.layers.Dense(128, kernel_regularizer=l2(0.01)),
            tf.keras.layers.LeakyReLU(alpha=0.01),

            tf.keras.layers.Dense(128, kernel_regularizer=l2(0.01)),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            
            tf.keras.layers.Dense(64, kernel_regularizer=l2(0.01)),
            tf.keras.layers.LeakyReLU(alpha=0.01),

            tf.keras.layers.Dense(self.action_size, activation="linear")
        ])

        opt = Adam(learning_rate=self.learning_rate_schedule, clipvalue=1.0)
        model.compile(loss="mse", optimizer=opt)
        logger.debug("Model built and compiled.")
        return model



    def update_target_model(self) -> None:
        """Update the target model with the weights of the main model."""
        self.target_model.set_weights(self.model.get_weights())
        logger.debug("Target model updated.")

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        logger.debug("Memory stored: action=%d, reward=%.2f, done=%s", action, reward, done)

    def act(self, state: np.ndarray) -> int:
        """Decide an action based on current state."""
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            logger.debug("Random action taken: %d", action)
            return action
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        action = np.argmax(q_values[0])
        logger.info("Action taken from model: %d", action)
        return action

    def replay(self, batch_size: int, episode: int) -> Optional[float]:
        """Replay experiences and update the network."""
        if len(self.memory) < batch_size:
            logger.debug("Not enough memories to replay. Memory size: %d", len(self.memory))
            return None

        minibatch = random.sample(self.memory, batch_size)
        losses = []

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t[0])

            history = self.model.fit(np.expand_dims(state, axis=0), target, epochs=1, verbose=0)
            loss = history.history['loss'][0]
            losses.append(loss)
            logger.debug("Training on state-action pair. Loss: %.5f", loss)
            global_step = episode * 1000 + self.env.steps

            with self.tensorboard_writer.as_default():
                tf.summary.scalar('Loss', loss, step=global_step)
                tf.summary.histogram('Q-values', target, step=global_step)
                tf.summary.scalar('Epsilon', self.epsilon, step=global_step)
                self.tensorboard_writer.flush()

        mean_loss = np.mean(losses)
        logger.info("Episode %d, Mean loss: %.5f", episode, mean_loss)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)
            logger.debug("Epsilon decayed to: %.5f", self.epsilon)

        return mean_loss

    def log_loss_to_tensorboard(self, episode: int, loss: float) -> None:
        """Log the loss to TensorBoard with a unique step."""
        global_step = episode * 1000 + self.env.steps
        with self.tensorboard_writer.as_default():
            if loss:
                tf.summary.scalar('Loss', loss, step=global_step)
            self.tensorboard_writer.flush()
        logger.debug("Logged loss to TensorBoard at step %d.", global_step)

    def load(self, filepath: str) -> None:
        """Load model weights."""
        self.model.load_weights(filepath)
        logger.info("Model weights loaded from %s.", filepath)

    def save(self, filepath: str) -> None:
        """Save model weights."""
        self.model.save_weights(filepath)
        logger.info("Model weights saved to %s.", filepath)


def train_dqn(env: InsurEnv, tensorboard_writer: tf.summary.SummaryWriter, tensorboard_log_dir: str) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Train the DQN agent in the given environment."""
    agent = DQNAgent(env, tensorboard_writer, tensorboard_log_dir)
    episode_rewards = []
    full_loss_data = []
    total_steps = 0  # Keep track of total steps across all episodes

    try:
        for e in range(EPISODES):
            state, _ = env.reset()
            cumulative_reward = 0.0
            done = False
            episode_steps = 0  # Track steps within this episode

            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                cumulative_reward += reward
                episode_steps += 1
                total_steps += 1  # Increment the total step count

                loss = agent.replay(BATCH_SIZE, e)
                full_loss_data.append({"episode": e, "step": episode_steps, "loss": loss})

                # Log to TensorBoard with the correct global step (cumulative steps)
                with tensorboard_writer.as_default():
                    tf.summary.scalar('Epsilon', agent.epsilon, step=total_steps)
                    tf.summary.scalar('Reward per Step', cumulative_reward, step=total_steps)
                    if loss:
                        tf.summary.scalar("Loss_by_step", loss, step=total_steps)
                    tensorboard_writer.flush()

            logger.info("Episode: %d/%d, Reward: %.2f, Epsilon: %.5f", e + 1, EPISODES, cumulative_reward, agent.epsilon)
            episode_rewards.append(cumulative_reward)

            agent.update_target_model()

            # Optionally save model weights every 10 episodes
            filepath = f".temp/data/experiment_results/dqn_weights_{e}.weights.h5"
            agent.save(filepath)
            agent.log_loss_to_tensorboard(e, loss)

    except Exception as e:
        logger.exception("Error during training: %s", e)
        raise

    return episode_rewards, full_loss_data


def run_dqn_with_insurenv(budget: int) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Set up the environment and train the DQN agent for a given budget."""
    try:
        data = load_data()
        state_columns = get_state_columns(data)

        env = InsurEnv(
            data=data,
            index_column=index_column,
            state_columns=state_columns,
            value_column=value_column,
            budget=budget,
        )
        logger.info("Environment created successfully with budget %d.", budget)

        tensorboard_writer, tensorboard_log_dir = create_tensorboard_writer(LOG_DIR)

        return train_dqn(env, tensorboard_writer, tensorboard_log_dir)

    except Exception as e:
        logger.exception("Error in setting up the environment or training: %s", e)
        raise


if __name__ == "__main__":
    os.makedirs(".temp/data/experiment_results", exist_ok=True)
    logger.info("Starting DQN training for multiple budgets.")

    for budget in BUDGET_LIST:
        logger.info(f"Training with budget: {budget}")
        episode_rewards, loss = run_dqn_with_insurenv(budget)

        loss_df = pd.DataFrame(loss)
        loss_df.to_parquet(f".temp/data/experiment_results/loss_dqn_budget_{budget}.parquet", index=False)
        logger.info(f"Training completed and results saved for budget: {budget}")
