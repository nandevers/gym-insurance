import logging
import os
import random
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.constraints import max_norm
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l1_l2, l2

from gym_insurance.constants import BUDGET_LIST, DATAPATH
from gym_insurance.envs.insurenv2 import InsurEnv  # Assuming InsurEnv is defined in insur_env.py

SOURCE_DIR = ".temp/data/experiment_results/insurenv"
DESTINATION_DIR = ".temp/data/experiment_results/dqn/results"

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# TensorFlow optimization settings for better CPU usage
tf.config.threading.set_intra_op_parallelism_threads(7)
tf.config.threading.set_inter_op_parallelism_threads(10)
tf.config.set_soft_device_placement(True)


# Hyperparameters
EPISODES = 500  # Increased to allow for more training
MIN_MEMORY = 64  # Minimum memory for replay
BATCH_SIZE = 32  # Lowered to match original DQN paper
MAX_MEMORY_SIZE = 40_000  # Increased replay memory size
GAMMA = 0.99  # Discount factor as in the original DQN
LEARNING_RATE = 0.001  # Learning rate from the original DQN
EPSILON = 1.0  # **Initial exploration rate**
EPSILON_DECAY = 0.7  # Gradual decay over time
MIN_EPSILON = 0.1  # As used in the original paper
TARGET_UPDATE_FREQ = 10000  # Fixed target network update frequency
MAX_STEPS = 300  # Max steps per episode
MIN_VALOR_INDENIZACAO = 100
index_column = "id_proposta"
value_column = "valor_indenizacao"

LOG_DIR = ".temp/logs"
# Setup logging
logger = logging.getLogger("DQN_Training")
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
os.makedirs(Path(LOG_DIR) / "dqn", exist_ok=True)
file_handler = logging.FileHandler(Path(LOG_DIR) / "dqn/dqn_training.log")

console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

# Create formatters and add to handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def create_tensorboard_writer(log_dir: str, budget: int) -> tf.summary.SummaryWriter:
    """Create a TensorBoard writer with a unique subdirectory for each budget."""
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = f"{log_dir}/budget_{budget}/{current_time}"
    os.makedirs(run_log_dir, exist_ok=True)
    return tf.summary.create_file_writer(run_log_dir), run_log_dir

def load_data() -> pd.DataFrame:
    """Load and preprocess data for the environment."""
    try:
        data_path = Path(DATAPATH) / "processed/psr_train_and_test_set.parquet"
        data = pd.read_parquet(data_path).query("valor_indenizacao > 100 and sample=='train'")
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
    def __init__(
        self,
        env: InsurEnv,
        tensorboard_writer: tf.summary.SummaryWriter,
        tensorboard_log_dir: str,
        budget: int,
    ) -> None:
        """Initialize the DQNAgent with the environment and specific budget."""
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=MAX_MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON  # **Start with high exploration**
        self.epsilon_decay = EPSILON_DECAY
        self.min_epsilon = MIN_EPSILON
        self.learning_rate_schedule = ExponentialDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=10,
            decay_rate=0.96,
            staircase=True,
        )
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.losses = []  # Track training losses
        self.tensorboard_writer = tensorboard_writer
        self.budget = budget  # Track the budget

        # Setup the Keras TensorBoard callback specific to the budget
        self.tensorboard_callback = TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
        )
        logger.debug(
            "DQNAgent initialized with state_size=%d, action_size=%d for budget %d",
            self.state_size,
            self.action_size,
            self.budget,
        )


    def build_model(self) -> tf.keras.Model:
        """Build a deeper and wider Q-network model with extensive regularization."""
        model = Sequential(
            [
                # First layer with L1-L2 regularization, max norm constraint, and activity regularizer
                tf.keras.layers.Dense(
                    1024,  # Increased the number of units
                    input_dim=self.state_size,
                    kernel_regularizer=l1_l2(l1=0.05, l2=0.1),
                    activity_regularizer=l2(0.01),
                    kernel_constraint=max_norm(3),
                ),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),

                # Second layer with strong L2 regularization and dropout
                tf.keras.layers.Dense(
                    512,  # Increased units
                    kernel_regularizer=l2(0.2),
                    activity_regularizer=l2(0.01),
                    kernel_constraint=max_norm(3),
                ),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),

                # Third layer with higher L2 and activity regularization
                tf.keras.layers.Dense(
                    512,  # Increased units
                    kernel_regularizer=l2(0.2),
                    activity_regularizer=l2(0.01),
                    kernel_constraint=max_norm(3),
                ),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),

                # Fourth layer with L2 regularization and activity regularization
                tf.keras.layers.Dense(
                    256,  # Increased units
                    kernel_regularizer=l2(0.2),
                    activity_regularizer=l2(0.01),
                    kernel_constraint=max_norm(3),
                ),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),

                # Fifth layer with strong regularization
                tf.keras.layers.Dense(
                    256,  # Increased units
                    kernel_regularizer=l2(0.2),
                    activity_regularizer=l2(0.01),
                    kernel_constraint=max_norm(3),
                ),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),

                # Sixth layer with high dropout and regularization
                tf.keras.layers.Dense(
                    128,  # Increased units
                    kernel_regularizer=l2(0.1),
                    activity_regularizer=l2(0.01),
                    kernel_constraint=max_norm(3),
                ),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),

                # Seventh layer with strong regularization
                tf.keras.layers.Dense(
                    128,  # Increased units
                    kernel_regularizer=l2(0.1),
                    activity_regularizer=l2(0.01),
                    kernel_constraint=max_norm(3),
                ),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),

                # Eighth layer with high dropout and regularization
                tf.keras.layers.Dense(
                    128,  # Increased units
                    kernel_regularizer=l2(0.1),
                    activity_regularizer=l2(0.01),
                    kernel_constraint=max_norm(3),
                ),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.Dropout(0.5),

                # Ninth layer with regularization
                tf.keras.layers.Dense(
                    128,  # Increased units
                    kernel_regularizer=l2(0.1),
                    activity_regularizer=l2(0.01),
                    kernel_constraint=max_norm(3),
                ),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),

                # Output layer
                tf.keras.layers.Dense(self.action_size, activation="linear"),
            ]
        )

        # Adam optimizer with reduced learning rate clipping and weight decay for implicit regularization
        opt = Adam(learning_rate=self.learning_rate_schedule, clipvalue=.5)

        model.compile(loss="mse", optimizer=opt)
        logger.debug(
            "Deeper and wider model built and compiled with extensive regularization techniques."
        )
        return model


    def update_target_model(self) -> None:
        """Update the target model with the weights of the main model."""
        self.target_model.set_weights(self.model.get_weights())
        logger.debug("Target model updated.")

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        logger.debug(
            "Memory stored: action=%d, reward=%.2f, done=%s", action, reward, done
        )
        
    def act(self, state: np.ndarray) -> int:
        """Decide an action based on current state using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:  # **Epsilon-greedy logic**
            action = random.randrange(self.action_size)  # **Take random actiona*
            logger.debug("Random action taken: %d", action)
            return action
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        action = np.argmax(q_values[0])  # **Take action predicted by the model**
        logger.info("Action taken from model: %d", action)
        return action

    def replay(
        self, batch_size: int, episode: int, total_steps: int
    ) -> Optional[float]:
        """Replay experiences and update the network, adding budget to TensorBoard logs."""
        if len(self.memory) < MIN_MEMORY:
            logger.debug(
                "Not enough memories to replay. Memory size: %d", len(self.memory)
            )
            return None

        minibatch = random.sample(self.memory, batch_size)
        losses = []
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
            logger.debug("Training on state-action pair. Loss: %.5f", loss)
            
            with self.tensorboard_writer.as_default():
                # Log with budget information
                tf.summary.scalar(f"Loss_budget_{self.budget}", loss, step=total_steps)
                tf.summary.histogram(
                    f"Q-values_budget_{self.budget}", target, step=total_steps
                )
                tf.summary.scalar(
                    f"Epsilon_budget_{self.budget}", self.epsilon, step=total_steps
                )
                self.tensorboard_writer.flush()

        mean_loss = np.mean(losses)
        logger.info(
            "Episode %d, Mean loss: %.5f for budget %d", episode, mean_loss, self.budget
        )

        # **Epsilon decay after each episode or step**
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)
            logger.debug(
                "Epsilon decayed to: %.5f for budget %d", self.epsilon, self.budget
            )

        return mean_loss

# Rest of your code remains unchanged
    def log_loss_to_tensorboard(
        self, episode: int, loss: float, total_steps: int
    ) -> None:
        """Log the loss to TensorBoard with budget-specific logging."""
        with self.tensorboard_writer.as_default():
            if loss:
                tf.summary.scalar(f"Loss_budget_{self.budget}", loss, step=total_steps)
            self.tensorboard_writer.flush()
        logger.debug(
            "Logged loss to TensorBoard at step %d for budget %d.",
            total_steps,
            self.budget,
        )

    def load(self, filepath: str) -> None:
        """Load model weights."""
        self.model.load_weights(filepath)
        logger.info(
            "Model weights loaded from %s for budget %d.", filepath, self.budget
        )

    def save(self, filepath: str) -> None:
        """Save model weights."""
        self.model.save_weights(filepath)
        logger.info("Model weights saved to %s for budget %d.", filepath, self.budget)

def train_dqn(
    env: InsurEnv,
    tensorboard_writer: tf.summary.SummaryWriter,
    tensorboard_log_dir: str,
    budget: int,
    early_stopping_patience: int = 6,  # Number of episodes without improvement to stop
    early_stopping_min_delta: float = 0.1,  # Minimum improvement to reset patience
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Train the DQN agent in the given environment with early stopping."""
    agent = DQNAgent(env, tensorboard_writer, tensorboard_log_dir, budget)
    episode_rewards = []
    full_loss_data = []

    best_loss = float("inf")  # Best loss observed during training
    patience_counter = 0  # Counter for early stopping

    episode_steps = 0  # Track steps within this episode
    try:
        for e in range(EPISODES):
            state, _ = env.reset()
            cumulative_reward = 0.0
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                cumulative_reward += reward
                episode_steps += 1

                # Replay and calculate the loss
                loss = agent.replay(batch_size=BATCH_SIZE, episode=e, total_steps=episode_steps)
                full_loss_data.append(
                    {
                        "episode": e,
                        "step": episode_steps,
                        "loss": loss,
                        "budget": budget,
                    }
                )

                # Log metrics to TensorBoard
                with tensorboard_writer.as_default():
                    tf.summary.scalar(
                        f"Epsilon_budget_{budget}", agent.epsilon, step=episode_steps
                    )
                    tf.summary.scalar(
                        f"Reward_per_Step_budget_{budget}",
                        cumulative_reward,
                        step=episode_steps,
                    )
                    if loss:
                        tf.summary.scalar(
                            f"Loss_by_step_budget_{budget}", loss, step=episode_steps
                        )
                    tensorboard_writer.flush()

            # Update reward tracking
            logger.info(
                "Episode: %d/%d, Reward: %.2f, Epsilon: %.5f for budget %d",
                e + 1,
                EPISODES,
                cumulative_reward,
                agent.epsilon,
                budget,
            )
            episode_rewards.append(cumulative_reward)


            # Save the model every 20 episodes
            if e % 30:
                # Update target model periodically
                agent.update_target_model()
                filepath = f".temp/data/experiment_results/dqn/dqn_weights_budget_{budget}_episode_{e}.weights.h5"
                agent.save(filepath)

            # Log loss for this episode to TensorBoard
            agent.log_loss_to_tensorboard(e, loss, total_steps=episode_steps)

            # Early stopping logic
            if loss and loss < best_loss - early_stopping_min_delta:
                best_loss = loss
                patience_counter = 0  # Reset patience counter
                logger.info("Improved loss to %.5f at episode %d", loss, e)
            else:
                patience_counter += 1
                logger.info(
                    "No improvement in loss for %d episodes. Patience counter: %d",
                    patience_counter,
                    early_stopping_patience,
                )

            # Stop training if no improvement after 'patience' episodes
            if patience_counter >= early_stopping_patience:
                logger.info(
                    "Early stopping triggered. No improvement in loss for %d episodes.",
                    early_stopping_patience,
                )
                break

    except Exception as exc:
        logger.exception("Error during training for budget %d: %s", budget, exc)
        raise

    return episode_rewards, full_loss_data



def run_dqn_with_insurenv(budget: int) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Set up the environment and train the DQN agent for a given budget."""

    data = load_data().query(f"valor_indenizacao<={budget}")
    state_columns = get_state_columns(data)

    env = InsurEnv(
        data=data,
        index_column=index_column,
        state_columns=state_columns,
        value_column=value_column,
        budget=budget,
        max_steps=MAX_STEPS,
    )
    logger.info("Environment created successfully with budget %d.", budget)

    tensorboard_writer, tensorboard_log_dir = create_tensorboard_writer(
            LOG_DIR, budget
        )

    return train_dqn(env, tensorboard_writer, tensorboard_log_dir, budget)



def move_parquet_files(source_dir, destination_dir):
    """
    Move all parquet files from source_dir (including subdirectories) to destination_dir.
    
    :param source_dir: The directory to move files from.
    :param destination_dir: The directory to move files to.
    """
    # Ensure destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        
    # Walk through all files in source directory and subdirectories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".parquet"):
                # Full path of the source file
                source_file_path = os.path.join(root, file)
                
                # Move file to destination directory
                destination_file_path = os.path.join(destination_dir, file)
                
                print(f"Moving file: {source_file_path} to {destination_file_path}")
                
                shutil.move(source_file_path, destination_file_path)




if __name__ == "__main__":
    os.makedirs(".temp/data/experiment_results/dqn", exist_ok=True)
    logger.info("Starting DQN training for multiple budgets.")

    for budget in BUDGET_LIST:
        logger.info(f"Training with budget: {budget}")
        episode_rewards, loss = run_dqn_with_insurenv(budget)

        loss_df = pd.DataFrame(loss)
        loss_df.to_parquet(
            f".temp/data/experiment_results/dqn/loss_dqn_budget_{budget}.parquet",
            index=False,
        )
        logger.info(f"Training completed and results saved for budget: {budget}")
        move_parquet_files(SOURCE_DIR, DESTINATION_DIR)
