import gym as gym
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from gym_insurance.constants import BUDGET_LIST, DATAPATH, logger


def get_state_columns(data):
    return data.columns[5:-1]


def load_data():
    return (
        pd.read_parquet(DATAPATH / "processed/psr_train_and_test_set.parquet")
        .query("valor_indenizacao==valor_indenizacao")
        .sample(1000)
    )


index_column = "id_proposta"
value_column = "valor_indenizacao"

for i, b in enumerate(BUDGET_LIST):
    logger.info(f"Evaluating DQN Agent for {b}")
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

    model = DQN.load(f"models/dqn/dqn_insurenv_{b}.zip", env)

    # Define the number of episodes for evaluation
    num_episodes = 1

    # Evaluate the policy and get detailed results
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=num_episodes,
        return_episode_rewards=True,
        deterministic=True,
    )

    # Create a DataFrame from the results
    evaluation_results = pd.DataFrame(
        {
            "Episode": range(num_episodes),
            "Reward": episode_rewards,
            "Length": episode_lengths,
        }
    )

    # Save the results to a Parquet file
    evaluation_results.to_parquet(DATAPATH / f"dqn_detailed_results_{b}.parquet")
