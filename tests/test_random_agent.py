import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from gym_insurance.envs.insurenv import InsurEnv

size = 2000

# Create a sample dataset
data = pd.DataFrame(
    {
        "index_column": range(size),
        "value_column": np.random.rand(size) * 1000,
        "feature_1": np.random.rand(size),
        "feature_2": np.random.rand(size),
    }
)

# Initialize the InsurEnv environment


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, observation):
        return self.action_space.sample()


def run_random_agent(num_episodes, budget):
    env = InsurEnv(
        data=data,
        index_column="index_column",
        value_column="value_column",
        state_columns=["feature_1", "feature_2"],
        budget=budget,
    )
    episode_results = []
    agent = RandomAgent(env.action_space)

    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        acts = []
        while not done:
            action = agent.select_action(observation)
            observation, reward, done, info = env.step(action)
            # Optionally, render the environment
            # env.render()
        print(f"Episode {episode + 1} finished")
        df = env.results.copy(deep=True)
        df["episode"] = episode + 1
        episode_results.append(df)
    env.close()
    return episode_results


def run_bootstrap_random_agent(budget, k=10):
    bootstrap_results = []
    num_episodes = 100

    for i in range(k):
        results = run_random_agent(num_episodes=num_episodes, budget=budget)
        results = pd.concat(results, axis=0)
        results["boostrap_batch"] = i + 1
        bootstrap_results.append(results)

    return pd.concat(bootstrap_results, axis=0)


def run_budget_boostrap_random_agent():
    # budget_list = [1_000_000, 5_000_000, 10_000_000, 15_000_000]
    budget_list = [1_00000000]
    budget_results = []
    for b in budget_list:
        results = run_bootstrap_random_agent(budget=b, k=10)
        results = pd.concat(results, axis=0)
        budget_results.append(results)
    return pd.concat(budget_results, axis=0)


# Run the random agent
results = run_budget_boostrap_random_agent()
results
