import gymnasium as gym
from stable_baselines3 import A2C
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_insurance  # This will register the environment


# Create a sample dataset
data = pd.DataFrame(
    {
        "index_column": range(100),
        "value_column": np.random.rand(100) * 1000,
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
    }
)

# Initialize the InsurEnv environment
env = gym.make(
    "InsurEnv-v0",
    data=data,
    index_column="index_column",
    value_column="value_column",
    state_columns=["feature_1", "feature_2"],
    budget=100000,
)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
