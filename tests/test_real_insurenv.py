# =============================
import pandas as pd
from dotenv import load_dotenv
from keras.layers import Dense, Flatten, Input
from rl.policy import BoltzmannQPolicy
from keras.models import Model, Sequential
from keras.optimizers import adam_v2
from rl.agents import DQNAgent
from rl.memory import SequentialMemory

from gym_insurance.envs.insurenv import InsurEnv
from handle_data import dump, load

assert load_dotenv()
import numpy as np

data = pd.read_csv("../DeFraud/data/processed/psr_train_set.csv")

# not fill the mean, leave missing
data = data.query("valor_indenização!=1147131.5")
env = InsurEnv(
    data=data,
    value_column="valor_indenização",
    state_columns=data.columns[5:-1].to_list(),
    budget=100000,
)

env.reset()
episodes = 100
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        # env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print("Episode:{} Score:{}".format(episode, score))
