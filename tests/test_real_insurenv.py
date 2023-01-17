# =============================
import pandas as pd
from dotenv import load_dotenv
from keras.layers import Dense, Flatten, Input
from rl.policy import BoltzmannQPolicy
from keras.models import Model, Sequential
from keras.optimizers import adam_v2
from rl.agents import DQNAgent
from rl.memory import SequentialMemory

from gym_insurance import InsurEnv
from handle_data import dump, load

assert load_dotenv()
import numpy as np

data = load("Train Set", RAW=False)
cols = [
    "nr_previous_proposals",
    f"max_{'nr_produtividade_estimada'}",
    f"min_{'nr_produtividade_estimada'}",
    f"min_{'vl_limite_garantia'}",
    f"max_{'vl_limite_garantia'}",
    "first_digit",
    "second_digit",
]

env = InsurEnv(
    data=data,
    id_col=["nr_documento_segurado", "nr_proposta"],
    value_column="valor_indenização",
    state_columns=cols,
    budget=100000,
    time_col=None,
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
