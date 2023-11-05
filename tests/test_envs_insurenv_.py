import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

import numpy as np
import pandas as pd

import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio


import matplotlib.pyplot as plt
from scipy.stats import chisquare
from gym_insurance.envs.insurenv import InsurEnv



import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    # Code that produces the runtime warnings
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon, kruskal

def compare_non_iid(g1, g2):
    # Converting lists to numpy arrays
    g1 = np.array(g1)
    g2 = np.array(g2)
    
    # Checking if the samples have the same length for paired tests
    paired = len(g1) == len(g2)
    
    # Visual Inspection: Time Series Plot
    plt.figure(figsize=(10, 5))
    plt.plot(g1, label='Sample 1', marker='o')
    plt.plot(g2, label='Sample 2', marker='o')
    plt.title('Time Series Plot')
    plt.legend()
    plt.show()
    
    # Visual Inspection: Boxplot
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=[g1, g2])
    plt.xticks(ticks=[0, 1], labels=['Sample 1', 'Sample 2'])
    plt.title('Boxplot')
    plt.show()
    
    # Statistical Tests
    print("Statistical Tests:")
    if paired:
        # Wilcoxon Signed-Rank Test (for paired samples)
        w_stat, w_p_value = wilcoxon(g1, g2)
        print("Wilcoxon Signed-Rank Test: Statistic = {:.4f}, P-value = {:.4f}".format(w_stat, w_p_value))
    else:
        # Mann-Whitney U Test (for independent samples)
        mw_stat, mw_p_value = mannwhitneyu(g1, g2)
        print("Mann-Whitney U Test: Statistic = {:.4f}, P-value = {:.4f}".format(mw_stat, mw_p_value))
        
        # Kruskal-Wallis H Test (for independent samples)
        kw_stat, kw_p_value = kruskal(g1, g2)
        print("Kruskal-Wallis H Test: Statistic = {:.4f}, P-value = {:.4f}".format(kw_stat, kw_p_value))


def compare_distributions(g1, g2):
    # Calculating Descriptive Statistics
    print("Descriptive Statistics:")
    print("Sample 1: Mean = {:.2f}, Median = {:.2f}, Std Dev = {:.2f}".format(np.mean(g1), np.median(g1), np.std(g1)))
    print("Sample 2: Mean = {:.2f}, Median = {:.2f}, Std Dev = {:.2f}".format(np.mean(g2), np.median(g2), np.std(g2)))
    print()
    
    # Visual Inspection: Histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(g1, kde=False, label='Sample 1', color='blue', bins=30)
    sns.histplot(g2, kde=False, label='Sample 2', color='red', bins=30, alpha=0.6)
    plt.title('Histogram')
    plt.legend()
    plt.show()
    
    # Visual Inspection: Boxplot
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=[g1, g2])
    plt.xticks(ticks=[0, 1], labels=['Sample 1', 'Sample 2'])
    plt.title('Boxplot')
    plt.show()
    
    # Statistical Tests
    print("Statistical Tests:")
    # Kolmogorov-Smirnov Test
    ks_stat, ks_p_value = stats.ks_2samp(g1, g2)
    print("Kolmogorov-Smirnov Test: Statistic = {:.4f}, P-value = {:.4f}".format(ks_stat, ks_p_value))
    
    # Mann-Whitney U Test
    mw_stat, mw_p_value = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    print("Mann-Whitney U Test: Statistic = {:.4f}, P-value = {:.4f}".format(mw_stat, mw_p_value))
    
    # T-Test (assuming equal variance)
    t_stat, t_p_value = stats.ttest_ind(g1, g2, equal_var=True)
    print("T-Test: Statistic = {:.4f}, P-value = {:.4f}".format(t_stat, t_p_value))
    
    # Levene's Test for equality of variances
    lev_stat, lev_p_value = stats.levene(g1, g2)
    print("Levene's Test: Statistic = {:.4f}, P-value = {:.4f}".format(lev_stat, lev_p_value))



logging.basicConfig(level=logging.INFO)


TEMP_PATH = ".temp/"
FILENAME = "insurance_claims.parquet"


def setup_environment():
    file_path = os.path.join(TEMP_PATH, FILENAME)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    data = pd.read_parquet(file_path)
    logging.info("Data columns: %s", data.columns.tolist())


    index_column = "claim_id"
    value_column = "amount"
    state_columns = [
        "customer_age",
        "claim_type",
        "days_to_report",
        "previous_claims",
        "feature_1",
        "feature_2",
        "feature_3",
    ]
    budget = 9999999

    return InsurEnv(
        data=data,
        index_column=index_column,
        value_column=value_column,
        state_columns=state_columns,
        budget=budget,
    )


def test_reward_randomness(env, num_episodes=100, max_steps_per_episode=1000, num_bootstrap_samples=1000):
    rewards = {}
    # Collect rewards

    cumulative_rewards = 0
    approved_cumulative_rewards = {}
    rejected_cumulative_rewards = {}
    for _ in range(num_episodes):
        state = env.reset()
        for step in range(max_steps_per_episode):
            action = np.random.choice([0, 1])  # Random decision
            next_state, reward, done, _ = env.step(action)
            cumulative_rewards +=reward
            if action == 1:
                approved_cumulative_rewards[step] = cumulative_rewards
            else:
                rejected_cumulative_rewards[step] = cumulative_rewards
            if reward != 0:
                rewards[step] =cumulative_rewards
            if done:
                break
    return list(approved_cumulative_rewards), list(rejected_cumulative_rewards)


def save_histograms(g1, g2, file_name):
    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Plot the histograms
    sns.histplot(g1, kde=False, color='blue', label='Group 1')
    sns.histplot(g2, kde=False, color='red', label='Group 2')

    # Add labels and legend
    plt.title('Histograms of Group 1 and Group 2')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.legend(loc='upper right')

    # Save the figure
    plt.savefig(file_name)
    plt.close()



if __name__ == "__main__":
    env = setup_environment()
    g1, g2 = test_reward_randomness(env)
    save_histograms(g1, g2, 'histogram.png')
    compare_distributions(g1, g2)
    compare_non_iid(g1,g2)