import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from constants import DATAPATH, BUDGET_LIST, logger
import plotly.express as px
from scipy.stats import chi2_contingency, mannwhitneyu, ks_2samp, f_oneway
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import numpy as np

# Define the paths using pathlib
DATAPATH = Path("C:/Repos/gym-insurance/experiment_results/data")
PLOTSPATH = Path("C:/Repos/gym-insurance/experiment_results/plots/decisions")


def load_data(filename):
    return pd.read_parquet(filename)


def plot_decision_frequency(df, budget):
    plt.figure(figsize=(8, 6))
    sns.countplot(x="decision", data=df)
    plt.title("Decision Frequency")
    plt.savefig(PLOTSPATH / f"decision_frequency_{budget}.png")
    plt.clf()
    # Chi-squared test can be added here


def plot_decision_vs_rewards(df, budget):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="rewards", y="decision", data=df)
    plt.title("Decision vs. Rewards")
    plt.savefig(PLOTSPATH / f"decision_vs_rewards_{budget}.png")
    plt.clf()
    # Logistic regression analysis can be performed here


def plot_decision_distribution_across_steps(df, budget):
    plt.figure(figsize=(8, 6))
    sns.lineplot(x="steps", y="decision", data=df)
    plt.title("Decision Distribution Across Steps")
    plt.savefig(PLOTSPATH / f"decision_distribution_across_steps_{budget}.png")
    plt.clf()
    # Logistic regression analysis can be performed here


def plot_boxplot_budget_by_decision(df, budget):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="decision", y="budget", data=df)
    plt.title("Budget Distribution by Decision")
    plt.savefig(PLOTSPATH / f"budget_by_decision_{budget}.png")
    plt.clf()
    # Mann-Whitney U test can be performed here


def plot_correlation_heatmap(df, budget):
    plt.figure(figsize=(8, 6))  # Adjusted figure size
    # Select specific columns for correlation
    selected_columns = ["wc", "woc", "rewards", "budget"]
    selected_df = df[selected_columns]
    corr = selected_df.corr()

    # Plotting the heatmap
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(PLOTSPATH / f"correlation_heatmap_{budget}.png")
    plt.clf()
    # Additional statistical analysis can


def plot_cumulative_rewards_over_time(df, budget):
    plt.figure(figsize=(8, 6))
    sns.lineplot(x="steps", y="cum_rewards", hue="decision", data=df)
    plt.title("Cumulative Rewards Over Time by Decision")
    plt.savefig(PLOTSPATH / f"cumulative_rewards_over_time_{budget}.png")
    plt.clf()
    # Time-series regression analysis can be added here


def plot_histogram_steps_by_decision(df, budget):
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x="steps", hue="decision", multiple="stack")
    plt.title("Histogram of Steps by Decision")
    plt.savefig(PLOTSPATH / f"histogram_steps_by_decision_{budget}.png")
    plt.clf()
    # Kolmogorov-Smirnov test can be performed here


def plot_decision_impact_on_event_types(df, budget):
    if "event_type" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x="event_type", hue="decision", data=df)
        plt.title("Decision Impact on Event Types")
        plt.savefig(PLOTSPATH / f"decision_impact_on_event_types_{budget}.png")
        plt.clf()
        # Chi-squared test for independence can be performed here


def plot_decision_vs_avg_indemnization(df, budget):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="qt_valor_indenizacao", y="decision", data=df)
    plt.title("Decision vs. Average Indemnization")
    plt.savefig(PLOTSPATH / f"decision_vs_avg_indemnization_{budget}.png")
    plt.clf()
    # Additional analysis can be added here


def plot_proportion_of_decisions_over_episodes(df, budget):
    plt.figure(figsize=(8, 6))
    sns.barplot(x="episodes", y="decision", data=df, estimator=np.mean)
    plt.title("Proportion of Decisions Over Episodes")
    plt.savefig(PLOTSPATH / f"proportion_of_decisions_over_episodes_{budget}.png")
    plt.clf()
    # Additional analysis can be added here


def main():
    PLOTSPATH.mkdir(parents=True, exist_ok=True)
    for b in BUDGET_LIST:
        logger.info(f"Calculating analytics for {b}")
        df = load_data(DATAPATH / f"results_{b}.paquet")

        # Generate additional plots
        plot_cumulative_rewards_over_time(df, b)
        plot_histogram_steps_by_decision(df, b)
        plot_decision_impact_on_event_types(df, b)
        plot_decision_vs_avg_indemnization(df, b)
        plot_proportion_of_decisions_over_episodes(df, b)
        plot_decision_frequency(df, b)
        plot_decision_vs_rewards(df, b)
        plot_decision_distribution_across_steps(df, b)
        plot_boxplot_budget_by_decision(df, b)
        plot_correlation_heatmap(df, b)


if __name__ == "__main__":
    main()
