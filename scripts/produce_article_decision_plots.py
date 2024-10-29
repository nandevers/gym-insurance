from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gym_insurance.constants import BUDGET_LIST, DATAPATH, logger

# Define the paths using pathlib
PLOTSPATH = Path(".temp/data/plots")

# Global plot settings for consistency
sns.set(style="whitegrid")
plt.rcParams.update(
    {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.figsize": (10, 7),
    }
)


def load_data(filename):
    return pd.read_parquet(filename)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(PLOTSPATH / filename)
    plt.clf()


def plot_decision_frequency(df, budget):
    plt.figure()
    sns.countplot(x="decision", data=df, palette="Blues_r")
    plt.title("Frequency of Decision Outcomes")
    plt.xlabel("Decision Type")
    plt.ylabel("Count")
    for p in plt.gca().patches:
        plt.gca().annotate(
            f"{p.get_height()}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="baseline",
        )
    save_plot(f"decision_frequency_{budget}.png")


def plot_decision_vs_rewards(df, budget):
    plt.figure()
    sns.scatterplot(x="rewards", y="decision", data=df, alpha=0.6, edgecolor=None)
    plt.title("Relationship Between Decision and Rewards")
    plt.xlabel("Rewards")
    plt.ylabel("Decision")
    plt.axhline(
        0.5, color="gray", linestyle="--"
    )  # Adding a reference line for better interpretation
    save_plot(f"decision_vs_rewards_{budget}.png")


def plot_decision_distribution_across_steps(df, budget):
    plt.figure()
    sns.lineplot(x="steps", y="decision", data=df, ci=None, color="b")
    plt.title("Decision Trends Across Steps")
    plt.xlabel("Steps")
    plt.ylabel("Decision")
    plt.axhline(0.5, color="gray", linestyle="--")  # Adding a reference line
    save_plot(f"decision_distribution_across_steps_{budget}.png")


def plot_boxplot_budget_by_decision(df, budget):
    plt.figure()
    sns.boxplot(x="decision", y="budget", data=df, palette="Blues", showmeans=True)
    plt.title("Budget Distribution by Decision")
    plt.xlabel("Decision Type")
    plt.ylabel("Budget (in millions)")
    save_plot(f"budget_by_decision_{budget}.png")


def plot_correlation_heatmap(df, budget):
    plt.figure()
    selected_columns = ["wc", "woc", "rewards", "budget"]
    corr = df[selected_columns].corr()
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", cbar_kws={"label": "Correlation Coefficient"}
    )
    plt.title("Correlation Heatmap for Key Variables")
    save_plot(f"correlation_heatmap_{budget}.png")


def plot_cumulative_rewards_over_time(df, budget):
    plt.figure()
    sns.lineplot(
        x="steps",
        y="cum_rewards",
        hue="decision",
        data=df,
        ci="sd",
        palette="coolwarm",
        linewidth=2,
    )
    plt.title("Cumulative Rewards Over Time by Decision Type")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Rewards")
    save_plot(f"cumulative_rewards_over_time_{budget}.png")


def plot_histogram_steps_by_decision(df, budget):
    plt.figure()
    sns.histplot(data=df, x="steps", hue="decision", multiple="stack", palette="Set2")
    plt.title("Distribution of Steps by Decision")
    plt.xlabel("Steps")
    plt.ylabel("Count")
    save_plot(f"histogram_steps_by_decision_{budget}.png")


def plot_decision_impact_on_event_types(df, budget):
    if "event_type" in df.columns:
        plt.figure()
        sns.countplot(x="event_type", hue="decision", data=df, palette="Set1")
        plt.title("Impact of Decision on Event Types")
        plt.xlabel("Event Type")
        plt.ylabel("Count")
        save_plot(f"decision_impact_on_event_types_{budget}.png")


def plot_decision_vs_avg_indemnization(df, budget):
    plt.figure()
    sns.scatterplot(x="qt_valor_indenizacao", y="decision", data=df, alpha=0.6)
    plt.title("Decision vs. Indemnization Amounts")
    plt.xlabel("Indemnization Amount")
    plt.ylabel("Decision")
    plt.axhline(0.5, color="gray", linestyle="--")
    save_plot(f"decision_vs_avg_indemnization_{budget}.png")


def plot_proportion_of_decisions_over_episodes(df, budget):
    plt.figure()
    sns.barplot(
        x="episodes",
        y="decision",
        data=df,
        estimator=np.mean,
        ci="sd",
        palette="coolwarm",
    )
    plt.title("Proportion of Decisions Over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Average Decision")
    save_plot(f"proportion_of_decisions_over_episodes_{budget}.png")


def main():
    PLOTSPATH.mkdir(parents=True, exist_ok=True)
    for b in BUDGET_LIST:
        logger.info(f"Generating plots for budget {b}")
        df = load_data(DATAPATH / f"experiment_results/results_{b}.parquet")

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
