import pandas as pd
import os
from constants import DATAPATH, BUDGET_LIST, logger
from pathlib import Path

DATAPATH = Path("C:\\Repos\\gym-insurance\\experiment_results\\data")


def load_data(filename):
    return pd.read_parquet(filename)


def analyze_data(df):
    # Grouping by both episodes and decision
    grouped = df.groupby(["episodes", "decision"])

    # Calculating metrics for each group
    group_analytics = grouped.apply(calculate_metrics_per_group)

    return group_analytics


def calculate_metrics_per_group(group_df):
    metrics = {}
    metrics["total_rewards"] = group_df["rewards"].sum()
    metrics["average_rewards"] = group_df["rewards"].mean()
    metrics["max_reward"] = group_df["rewards"].max()
    metrics["cumulative_rewards_over_time"] = group_df["cum_rewards"].iloc[-1]
    metrics["average_cumulative_rewards"] = group_df["cum_rewards"].mean()
    metrics["budget_utilization"] = (
        group_df["initial_budget"] - group_df["budget"]
    ).mean()
    metrics["average_steps"] = group_df["steps"].mean()
    metrics["success_rate"] = (group_df["rewards"] > 0).mean()
    metrics["average_value_of_indemnization"] = group_df["qt_valor_indenizacao"].mean()
    metrics["std_dev_rewards"] = group_df["rewards"].std()
    return pd.Series(metrics)


def save_analytics(analytics, budget):
    analytics.to_parquet(DATAPATH / f"decision_episode_analytics_{budget}.paquet")


def main():
    for b in BUDGET_LIST:
        logger.info(f"Calculating Decision Analytics for {b}")
        df = load_data(
            DATAPATH / f"results_{b}.paquet"
        )  # Replace with your data file path
        analytics = analyze_data(df)
        save_analytics(analytics, b)


if __name__ == "__main__":
    main()
