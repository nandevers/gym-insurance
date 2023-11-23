import pandas as pd
import logging
import os
from constants import DATAPATH, BUDGET_LIST, logger
from pathlib import Path


def load_data(filename):
    return pd.read_parquet(filename)


def analyze_data(df):
    grouped = df.groupby("episodes")

    # Calculating metrics for each episode
    episode_analytics = grouped.apply(calculate_metrics_per_episode)

    return episode_analytics


def calculate_metrics_per_episode(episode_df):
    metrics = {}
    metrics["total_rewards"] = episode_df["rewards"].sum()
    metrics["average_rewards"] = episode_df["rewards"].mean()
    metrics["max_reward"] = episode_df["rewards"].max()
    metrics["cumulative_rewards_over_time"] = episode_df["cum_rewards"].iloc[-1]
    metrics["average_cumulative_rewards"] = episode_df["cum_rewards"].mean()
    metrics["budget_utilization"] = (
        episode_df["initial_budget"] - episode_df["budget"]
    ).mean()
    metrics["average_steps"] = episode_df["steps"].mean()
    metrics["success_rate"] = (episode_df["rewards"] > 0).mean()
    metrics["average_value_of_indemnization"] = episode_df[
        "qt_valor_indenizacao"
    ].mean()
    metrics["std_dev_rewards"] = episode_df["rewards"].std()
    return pd.Series(metrics)


def save_analytics(analytics, budget):
    file_path = DATAPATH / f"episode_analytics_{budget}.paquet"
    logger.info(f"Saving data to {file_path}")
    analytics.to_parquet(file_path)


def main():
    for b in BUDGET_LIST:
        logger.info(f"Calculating analytics for {b}")
        df = load_data(DATAPATH / f"results_{b}.paquet")
        logger.info(df.head())
        analytics = analyze_data(df)
        save_analytics(analytics, budget=b)


if __name__ == "__main__":
    main()
