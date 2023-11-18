import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

# Define the paths using pathlib for better path handling
DATAPATH = Path("C:/Repos/gym-insurance/experiment_results/data")
PLOTSPATH = Path("C:/Repos/gym-insurance/experiment_results/plots")


def load_data(filename):
    # Load the data from the specified path
    return pd.read_parquet(DATAPATH / filename)


def rewards_analysis_plots(df):
    # Ensure the plots path directory exists
    (PLOTSPATH / "rewards_analysis").mkdir(parents=True, exist_ok=True)

    # Rewards Over Time Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="steps", y="rewards")
    plt.title("Rewards Over Time")
    plt.savefig(PLOTSPATH / "rewards_analysis/rewards_over_time.png")
    plt.clf()

    # Cumulative Rewards Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="steps", y="cum_rewards")
    plt.title("Cumulative Rewards Over Steps")
    plt.savefig(PLOTSPATH / "rewards_analysis/cumulative_rewards.png")
    plt.clf()

    # Histogram of Rewards
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="rewards", kde=True)
    plt.title("Histogram of Rewards")
    plt.savefig(PLOTSPATH / "rewards_analysis/histogram_of_rewards.png")
    plt.clf()

    # Scatter Plot of Rewards vs. Steps
    fig = px.scatter(df, x="steps", y="rewards", title="Rewards vs. Steps")
    fig.write_image(str(PLOTSPATH / "rewards_analysis/scatter_rewards_vs_steps.png"))

    # Reward Distribution by Event Type (if applicable)
    if "event_type" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x="rewards", hue="event_type", multiple="stack")
        plt.title("Reward Distribution by Event Type")
        plt.savefig(
            PLOTSPATH / "rewards_analysis/reward_distribution_by_event_type.png"
        )
        plt.clf()


def main():
    b = 1_000_000
    filename = f"results_{b}.paquet"
    df = load_data(filename)  # Replace with your data file path
    rewards_analysis_plots(df)


if __name__ == "__main__":
    main()
