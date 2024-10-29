import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator

# Directory where the TensorBoard log files are stored
log_dir = ".t   emp/logs"

# List all files in the log directory
event_files = [f for f in os.listdir(log_dir) if "tfevents" in f]


# Function to plot and save figures from scalars
def extract_tensorboard_scalars(event_file_path, save_dir="./tensorboard_figures"):
    # Load the event file
    event_acc = EventAccumulator(event_file_path)
    event_acc.Reload()  # Load the events from the file

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Extract scalar tags (e.g., accuracy, loss, etc.)
    scalar_tags = event_acc.Tags().get("scalars", [])
    for tag in scalar_tags:
        # Retrieve all scalar events for the current tag
        events = event_acc.Scalars(tag)

        # Extract steps and values
        steps = [e.step for e in events]
        values = [e.value for e in events]

        # Plot and save the figure
        plt.figure()
        plt.plot(steps, values)
        plt.xlabel("Step")
        plt.ylabel(tag)
        plt.title(f"{tag} over time")
        plt.savefig(os.path.join(save_dir, f"{tag}.png"))
        plt.close()


# Function to plot and save figures from histograms
def extract_tensorboard_histograms(
    event_file_path, save_dir="./tensorboard_histograms"
):
    event_acc = EventAccumulator(event_file_path)
    event_acc.Reload()

    os.makedirs(save_dir, exist_ok=True)

    histogram_tags = event_acc.Tags().get("histograms", [])
    for tag in histogram_tags:
        events = event_acc.Histograms(tag)
        for i, event in enumerate(events):
            values = event.histogram_value
            # Customize how to plot the histogram based on event data.
            plt.figure()
            plt.hist(values, bins=30)
            plt.title(f"Histogram of {tag} at step {event.step}")
            plt.savefig(os.path.join(save_dir, f"{tag}_step_{event.step}.png"))
            plt.close()


# Loop through all event files and extract both scalars and histograms
for event_file in event_files:
    event_file_path = os.path.join(log_dir, event_file)

    # Extract and save scalar figures
    extract_tensorboard_scalars(event_file_path, save_dir="./tensorboard_figures")

    # Extract and save histogram figures
    extract_tensorboard_histograms(event_file_path, save_dir="./tensorboard_histograms")
