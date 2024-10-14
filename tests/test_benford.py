# benford_analysis.py

import os
from pathlib import Path
from matplotlib import pyplot as plt
import benfordslaw.benfordslaw as bf  # Install via pip: pip install benfordslaw
import numpy as np

# Mapping for plot labeling
decisions = {1: "approved", 0: "rejected", "all": "all"}


class Benford:
    def __init__(self, d=1):
        self.d = d
        self.bfs = [bf(pos=dp, verbose=0, alpha=0.01) for dp in range(1, d + 1)]

    def fit(self, x):
        for bf_instance in self.bfs:
            bf_instance.fit(np.array(x))

    def distance(self, x):
        self.fit(x)
        max_diff = 0.0
        for dp, bf_instance in enumerate(self.bfs, start=1):
            theoretical = bf_instance.leading_digits
            observed = bf_instance.fit(np.array(x))["percentage_emp"][:, 1]
            diff = np.abs(theoretical - observed)
            current_max = np.max(diff)
            print(f"Digit Position {dp}:")
            print(f"Theoretical Proportions: {theoretical}")
            print(f"Observed Proportions: {observed}")
            print(f"Maximum Absolute Difference: {current_max}\n")
            if current_max > max_diff:
                max_diff = current_max
        print(f"Overall Benford Distance: {max_diff}\n")
        return max_diff

    def chi_test(self, x):
        self.fit(x)
        p_values = []
        for dp, bf_instance in enumerate(self.bfs, start=1):
            p_val = bf_instance.fit(np.array(x))["P_significant"]
            p_values.append(p_val)
        return p_values

    def plot(self, x):
        self.fit(x)
        figures = []
        for dp, bf_instance in enumerate(self.bfs, start=1):
            fig, ax = bf_instance.plot(title=f"Digit Position {dp}")
            figures.append(fig)
        return figures

    def save_plot(self, x, path, decision=None):
        self.fit(x)
        os.makedirs(path, exist_ok=True)
        for dp, bf_instance in enumerate(self.bfs, start=1):
            decision_label = decisions.get(decision, "all")
            file_path = os.path.join(path, f"{decision_label}_DIGIT_{dp}.png")
            fig, ax = bf_instance.plot(title=f"Digit Position {dp} - {decision_label}")
            fig.savefig(file_path)
            plt.close(fig)


def generate_indemnity_payments(n=1000, min_value=1, max_value=10000, seed=42):
    np.random.seed(seed)
    # Log-normal distribution to mimic real-life skewed data
    mean = np.log(1000)
    sigma = 1.0
    data = np.random.lognormal(mean=mean, sigma=sigma, size=n)
    data = np.clip(data, min_value, max_value)
    return np.round(data).astype(int).tolist()


def main():
    # Generate sample indemnity payment data
    dataset_x = generate_indemnity_payments(n=1000, seed=42)
    dataset_y = generate_indemnity_payments(n=1000, seed=24)

    # Initialize Benford analyzer for first 3 digit positions
    benford_analyzer = Benford(d=3)

    # Analyze Dataset X
    print("Analyzing Dataset X:")
    distance_x = benford_analyzer.distance(dataset_x)
    print(f"Benford Distance for Dataset X: {distance_x}\n")

    # Analyze Dataset Y
    print("Analyzing Dataset Y:")
    distance_y = benford_analyzer.distance(dataset_y)
    print(f"Benford Distance for Dataset Y: {distance_y}\n")

    # Perform Chi-Squared Tests
    chi_test_x = benford_analyzer.chi_test(dataset_x)
    chi_test_y = benford_analyzer.chi_test(dataset_y)

    print("Chi-Squared P-Values for Dataset X:")
    for dp, p_val in enumerate(chi_test_x, start=1):
        print(f"Digit Position {dp}: P-Value = {p_val}")
    print("\n")

    print("Chi-Squared P-Values for Dataset Y:")
    for dp, p_val in enumerate(chi_test_y, start=1):
        print(f"Digit Position {dp}: P-Value = {p_val}")
    print("\n")

    # Save Plots
    save_path = Path("benford_plots")
    benford_analyzer.save_plot(dataset_x, save_path, decision=1)
    benford_analyzer.save_plot(dataset_y, save_path, decision=0)
    print(f"Plots have been saved to: {save_path.resolve()}")


if __name__ == "__main__":
    main()
