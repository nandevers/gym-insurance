import os
import random
from math import log10
from matplotlib import pyplot as plt

import benfordslaw.benfordslaw as bf
import numpy as np
import pandas as pd
from benfordslaw.benfordslaw import _count_digit
from scipy.spatial import distance

decisions = {1: "approved", 0: "rejected", "decision": "all"}


import os
from math import log10
from matplotlib import pyplot as plt

import benfordslaw.benfordslaw as bf
import numpy as np
import pandas as pd
from benfordslaw.benfordslaw import _count_digit
from scipy.spatial import distance

# Define decision mapping
decisions = {1: "approved", 0: "rejected", "decision": "all"}


# benford_analysis.py

import os
from pathlib import Path
from matplotlib import pyplot as plt
import benfordslaw.benfordslaw as bf  # Install via pip: pip install benfordslaw
import numpy as np

# Mapping for plot labeling
decisions = {1: "approved", 0: "rejected", "all": "all"}


#class Benford:
#    def __init__(self, d=1):
#        self.d = d
#        self.bfs = [bf(pos=dp, verbose=0, alpha=0.01) for dp in range(1, d + 1)]
#
#    def fit(self, x):
#        for bf_instance in self.bfs:
#            bf_instance.fit(np.array(x))
#
#    def distance(self, x):
#        self.fit(x)
#        max_diff = 0.0
#        for dp, bf_instance in enumerate(self.bfs, start=1):
#            theoretical = bf_instance.leading_digits
#            observed = bf_instance.fit(np.array(x))["percentage_emp"][:, 1]
#            diff = np.abs(theoretical - observed)
#            current_max = np.max(diff)
#            print(f"Digit Position {dp}:")
#            print(f"Theoretical Proportions: {theoretical}")
#            print(f"Observed Proportions: {observed}")
#            print(f"Maximum Absolute Difference: {current_max}\n")
#            if current_max > max_diff:
#                max_diff = current_max
#        print(f"Overall Benford Distance: {max_diff}\n")
#        return max_diff / 100
#
#    def chi_test(self, x):
#        self.fit(x)
#        p_values = []
#        for dp, bf_instance in enumerate(self.bfs, start=1):
#            p_val = bf_instance.fit(np.array(x))["P_significant"]
#            p_values.append(p_val)
#        return p_values
#
#    def plot(self, x):
#        self.fit(x)
#        figures = []
#        for dp, bf_instance in enumerate(self.bfs, start=1):
#            fig, ax = bf_instance.plot(title=f"Digit Position {dp}")
#            figures.append(fig)
#        return figures
#
#    def save_plot(self, x, path, decision=None):
#        self.fit(x)
#        os.makedirs(path, exist_ok=True)
#        for dp, bf_instance in enumerate(self.bfs, start=1):
#            decision_label = decisions.get(decision, "all")
#            file_path = os.path.join(path, f"{decision_label}_DIGIT_{dp}.png")
#            fig, ax = bf_instance.plot(title=f"Digit Position {dp} - {decision_label}")
#            fig.savefig(file_path)
#            plt.close(fig)


class Benford(object):
   def __init__(self, d=2) -> None:
       self.d = d
       self.drange = range(1, d + 1)
       self.bfs = [bf(pos=d, verbose=0, alpha=0.01) for d in self.drange]

   def update_drange(self, d):
       self.drange = range(1, d + 1)

   def _fit(self, x):
       return [self.bfs[d - 1].fit(x) for d in self.drange]

   def _vec_dist(self, x):
       emp_dist = self._fit(x)
       return [emp_dist[d - 1]["percentage_emp"][:, 1] for d in self.drange]

   def distance(self, x):
       dists = 0
       emp_dist = self._vec_dist(x)
       _dist = []
       for d in self.drange:
           len(self.bfs[d - 1].leading_digits) == emp_dist[d - 1]
           _dist.append(
               np.max(self.bfs[d - 1].leading_digits - emp_dist[d - 1])
               / np.linalg.norm(self.bfs[d - 1].leading_digits - emp_dist[d - 1])
           )
       _dist_mean = np.mean(_dist)
       if np.isnan(_dist_mean):
           return 1
       else:
           return _dist_mean

   def chi_test(self, x):
       tests = self._fit(x)
       return [tests[d - 1]["P_significant"] for d in self.drange]

   def plot(self, x):
       self._fit(x)
       return [self.bfs[d - 1].plot(title=f"d={d}")[0] for d in self.drange]

   def save_plot(self, x, path, decision=None):
       self._fit(x)
       for d in self.drange:
           os.makedirs(path, exist_ok=True)
           file_path = path / f"{decisions[decision]}_DIGIT_{d}.png"
           _plot = self.bfs[d - 1].plot(
               title=f"""Distribution of d={d} and n:{len(x)} for Decision: {decision}"""
           )[0]
           plt.close()
           _plot.savefig(file_path)

#
# class BenfordRandom:
#    def __init__(self):
#        self.first_digit_distribution = [log10(1 + 1 / i) for i in range(1, 10)]
#
#    def rand(self):
#        rand_p = random.random()
#        for i in range(1, 10):
#            if rand_p < self.first_digit_distribution[i - 1]:
#                return i
#        return 9
#
#    def randint(self, a, b, size=None):
#        if size is None:
#            return self.rand() * (10 ** (int(log10(b)) - 1)) + random.randint(a, b)
#        else:
#            return np.array(
#                [
#                    self.rand() * (10 ** (int(log10(b)) - 1)) + random.randint(a, b)
#                    for _ in range(size)
#                ]
#            )
#
