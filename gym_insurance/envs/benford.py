import random
from math import log10

import numpy as np
import pandas as pd
import benfordslaw.benfordslaw as bf
from benfordslaw.benfordslaw import _count_digit


class Benford(object):
    def __init__(self, d=2) -> None:
        self.bfs = [bf(pos=d + 1, verbose=0) for d in range(d)]
        self.d = d

    def _fit(self, x):
        return [self.bfs[d].fit(x) for d in range(self.d)]

    def _vec_dist(self, x):
        emp_dist = self._fit(x)
        return [emp_dist[d]["percentage_emp"][:, 1] for d in range(self.d)]

    def distance(self, x):
        emp_dist = self._vec_dist(x)
        dists = [
            np.linalg.norm(self.bfs[d].leading_digits - emp_dist[d])
            for d in range(self.d)
        ]
        return np.mean(dists)


class BenfordRandom:
    def __init__(self):
        self.first_digit_distribution = [log10(1 + 1 / i) for i in range(1, 10)]

    def rand(self):
        rand_p = random.random()
        for i in range(1, 10):
            if rand_p < self.first_digit_distribution[i - 1]:
                return i
        return 9

    def randint(self, a, b, size=None):
        if size is None:
            return self.rand() * (10 ** (int(log10(b)) - 1)) + random.randint(a, b)
        else:
            return np.array(
                [
                    self.rand() * (10 ** (int(log10(b)) - 1)) + random.randint(a, b)
                    for _ in range(size)
                ]
            )
