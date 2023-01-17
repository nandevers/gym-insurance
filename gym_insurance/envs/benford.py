import random
import numpy as np
from math import log10

class BenfordRandom:
    def __init__(self):
        self.first_digit_distribution = [log10(1 + 1/i) for i in range(1, 10)]

    def rand(self):
        rand_p = random.random()
        for i in range(1, 10):
            if rand_p < self.first_digit_distribution[i-1]:
                return i
        return 9

    def randint(self, a, b, size=None):
        if size is None:
            return self.rand() * (10 ** (int(log10(b)) - 1)) + random.randint(a, b)
        else:
            return np.array([self.rand() * (10 ** (int(log10(b)) - 1)) + random.randint(a, b) for _ in range(size)])
