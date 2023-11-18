import math
import numpy as np
from collections import defaultdict


class ABenford(object):
    def __init__(self, digit_length, threshold):
        self.digit_length = digit_length
        self.threshold = threshold
        self._variance = defaultdict(int)
        self._inference = defaultdict(int)
        self._inference_bool = defaultdict(int)
        self.n = defaultdict(int)

    def compute_observed_frequencies(self, data):
        """
        Compute the observed frequencies of leading digit sequences for the given data.

        Parameters:
            data (list): a list of integers

        Returns:
            defaultdict: a dictionary with leading digit sequences as keys and their frequencies as values
        """
        observed_frequencies = defaultdict(int)
        for number in data:
            n = str(number)
            for i in range(1, len(n) + 1):
                if len(n) == i:
                    observed_frequencies[str(number)[:i]] += 1
        return observed_frequencies

    def compute_expected_frequencies(self, data):
        """
        Compute the expected frequencies of leading digit sequences for the given data.

        Parameters:
            data (list): a list of integers

        Returns:
            tuple: a tuple containing
                - c (dict): a dictionary with the ci constants for each leading digit sequence
                - observed (defaultdict): a dictionary with observed frequencies of leading digit sequences
                - ni (defaultdict): a dictionary with the ni values for each leading digit sequence
                - expected_frequencies (defaultdict): a dictionary with expected frequencies of leading digit sequences
        """
        c = {}
        observed_frequencies = self.compute_observed_frequencies(data)
        expected_frequencies = defaultdict(int)

        length = [len(str(o)) for o in observed_frequencies.keys()]
        min_length = min(int(l) for l in length)
        max_length = max((int(l) for l in length)) + 1

        for i in range(min_length, max_length):
            self.observations = [
                o[0] for o in observed_frequencies.items() if len(str(o[0])) == i
            ]

            self.n[i] = sum(
                [o[1] for o in observed_frequencies.items() if len(str(o[0])) == i]
            )

            # Step 2: Count the frequency of each leading digit sequence
            inner = [
                (observed_frequencies[d] / math.log10(1 + 1 / int(d)))
                for d in self.observations
            ]

            c[i] = (1 / self.n[i]) * sum(inner)

            # Step 3: Compute the expected frequencies of each leading digit sequence
            for o in observed_frequencies:
                if len(str(o)) == i:
                    expected_frequencies[o] = c[i] * math.log10(1 + 1 / int(o)) * int(o)

        self.observed_frequencies = observed_frequencies
        self.expected_frequencies = expected_frequencies
        self.c = c

        return (
            c,
            self.observed_frequencies,
            self.n,
            self.expected_frequencies,
        )

    def compare_observed_and_expected_frequencies(self):
        """
        Compare the observed and expected frequencies of leading digit sequences.

        Returns:
            dict: a dictionary with leading digit sequence as the key and a tuple of the observed and expected frequencies as the value
        """
        result = {}
        for leading_digits in self.observed_frequencies:
            result[leading_digits] = (
                self.observed_frequencies[leading_digits],
                self.expected_frequencies[leading_digits],
            )
        return result

    def compute_variance(self):
        """
        Compute the variance between observed and expected distributions of leading digit sequences.

        Returns:
            float: the variance value
        """
        for i in range(0, 9):
            of = [
                int(o[1])
                for o in self.observed_frequencies.items()
                if len(str(o[0])) == i
            ]
            ef = [
                o[1] for o in self.expected_frequencies.items() if len(str(o[0])) == i
            ]
            if len(of) + len(ef) > 0:
                self._variance[i] = sum(
                    [(o - e) ** 2 / self.n[i] for o, e in zip(of, ef)]
                )
                self._inference[i] = [
                    (o - e) / np.sqrt(self._variance[i]) for o, e in zip(of, ef)
                ]
                self._inference_bool[i] = [
                    self.threshold < v for v in self._inference[i]
                ]
