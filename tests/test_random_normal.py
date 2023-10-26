from gym_insurance.envs.benford import BenfordRandom
from gym_insurance.envs.abenford import ABenford
import numpy as np

rand_array = np.random.randint(900, 1000, 100000, dtype="int")
rand_array2 = np.random.randint(1, 1000, 100000, dtype="int")

data = [*rand_array, *rand_array2]
check = ABenford(2, 1.95)
check.compute_observed_frequencies(data)

(
    c,
    observed_frequencies,
    ni,
    expected_frequencies,
) = check.compute_expected_frequencies(data)

comparison = check.compare_observed_and_expected_frequencies()
variance = check.compute_variance()

print(check._variance)
print(check._inference)
print(check._inference_bool)
check.c

results = []
for i in check._inference.items():
    obs = [o[1] for o in check.observed_frequencies.items() if len(str(o[0])) == i[0]]
    exp = [o[1] for o in check.expected_frequencies.items() if len(str(o[0])) == i[0]]

    for p, j in enumerate(i[1]):
        if p > 0:
            results.append(
                [p, np.abs(obs[p] - exp[p]), obs[p], exp[p], j, math.log10(1 + 1 / p)]
            )
        else:
            results.append([p, np.abs(obs[p] - exp[p]), obs[p], exp[p], j, 0])

import pandas as pd

results = pd.DataFrame(
    results, columns=["p", "diff", "obs", "exp", "j", "benf"]
).sort_values("diff")
sum(results.exp < 0)
results.query("p==145")
results.j.describe()

import math

check.expected_frequencies["1"] / check.n[1]
check.observed_frequencies["1"] / check.n[1]
sum([o[1] for o in check.observed_frequencies.items() if len(str(o[0])) == 1])
math.log10(1 + 1 / 1)
math.log10(1 + 1 / 2)


from gym_insurance.envs.benford import BenfordRandom
from gym_insurance.envs.abenford import ABenford
import scipy.stats as stats


if __name__ == "__main__":
    # Create a power-law distribution object with a = 1, loc = 1
    benford = stats.powerlaw(a=1, loc=1)
    # Calculate the 95% upper confidence level using the inverse CDF (PPF)
    upper_conf_level = benford.ppf(0.80)
    print(f"95% upper confidence level: {upper_conf_level:.4f}")

    # Test for Benford distribution
    benford_random = BenfordRandom()
    data = benford_random.randint(1, 1000, size=100000)

    check = ABenford(2, 1.95)
    (
        c,
        observed_frequencies,
        ni,
        expected_frequencies,
    ) = check.compute_expected_frequencies(data)

    comparison = check.compare_observed_and_expected_frequencies()
    variance = check.compute_variance()

    print(check._variance)
    print(check._inference)
    print(check._inference_bool)
    print(check.n)
