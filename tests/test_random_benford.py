from gym_insurance.envs.benford import BenfordRandom
from gym_insurance.envs.abenford import ABenford

# Test for Benford distribution
benford_random = BenfordRandom()
data = benford_random.randint(1, 1000, size=100000)

check = ABenford(2)
(
    c,
    observed,
    observed_frequencies,
    ni,
    expected_frequencies,
) = check.compute_expected_frequencies(data)

comparison = check.compare_observed_and_expected_frequencies()
variance = check.compute_variance()

print(check._variance)
print(check._inference)

check._inference[2]

for i in check._inference.items():
    print(sum([j>1 for j in i[1] ]))

# Test for Normal distribution
import numpy as np
rand_array = np.random.randint(100, 1000, 100000, dtype="int")
data = rand_array

check = ABenford(2)
check.compute_observed_frequencies(data)
max_length = [len(str(o)) for o in check.compute_observed_frequencies(data).keys()]
max_length = max((int(m) for m in max_length))

(
    c,
    observed,
    observed_frequencies,
    ni,
    expected_frequencies,
) = check.compute_expected_frequencies(data)
sum((int(e) > 1 for e in expected_frequencies))

comparison = check.compare_observed_and_expected_frequencies()
variance = check.compute_variance()

print(check._variance)
print(check._inference)


check._inference

for i in check._inference.items():
    print(sum([j>1 for j in i[1] if i is not isinstance(i, int)]))
