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
