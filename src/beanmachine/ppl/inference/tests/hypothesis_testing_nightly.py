# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
from sys import float_info

import torch.distributions as dist
from beanmachine.ppl.testlib.hypothesis_testing import (
    inverse_normal_cdf,
    mean_equality_hypothesis_confidence_interval,
    mean_equality_hypothesis_test,
)
from numpy import sqrt
from torch import manual_seed, mean, tensor


class HypothesisTestingTest(unittest.TestCase):
    def setUp(self):
        pass

    # Tests for the hypothsis test

    # Uniformly distributed random numbers
    def random(self, min_bound, max_bound):
        return dist.uniform.Uniform(min_bound, max_bound).sample()

    # Determining the range of floating point values we will explore
    def float_exponent_range(self, safety_factor=10):
        """Provided exponents for range of floating point
        numbers we are willing to test. The parameter
        safety_factor should always be greater than 1, and
        is used to avoid pressing extreme values."""
        exp_min = float_info.min_10_exp / safety_factor
        exp_max = float_info.max_10_exp / safety_factor
        return exp_min, exp_max

    # Determining the range of distribution of means and stds we will explore
    def random_mean_and_std(self, exp_min, exp_max):
        """Generates a mean and std from a `reasonable` range
        of possible test values. Please note that this generator
        is by no means `exhaustive`. The purpose of the method
        is to simply provide a based set of values for checking
        our basic hypothesis tests."""
        exp_mean = self.random(exp_min, exp_max)
        exp_std = exp_mean + self.random(-3, 3)
        true_mean = tensor(self.random(-1, 1) * 10 ** exp_mean)
        true_std = tensor(self.random(0, 1) * 10 ** exp_std)
        return true_mean, true_std

    # Main procedure for testing the hypothesis test
    # It works by checking the p-value semantics of the mean equality
    # hypothesis test. By default it uses 100 batches of
    # 1000 samples of 1000 elements.

    def run_mean_equality_hypothesis_test_on_synthetic_samples(
        self, samples=1000, sample_size=1000, p_value=0.01, random_seed=None
    ):
        """Generates as many samples as provided by the parameter of that
        name, and uses them to check that the mean_equality_hypothesis_test
        demonstrates type I error (fails to recognize equality) at a rate
        comparable to the provided p-value. In addition to this basic check,
        this test also compare hypothesis_test to confidence_interval."""
        if random_seed is not None:
            manual_seed(random_seed)
        accepted_test = 0
        accepted_interval_not_test = 0
        accepted_test_not_interval = 0
        exp_min, exp_max = self.float_exponent_range()
        for _ in range(0, samples):
            true_mean, true_std = self.random_mean_and_std(exp_min, exp_max)
            d = dist.normal.Normal(loc=true_mean, scale=true_std)
            sample_size = tensor([sample_size])
            r = d.sample(sample_size)
            sample_mean = mean(r)
            # Record hypothesis_test_behavior
            accept_test = mean_equality_hypothesis_test(
                sample_mean, true_mean, true_std, sample_size, p_value
            )
            if accept_test:
                accepted_test += 1
            # Compare hypothesis_test to confidence_interval
            lower_bound, upper_bound = mean_equality_hypothesis_confidence_interval(
                true_mean, true_std, sample_size, p_value
            )
            below_upper = min(lower_bound <= sample_mean).item()
            above_lower = min(sample_mean <= upper_bound).item()
            accept_interval = below_upper and above_lower
            # accept_interval = min(lower_bound <= sample_mean <= upper_bound).item()
            if accept_test and not accept_interval:
                accepted_test_not_interval += 1
            if accept_interval and not accept_test:
                accepted_interval_not_test += 1

        observed_p_value = 1 - accepted_test / samples
        return observed_p_value, accepted_test_not_interval, accepted_interval_not_test

    # Test function for the hypothesis test. Normal operation is to
    # take no arguments. Auding can be done by changing the random_seed.
    # An audit would pass if the test returns False for only an alpha
    # fraction of the random_seeds used for audting.
    def check_mean_equality_hypothesis_test(
        self, runs=100, samples=1000, p_value=0.01, alpha=0.01, random_seed=42
    ):
        """Check that the hypothesis tests are working as expected,
        that is, their promised p-value is about the same as the rate at which
        they fail. The idea here is that we run a series of checks, and treat
        this as a binomial distribution.
        Note, the alpha for this test should not be confused with the
        p-value of the individual tests.
        Yes, this method is using hypothesis testing to test our hypothesis
        testing method."""
        run_results = [
            self.run_mean_equality_hypothesis_test_on_synthetic_samples(
                samples=samples, p_value=p_value, random_seed=random_seed + i
            )
            for i in range(0, runs)
        ]
        observed_p_values = []
        accepted_test_not_interval = 0
        accepted_interval_not_test = 0
        for oti in run_results:
            o, t, i = oti
            observed_p_values += [o]
            accepted_test_not_interval += t
            accepted_interval_not_test += i
        bound = inverse_normal_cdf(1 - alpha / 2)
        true_std = sqrt(p_value * (1 - p_value))  # For binomial distribution
        binomial_results = [
            -bound <= (observed_p_value - p_value) * sqrt(samples) / true_std <= bound
            for observed_p_value in observed_p_values
        ]
        observed_success = sum(binomial_results)
        expected_success = round(runs * (1 - alpha))
        return (
            observed_success,
            expected_success,
            accepted_test_not_interval,
            accepted_interval_not_test,
        )

    def test_hypothesis_testing_run(self):
        (
            observed,
            expected,
            accepted_test_not_interval,
            accepted_interval_not_test,
        ) = self.check_mean_equality_hypothesis_test(10)
        self.assertEqual(observed, expected, "Hypothesis test p-value semantics fails")
        self.assertTrue(expected > 0, "Expected success rate should be positive")
        self.assertEqual(accepted_test_not_interval, 0, "Interval can be too small")
        self.assertEqual(accepted_interval_not_test, 0, "Interval can be too big")
