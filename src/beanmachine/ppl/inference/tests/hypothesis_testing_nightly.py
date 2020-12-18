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
    """This class tests the hypothesis test codes."""

    # Uniformly distributed random numbers
    def random(self, min_bound, max_bound):
        # TODO: Consider replacing with: (max_bound - min_bound) * torch.rand(size) + min_bound
        # where size = (max_bound+min_bound).size()
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
        true_mean = self.random(-1, 1) * 10 ** exp_mean
        true_std = self.random(0, 1) * 10 ** exp_std
        return true_mean, true_std

    # Main procedure for testing the hypothesis test
    # It works by checking the significance level (alpha) semantics
    # of the mean equality hypothesis test.

    def run_mean_equality_hypothesis_test_on_synthetic_samples(
        self, samples, sample_size, alpha, random_seed=42
    ):
        """Generates as many samples as provided by the parameter of that
        name, and performs the mean_equality_hypothesis_test
        on each of these samples. Since we use the mean and standard
        devaiation of the distribution, which are known, the hypothesis
        test *should* faile at a rate fo alpha. In order for this to be
        checked, we return the observed_alpha rate. In addition, we check
        that the hypothesis_test to confidence_interval methods are consistent,
        and return a count of any potential discrepancies between them."""
        manual_seed(random_seed)
        accepted_test = 0
        exp_min, exp_max = self.float_exponent_range()
        for _ in range(0, samples):
            true_mean, true_std = self.random_mean_and_std(exp_min, exp_max)
            d = dist.normal.Normal(loc=true_mean, scale=true_std)
            sample_size = tensor([sample_size])
            r = d.sample(sample_size)
            sample_mean = mean(r)
            # Record hypothesis_test_behavior for this single sample
            accept_test = mean_equality_hypothesis_test(
                sample_mean, true_mean, true_std, sample_size, alpha
            )
            if accept_test:
                accepted_test += 1
            # Compare hypothesis_test to confidence_interval
            lower_bound, upper_bound = mean_equality_hypothesis_confidence_interval(
                true_mean, true_std, sample_size, alpha
            )
            below_upper = (lower_bound <= sample_mean).all()
            above_lower = (sample_mean <= upper_bound).all()
            accept_interval = below_upper and above_lower
            # accept_interval = min(lower_bound <= sample_mean <= upper_bound).item()
            self.assertFalse(
                accept_test and not accept_interval, "Interval can be too small"
            )
            self.assertFalse(
                accept_interval and not accept_test, "Interval can be too big"
            )

        observed_alpha = 1 - accepted_test / samples
        return observed_alpha

    # Test function for the hypothesis test. Normal operation is to
    # take no arguments. Auditing can be done by changing the random_seed.
    # An audit would pass if the test returns False for only an alpha
    # fraction of the random_seeds on average. Since this is a stochastic
    # correctness criteria, we use alpha_meta for this (meta-)test.
    def test_mean_equality_hypothesis_test(
        self, runs=1000, samples=100, alpha=0.01, alpha_meta=0.01, random_seed=42
    ):
        """Check that the hypothesis tests are working as expected,
        that is, their promised alpha is about the same as the rate at which
        they fail. The idea here is that we run a series of checks, and treat
        this as a binomial distribution.
        Note, the alpha_meta for this test should not be confused with the
        alpha of the individual tests.
        Yes, this method is using hypothesis testing to test our hypothesis
        testing method. We call this a meta-test.

        Note:
        1) We do the meta-test multiple times (runs)
        2) Each meta-test is a Bernoulli trial. The probability of failure
           should be exactly alpha.
        3) We check that the total runs of the meta-test have an observed
           failure rate that is equal to alpha. We do this by checking
           that it falls within the alpha_meta CI.
        """
        observed_alphas = [
            self.run_mean_equality_hypothesis_test_on_synthetic_samples(
                samples=samples,
                sample_size=100,
                alpha=alpha,
                random_seed=(random_seed + i) * i,
            )
            for i in range(0, runs)
        ]

        # Meta-test
        true_mean = alpha  # For binomial meta-test distribution
        true_std = sqrt(alpha * (1 - alpha))
        bound = inverse_normal_cdf(1 - alpha_meta / 2)
        binomial_results = [
            -bound <= (observed_alpha - true_mean) * sqrt(samples) / true_std <= bound
            for observed_alpha in observed_alphas
        ]

        # Notice that the meta-tests gives us a series of booleans. How do we interpret
        # those? That's what we need the meta-meta-test

        # Meta-meta-test.
        true_mean = (
            1 - alpha_meta
        )  # So, we'll use alpha_meta for both the meta- and meta-meta- test
        true_std = sqrt(alpha_meta * (1 - alpha_meta))
        observed_mean = sum(binomial_results) / runs
        bound = inverse_normal_cdf(
            1 - alpha_meta / 2
        )  # Yes, this is the same as for meta-test

        meta_meta_test = (
            -bound <= (observed_mean - true_mean) * sqrt(runs) / true_std <= bound
        )

        message = str(
            (
                "true_mean = "
                + str(true_mean)
                + ", "
                + " observed_mean = "
                + str(observed_mean)
                + ", "
                + "adjusted_bound = "
                + str(bound * true_std / sqrt(runs)),
            )
        )

        self.assertTrue(
            meta_meta_test,
            "Unable to confirm significance level (alpha) semantics: " + message,
        )
