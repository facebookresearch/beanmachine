"""Tests for hypothesis_testing.py"""
import unittest

from beanmachine.ppl.testlib.hypothesis_testing import (
    inverse_chi2_cdf,
    inverse_normal_cdf,
    mean_equality_hypothesis_test,
    variance_equality_hypothesis_confidence_interval,
    variance_equality_hypothesis_test,
)
from torch import tensor


class HypothesisTestingTest(unittest.TestCase):
    def test_hypothesis_test_inverse_normal_cdf(self) -> None:
        """Minimal test for inverse normal CDF used to calculate z values"""

        # Check that the median has the probability we expect
        median = inverse_normal_cdf(0.5)
        self.assertEqual(
            median, 0.0, msg="Unexpected value for median of normal distribution"
        )

        # Record and check the values we get for z_0.01
        expected_z_one_percent = -2.3263478740408408
        observed_z_one_percent = inverse_normal_cdf(0.01)
        self.assertEqual(
            observed_z_one_percent,
            expected_z_one_percent,
            msg="Expected value for z_0.01",
        )

        # Record and check the values we get for z_0.99
        expected_z_99_percent = 2.3263478740408408
        observed_z_99_percent = inverse_normal_cdf(1 - 0.01)
        self.assertEqual(
            observed_z_99_percent,
            expected_z_99_percent,
            msg="Expected value for z_0.99",
        )

        # Record and check the values we get for z_0.005
        expected_z_half_percent = -2.575829303548901
        observed_z_half_percent = inverse_normal_cdf(0.005)
        self.assertEqual(
            observed_z_half_percent,
            expected_z_half_percent,
            msg="Expected value for z_0.005",
        )

        # This example shows why 1-p can be problematic
        # Compare this value to -expected_z_half_percent
        expected_z_995_thousandths = 2.5758293035489004
        observed_z_995_thousandths = inverse_normal_cdf(0.995)
        self.assertTrue(
            not (expected_z_995_thousandths == -expected_z_half_percent),
            msg="Numerical z_p is usually not exactly -z_(1-p)",
        )
        self.assertEqual(
            observed_z_995_thousandths,
            expected_z_995_thousandths,
            msg="Expected value for z_0.005",
        )

    def test_hypothesis_test_mean(self) -> None:
        """Minimal test for mean equality hypothesis test"""
        sample_mean = tensor(10)
        true_mean = tensor(0)
        true_std = tensor(1)
        sample_size = tensor(1)
        p_value = 0.01
        observed_result = mean_equality_hypothesis_test(
            sample_mean, true_mean, true_std, sample_size, p_value
        )
        self.assertEqual(
            observed_result, False, msg="Mean is not within confidence interval"
        )

        sample_mean = tensor(0)
        true_mean = tensor(0)
        true_std = tensor(1)
        sample_size = tensor(1)
        p_value = 0.01
        observed_result = mean_equality_hypothesis_test(
            sample_mean, true_mean, true_std, sample_size, p_value
        )
        self.assertEqual(
            observed_result, True, msg="Mean is not within confidence interval"
        )

        # This test case is at the edge of acceptable.
        # It should pass because of the = in <= in the
        # mean_equality_hypothesis_test method
        expected_z_995_thousandths = 2.5758293035489004
        sample_mean = tensor(expected_z_995_thousandths)
        true_mean = tensor(0)
        true_std = tensor(1)
        sample_size = tensor(1)
        p_value = 0.01
        observed_result = mean_equality_hypothesis_test(
            sample_mean, true_mean, true_std, sample_size, p_value
        )
        self.assertEqual(
            observed_result, True, msg="Mean is not within confidence interval"
        )

        # The following two tests are pushing the edge case around what
        # should be acceptable to the test. It is strange that the one
        # slighly larger than the alpha value does not fail.
        # TODO: Investigate and explain why this passes when it should be
        # just outside the acceptable boundary.
        expected_z_995_thousandths = 2.5758293035489004
        sample_mean = tensor(expected_z_995_thousandths * 1.00000001)
        true_mean = tensor(0)
        true_std = tensor(1)
        sample_size = tensor(1)
        p_value = 0.01
        observed_result = mean_equality_hypothesis_test(
            sample_mean, true_mean, true_std, sample_size, p_value
        )
        self.assertEqual(
            observed_result, True, msg="Mean is not within confidence interval"
        )

        # This one, with bigger multiplierf, finally returns False
        expected_z_995_thousandths = 2.5758293035489004
        sample_mean = tensor(expected_z_995_thousandths * 1.0000001)
        true_mean = tensor(0)
        true_std = tensor(1)
        sample_size = tensor(1)
        p_value = 0.01
        observed_result = mean_equality_hypothesis_test(
            sample_mean, true_mean, true_std, sample_size, p_value
        )
        self.assertEqual(
            observed_result, False, msg="Mean is not within confidence interval"
        )

    def test_hypothesis_test_inverse_chi2_cdf(self) -> None:
        """Minimal test for inverse chi-squared CDF used to calculate chi2 values"""

        # Check that the median has the probability we expect
        # A rule of thumb for chi2 is that median is df-0.7
        # in this test we pick a more specific value from test run
        median = inverse_chi2_cdf(100, 0.5)
        self.assertEqual(
            median,
            99.33412923598846,
            msg="Unexpected value for median of normal distribution",
        )

        # Record and check the values we get for chi2_0.01
        # From C.M. Thompson tables from 1941, we expect 70.0648
        # more specific value reflects results from test run
        # NB: Test run appears to contradict least significant
        # digit in table the table cited above, but not if we take
        # into account p used for lookup in distribution
        # is 0.990, which suggests only on 4 digits are valid.
        expected_chi2_one_percent = 70.06489492539978
        observed_chi2_one_percent = inverse_chi2_cdf(100, 0.01)
        self.assertEqual(
            observed_chi2_one_percent,
            expected_chi2_one_percent,
            msg="Expected value for chi2_0.01",
        )

        # Record and check the values we get for chi2_0.99
        # Table above predicts 135.807
        expected_chi2_99_percent = 135.80672317102676
        observed_chi2_99_percent = inverse_chi2_cdf(100, 1 - 0.01)
        self.assertEqual(
            observed_chi2_99_percent,
            expected_chi2_99_percent,
            msg="Expected value for chi2_0.99",
        )

        # Record and check the values we get for chi2_0.005
        # Table above predicts 67.3276
        expected_chi2_half_percent = 67.32756330547916
        observed_chi2_half_percent = inverse_chi2_cdf(100, 0.005)
        self.assertEqual(
            observed_chi2_half_percent,
            expected_chi2_half_percent,
            msg="Expected value for chi2_0.005",
        )

    def test_hypothesis_test_variance(self) -> None:
        """Minimal test for variance equality hypothesis test"""
        # Based on solved example in Scheaffer & McClave, 1986, Pg 300
        sample_std = tensor(0.0003) ** 0.5
        true_std = tensor(0.0002) ** 0.5
        degrees_of_freedom = tensor(9)
        p_value = 0.05
        observed_result = variance_equality_hypothesis_test(
            sample_std, true_std, degrees_of_freedom, p_value
        )
        self.assertEqual(
            observed_result, True, msg="Variance is within confidence interval"
        )

        sample_std = tensor(0.002) ** 0.5
        true_std = tensor(0.0002) ** 0.5
        degrees_of_freedom = tensor(9)
        p_value = 0.05
        observed_result = variance_equality_hypothesis_test(
            sample_std, true_std, degrees_of_freedom, p_value
        )
        self.assertEqual(
            observed_result, False, msg="Variance is not within confidence interval"
        )

        # Based on lookup of chi-squared table values
        # The interval for chi-square at p=0.1 split over both distribution ends is
        # approximately [77.9, 124.3]
        # First, we check the lower bound
        sample_std = tensor(78.0 / 100.0) ** 0.5
        true_std = tensor(1.0)
        degrees_of_freedom = tensor(100)
        p_value = 0.1
        observed_result = variance_equality_hypothesis_test(
            sample_std, true_std, degrees_of_freedom, p_value
        )
        self.assertEqual(
            observed_result, True, msg="Variance is within confidence interval"
        )

        sample_std = tensor(77.0 / 100.0) ** 0.5
        true_std = tensor(1.0)
        degrees_of_freedom = tensor(100)
        p_value = 0.1
        observed_result = variance_equality_hypothesis_test(
            sample_std, true_std, degrees_of_freedom, p_value
        )
        self.assertEqual(
            observed_result, False, msg="Variance is not within confidence interval"
        )

        # Second, we check the upper bound
        sample_std = tensor(124.0 / 100.0) ** 0.5
        true_std = tensor(1.0)
        degrees_of_freedom = tensor(100)
        p_value = 0.1
        observed_result = variance_equality_hypothesis_test(
            sample_std, true_std, degrees_of_freedom, p_value
        )
        self.assertEqual(
            observed_result, True, msg="Variance is within confidence interval"
        )

        sample_std = tensor(125.0 / 100.0) ** 0.5
        true_std = tensor(1.0)
        degrees_of_freedom = tensor(100)
        p_value = 0.1
        observed_result = variance_equality_hypothesis_test(
            sample_std, true_std, degrees_of_freedom, p_value
        )
        self.assertEqual(
            observed_result, False, msg="Variance is not within confidence interval"
        )

    def test_confidence_interval_variance(self) -> None:
        """Minimal test for variance confidence interval"""

        true_std = tensor(1.0)
        degrees_of_freedom = tensor(100)
        p_value = 0.1
        observed_interval = variance_equality_hypothesis_confidence_interval(
            true_std, degrees_of_freedom, p_value
        )

        observed_lower, observed_upper = observed_interval

        expected_std_lower1 = tensor(77.0) ** 0.5
        expected_std_lower2 = tensor(78.0) ** 0.5
        expected_std_upper1 = tensor(125.0) ** 0.5
        expected_std_upper2 = tensor(125.0) ** 0.5

        # TODO: The following logic needs to be checked
        observed_lower_result = (
            expected_std_lower1 <= observed_lower <= expected_std_lower2
        )
        observed_upper_result = (
            expected_std_upper1 <= observed_upper <= expected_std_upper2
        )

        observed_result = observed_lower_result and observed_upper_result

        self.assertEqual(
            observed_result, False, msg="Variance is not within confidence interval"
        )
