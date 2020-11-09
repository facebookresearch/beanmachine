# Copyright (c) Facebook, Inc. and its affiliates.
import scipy.stats as stats
from numpy import sqrt
from torch import Tensor, abs, max, min, prod, tensor


# This module defines hypothesis tests for equal means and equal variance

# Helper functions:

# Inverse of CDF of normal distribution at given probability
inverse_normal_cdf = stats.norm.ppf

# Inverse of CDF of chi-squared distribution at given probability
def inverse_chi2_cdf(df, p):
    return stats.chi2(df).ppf(p)


# Hypothesis test for equality of sample mean to a true mean
def mean_equality_hypothesis_test(
    sample_mean: Tensor,
    true_mean: Tensor,
    true_std: Tensor,
    sample_size: Tensor,
    p_value,
):
    """Test for the null hypothesis that the mean of a Gaussian
    distribution is within the central 1 - p-value confidence
    interval (CI) for a sample of size sample_size. We also apply an adjustment
    that takes into account that we do the test pointwise independently
    for each element of the tensor. This is basically the Dunn-Šidák
    correction,
    https://en.wikipedia.org/wiki/%C5%A0id%C3%A1k_correction"""
    if min(sample_size).item() <= 0:
        return False
    dimensions = prod(tensor(Tensor.size(true_mean))).item()
    if dimensions == 0:
        return False
    if max(true_std <= 0).item():
        return False
    adjusted_p_value = 1 - (1 - p_value) ** (1.0 / dimensions)
    test_result = max(
        abs(sample_mean - true_mean) * sqrt(sample_size) / true_std
    ).item() <= inverse_normal_cdf(1 - adjusted_p_value / 2)
    return test_result


# The following function explicitly constructs a confidence interval.
# This provides an alternative way for performing the hypothesis test,
# but which also makes reporting test failures easier.
def mean_equality_hypothesis_confidence_interval(
    true_mean: Tensor, true_std: Tensor, sample_size: Tensor, p_value
):
    """Test for the null hypothesis that the mean of a Gaussian
    distribution is within the central 1 - p-value confidence
    interval (CI) for a sample of size sample_size. We also apply an adjustment
    that takes into account that we do the test pointwise independently
    for each element of the tensor. This is basically the Dunn-Šidák
    correction,
    https://en.wikipedia.org/wiki/%C5%A0id%C3%A1k_correction"""
    if min(sample_size).item() <= 0:
        return False
    dimensions = prod(tensor(Tensor.size(true_mean))).item()
    if dimensions == 0:
        return None
    if max(true_std == 0).item():
        return None
    adjusted_p_value = 1 - (1 - p_value) ** (1.0 / dimensions)
    bound_std = true_std / sqrt(sample_size)
    z_score = inverse_normal_cdf(1 - adjusted_p_value / 2)
    # TODO: We use z_{1-alpha} instead of -z_alpha for compatibility
    # with mean_equality_hypothesis_test. Ideally, both should be
    # changed to use the unmodified bounds. In any case, the two
    # functions should be matched for consistency
    lower_bound = true_mean - bound_std * z_score
    upper_bound = true_mean + bound_std * z_score
    return lower_bound, upper_bound


# Hypothesis test for equality of sample variance to a true variance
def variance_equality_hypothesis_test(
    sample_std: Tensor,
    true_std: Tensor,
    degrees_of_freedom: Tensor,
    p_value,
):
    """Test for the null hypothesis that the variance of a Gaussian
    distribution is within the central 1 - p-value confidence
    interval (CI) for a sample of effective sample size (ESS)
    degrees_of_freedom. We also apply an adjustment that takes
    into account that we do the test pointwise independently
    for each element of the tensor. This is basically the Dunn-Šidák
    correction,
    https://en.wikipedia.org/wiki/%C5%A0id%C3%A1k_correction"""
    if min(degrees_of_freedom).item() <= 0:
        return False
    dimensions = prod(tensor(Tensor.size(true_std))).item()
    if dimensions == 0:
        return False
    if max(true_std <= 0).item():
        return False
    adjusted_p_value = 1 - (1 - p_value) ** (1.0 / dimensions)
    test_statistic = degrees_of_freedom * (sample_std / true_std) ** 2
    lower_bound = inverse_chi2_cdf(degrees_of_freedom, adjusted_p_value / 2)
    upper_bound = inverse_chi2_cdf(degrees_of_freedom, 1 - adjusted_p_value / 2)
    lower_bound_result = lower_bound <= min(test_statistic).item()
    upper_bound_result = max(test_statistic).item() <= upper_bound
    test_result = lower_bound_result and upper_bound_result
    return test_result


# The following function explicitly constructs a confidence interval.
# This provides an alternative way for performing the hypothesis test,
# but which also makes reporting test failures easier.
def variance_equality_hypothesis_confidence_interval(
    true_std: Tensor, degrees_of_freedom: Tensor, p_value
):
    """Test for the null hypothesis that the mean of a Gaussian
    distribution is within the central 1 - p-value confidence
    interval (CI) for a sample of size sample_size. We also apply an adjustment
    that takes into account that we do the test pointwise independently
    for each element of the tensor. This is basically the Dunn-Šidák
    correction,
    https://en.wikipedia.org/wiki/%C5%A0id%C3%A1k_correction"""
    if min(degrees_of_freedom).item() <= 0:
        return False
    dimensions = prod(tensor(Tensor.size(true_std))).item()
    if dimensions == 0:
        return None
    if max(true_std == 0).item():
        return None
    adjusted_p_value = 1 - (1 - p_value) ** (1.0 / dimensions)
    bound_std = true_std / sqrt(degrees_of_freedom)
    z_score = inverse_normal_cdf(1 - adjusted_p_value / 2)
    # TODO: We use z_{1-alpha} instead of -z_alpha for compatibility
    # with mean_equality_hypothesis_test. Ideally, both should be
    # changed to use the unmodified bounds. In any case, the two
    # functions should be matched for consistency
    lower_bound = true_std - bound_std * z_score
    upper_bound = true_std + bound_std * z_score
    return lower_bound, upper_bound
