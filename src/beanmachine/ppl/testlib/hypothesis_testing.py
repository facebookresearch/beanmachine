# Copyright (c) Facebook, Inc. and its affiliates.
import scipy.stats as stats
from numpy import sqrt
from torch import Tensor, abs, max, min, prod, tensor


# This module defines the function `mean_equality_hypothesis_test` which
# we would like to evaluate for use with conjugate pair tests. The definition
# of the function is short (less than 20 lines) and most of the code is a
# test that checks that validates its p-value semantics. 100 batches of
# 1000 samples of 1000 elements are used for this basic test.

# The hypothesis test

# Inverse of CDF of normal distribution at given probability
inverse_normal_cdf = stats.norm.ppf


# The hypothesis test proper
def mean_equality_hypothesis_confidence_interval(
    true_mean: Tensor, true_std: Tensor, sample_size: Tensor, p_value
):
    """Test for the null hypothesis that the mean of a Gaussian
    distribution is within the central 1 - p-value confidence
    interval (CI) for a sample of size sample_size. An adjustment
    just takes into account that we do the test pointwise independently
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


def mean_equality_hypothesis_test(
    sample_mean: Tensor,
    true_mean: Tensor,
    true_std: Tensor,
    sample_size: Tensor,
    p_value,
):
    """Test for the null hypothesis that the mean of a Gaussian
    distribution is within the central 1 - p-value confidence
    interval (CI) for a sample of size sample_size. An adjustment
    just takes into account that we do the test pointwise independently
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
