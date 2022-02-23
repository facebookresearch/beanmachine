# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end test of realistic linear regression model"""

# This is copied from bento workbook N140350, simplified, and
# modified to use BMG inference.

import pytest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import Normal, Uniform


@bm.random_variable
def theta_0():
    return Normal(0.0, 1.0)


@bm.random_variable
def theta_1():
    return Normal(0.0, 1.0)


# @bm.random_variable
# def error():
#     return Uniform(0.0, 1.0)


@bm.random_variable
def x(i):
    return Normal(0.0, 1.0)


@bm.random_variable
def y(i):
    return Normal(theta_0() + theta_1() * x(i), 1.0)


def test_linear_regression_inference(benchmark) -> None:
    # We start by generating some test data; we can use the inference engine
    # as a random number generator if we have no observations.
    #
    # Generate an intercept, slope, and n points such that:
    #
    # y(i) = theta_0() + theta_1() * x(i) + some normal error

    n = 100
    x_rvs = [x(i) for i in range(n)]
    y_rvs = [y(i) for i in range(n)]

    test_samples = BMGInference().infer(
        [theta_0(), theta_1()] + x_rvs + y_rvs, {}, 1
    )
    true_intercept = test_samples[theta_0()][0].item()
    true_slope = test_samples[theta_1()][0].item()
    points = [(test_samples[x(i)][0], test_samples[y(i)][0]) for i in range(n)]

    # We are only pseudo-random here so we should always get the same result.
    expected_true_intercept = -0.05
    expected_true_slope = -0.44

    assert abs(true_intercept - expected_true_intercept) < 0.1
    assert abs(true_slope - expected_true_slope) < 0.5

    # If we then run inference when observing the set of (x, y) points we generated,
    # what slope and intercept do we infer? It should be close to the actual values.

    observed_xs = {x(i): points[i][0] for i in range(n)}
    observed_ys = {y(i): points[i][1] for i in range(n)}
    observations = {**observed_xs, **observed_ys}
    queries = [theta_0(), theta_1()]
    num_samples = 1000

    samples = benchmark(BMGInference().infer, queries, observations, num_samples)

    inferred_intercept = samples[theta_0()].mean()
    inferred_slope = samples[theta_1()].mean()

    expected_inferred_int = -0.05
    expected_inferred_slope = -0.33

    assert abs(inferred_intercept - expected_inferred_int) < 0.2
    assert abs(inferred_slope - expected_inferred_slope) < 0.5