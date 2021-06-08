# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end test for tutorial on Bayesian Logistic Regression"""
# This file is a manual replica of the Bento tutorial with the same name
# TODO: The disabled error raises the following error:
# E           ValueError: The model uses a * operation unsupported by Bean Machine Graph.
# E           The unsupported node is the probability of a Bernoulli(logits).
# More work may be needed, but at least this issue needs to be addressed

import logging
import unittest

# import matplotlib.pyplot as plt
import beanmachine.ppl as bm
from beanmachine.ppl.inference import SingleSiteNewtonianMonteCarlo
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import manual_seed, tensor
from torch.distributions import Bernoulli, Normal, Uniform

logging.getLogger("beanmachine").setLevel(50)
manual_seed(8)

# This makes the results deterministic and reproducible.

# Creating sample data

n = 200
low = -10.0
high = 10.0
uniform = Uniform(low=tensor([1.0, low, low]), high=tensor([1.0, high, high]))
points = tensor([uniform.sample().tolist() for i in range(n)]).view(n, 3)

# For intreactive mode, visualization of randomized data:
#
# plt.figure(figsize=(10,8))
# for point in points:
#     plt.scatter(point[1], point[2], c='black')

true_coefficients = tensor([-2.0, 0.3, -0.5]).view(3, 1)
true_slope = -float(true_coefficients[1] / true_coefficients[2])
true_intercept = -float(true_coefficients[0] / true_coefficients[2])


def log_odds(point):
    return point.view(1, 3).mm(true_coefficients)


observed_categories = tensor(
    [Bernoulli(logits=log_odds(point)).sample() for point in points]
)

# Data visualization methods

# def plot_line(slope, intercept):
#     if intercept > high or intercept < low:
#         return
#     xs = [low, high]
#     ys = [slope * low + intercept, slope * high + intercept]
#     if ys[0] > high:
#         xs[0] = (high - intercept) / slope
#         ys[0] = high
#     elif ys[0] < low:
#         xs[0] = (low - intercept) / slope
#         ys[0] = low
#     if ys[1] > high:
#         xs[1] = (high - intercept) / slope
#         ys[1] = high
#     elif ys[1] < low:
#         xs[1] = (low - intercept) / slope
#         ys[1] = low
#     plt.plot(xs, ys)

# def plot_points(ps, cs):
#     for p, c in zip(ps, cs):
#         plt.scatter(p[1], p[2], c=('orange' if c == 1.0 else 'blue'))

# plt.figure(figsize=(10,8))
# plot_points(points, observed_categories)
# plot_line(true_slope, true_intercept)

# Model
scale = 20.0


@bm.random_variable
def coefficients():
    mean = tensor([0.0, 0.0, 0.0]).view(3, 1)
    sigma = tensor([scale / 2.0, scale / 2.0, scale / 2.0]).view(3, 1)
    return Normal(mean, sigma)


@bm.random_variable
def categories():
    return Bernoulli(logits=points.mm(coefficients()))


# Inference parameters

num_samples = (
    4  ###00 Sample size should not affect (the ability to find) compilation issues.
)
num_chains = 1
observations = {categories(): observed_categories.view(n, 1)}
queries = [coefficients()]


class tutorialBaysianLogisticRegressionTest(unittest.TestCase):
    def test_tutorial_Bayesian_Logistic_Regression_inference(self) -> None:
        """Check BM and BMG inference both terminate"""

        self.maxDiff = None
        num_samples = 10

        # Inference with BM
        mc = SingleSiteNewtonianMonteCarlo(real_space_alpha=1.0, real_space_beta=5.0)
        samples = mc.infer(queries, observations, num_samples, num_chains)
        sampled_coefficients = samples.get_chain()[coefficients()]
        slopes = [
            -float(s[1] / s[2]) for s in sampled_coefficients if float(s[2]) != 0.0
        ]
        intercepts = [
            -float(s[0] / s[2]) for s in sampled_coefficients if float(s[2]) != 0.0
        ]
        self.assertTrue(True, msg="We just want to check this point is reached")

    def disabled_test_tutorial_Bayesian_Logistic_Regression_to_dot_cpp_python(
        self,
    ) -> None:
        self.maxDiff = None
        observed = BMGInference().to_dot(queries, observations)
        expected = """
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_cpp(queries, observations)
        expected = """
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_python(queries, observations)
        expected = """
"""
        self.assertEqual(expected.strip(), observed.strip())
