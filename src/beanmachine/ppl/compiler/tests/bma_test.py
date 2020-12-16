# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end test of realistic coin flip model"""
import sys
import unittest

from beanmachine.ppl.compiler.bm_to_bmg import infer


source = """
import beanmachine.ppl as bm
import torch
from torch import tensor
from torch.distributions import HalfCauchy, Normal, StudentT

class Group:
    level = 2

class Team:
    group: Group
    level = 1

    def __init__(self, group: Group):
        self.group = group

class Experiment:
    result: float
    stddev: float
    team: Team
    level = 0

    def __init__(self, result: float, stddev: float, team: Team):
        self.result = result
        self.stddev = stddev
        self.team = team

group1 = Group()
group2 = Group()
team1 = Team(group1)
team2 = Team(group1)
team3 = Team(group2)
team4 = Team(group2)

# I generated sample values for everything that conform to this model:

# * true value is 10.0
# * experiment bias stddev is 2.10
# * team bias stddev is 1.34
# * group bias stddev is 1.52
# * experiment biases are
#   -0.82, -1.58, 0.45, 0.23, 1.30, -1.25, -1.26, -1.14
# * team biases are -2.19, -1.41, -0.26, 1.16
# * group biases are 0.19, 0.79
# * experiment stddevs and results are given below.

experiments = [
    Experiment(7.36, 0.3, team1),
    Experiment(6.47, 0.5, team1),
    Experiment(8.87, 0.2, team2),
    Experiment(9.17, 1.0, team2),
    Experiment(11.19, 2.4, team3),
    Experiment(10.30, 1.5, team3),
    Experiment(11.06, 0.9, team4),
    Experiment(10.74, 0.8, team4),
]

@bm.random_variable
def true_value():
    return StudentT(1.0)

@bm.random_variable
def bias_size(level):
    return HalfCauchy(1.0)

@bm.random_variable
def node_bias(node):
    return Normal(0, bias_size(node.level))

@bm.random_variable
def result(experiment):
    mean = (
        true_value()
        + node_bias(experiment)
        + node_bias(experiment.team)
        + node_bias(experiment.team.group)
    )
    return Normal(mean, experiment.stddev)

queries = [true_value(), bias_size(0), bias_size(1), bias_size(2)]
observations = {}
for x in experiments:
    observations[result(x)] = tensor(x.result)
"""

# Eight experiments, four teams, two groups, is very little data to
# make good inferences from, so we should expect that the inference
# engine does not get particularly close.

# The true value is 10.0, but the observations given best match
# a true value of 8.15.

expected_true_value = 8.15

# True exp bias size was 2.10 but observations given best match
# a exp bias size of 0.42

expected_exp_bias = 0.42

# True team bias size was 1.32 but observations given best match
# a team bias of 1.03

expected_team_bias = 1.03

# True group bias size was 1.52 but observations given best match
# a group bias of 2.17

expected_group_bias = 2.17


def average(items):
    return sum(items) / len(items)


class BMATest(unittest.TestCase):
    @unittest.skipIf(
        sys.platform.startswith("darwin"),
        reason="Numerical behavior seems to be different on MacOS",
    )
    def test_bma_inference(self) -> None:
        """test_bma_inference from bma_test.py"""

        # We've got a prior on the coin of Beta(2,2), so it is most
        # likely to be actually fair, but still with some probability
        # of being unfair in either direction.
        #
        # We flip the coin four times and get heads 25% of the time,
        # so this is some evidence that the true fairness of the coin is
        # closer to 25% than 50%.
        #
        # We sample 1000 times from the posterior and take the average;
        # it should come out that the true fairness is now most likely
        # to be around 37%.

        self.maxDiff = None
        observed = infer(source)
        observed_true_value = average([x[0] for x in observed])
        observed_exp_bias = average([x[1] for x in observed])
        observed_team_bias = average([x[2] for x in observed])
        observed_group_bias = average([x[3] for x in observed])

        self.assertAlmostEqual(observed_true_value, expected_true_value, delta=0.1)
        self.assertAlmostEqual(observed_exp_bias, expected_exp_bias, delta=0.1)
        self.assertAlmostEqual(observed_team_bias, expected_team_bias, delta=0.1)
        self.assertAlmostEqual(observed_group_bias, expected_group_bias, delta=0.1)
