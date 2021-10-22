# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end compiler test for Bayesian Meta-Analysis model"""

import platform
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
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


class BMATest(unittest.TestCase):
    @unittest.skipIf(
        platform.system() in ["Darwin", "Windows"],
        reason="Numerical behavior seems to be different on MacOS/Windows",
    )
    def test_bma_inference(self) -> None:
        queries = [true_value(), bias_size(0), bias_size(1), bias_size(2)]
        observations = {result(x): tensor(x.result) for x in experiments}

        # Eight experiments, four teams, two groups, is very little data to
        # make good inferences from, so we should expect that the inference
        # engine does not get particularly close.

        # The true value is 10.0, but the observations given best match
        # a true value of 8.15.

        expected_true_value = 8.15

        # True exp bias size was 2.10 but observations given best match
        # a exp bias size of 0.70

        expected_exp_bias = 0.70

        # True team bias size was 1.32 but observations given best match
        # a team bias of 1.26

        expected_team_bias = 1.26

        # True group bias size was 1.52 but observations given best match
        # a group bias of 1.50

        expected_group_bias = 1.50

        mcsamples = BMGInference().infer(queries, observations, 1000, 1)

        queries = [true_value(), bias_size(0), bias_size(1), bias_size(2)]

        observed_true_value = mcsamples[true_value()].mean()
        observed_exp_bias = mcsamples[bias_size(0)].mean()
        observed_team_bias = mcsamples[bias_size(1)].mean()
        observed_group_bias = mcsamples[bias_size(2)].mean()

        self.assertAlmostEqual(observed_true_value, expected_true_value, delta=0.1)
        self.assertAlmostEqual(observed_exp_bias, expected_exp_bias, delta=0.1)
        self.assertAlmostEqual(observed_team_bias, expected_team_bias, delta=0.1)
        self.assertAlmostEqual(observed_group_bias, expected_group_bias, delta=0.1)
