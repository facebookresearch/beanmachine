# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end test of forward sampling a multivariate normal via beanstalk"""
import unittest

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import MultivariateNormal


# We have x ~ N([0, 0], I_2) and would like to forward sample it with BMG
@bm.random_variable
def x():
    return MultivariateNormal(torch.zeros(2), torch.eye(2))


queries = [x()]
observations = {}


expected_dot = """
TODO
"""


class MultivariateNormalTest(unittest.TestCase):
    @unittest.skip
    def test_multivariate_normal_(self) -> None:
        self.maxDiff = None
        bmg = BMGInference()
        samples = bmg.infer(queries, observations, 1000)

        hat_E_x = samples[x()].mean()
        self.assertAlmostEqual(first=hat_E_x[0], second=0.0, delta=0.05)
        self.assertAlmostEqual(first=hat_E_x[1], second=0.0, delta=0.05)

    @unittest.skip
    def test_multivariate_normal_to_dot(self) -> None:
        self.maxDiff = None
        bmg = BMGInference()
        observed = bmg.to_dot(queries, observations)
        self.assertEqual(expected_dot.strip(), observed.strip())
