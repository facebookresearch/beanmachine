# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end test of realistic coin flip model"""
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Beta


@bm.random_variable
def beta():
    return Beta(2.0, 2.0)


@bm.random_variable
def flip(n):
    return Bernoulli(beta())


class CoinFlipTest(unittest.TestCase):
    def test_inference(self) -> None:
        """test_inference from coin_flip_test.py"""

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
        queries = [beta()]
        observations = {
            flip(0): tensor(0.0),
            flip(1): tensor(0.0),
            flip(2): tensor(1.0),
            flip(3): tensor(0.0),
        }
        num_samples = 1000
        inference = BMGInference()
        mcsamples = inference.infer(queries, observations, num_samples)
        samples = mcsamples[beta()]
        observed = samples.mean()
        expected = 0.37
        self.assertAlmostEqual(first=observed, second=expected, delta=0.05)
