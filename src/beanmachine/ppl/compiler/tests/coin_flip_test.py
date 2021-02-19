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
    def test_coin_flip_inference(self) -> None:
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

    def test_coin_flip_to_dot(self) -> None:
        self.maxDiff = None
        queries = [beta()]
        observations = {
            flip(0): tensor(0.0),
            flip(1): tensor(0.0),
            flip(2): tensor(1.0),
            flip(3): tensor(0.0),
        }
        inference = BMGInference()
        observed = inference.to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=Beta];
  N02[label=Sample];
  N03[label=Bernoulli];
  N04[label=Sample];
  N05[label="Observation False"];
  N06[label=Sample];
  N07[label="Observation False"];
  N08[label=Sample];
  N09[label="Observation True"];
  N10[label=Sample];
  N11[label="Observation False"];
  N12[label=Query];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N02 -> N03;
  N02 -> N12;
  N03 -> N04;
  N03 -> N06;
  N03 -> N08;
  N03 -> N10;
  N04 -> N05;
  N06 -> N07;
  N08 -> N09;
  N10 -> N11;
}
        """
        self.assertEqual(observed.strip(), expected.strip())
