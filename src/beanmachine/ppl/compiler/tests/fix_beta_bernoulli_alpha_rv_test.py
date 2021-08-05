# Copyright (c) Facebook, Inc. and its affiliates.
"""Beta-Bernoulli model conjugacy transformation check when
   hyperparameter is a random variable."""

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.examples.conjugate_models.beta_bernoulli import BetaBernoulliModel
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Beta


class BetaBernoulliAlphaRVModel(BetaBernoulliModel):
    def __init__(self):
        self.beta_ = 2.0

    @bm.random_variable
    def alpha(self):
        return Beta(5.0, 1.0)

    @bm.random_variable
    def theta(self):
        return Beta(self.alpha(), self.beta_)


class BetaBernoulliWithAlphaAsRVConjugateTest(unittest.TestCase):
    def test_conjugate_graph(self) -> None:
        """
        Test to check that Beta-Bernoulli conjugate transformation
        is not be applied when parameters of Beta distribution are
        random variables.
        """
        self.maxDiff = None
        model = BetaBernoulliAlphaRVModel()
        queries = [model.theta()]
        observations = {
            model.y(0): tensor(0.0),
            model.y(1): tensor(0.0),
            model.y(2): tensor(1.0),
            model.y(3): tensor(0.0),
        }
        num_samples = 1000
        bmg = BMGInference()

        # This is the model before beta-bernoulli conjugate rewrite is applied
        expected_bmg = bmg.to_dot(queries, observations, num_samples)

        # This is the model after beta-bernoulli conjugate rewrite is applied
        skip_optimizations = set()
        observed_bmg = bmg.to_dot(
            queries, observations, num_samples, skip_optimizations=skip_optimizations
        )

        self.assertEqual(expected_bmg.strip(), observed_bmg.strip())
