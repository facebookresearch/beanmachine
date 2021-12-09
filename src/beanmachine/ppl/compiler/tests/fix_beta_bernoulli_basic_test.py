# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare original and conjugate prior transformed
   Beta-Bernoulli model"""

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Beta


class HeadsRateModel(object):
    """Original, untransformed model"""

    @bm.random_variable
    def theta(self):
        return Beta(2.0, 2.0)

    @bm.random_variable
    def y(self, i):
        return Bernoulli(self.theta())

    def run(self):
        queries = [self.theta()]
        observations = {
            self.y(0): tensor(0.0),
            self.y(1): tensor(0.0),
            self.y(2): tensor(1.0),
            self.y(3): tensor(0.0),
        }
        num_samples = 1000
        bmg = BMGInference()
        skip_optimizations = set()
        posterior = bmg.infer(
            queries, observations, num_samples, 1, skip_optimizations=skip_optimizations
        )
        bmg_graph = bmg.to_dot(
            queries, observations, num_samples, skip_optimizations=skip_optimizations
        )
        theta_samples = posterior[self.theta()][0]
        return theta_samples, bmg_graph


class HeadsRateModelTransformed(object):
    """Conjugate Prior Transformed model"""

    @bm.random_variable
    def theta(self):
        return Beta(2.0, 2.0)

    @bm.random_variable
    def y(self, i):
        return Bernoulli(self.theta())

    @bm.random_variable
    def theta_transformed(self):
        # Analytical posterior Beta(alpha + sum y_i, beta + n - sum y_i)
        return Beta(2.0 + 1.0, 2.0 + (4.0 - 1.0))

    def run(self):
        # queries = [self.theta()]
        queries_transformed = [self.theta_transformed()]
        # observations = {
        #     self.y(0): tensor(0.0),
        #     self.y(1): tensor(0.0),
        #     self.y(2): tensor(1.0),
        #     self.y(3): tensor(0.0),
        # }
        observations_transformed = {}
        num_samples = 1000
        bmg = BMGInference()
        # posterior = bmg.infer(queries, observations, num_samples)
        posterior_transformed = bmg.infer(
            queries_transformed, observations_transformed, num_samples
        )
        # theta_samples = posterior[self.theta](0)
        bmg_graph = bmg.to_dot(queries_transformed, observations_transformed)
        theta_samples_transformed = posterior_transformed[self.theta_transformed()][0]
        return theta_samples_transformed, bmg_graph


class HeadsRateModelTest(unittest.TestCase):
    def test_beta_bernoulli_conjugate_graph(self) -> None:
        _, heads_rate_model_graph = HeadsRateModel().run()
        _, heads_rate_model_transformed_graph = HeadsRateModelTransformed().run()

        self.assertEqual(heads_rate_model_graph, heads_rate_model_transformed_graph)
