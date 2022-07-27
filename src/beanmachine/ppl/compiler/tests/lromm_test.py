# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compilation test of Todd's Linear Regression Outliers Marginalized model"""
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.distributions.unit import Unit
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import logaddexp, tensor
from torch.distributions import Beta, Gamma, Normal


_x_obs = tensor([0, 3, 9])
_y_obs = tensor([33, 68, 34])
_err_obs = tensor([3.6, 3.9, 2.6])


@bm.random_variable
def beta_0():
    return Normal(0, 10)


@bm.random_variable
def beta_1():
    return Normal(0, 10)


@bm.random_variable
def sigma_out():
    return Gamma(1, 1)


@bm.random_variable
def theta():
    return Beta(2, 5)


@bm.random_variable
def y():
    mu = beta_0() + beta_1() * _x_obs
    ns = Normal(mu, sigma_out())
    ne = Normal(mu, _err_obs)
    log_likelihood_outlier = theta().log() + ns.log_prob(_y_obs)
    log_likelihood = (1 - theta()).log() + ne.log_prob(_y_obs)
    return Unit(logaddexp(log_likelihood_outlier, log_likelihood))


class LROMMTest(unittest.TestCase):
    def test_lromm_to_dot(self) -> None:
        self.maxDiff = None
        queries = [beta_0(), beta_1(), sigma_out(), theta()]
        observations = {y(): _y_obs}

        # TODO: This test demonstrates that we crash when attempting
        # to compile a model with an unsupported distribution that
        # does not come from the torch distributions module.

        with self.assertRaises(RuntimeError) as ex:
            BMGInference().to_dot(queries, observations)
        expected = """
Could not infer dtype of LogAddExpNode
        """
        observed = str(ex.exception)
        self.assertEqual(observed.strip(), expected.strip())
