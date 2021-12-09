# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit test for NormalEig class"""
import unittest

import torch
from beanmachine.ppl.legacy.inference.proposer.normal_eig import NormalEig
from torch.distributions.multivariate_normal import MultivariateNormal


class NormalEigTest(unittest.TestCase):
    def test_normal_eig(self) -> None:
        covar = torch.Tensor([[1, 0.1, 0], [0.1, 2, 0.5], [0, 0.5, 3]])
        evals, evecs = torch.linalg.eigh(covar)
        mean = torch.Tensor([1.0, 3.5, -1.2])
        # we want to test that both distributions are identical
        ref_dist = MultivariateNormal(mean, covar)
        test_dist = NormalEig(mean, evals, evecs)
        # density at the mean should be equal
        self.assertAlmostEqual(
            ref_dist.log_prob(mean).item(), test_dist.log_prob(mean).item(), 2
        )
        # density at a random sample should also be equal
        val = test_dist.sample()
        self.assertEqual(val.shape, torch.Size([3]))
        self.assertAlmostEqual(
            ref_dist.log_prob(val).item(), test_dist.log_prob(val).item(), 2
        )
        # test that the empirical mean is correct
        emp_mean = sum(test_dist.sample() for _ in range(10000)) / 10000
        self.assertTrue(((mean - emp_mean).abs() < 0.1).all())
        # test that the empirical covariance is correct

        def outerprod(x):
            return torch.ger(x, x)

        emp_covar = (
            sum(outerprod(test_dist.sample() - mean) for _ in range(2000)) / 2000
        )
        self.assertTrue(((covar - emp_covar).abs() < 0.2).all())
