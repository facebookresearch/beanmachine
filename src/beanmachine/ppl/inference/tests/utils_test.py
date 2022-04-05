# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
import torch.testing


@bm.random_variable
def foo():
    return dist.Normal(0.0, 1.0)


def test_set_random_seed():
    def sample_with_seed(seed):
        bm.seed(seed)
        return bm.SingleSiteAncestralMetropolisHastings().infer(
            [foo()], {}, num_samples=20, num_chains=1
        )

    samples1 = sample_with_seed(123)
    samples2 = sample_with_seed(123)
    assert torch.allclose(samples1[foo()], samples2[foo()])


class TestMakeQueriesAndObservations(unittest.TestCase):
    def setUp(self) -> None:
        def a_variable() -> dist.Distribution:
            return dist.Normal(0, 1)

        self.alpha = bm.random_variable(a_variable)

        def b_variable() -> dist.Distribution:
            return dist.Beta(1, 1)

        self.beta = bm.random_variable(b_variable)

    def test_make_queries(self) -> None:
        with self.assertRaises(TypeError):
            bm.inference.utils.make_queries([self.alpha(), self.beta(), torch.ones(2)])

        self.assertEqual(
            bm.inference.utils.make_queries([self.alpha(), self.beta()]),
            [self.alpha(), self.beta()],
        )

    def test_make_observations(self) -> None:
        with self.assertRaises(TypeError):
            bm.inference.utils.make_observations(
                {
                    self.alpha(): torch.ones(2),
                    self.beta(): torch.zeros(2),
                    "c": torch.ones(2),
                }
            )

        expected = {self.alpha(): torch.ones(2), self.beta(): torch.zeros(2)}
        actual = bm.inference.utils.make_observations(
            {self.alpha(): torch.ones(2), self.beta(): torch.zeros(2)}
        )
        self.assertEqual(actual.keys(), expected.keys())
        for k in actual:
            torch.testing.assert_allclose(actual[k], expected[k])
