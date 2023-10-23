# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bm_inference import BMInference
from beanmachine.ppl.inference.nuts_inference import GlobalNoUTurnSampler
from torch import tensor
from torch.distributions import Bernoulli, Beta, Normal


@bm.random_variable
def beta():
    return Beta(2.0, 2.0)


@bm.random_variable
def flip(n):
    return Bernoulli(beta())


@bm.random_variable
def flip_2(n):
    return Bernoulli(beta() * 0.5)


@bm.random_variable
def normal(n):
    return Normal(flip_2(n), 1.0)


class CoinFlipTest(unittest.TestCase):
    def test_gen_bm_python_simple(self) -> None:
        self.maxDiff = None
        queries = [beta()]
        observations = {
            flip(0): tensor(0.0),
            flip(1): tensor(0.0),
            flip(2): tensor(1.0),
            flip(3): tensor(0.0),
        }
        observed = BMInference().to_python(
            queries,
            observations,
            num_samples=1000,
            num_chains=1,
            num_adaptive_samples=500,
        )
        expected = """
import beanmachine.ppl as bm
import torch
from beanmachine.ppl.inference.nuts_inference import GlobalNoUTurnSampler
v0 = 2.0
@bm.random_variable
def rv0():
\treturn torch.distributions.Beta(v0, v0)
v1 = rv0()
@bm.random_variable
def rv1(i):
\treturn torch.distributions.Bernoulli(v1.wrapper(*v1.arguments))
v2 = rv1(1,)
v3 = rv1(2,)
v4 = rv1(3,)
v5 = rv1(4,)
opt_queries = [v1]
opt_observations = {v2 : torch.tensor(0.0),v3 : torch.tensor(0.0),v4 : torch.tensor(1.0),v5 : torch.tensor(0.0)}
samples = GlobalNoUTurnSampler().infer(
                opt_queries,
                opt_observations,
                num_samples=1000,
                num_chains=1,
                num_adaptive_samples=500
            )
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed_samples = BMInference().infer(
            queries,
            observations,
            num_samples=1000,
            num_chains=1,
            num_adaptive_samples=500,
        )
        expected_samples = GlobalNoUTurnSampler().infer(
            queries,
            observations,
            num_samples=1000,
            num_chains=1,
            num_adaptive_samples=500,
        )
        observed_mean = observed_samples[beta()].mean()
        expected_mean = expected_samples[beta()].mean()
        self.assertAlmostEqual(expected_mean, observed_mean, delta=0.05)

    def test_gen_bm_python_rv_operations(self) -> None:
        self.maxDiff = None
        queries = [beta(), normal(0), normal(1)]
        observations = {
            flip_2(0): tensor(0.0),
        }
        observed = BMInference().to_python(queries, observations, num_samples=1000)
        expected = """
import beanmachine.ppl as bm
import torch
from beanmachine.ppl.inference.nuts_inference import GlobalNoUTurnSampler
v0 = 2.0
@bm.random_variable
def rv0():
\treturn torch.distributions.Beta(v0, v0)
v1 = rv0()
v2 = 0.5
@bm.functional
def f3():
\treturn torch.multiply(v1.wrapper(*v1.arguments), v2)
@bm.random_variable
def rv1(i):
\treturn torch.distributions.Bernoulli(f3())
v4 = rv1(1,)
@bm.functional
def f5():
\treturn (v4.wrapper(*v4.arguments))
v6 = 1.0
@bm.random_variable
def rv2():
\treturn torch.distributions.Normal(f5(), v6)
v7 = rv2()
v8 = rv1(2,)
@bm.functional
def f9():
\treturn (v8.wrapper(*v8.arguments))
@bm.random_variable
def rv3():
\treturn torch.distributions.Normal(f9(), v6)
v10 = rv3()
opt_queries = [v1,v7,v10]
opt_observations = {v4 : torch.tensor(0.0)}
samples = GlobalNoUTurnSampler().infer(
                opt_queries,
                opt_observations,
                num_samples=1000,
                num_chains=4,
                num_adaptive_samples=0
            )
"""
        self.assertEqual(expected.strip(), observed.strip())
