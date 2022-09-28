#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Bernoulli compiler tests

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch.distributions import Bernoulli, Beta


_bern_ext = Bernoulli(0.5)


@bm.random_variable
def bern_1():
    # Distribution created externally to random variable
    return _bern_ext


@bm.random_variable
def bern_2():
    # Distribution created in random variable, named argument
    return Bernoulli(probs=0.25)


@bm.random_variable
def beta():
    return Beta(2.0, 2.0)


@bm.random_variable
def bern_3():
    # Distribution parameterized by another rv
    return Bernoulli(beta())


@bm.random_variable
def bern_4():
    # Bernoullis with constant logits are treated as though we had
    # the probs instead. Notice that this is deduplicated in the graph
    # with Bern(0.5) (of course it is a different sample because it
    # is a different RV).
    return Bernoulli(logits=0.0)


@bm.random_variable
def bern_5():
    # Bernoullis with stochastic logits become a different kind of node.
    return Bernoulli(logits=beta())


class BernoulliTest(unittest.TestCase):
    def test_bernoulli(self) -> None:
        self.maxDiff = None

        queries = [
            bern_1(),
            bern_2(),
            bern_3(),
            bern_4(),
            bern_5(),
        ]
        observations = {}
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=Query];
  N04[label=0.25];
  N05[label=Bernoulli];
  N06[label=Sample];
  N07[label=Query];
  N08[label=2.0];
  N09[label=Beta];
  N10[label=Sample];
  N11[label=Bernoulli];
  N12[label=Sample];
  N13[label=Query];
  N14[label=Sample];
  N15[label=Query];
  N16[label=ToReal];
  N17[label="Bernoulli(logits)"];
  N18[label=Sample];
  N19[label=Query];
  N00 -> N01;
  N01 -> N02;
  N01 -> N14;
  N02 -> N03;
  N04 -> N05;
  N05 -> N06;
  N06 -> N07;
  N08 -> N09;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N10 -> N16;
  N11 -> N12;
  N12 -> N13;
  N14 -> N15;
  N16 -> N17;
  N17 -> N18;
  N18 -> N19;
}
        """
        self.assertEqual(expected.strip(), observed.strip())
