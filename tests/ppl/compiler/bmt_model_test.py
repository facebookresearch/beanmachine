# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compilation test of Todd's Bayesian Multiple Testing model"""
import unittest

import beanmachine.ppl as bm
import torch.distributions as dist
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor


@bm.random_variable
def theta():
    return dist.Beta(2, 5)


@bm.random_variable
def sigma():
    return dist.HalfCauchy(5)


@bm.random_variable
def tau():
    return dist.HalfCauchy(5)


@bm.random_variable
def z(i):
    return dist.Bernoulli(theta())


@bm.random_variable
def mu(i):
    return dist.Normal(0, tau())


@bm.random_variable
def x(i):
    return dist.Normal(mu(i) * z(i), sigma())


class BMTModelTest(unittest.TestCase):
    def test_bmt_to_dot(self) -> None:

        self.maxDiff = None
        x_obs = [3.0, -0.75, 2.0, -0.3]
        n_obs = len(x_obs)
        queries = (
            [theta(), sigma(), tau()]
            + [z(i) for i in range(n_obs)]
            + [mu(i) for i in range(n_obs)]
        )
        observations = {x(i): tensor(v) for i, v in enumerate(x_obs)}
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label=5.0];
  N01[label=HalfCauchy];
  N02[label=Sample];
  N03[label=0.0];
  N04[label=Normal];
  N05[label=Sample];
  N06[label=2.0];
  N07[label=Beta];
  N08[label=Sample];
  N09[label=Bernoulli];
  N10[label=Sample];
  N11[label=Sample];
  N12[label=if];
  N13[label=Normal];
  N14[label=Sample];
  N15[label="Observation 3.0"];
  N16[label=Sample];
  N17[label=Sample];
  N18[label=if];
  N19[label=Normal];
  N20[label=Sample];
  N21[label="Observation -0.75"];
  N22[label=Sample];
  N23[label=Sample];
  N24[label=if];
  N25[label=Normal];
  N26[label=Sample];
  N27[label="Observation 2.0"];
  N28[label=Sample];
  N29[label=Sample];
  N30[label=if];
  N31[label=Normal];
  N32[label=Sample];
  N33[label="Observation -0.30000001192092896"];
  N34[label=Query];
  N35[label=Query];
  N36[label=Query];
  N37[label=Query];
  N38[label=Query];
  N39[label=Query];
  N40[label=Query];
  N41[label=Query];
  N42[label=Query];
  N43[label=Query];
  N44[label=Query];
  N00 -> N01;
  N00 -> N07;
  N01 -> N02;
  N01 -> N11;
  N02 -> N04;
  N02 -> N36;
  N03 -> N04;
  N03 -> N12;
  N03 -> N18;
  N03 -> N24;
  N03 -> N30;
  N04 -> N05;
  N04 -> N16;
  N04 -> N22;
  N04 -> N28;
  N05 -> N12;
  N05 -> N41;
  N06 -> N07;
  N07 -> N08;
  N08 -> N09;
  N08 -> N34;
  N09 -> N10;
  N09 -> N17;
  N09 -> N23;
  N09 -> N29;
  N10 -> N12;
  N10 -> N37;
  N11 -> N13;
  N11 -> N19;
  N11 -> N25;
  N11 -> N31;
  N11 -> N35;
  N12 -> N13;
  N13 -> N14;
  N14 -> N15;
  N16 -> N18;
  N16 -> N42;
  N17 -> N18;
  N17 -> N38;
  N18 -> N19;
  N19 -> N20;
  N20 -> N21;
  N22 -> N24;
  N22 -> N43;
  N23 -> N24;
  N23 -> N39;
  N24 -> N25;
  N25 -> N26;
  N26 -> N27;
  N28 -> N30;
  N28 -> N44;
  N29 -> N30;
  N29 -> N40;
  N30 -> N31;
  N31 -> N32;
  N32 -> N33;
}
        """
        self.assertEqual(expected.strip(), observed.strip())
