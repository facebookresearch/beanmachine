#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import exp
from torch.distributions import Normal


@bm.random_variable
def X():
    return Normal(0.0, 3.0)


@bm.random_variable
def Y():
    return Normal(loc=0.0, scale=exp(X() * 0.5))


class NealsFunnelTest(unittest.TestCase):
    def test_neals_funnel(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [X(), Y()]
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=3.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Query];
  N05[label=0.5];
  N06[label="*"];
  N07[label=Exp];
  N08[label=Normal];
  N09[label=Sample];
  N10[label=Query];
  N00 -> N02;
  N00 -> N08;
  N01 -> N02;
  N02 -> N03;
  N03 -> N04;
  N03 -> N06;
  N05 -> N06;
  N06 -> N07;
  N07 -> N08;
  N08 -> N09;
  N09 -> N10;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
