#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# BM -> BMG compiler index tests

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Binomial, Normal

# Simplexes are tested in dirichlet_test.py
# TODO: Test array of Booleans


@bm.random_variable
def flip():
    return Bernoulli(0.5)


@bm.random_variable
def real():
    return Normal(tensor([1.5, -1.5])[flip()], 1.0)


@bm.random_variable
def pos_real():
    return Normal(0.0, tensor([1.5, 1.5])[flip()])


@bm.random_variable
def neg_real():
    return Bernoulli(tensor([-1.5, -1.5])[flip()].log())


@bm.random_variable
def prob():
    return Bernoulli(tensor([0.5, 0.25])[flip()])


@bm.functional
def natural():
    return Binomial(0.5, tensor([2, 3])[flip()])


class IndexTest(unittest.TestCase):
    def test_index_constant_vector_stochastic_index(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot([pos_real()], {})
        expected = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0.0];
  N04[label="[1.5,1.5]"];
  N05[label=1];
  N06[label=0];
  N07[label=if];
  N08[label=index];
  N09[label=Normal];
  N10[label=Sample];
  N11[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N07;
  N03 -> N09;
  N04 -> N08;
  N05 -> N07;
  N06 -> N07;
  N07 -> N08;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # TODO: real
        # TODO: neg_real
        # TODO: prob
        # TODO: natural
        # TODO: Boolean
