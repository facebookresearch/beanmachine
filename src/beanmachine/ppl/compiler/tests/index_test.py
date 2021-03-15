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
    return Normal(0.0, tensor([1.5, 2.5])[flip()])


@bm.random_variable
def neg_real():
    return Bernoulli(tensor([-1.5, -2.5])[flip()].exp())


@bm.random_variable
def prob():
    return Bernoulli(tensor([0.5, 0.25])[flip()])


@bm.random_variable
def natural():
    return Binomial(tensor([2, 3])[flip()], 0.75)


class IndexTest(unittest.TestCase):
    def test_index_constant_vector_stochastic_index(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot(
            [pos_real(), real(), neg_real(), prob(), natural()],
            {},
        )
        expected = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0.0];
  N04[label="[1.5,2.5]"];
  N05[label=1];
  N06[label=0];
  N07[label=if];
  N08[label=index];
  N09[label=Normal];
  N10[label=Sample];
  N11[label=Query];
  N12[label="[1.5,-1.5]"];
  N13[label=index];
  N14[label=1.0];
  N15[label=Normal];
  N16[label=Sample];
  N17[label=Query];
  N18[label="[-1.5,-2.5]"];
  N19[label=index];
  N20[label=Exp];
  N21[label=Bernoulli];
  N22[label=Sample];
  N23[label=Query];
  N24[label="[0.5,0.25]"];
  N25[label=index];
  N26[label=Bernoulli];
  N27[label=Sample];
  N28[label=Query];
  N29[label="[2,3]"];
  N30[label=index];
  N31[label=0.75];
  N32[label=Binomial];
  N33[label=Sample];
  N34[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N07;
  N03 -> N09;
  N04 -> N08;
  N05 -> N07;
  N06 -> N07;
  N07 -> N08;
  N07 -> N13;
  N07 -> N19;
  N07 -> N25;
  N07 -> N30;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N12 -> N13;
  N13 -> N15;
  N14 -> N15;
  N15 -> N16;
  N16 -> N17;
  N18 -> N19;
  N19 -> N20;
  N20 -> N21;
  N21 -> N22;
  N22 -> N23;
  N24 -> N25;
  N25 -> N26;
  N26 -> N27;
  N27 -> N28;
  N29 -> N30;
  N30 -> N32;
  N31 -> N32;
  N32 -> N33;
  N33 -> N34;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
