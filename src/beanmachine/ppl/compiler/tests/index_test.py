#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# BM -> BMG compiler index tests

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Binomial, HalfCauchy, Normal


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


@bm.random_variable
def normal():
    return Normal(0.0, 1.0)


@bm.random_variable
def hc():
    return HalfCauchy(0.0)


@bm.random_variable
def optimize_away_index():
    t = tensor([normal(), hc()])
    return Normal(t[0], t[1])


@bm.functional
def column_index():
    t = tensor([[normal(), hc()], [hc(), normal()]])
    return t[flip()][flip()]


class IndexTest(unittest.TestCase):
    def test_index_constant_vector_stochastic_index(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot(
            [pos_real(), real(), neg_real(), prob(), natural()], {},
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

    def test_index_stochastic_tensor_constant_index(self) -> None:
        self.maxDiff = None

        # Here we demonstrate that we can make a tensor containing graph
        # nodes and index into that with a constant; the indexing operation
        # is optimized out.

        observed = BMGInference().to_dot([optimize_away_index()], {})
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=0.0];
  N5[label=HalfCauchy];
  N6[label=Sample];
  N7[label=Normal];
  N8[label=Sample];
  N9[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N7;
  N4 -> N5;
  N5 -> N6;
  N6 -> N7;
  N7 -> N8;
  N8 -> N9;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_column_index(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot([column_index()], {})
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=0.0];
  N05[label=HalfCauchy];
  N06[label=Sample];
  N07[label=0.5];
  N08[label=Bernoulli];
  N09[label=Sample];
  N10[label=2];
  N11[label=ToReal];
  N12[label=ToMatrix];
  N13[label=1];
  N14[label=0];
  N15[label=if];
  N16[label=ColumnIndex];
  N17[label=index];
  N18[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N03 -> N12;
  N03 -> N12;
  N04 -> N05;
  N05 -> N06;
  N06 -> N11;
  N07 -> N08;
  N08 -> N09;
  N09 -> N15;
  N10 -> N12;
  N10 -> N12;
  N11 -> N12;
  N11 -> N12;
  N12 -> N16;
  N13 -> N15;
  N14 -> N15;
  N15 -> N16;
  N15 -> N17;
  N16 -> N17;
  N17 -> N18;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
