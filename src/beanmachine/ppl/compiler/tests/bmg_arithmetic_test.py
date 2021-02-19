#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# BM -> BMG compiler arithmetic tests

import unittest

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import Bernoulli, Beta, HalfCauchy, Normal


@bm.random_variable
def beta():
    return Beta(2.0, 2.0)


@bm.random_variable
def norm():
    return Normal(0.0, 1.0)


@bm.random_variable
def hc():
    return HalfCauchy(1.0)


@bm.functional
def expm1_prob():
    return beta().expm1()


@bm.functional
def expm1_real():
    return torch.expm1(norm())


@bm.functional
def expm1_negreal():
    return torch.Tensor.expm1(-hc())


@bm.functional
def logistic_prob():
    return beta().sigmoid()


@bm.functional
def logistic_real():
    return torch.sigmoid(norm())


@bm.functional
def logistic_negreal():
    return torch.Tensor.sigmoid(-hc())


@bm.random_variable
def ordinary_arithmetic(n):
    return Bernoulli(torch.tensor(0.5) + torch.log(torch.exp(n * torch.tensor(0.1))))


@bm.random_variable
def stochastic_arithmetic():
    s = 0.0
    for n in [0, 1]:
        s = s + torch.log(torch.tensor(0.01)) * ordinary_arithmetic(n)
    return Bernoulli(1 - torch.exp(torch.log(torch.tensor(0.99)) + s))


class BMGArithmeticTest(unittest.TestCase):
    def test_bmg_arithmetic_expm1(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot([expm1_prob()], {})
        expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=ToPosReal];
  N4[label=ExpM1];
  N5[label=Query];
  N0 -> N1;
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
}"""
        self.assertEqual(observed.strip(), expected.strip())

        observed = BMGInference().to_dot([expm1_real()], {})
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=ExpM1];
  N5[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
}"""
        self.assertEqual(observed.strip(), expected.strip())

        observed = BMGInference().to_dot([expm1_negreal()], {})
        expected = """
digraph "graph" {
  N0[label=1.0];
  N1[label=HalfCauchy];
  N2[label=Sample];
  N3[label="-"];
  N4[label=ExpM1];
  N5[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
}"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_logistic(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot([logistic_prob()], {})
        expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=ToReal];
  N4[label=Logistic];
  N5[label=Query];
  N0 -> N1;
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
}"""
        self.assertEqual(observed.strip(), expected.strip())

        observed = BMGInference().to_dot([logistic_real()], {})
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Logistic];
  N5[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
}"""
        self.assertEqual(observed.strip(), expected.strip())

        observed = BMGInference().to_dot([logistic_negreal()], {})
        expected = """
digraph "graph" {
  N0[label=1.0];
  N1[label=HalfCauchy];
  N2[label=Sample];
  N3[label="-"];
  N4[label=ToReal];
  N5[label=Logistic];
  N6[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
  N5 -> N6;
}"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_misc_arithmetic(self) -> None:
        self.maxDiff = None
        observed = BMGInference().to_dot([stochastic_arithmetic()], {})
        expected = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0.6000000238418579];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label=-0.010050326585769653];
  N07[label=-4.605170249938965];
  N08[label=0.0];
  N09[label=if];
  N10[label=if];
  N11[label="+"];
  N12[label="+"];
  N13[label=Exp];
  N14[label=complement];
  N15[label=Bernoulli];
  N16[label=Sample];
  N17[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N09;
  N03 -> N04;
  N04 -> N05;
  N05 -> N10;
  N06 -> N12;
  N07 -> N09;
  N07 -> N10;
  N08 -> N09;
  N08 -> N10;
  N09 -> N11;
  N10 -> N11;
  N11 -> N12;
  N12 -> N13;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
  N16 -> N17;
}
"""
        self.assertEqual(observed.strip(), expected.strip())
