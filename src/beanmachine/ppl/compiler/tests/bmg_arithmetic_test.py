#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# BM -> BMG compiler arithmetic tests

import unittest

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import Beta, HalfCauchy, Normal


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
