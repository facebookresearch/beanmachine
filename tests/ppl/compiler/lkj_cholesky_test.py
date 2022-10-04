#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# LKJ Cholesky compiler tests

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch.distributions import HalfNormal, LKJCholesky


@bm.random_variable
def lkj_chol_1():
    return LKJCholesky(3, 2.0)


@bm.random_variable
def lkj_chol_2():
    # Distribution created in random variable, named argument
    return LKJCholesky(concentration=2.0, dim=3)


@bm.random_variable
def half_normal():
    return HalfNormal(1.0)


@bm.random_variable
def lkj_chol_3():
    # Distribution parameterized by another rv
    return LKJCholesky(3, half_normal())


@bm.random_variable
def bad_lkj_chol_1():
    # LKJ Cholesky must have dimension at least 2
    return LKJCholesky(1, half_normal())


@bm.random_variable
def bad_lkj_chol_2():
    # LKJ Cholesky must have natural dimension
    return LKJCholesky(3.5, half_normal())


@bm.random_variable
def bad_lkj_chol_3():
    # LKJ Cholesky must have a positive concentration value
    return LKJCholesky(3, -2.0)


class LKJCholeskyTest(unittest.TestCase):

    expected_simple_case = """
digraph "graph" {
  N0[label=3];
  N1[label=2.0];
  N2[label=LKJCholesky];
  N3[label=Sample];
  N4[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
}
        """.strip()

    expected_random_parameter_case = """
digraph "graph" {
  N0[label=1.0];
  N1[label=HalfNormal];
  N2[label=Sample];
  N3[label=3];
  N4[label=LKJCholesky];
  N5[label=Sample];
  N6[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N4;
  N3 -> N4;
  N4 -> N5;
  N5 -> N6;
}
    """.strip()

    def test_lkj_chol_1(self) -> None:
        observed = BMGInference().to_dot([lkj_chol_1()], {})
        self.assertEqual(self.expected_simple_case, observed.strip())

    def test_lkj_chol_2(self) -> None:
        queries = [lkj_chol_2()]
        observed = BMGInference().to_dot(queries, {})
        self.assertEqual(self.expected_simple_case, observed.strip())

    def test_lkj_chol_3(self) -> None:
        queries = [lkj_chol_3()]
        observed = BMGInference().to_dot(queries, {})
        self.assertEqual(self.expected_random_parameter_case, observed.strip())

    def test_bad_lkj_chol_1(self) -> None:
        queries = [bad_lkj_chol_1()]
        self.assertRaises(ValueError, lambda: BMGInference().infer(queries, {}, 1))

    def test_bad_lkj_chol_2(self) -> None:
        queries = [bad_lkj_chol_2()]
        self.assertRaises(ValueError, lambda: BMGInference().infer(queries, {}, 1))

    def test_bad_lkj_chol_3s(self) -> None:
        queries = [bad_lkj_chol_3()]
        self.assertRaises(ValueError, lambda: BMGInference().infer(queries, {}, 1))
