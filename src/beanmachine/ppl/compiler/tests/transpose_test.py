#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Normal


@bm.random_variable
def norm(x):
    return Normal(0.0, 1.0)


@bm.functional
def norm_array():
    return tensor([[norm(0), norm(1)], [norm(2), norm(3)]])


@bm.functional
def transpose_1():
    return torch.transpose(norm_array(), 0, 1)


@bm.functional
def transpose_2():
    return torch.transpose(norm_array(), 1, 0)


@bm.functional
def transpose_3():
    return norm_array().transpose(0, 1)


# Fails due to invalid dimensions
@bm.functional
def unsupported_transpose_1():
    return torch.transpose(norm_array(), 3, 2)


# Fails due to invalid dimension
@bm.functional
def unsupported_transpose_2():
    return norm_array().transpose(3, 1)


# Fails due to invalid (non-int) dimension
@bm.functional
def unsupported_transpose_3():
    return norm_array().transpose(3.2, 1)


@bm.functional
def scalar_transpose():
    return torch.transpose(norm(0), 0, 1)


@bm.functional
def scalar_transpose_2():
    return torch.transpose(tensor([norm(0)]), 0, 1)


class TransposeTest(unittest.TestCase):

    dot_from_normal = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=Sample];
  N06[label=Sample];
  N07[label=2];
  N08[label=ToMatrix];
  N09[label=Transpose];
  N10[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N02 -> N05;
  N02 -> N06;
  N03 -> N08;
  N04 -> N08;
  N05 -> N08;
  N06 -> N08;
  N07 -> N08;
  N07 -> N08;
  N08 -> N09;
  N09 -> N10;
}
""".strip()

    def test_transpose_1(self) -> None:
        queries = [transpose_1()]
        dot = BMGInference().to_dot(queries, {})
        self.assertEqual(dot.strip(), self.dot_from_normal)

    def test_transpose_2(self) -> None:
        queries = [transpose_2()]
        dot = BMGInference().to_dot(queries, {})
        self.assertEqual(dot.strip(), self.dot_from_normal)

    def test_transpose_3(self) -> None:
        queries = [transpose_3()]
        dot = BMGInference().to_dot(queries, {})
        self.assertEqual(dot.strip(), self.dot_from_normal)

    def test_unsupported_transpose_1(self) -> None:
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([unsupported_transpose_1()], {}, 1)
        expected = """
Unsupported dimension arguments for transpose: 3 and 2
        """
        self.assertEqual(expected.strip(), str(ex.exception).strip())

    def test_unsupported_transpose_2(self) -> None:
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([unsupported_transpose_2()], {}, 1)
        expected = """
Unsupported dimension arguments for transpose: 3 and 1
        """
        self.assertEqual(expected.strip(), str(ex.exception).strip())

    def test_unsupported_transpose_3(self) -> None:
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([unsupported_transpose_3()], {}, 1)
        expected = """
Unsupported dimension arguments for transpose: 3.2 and 1
        """
        self.assertEqual(expected.strip(), str(ex.exception).strip())

    def test_scalar_transpose(self) -> None:
        queries = [scalar_transpose()]
        dot = BMGInference().to_dot(queries, {})
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
}
        """
        self.assertEqual(dot.strip(), expected.strip())

    def test_1x1_transpose(self) -> None:
        queries = [scalar_transpose_2()]
        dot = BMGInference().to_dot(queries, {})
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
}
        """
        self.assertEqual(dot.strip(), expected.strip())
