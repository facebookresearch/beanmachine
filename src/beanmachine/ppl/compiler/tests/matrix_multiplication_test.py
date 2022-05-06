# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Normal


m1 = tensor([[12.0, 13.0], [14.0, 15.0]])
m2 = tensor([[22.0, 23.0], [24.0, 25.0]])


@bm.random_variable
def norm_1():
    return Normal(0.0, 1.0)


@bm.functional
def norm():
    return torch.tensor([[1.0, 0.0], [0.0, norm_1()]])


@bm.functional
def mm():
    # Use both the instance and static forms.
    return torch.mm(m1.mm(norm()), m2)


@bm.functional
def matmul():
    return torch.matmul(m1.matmul(norm()), m2)


@bm.functional
def infix():
    return m1 @ norm() @ m2


@bm.functional
def op_matmul():
    return operator.matmul(operator.matmul(m1, norm()), m2)


# Matrix multiplication of single-valued tensors is turned into ordinary multiplication.
@bm.random_variable
def trivial_norm_matrix():
    return Normal(torch.tensor([0.0]), torch.tensor([1.0]))


@bm.functional
def trivial():
    return trivial_norm_matrix() @ trivial_norm_matrix()


class MatMulTest(unittest.TestCase):
    def test_matrix_multiplication(self) -> None:

        self.maxDiff = None
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label="[[12.0,13.0],\\\\n[14.0,15.0]]"];
  N05[label=2];
  N06[label=1.0];
  N07[label=ToMatrix];
  N08[label="@"];
  N09[label="[[22.0,23.0],\\\\n[24.0,25.0]]"];
  N10[label="@"];
  N11[label=Query];
  N00 -> N02;
  N00 -> N07;
  N00 -> N07;
  N01 -> N02;
  N02 -> N03;
  N03 -> N07;
  N04 -> N08;
  N05 -> N07;
  N05 -> N07;
  N06 -> N07;
  N07 -> N08;
  N08 -> N10;
  N09 -> N10;
  N10 -> N11;
}"""

        observed = BMGInference().to_dot([mm()], {})
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([matmul()], {})
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([infix()], {})
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([op_matmul()], {})
        self.assertEqual(expected.strip(), observed.strip())

        expected_trivial = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label="*"];
  N5[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N3 -> N4;
  N4 -> N5;
}"""
        observed = BMGInference().to_dot([trivial()], {})
        self.assertEqual(expected_trivial.strip(), observed.strip())
