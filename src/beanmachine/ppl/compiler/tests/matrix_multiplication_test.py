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
def norm():
    return Normal(0.0, 1.0)


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
def norm_matrix():
    return Normal(torch.tensor([0.0]), torch.tensor([1.0]))


@bm.functional
def trivial():
    return norm_matrix() @ norm_matrix()


class MatMulTest(unittest.TestCase):
    def test_matrix_multiplication(self) -> None:
        # TODO: Matrix multiplications should be accumulated but they are not
        # yet type analyzed or transformed into a BMG graph.

        self.maxDiff = None

        expected_accumulation = """
digraph "graph" {
  N0[label="[[12.0,13.0],\\\\n[14.0,15.0]]"];
  N1[label=0.0];
  N2[label=1.0];
  N3[label=Normal];
  N4[label=Sample];
  N5[label="@"];
  N6[label="[[22.0,23.0],\\\\n[24.0,25.0]]"];
  N7[label="@"];
  N8[label=Query];
  N0 -> N5;
  N1 -> N3;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
  N5 -> N7;
  N6 -> N7;
  N7 -> N8;
}"""
        expected_error = """
The model uses a matrix multiplication (@) operation unsupported by Bean Machine Graph.
The unsupported node was created in function call mm().
The model uses a matrix multiplication (@) operation unsupported by Bean Machine Graph.
The unsupported node was created in function call mm().
        """

        observed = BMGInference().to_dot([mm()], {}, after_transform=False)
        self.assertEqual(expected_accumulation.strip(), observed.strip())
        with self.assertRaises(ValueError) as ex:
            BMGInference().to_dot([mm()], {}, after_transform=True)
        self.assertEqual(expected_error.strip(), str(ex.exception))

        expected_error = """
The model uses a matrix multiplication (@) operation unsupported by Bean Machine Graph.
The unsupported node was created in function call matmul().
The model uses a matrix multiplication (@) operation unsupported by Bean Machine Graph.
The unsupported node was created in function call matmul().
        """

        observed = BMGInference().to_dot([matmul()], {}, after_transform=False)
        self.assertEqual(expected_accumulation.strip(), observed.strip())
        with self.assertRaises(ValueError) as ex:
            BMGInference().to_dot([matmul()], {}, after_transform=True)
        self.assertEqual(expected_error.strip(), str(ex.exception))

        expected_error = """
The model uses a matrix multiplication (@) operation unsupported by Bean Machine Graph.
The unsupported node was created in function call infix().
The model uses a matrix multiplication (@) operation unsupported by Bean Machine Graph.
The unsupported node was created in function call infix().
        """

        observed = BMGInference().to_dot([infix()], {}, after_transform=False)
        self.assertEqual(expected_accumulation.strip(), observed.strip())
        with self.assertRaises(ValueError) as ex:
            BMGInference().to_dot([infix()], {}, after_transform=True)
        self.assertEqual(expected_error.strip(), str(ex.exception))

        expected_error = """
The model uses a matrix multiplication (@) operation unsupported by Bean Machine Graph.
The unsupported node was created in function call op_matmul().
The model uses a matrix multiplication (@) operation unsupported by Bean Machine Graph.
The unsupported node was created in function call op_matmul().
        """

        observed = BMGInference().to_dot([op_matmul()], {}, after_transform=False)
        self.assertEqual(expected_accumulation.strip(), observed.strip())
        with self.assertRaises(ValueError) as ex:
            BMGInference().to_dot([op_matmul()], {}, after_transform=True)
        self.assertEqual(expected_error.strip(), str(ex.exception))

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
