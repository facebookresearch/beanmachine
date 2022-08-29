# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.compiler.copy_and_replace import copy_and_replace
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.compiler.runtime import BMGRuntime
from beanmachine.ppl.compiler.tensorizer_transformer import Tensorizer
from torch import mm, tensor
from torch.distributions import Normal


@bm.random_variable
def norm(n):
    return Normal(tensor(0.0), tensor(1.0))


@bm.functional
def make_matrix(n):
    return tensor([[norm(n), norm(n)], [norm(n), 1.25]])


@bm.functional
def make_tensor(n):
    return tensor(
        [[[norm(n), norm(n)], [norm(n), 2.35]], [[norm(n), norm(n)], [norm(n), 1.25]]]
    )


@bm.functional
def operators_are_tensorized():
    return (make_matrix(0) + make_matrix(1)).exp().sum()


@bm.functional
def matrix_scale_lhs():
    return make_matrix(1) * norm(2)


@bm.functional
def matrix_scale_rhs():
    return norm(1) * make_matrix(2)


@bm.functional
def scalar_mult():
    return norm(1) * norm(2)


@bm.functional
def non_matrix_tensor_mult_lhs():
    return make_tensor(1) * norm(2)


@bm.functional
def non_matrix_tensor_mult_rhs():
    return norm(6) * make_tensor(5)


@bm.functional
def mm_mismatch():
    return mm(make_tensor(1), tensor([3.6, 3.1, 3.5]))


class TensorizeTransformerTest(unittest.TestCase):
    def test_tensor_operators(self) -> None:
        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph([operators_are_tensorized()], {})
        transformed_graph, error_report = copy_and_replace(
            bmg, lambda c, s: Tensorizer(c, s)
        )
        before = to_dot(bmg)
        after = to_dot(transformed_graph)
        expected_before = """
        digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=1.25];
  N05[label=Tensor];
  N06[label=Sample];
  N07[label=Tensor];
  N08[label="+"];
  N09[label=Exp];
  N10[label=Sum];
  N11[label=Query];
  N00 -> N02[label=mu];
  N01 -> N02[label=sigma];
  N02 -> N03[label=operand];
  N02 -> N06[label=operand];
  N03 -> N05[label=0];
  N03 -> N05[label=1];
  N03 -> N05[label=2];
  N04 -> N05[label=3];
  N04 -> N07[label=3];
  N05 -> N08[label=left];
  N06 -> N07[label=0];
  N06 -> N07[label=1];
  N06 -> N07[label=2];
  N07 -> N08[label=right];
  N08 -> N09[label=operand];
  N09 -> N10[label=operand];
  N10 -> N11[label=operator];
}
        """
        expected_after = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=1.25];
  N05[label=Tensor];
  N06[label=Sample];
  N07[label=Tensor];
  N08[label=MatrixAdd];
  N09[label=MatrixExp];
  N10[label=MatrixSum];
  N11[label=Query];
  N00 -> N02[label=mu];
  N01 -> N02[label=sigma];
  N02 -> N03[label=operand];
  N02 -> N06[label=operand];
  N03 -> N05[label=0];
  N03 -> N05[label=1];
  N03 -> N05[label=2];
  N04 -> N05[label=3];
  N04 -> N07[label=3];
  N05 -> N08[label=left];
  N06 -> N07[label=0];
  N06 -> N07[label=1];
  N06 -> N07[label=2];
  N07 -> N08[label=right];
  N08 -> N09[label=operand];
  N09 -> N10[label=operand];
  N10 -> N11[label=operator];
}
        """
        self.assertEqual(expected_before.strip(), before.strip())
        self.assertEqual(expected_after.strip(), after.strip())

    def test_matrix_scale(self) -> None:
        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph(
            [
                matrix_scale_rhs(),
                matrix_scale_lhs(),
                non_matrix_tensor_mult_lhs(),
                non_matrix_tensor_mult_rhs(),
            ],
            {},
        )
        transformed_graph, error_report = copy_and_replace(
            bmg, lambda c, s: Tensorizer(c, s)
        )
        before = to_dot(bmg)
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=1.25];
  N06[label=Tensor];
  N07[label="*"];
  N08[label=Query];
  N09[label=Tensor];
  N10[label="*"];
  N11[label=Query];
  N12[label=2.35];
  N13[label=Tensor];
  N14[label="*"];
  N15[label=Query];
  N16[label=Sample];
  N17[label=Sample];
  N18[label=Tensor];
  N19[label="*"];
  N20[label=Query];
  N00 -> N02[label=mu];
  N01 -> N02[label=sigma];
  N02 -> N03[label=operand];
  N02 -> N04[label=operand];
  N02 -> N16[label=operand];
  N02 -> N17[label=operand];
  N03 -> N07[label=left];
  N03 -> N09[label=0];
  N03 -> N09[label=1];
  N03 -> N09[label=2];
  N03 -> N13[label=0];
  N03 -> N13[label=1];
  N03 -> N13[label=2];
  N03 -> N13[label=4];
  N03 -> N13[label=5];
  N03 -> N13[label=6];
  N04 -> N06[label=0];
  N04 -> N06[label=1];
  N04 -> N06[label=2];
  N04 -> N10[label=right];
  N04 -> N14[label=right];
  N05 -> N06[label=3];
  N05 -> N09[label=3];
  N05 -> N13[label=7];
  N05 -> N18[label=7];
  N06 -> N07[label=right];
  N07 -> N08[label=operator];
  N09 -> N10[label=left];
  N10 -> N11[label=operator];
  N12 -> N13[label=3];
  N12 -> N18[label=3];
  N13 -> N14[label=left];
  N14 -> N15[label=operator];
  N16 -> N19[label=left];
  N17 -> N18[label=0];
  N17 -> N18[label=1];
  N17 -> N18[label=2];
  N17 -> N18[label=4];
  N17 -> N18[label=5];
  N17 -> N18[label=6];
  N18 -> N19[label=right];
  N19 -> N20[label=operator];
}
        """
        self.assertEqual(expected.strip(), before.strip())
        after = to_dot(transformed_graph)
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=1.25];
  N06[label=Tensor];
  N07[label=MatrixScale];
  N08[label=Query];
  N09[label=Tensor];
  N10[label=MatrixScale];
  N11[label=Query];
  N12[label=2.35];
  N13[label=Tensor];
  N14[label=MatrixScale];
  N15[label=Query];
  N16[label=Sample];
  N17[label=Sample];
  N18[label=Tensor];
  N19[label=MatrixScale];
  N20[label=Query];
  N00 -> N02[label=mu];
  N01 -> N02[label=sigma];
  N02 -> N03[label=operand];
  N02 -> N04[label=operand];
  N02 -> N16[label=operand];
  N02 -> N17[label=operand];
  N03 -> N07[label=left];
  N03 -> N09[label=0];
  N03 -> N09[label=1];
  N03 -> N09[label=2];
  N03 -> N13[label=0];
  N03 -> N13[label=1];
  N03 -> N13[label=2];
  N03 -> N13[label=4];
  N03 -> N13[label=5];
  N03 -> N13[label=6];
  N04 -> N06[label=0];
  N04 -> N06[label=1];
  N04 -> N06[label=2];
  N04 -> N10[label=left];
  N04 -> N14[label=left];
  N05 -> N06[label=3];
  N05 -> N09[label=3];
  N05 -> N13[label=7];
  N05 -> N18[label=7];
  N06 -> N07[label=right];
  N07 -> N08[label=operator];
  N09 -> N10[label=right];
  N10 -> N11[label=operator];
  N12 -> N13[label=3];
  N12 -> N18[label=3];
  N13 -> N14[label=right];
  N14 -> N15[label=operator];
  N16 -> N19[label=left];
  N17 -> N18[label=0];
  N17 -> N18[label=1];
  N17 -> N18[label=2];
  N17 -> N18[label=4];
  N17 -> N18[label=5];
  N17 -> N18[label=6];
  N18 -> N19[label=right];
  N19 -> N20[label=operator];
}
        """
        self.assertEqual(expected.strip(), after.strip())

    def test_not_transformed(self) -> None:
        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph(
            [scalar_mult()],
            {},
        )
        transformed_graph, error_report = copy_and_replace(
            bmg, lambda c, s: Tensorizer(c, s)
        )
        observed = to_dot(transformed_graph)
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Sample];
  N5[label="*"];
  N6[label=Query];
  N0 -> N2[label=mu];
  N1 -> N2[label=sigma];
  N2 -> N3[label=operand];
  N2 -> N4[label=operand];
  N3 -> N5[label=left];
  N4 -> N5[label=right];
  N5 -> N6[label=operator];
}
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_errors(self) -> None:
        self.maxDiff = None
        # this case verifies that even if there is nothing replacable it will error out because the errors
        # in this graph prevent even checking whether this graph can be tensorized
        bmg = BMGRuntime().accumulate_graph([mm_mismatch()], {})
        transformed_graph, error_report = copy_and_replace(
            bmg, lambda c, s: Tensorizer(c, s)
        )
        if len(error_report.errors) == 1:
            error = error_report.errors[0].__str__()
            expected = """
The model uses a matrix multiplication (@) operation unsupported by Bean Machine Graph.
The dimensions of the operands are 2x2 and 3x1.
The unsupported node was created in function call mm_mismatch().
            """
            self.assertEqual(expected.strip(), error.strip())
        else:
            self.fail(
                "A single error message should have been generated. Tensorizing depends on sizing and a size cannot be inferred from an operation whose operand sizes are invalid."
            )
