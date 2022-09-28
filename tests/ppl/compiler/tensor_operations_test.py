# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.compiler.runtime import BMGRuntime
from beanmachine.ppl.inference import BMGInference
from torch import logsumexp, tensor
from torch.distributions import Bernoulli, Normal


@bm.random_variable
def norm(n):
    return Normal(tensor(0.0), tensor(1.0))


@bm.functional
def make_a_tensor():
    return tensor([norm(1), norm(1), norm(2), 1.25])


@bm.functional
def lse1():
    return make_a_tensor().logsumexp(dim=0)


@bm.functional
def lse2():
    return logsumexp(make_a_tensor(), dim=0)


@bm.functional
def lse_bad_1():
    # Dim cannot be anything but zero
    return logsumexp(make_a_tensor(), dim=1)


@bm.random_variable
def flip():
    return Bernoulli(0.5)


@bm.functional
def lse_bad_2():
    # keepdim cannot be anything but false
    return logsumexp(make_a_tensor(), dim=0, keepdim=flip())


class TensorOperationsTest(unittest.TestCase):
    def test_tensor_operations_1(self) -> None:
        self.maxDiff = None

        bmg = BMGRuntime().accumulate_graph([lse1()], {})
        observed = to_dot(bmg)
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=1.25];
  N06[label=Tensor];
  N07[label=0];
  N08[label=False];
  N09[label=LogSumExp];
  N10[label=Query];
  N00 -> N02[label=mu];
  N01 -> N02[label=sigma];
  N02 -> N03[label=operand];
  N02 -> N04[label=operand];
  N03 -> N06[label=0];
  N03 -> N06[label=1];
  N04 -> N06[label=2];
  N05 -> N06[label=3];
  N06 -> N09[label=operand];
  N07 -> N09[label=dim];
  N08 -> N09[label=keepdim];
  N09 -> N10[label=operator];
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # Do it again, but this time with the static method flavor of
        # logsumexp. We should get the same result.

        bmg = BMGRuntime().accumulate_graph([lse2()], {})
        observed = to_dot(bmg)
        self.assertEqual(expected.strip(), observed.strip())

        # Now try generating a BMG from them. The problem fixer should
        # remove the unsupported tensor node.

        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Sample];
  N5[label=1.25];
  N6[label=LogSumExp];
  N7[label=Query];
  N0 -> N2[label=mu];
  N1 -> N2[label=sigma];
  N2 -> N3[label=operand];
  N2 -> N4[label=operand];
  N3 -> N6[label=0];
  N3 -> N6[label=1];
  N4 -> N6[label=2];
  N5 -> N6[label=3];
  N6 -> N7[label=operator];
}
"""

        bmg = BMGRuntime().accumulate_graph([lse1()], {})
        observed = to_dot(bmg, after_transform=True)
        self.assertEqual(observed.strip(), expected.strip())

    def test_unsupported_logsumexp(self) -> None:

        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([lse_bad_1()], {}, 1)
        # TODO: Do a better job here. Say why the operation is unsupported.
        expected = """
The model uses a logsumexp operation unsupported by Bean Machine Graph.
The unsupported node was created in function call lse_bad_1().
        """
        self.assertEqual(expected.strip(), str(ex.exception).strip())

        expected = """
The node logsumexp cannot be sized.The operand sizes may be incompatible or the size may not be computable at compile time. The operand sizes are: [torch.Size([4]), torch.Size([]), torch.Size([])]
The unsizable node was created in function call lse_bad_2().
        """

        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([lse_bad_2()], {}, 1)
        self.assertEqual(expected.strip(), str(ex.exception).strip())
