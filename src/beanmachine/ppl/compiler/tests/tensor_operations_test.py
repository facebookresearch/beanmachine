# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from torch import logsumexp, tensor
from torch.distributions import Normal


@bm.random_variable
def norm(n):
    return Normal(tensor(0.0), tensor(1.0))


@bm.functional
def make_a_tensor():
    return tensor(norm(1), norm(1), norm(2), 1.25)


@bm.functional
def lse1():
    return make_a_tensor().logsumexp(dim=1)


@bm.functional
def lse2():
    return logsumexp(make_a_tensor(), dim=1)


class TensorOperationsTest(unittest.TestCase):
    def test_tensor_operations_1(self) -> None:
        self.maxDiff = None

        bmg = BMGraphBuilder()
        bmg.accumulate_graph([lse1()], {})
        observed = bmg.to_dot(
            point_at_input=True,
        )
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Sample];
  N5[label=1.25];
  N6[label=Tensor];
  N7[label=LogSumExp];
  N8[label=Query];
  N0 -> N2[label=mu];
  N1 -> N2[label=sigma];
  N2 -> N3[label=operand];
  N2 -> N4[label=operand];
  N3 -> N6[label=0];
  N3 -> N6[label=1];
  N4 -> N6[label=2];
  N5 -> N6[label=3];
  N6 -> N7[label=0];
  N7 -> N8[label=operator];
}"""
        self.assertEqual(observed.strip(), expected.strip())

        # Do it again, but this time with the static method flavor of
        # logsumexp. We should get the same result.

        bmg = BMGraphBuilder()
        bmg.accumulate_graph([lse2()], {})
        observed = bmg.to_dot(
            point_at_input=True,
        )
        self.assertEqual(observed.strip(), expected.strip())
