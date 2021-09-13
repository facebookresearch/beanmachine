# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.compiler.runtime import BMGRuntime
from torch import tensor
from torch.distributions import Bernoulli, Beta

# We need to be able to tell what size the tensor is
# when a model operates on multi-valued tensors.


@bm.random_variable
def coin():
    return Beta(tensor([[1.0, 2.0]]), 3.0)


@bm.random_variable
def flip():
    return Bernoulli(coin())


class SizerTest(unittest.TestCase):
    def test_sizer_1(self) -> None:
        self.maxDiff = None

        queries = [flip()]
        observations = {}
        bmg = BMGRuntime().accumulate_graph(queries, observations)
        observed = to_dot(bmg, node_sizes=True)
        expected = """
digraph "graph" {
  N0[label="[[1.0,2.0]]:[1,2]"];
  N1[label="[[3.0,3.0]]:[1,2]"];
  N2[label="Beta:[1,2]"];
  N3[label="Sample:[1,2]"];
  N4[label="Bernoulli:[1,2]"];
  N5[label="Sample:[1,2]"];
  N6[label="Query:[1,2]"];
  N0 -> N2[label=alpha];
  N1 -> N2[label=beta];
  N2 -> N3[label=operand];
  N3 -> N4[label=probability];
  N4 -> N5[label=operand];
  N5 -> N6[label=operator];
}
"""
        self.assertEqual(expected.strip(), observed.strip())
