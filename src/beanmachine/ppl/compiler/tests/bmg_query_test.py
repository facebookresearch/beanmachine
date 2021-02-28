# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import tensor


# Bean Machine allows queries on functionals that return constants;
# BMG does not. It would be nice though if a BM model that queried
# a constant worked when using BMGInference the same way that it
# does with other inference engines, for several reasons:
#
# (1) consistency of behaviour across inference engines
# (2) testing optimizations; if an optimization ends up producing
#     a constant, it's nice to be able to query that functional
#     and see that it does indeed produce a constant.
# (3) possible future error reporting; it would be nice to warn the
#     user that they are querying a constant because this could be
#     a bug in their model.
# (4) model development and debugging; a user might make a dummy functional
#     that just returns a constant now, intending to replace it with an
#     actual function later.  Or might force a functional to produce a
#     particular value to see how the model behaves in that case.
#
# This test verifies that we can query a constant functional.


@bm.functional
def c():
    return tensor(1.0)


class BMGQueryTest(unittest.TestCase):
    def test_constant_functional(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot([c()], {})
        expected = """
digraph "graph" {
  N0[label=1.0];
  N1[label=Query];
  N0 -> N1;
}"""
        self.assertEqual(expected.strip(), observed.strip())

        # We do not emit the query instruction when the queried node
        # is a constant.
        observed = BMGInference().to_cpp([c()], {})
        expected = """
graph::Graph g;
uint n0 = g.add_constant(torch::from_blob((float[]){1.0}, {}));
         """
        self.assertEqual(expected.strip(), observed.strip())

        # We do not emit the query instruction when the queried node
        # is a constant.
        observed = BMGInference().to_python([c()], {})
        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant(tensor(1.0))
        """
        self.assertEqual(expected.strip(), observed.strip())

        samples = BMGInference().infer([c()], {}, 10)
        observed = samples[c()]
        expected = "tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])"
        self.assertEqual(expected.strip(), str(observed).strip())
