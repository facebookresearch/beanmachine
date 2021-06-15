# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli


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


# TODO: Try multidimensional constant tensors.

# Two RVIDs but they both refer to the same query node:


@bm.random_variable
def flip():
    return Bernoulli(0.5)


@bm.functional
def flip2():
    return flip()


@bm.functional
def flip3():
    return flip() + 0


@bm.functional
def flip4():
    return 0 + flip()


# Here's a weird case. Normally query nodes are deduplicated but it is
# possible to end up with two distinct query nodes both referring to the
# same constant because of an optimization.


@bm.functional
def always_false_1():
    return 1 < flip()


@bm.functional
def always_false_2():
    # Boolean comparision optimizer turns both of these into False,
    # even though the queries were originally on different expressions
    # and therefore were different nodes.
    return flip() < 0


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

    def test_redundant_functionals(self) -> None:
        self.maxDiff = None

        # We see from the graph that we have two distinct RVIDs but they
        # both refer to the same query.  We need to make sure that BMG
        # inference works, and that we get the dictionary out that we expect.

        observed = BMGInference().to_dot([flip(), flip2()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
}
"""
        self.assertEqual(expected.strip(), str(observed).strip())

        samples = BMGInference().infer([flip(), flip2()], {}, 10)
        f = samples[flip()]
        f2 = samples[flip2()]
        self.assertEqual(str(f), str(f2))

        # A strange case: two queries on the same constant.

        observed = BMGInference().to_dot([always_false_1(), always_false_2()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=False];
  N4[label=Query];
  N5[label=Query];
  N0 -> N1;
  N1 -> N2;
  N3 -> N4;
  N3 -> N5;
}
"""
        self.assertEqual(expected.strip(), str(observed).strip())

        samples = BMGInference().infer([always_false_1(), always_false_2()], {}, 2)
        af1 = samples[always_false_1()]
        af2 = samples[always_false_2()]
        expected = "tensor([[False, False]])"
        self.assertEqual(expected, str(af1))
        self.assertEqual(expected, str(af2))

    def test_redundant_functionals_2(self) -> None:
        self.maxDiff = None

        # Here's a particularly weird one: we have what is initially two
        # distinct queries: flip() + 0 and 0 + flip(), but the graph optimizer
        # deduces that both queries refer to the same non-constant node.

        observed = BMGInference().to_dot([flip3(), flip4()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Query];
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N2 -> N4;
}
"""
        self.assertEqual(expected.strip(), str(observed).strip())

        samples = BMGInference().infer([flip3(), flip4()], {}, 10)
        f3 = samples[flip3()]
        f4 = samples[flip4()]
        self.assertEqual(str(f3), str(f4))
