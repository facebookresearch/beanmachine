# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    return tensor(2.5)


@bm.functional
def c2():
    return tensor([1.5, -2.5])


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


# BMG supports constant single values or tensors, but the tensors must
# be 1 or 2 dimensional; empty tensors and 3+ dimensional tensors
# need to produce an error.
@bm.functional
def invalid_tensor_1():
    return tensor([])


@bm.functional
def invalid_tensor_2():
    return tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])


class BMGQueryTest(unittest.TestCase):
    def test_constant_functional(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot([c(), c2()], {})
        expected = """
digraph "graph" {
  N0[label=2.5];
  N1[label=Query];
  N2[label="[1.5,-2.5]"];
  N3[label=Query];
  N0 -> N1;
  N2 -> N3;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_cpp([c(), c2()], {})
        # TODO: Is this valid C++? The API for adding constants
        # has changed but the code generator has not kept up.
        # Check if this is wrong and fix it.
        expected = """
graph::Graph g;
uint n0 = g.add_constant(torch::from_blob((float[]){2.5}, {}));
uint q0 = g.query(n0);
uint n1 = g.add_constant(torch::from_blob((float[]){1.5,-2.5}, {2}));
uint q1 = g.query(n1);
         """
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_python([c(), c2()], {})
        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_real(2.5)
q0 = g.query(n0)
n1 = g.add_constant_real_matrix(tensor([[1.5],[-2.5]]))
q1 = g.query(n1)
        """
        self.assertEqual(expected.strip(), observed.strip())

        samples = BMGInference().infer([c(), c2()], {}, 1, 1)
        observed = samples[c()]
        expected = "tensor([[2.5000]])"
        self.assertEqual(expected.strip(), str(observed).strip())
        observed = samples[c2()]
        expected = "tensor([[[ 1.5000, -2.5000]]], dtype=torch.float64)"
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
  N4[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N2 -> N4;
}
"""
        self.assertEqual(expected.strip(), str(observed).strip())

        samples = BMGInference().infer([flip(), flip2()], {}, 10, 1)
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

        samples = BMGInference().infer([always_false_1(), always_false_2()], {}, 2, 1)
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

        samples = BMGInference().infer([flip3(), flip4()], {}, 10, 1)
        f3 = samples[flip3()]
        f4 = samples[flip4()]
        self.assertEqual(str(f3), str(f4))

    def test_invalid_tensors(self) -> None:
        self.maxDiff = None

        with self.assertRaises(ValueError) as ex:
            BMGInference().to_dot([invalid_tensor_1(), invalid_tensor_2()], {})
        # TODO: This error message is horrid. Fix it.
        expected = (
            "The model uses a tensor "
            + "operation unsupported by Bean Machine Graph.\n"
            + "The unsupported node is the operator of a query.\n"
            + "The model uses a tensor operation unsupported by Bean Machine Graph.\n"
            + "The unsupported node is the operator of a query."
        )
        self.assertEqual(expected, str(ex.exception))
