#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

# Dirichlet compiler tests

import unittest

from beanmachine.graph import (
    AtomicType,
    DistributionType,
    Graph,
    InferenceType,
    OperatorType,
    ValueType,
    VariableType,
)
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from torch import tensor


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


# Support for Dirichlet distributions has recently been added to BMG;
# this is the first time that the compiler will have to deal with
# tensor-valued quantities directly so we anticipate having a number
# of problems to solve in type analysis and code generation that have
# been put off until now.
#
# We'll start by just taking the BMG code for a spin directly and see
# what gives errors and what gives results.

dirichlet = DistributionType.DIRICHLET
simplex = VariableType.COL_SIMPLEX_MATRIX
broadcast = VariableType.BROADCAST_MATRIX
real = AtomicType.REAL
prob = AtomicType.PROBABILITY
sample = OperatorType.SAMPLE
s3x1 = ValueType(simplex, prob, 3, 1)
r3x1 = ValueType(broadcast, real, 3, 1)
nmc = InferenceType.NMC
rejection = InferenceType.REJECTION


class DirichletTest(unittest.TestCase):
    def test_dirichlet_negative(self) -> None:
        self.maxDiff = None
        g = Graph()
        m1 = tensor([1.5, 1.0, 2.0])
        cm1 = g.add_constant_pos_matrix(m1)
        m2 = tensor([[1.5, 1.0], [2.0, 1.5]])
        cm2 = g.add_constant_pos_matrix(m2)
        two = g.add_constant(2)
        # Input must be a positive real matrix with one column.
        with self.assertRaises(ValueError):
            g.add_distribution(dirichlet, s3x1, [two])
        with self.assertRaises(ValueError):
            g.add_distribution(dirichlet, s3x1, [cm2])
        # Must be only one input
        with self.assertRaises(ValueError):
            g.add_distribution(dirichlet, s3x1, [cm1, two])
        # Output type must be simplex
        with self.assertRaises(ValueError):
            g.add_distribution(dirichlet, r3x1, [cm1])

    def test_dirichlet_sample(self) -> None:
        self.maxDiff = None
        g = Graph()
        m1 = tensor([1.5, 1.0, 2.0])
        cm1 = g.add_constant_pos_matrix(m1)
        d = g.add_distribution(dirichlet, s3x1, [cm1])
        ds = g.add_operator(sample, [d])
        g.query(ds)
        samples = g.infer(1, rejection)
        # samples has form [[array([[a1],[a2],[a3]])]]
        result = tensor(samples[0][0]).reshape([3])
        # We get a three-element simplex, so it should sum to 1.0.
        self.assertAlmostEqual(1.0, float(sum(result)))

    def test_constant_pos_real_matrix(self) -> None:

        # To make a BMG graph with a Dirichlet distribution the first thing
        # we'll need to do is make a positive real matrix as its input.
        # Demonstrate that we can add such a matrix to a graph builder,
        # do a type analysis, and generate C++ and Python code that builds
        # the graph.  Finally, actually build the graph.

        self.maxDiff = None

        bmg = BMGraphBuilder()
        c1 = bmg.add_pos_real_matrix(tensor(1.0))
        c2 = bmg.add_pos_real_matrix(tensor([1.0, 1.5]))
        c3 = bmg.add_pos_real_matrix(tensor([[1.0, 1.5], [2.0, 2.5]]))
        c4 = bmg.add_pos_real_matrix(tensor([1.0, 1.5]))

        # These should be deduplicated
        self.assertTrue(c4 is c2)

        # Verify that we can add these nodes to the graph, do a type analysis,
        # and survive the problem-fixing pass without generating an exception.
        bmg.add_query(c1)
        bmg.add_query(c2)
        bmg.add_query(c3)
        expected = """
digraph "graph" {
  N0[label="1.0:R+>=OH"];
  N1[label="Query:R+>=OH"];
  N2[label="[1.0,1.5]:MR+[1,2]>=MR+[1,2]"];
  N3[label="Query:MR+[1,2]>=MR+[1,2]"];
  N4[label="[[1.0,1.5],\\\\n[2.0,2.5]]:MR+[2,2]>=MR+[2,2]"];
  N5[label="Query:MR+[2,2]>=MR+[2,2]"];
  N0 -> N1;
  N2 -> N3;
  N4 -> N5;
}"""
        observed = bmg.to_dot(
            graph_types=True,
            inf_types=True,
            point_at_input=True,
            label_edges=False,
            after_transform=True,
        )
        self.assertEqual(expected.strip(), observed.strip())

        # We should be able to generate correct C++ and Python code to build
        # a graph that contains only positive constant matrices. Note that the
        # queries are not emitted into the graph because BMG does not allow
        # a query on a constant.
        expected = """
graph::Graph g;
Eigen::MatrixXd m0(1, 1)
m0 << 1.0;
uint n0 = g.add_constant_pos_matrix(m0);

Eigen::MatrixXd m2(2, 1)
m2 << 1.0, 1.5;
uint n2 = g.add_constant_pos_matrix(m2);

Eigen::MatrixXd m4(2, 2)
m4 << 1.0, 1.5, 2.0, 2.5;
uint n4 = g.add_constant_pos_matrix(m4);
        """
        observed = bmg.to_cpp()
        self.assertEqual(expected.strip(), observed.strip())

        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_matrix(tensor(1.0))

n2 = g.add_constant_pos_matrix(tensor([1.0,1.5]))

n4 = g.add_constant_pos_matrix(tensor([[1.0,1.5],[2.0,2.5]]))
        """
        observed = bmg.to_python()
        self.assertEqual(expected.strip(), observed.strip())

        # Let's actually get the graph
        g = bmg.to_bmg()
        expected = """
Node 0 type 1 parents [ ] children [ ] matrix<positive real> 1
Node 1 type 1 parents [ ] children [ ] matrix<positive real>   1 1.5
Node 2 type 1 parents [ ] children [ ] matrix<positive real>   1 1.5
 2 2.5"""
        observed = g.to_string()
        self.assertEqual(tidy(expected), tidy(observed))
