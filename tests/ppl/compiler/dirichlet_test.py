#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Dirichlet compiler tests

import unittest

import beanmachine.ppl as bm
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
from beanmachine.ppl.compiler.gen_bmg_cpp import to_bmg_cpp
from beanmachine.ppl.compiler.gen_bmg_graph import to_bmg_graph
from beanmachine.ppl.compiler.gen_bmg_python import to_bmg_python
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.compiler.runtime import BMGRuntime
from beanmachine.ppl.inference import BMGInference, SingleSiteNewtonianMonteCarlo
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import Size, tensor
from torch.distributions import Bernoulli, Dirichlet


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


def _rv_id() -> RVIdentifier:
    return RVIdentifier(lambda a, b: a, (1, 1))


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

# Here are some simple models we'll use to test the compiler.

# TODO: Test Dirichlets with non-constant inputs.


@bm.random_variable
def d0():
    return Dirichlet(tensor([]))


# Torch rejects this one.
# @bm.random_variable
# def d1a():
#     return Dirichlet(tensor(0.5))


@bm.random_variable
def d1b():
    return Dirichlet(tensor([1.0]))


@bm.random_variable
def d1c():
    return Dirichlet(tensor([[1.5]]))


@bm.random_variable
def d1d():
    return Dirichlet(tensor([[[2.0]]]))


# Torch rejects this one
# @bm.random_variable
# def d1e():
#     return Dirichlet(tensor([[[-2.0]]]))


@bm.random_variable
def d2a():
    return Dirichlet(tensor([2.5, 3.0]))


@bm.random_variable
def d2b():
    return Dirichlet(tensor([[3.5, 4.0]]))


@bm.random_variable
def d2c():
    return Dirichlet(tensor([[[4.5, 5.0]]]))


@bm.random_variable
def d23():
    return Dirichlet(tensor([[5.5, 6.0, 6.5], [7.0, 7.5, 8.0]]))


@bm.random_variable
def d3():
    return Dirichlet(tensor([1.0, 1.0, 1.0]))


@bm.functional
def d3_index_0():
    return d3()[0]


@bm.random_variable
def flip():
    return Bernoulli(0.5)


@bm.functional
def d2a_index_flip():
    return d2a()[flip()]


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
        bmg.add_query(c1, _rv_id())
        bmg.add_query(c2, _rv_id())
        bmg.add_query(c3, _rv_id())
        expected = """
digraph "graph" {
  N0[label="1.0:R+"];
  N1[label="Query:R+"];
  N2[label="[1.0,1.5]:MR+[2,1]"];
  N3[label="Query:MR+[2,1]"];
  N4[label="[[1.0,1.5],\\\\n[2.0,2.5]]:MR+[2,2]"];
  N5[label="Query:MR+[2,2]"];
  N0 -> N1;
  N2 -> N3;
  N4 -> N5;
}"""
        observed = to_dot(
            bmg,
            node_types=True,
            label_edges=False,
            after_transform=True,
        )
        self.assertEqual(expected.strip(), observed.strip())

        # We should be able to generate correct C++ and Python code to build
        # a graph that contains only positive constant matrices. Note that the
        # queries are not emitted into the graph because BMG does not allow
        # a query on a constant.
        #
        # NB: m2 is transposed from the source!
        expected = """
graph::Graph g;
Eigen::MatrixXd m0(1, 1)
m0 << 1.0;
uint n0 = g.add_constant_pos_matrix(m0);
uint q0 = g.query(n0);
Eigen::MatrixXd m1(2, 1)
m1 << 1.0, 1.5;
uint n1 = g.add_constant_pos_matrix(m1);
uint q1 = g.query(n1);
Eigen::MatrixXd m2(2, 2)
m2 << 1.0, 2.0, 1.5, 2.5;
uint n2 = g.add_constant_pos_matrix(m2);
uint q2 = g.query(n2);
        """
        observed = to_bmg_cpp(bmg).code
        self.assertEqual(expected.strip(), observed.strip())

        # Notice that constant matrices are always expressed as a
        # 2-d matrix, and we transpose them so that they are column-major.

        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_matrix(tensor([[1.0]]))
q0 = g.query(n0)
n1 = g.add_constant_pos_matrix(tensor([[1.0],[1.5]]))
q1 = g.query(n1)
n2 = g.add_constant_pos_matrix(tensor([[1.0,2.0],[1.5,2.5]]))
q2 = g.query(n2)
"""
        observed = to_bmg_python(bmg).code
        self.assertEqual(expected.strip(), observed.strip())

        # Let's actually get the graph.

        # Note that what was a row vector in the original code is now a column vector.
        expected = """
Node 0 type 1 parents [ ] children [ ] matrix<positive real> 1
Node 1 type 1 parents [ ] children [ ] matrix<positive real>   1
 1.5
Node 2 type 1 parents [ ] children [ ] matrix<positive real>   1   2
 1.5 2.5"""
        observed = to_bmg_graph(bmg).graph.to_string()
        self.assertEqual(tidy(expected), tidy(observed))

    def test_dirichlet_type_analysis(self) -> None:
        self.maxDiff = None
        queries = [d0(), d1b(), d1c(), d1d(), d2a(), d2b(), d2c(), d23()]
        bmg = BMGRuntime().accumulate_graph(queries, {})
        observed = to_dot(
            bmg,
            node_types=True,
            edge_requirements=True,
            after_transform=False,
            label_edges=False,
        )
        expected = """
digraph "graph" {
  N00[label="[]:T"];
  N01[label="Dirichlet:S[1,1]"];
  N02[label="Sample:S[1,1]"];
  N03[label="Query:S[1,1]"];
  N04[label="[1.0]:OH"];
  N05[label="Dirichlet:S[1,1]"];
  N06[label="Sample:S[1,1]"];
  N07[label="Query:S[1,1]"];
  N08[label="[[1.5]]:R+"];
  N09[label="Dirichlet:S[1,1]"];
  N10[label="Sample:S[1,1]"];
  N11[label="Query:S[1,1]"];
  N12[label="[[[2.0]]]:N"];
  N13[label="Dirichlet:S[1,1]"];
  N14[label="Sample:S[1,1]"];
  N15[label="Query:S[1,1]"];
  N16[label="[2.5,3.0]:MR+[2,1]"];
  N17[label="Dirichlet:S[2,1]"];
  N18[label="Sample:S[2,1]"];
  N19[label="Query:S[2,1]"];
  N20[label="[[3.5,4.0]]:MR+[2,1]"];
  N21[label="Dirichlet:S[2,1]"];
  N22[label="Sample:S[2,1]"];
  N23[label="Query:S[2,1]"];
  N24[label="[[[4.5,5.0]]]:T"];
  N25[label="Dirichlet:S[1,1]"];
  N26[label="Sample:S[1,1]"];
  N27[label="Query:S[1,1]"];
  N28[label="[[5.5,6.0,6.5],\\\\n[7.0,7.5,8.0]]:MR+[3,2]"];
  N29[label="Dirichlet:S[3,1]"];
  N30[label="Sample:S[3,1]"];
  N31[label="Query:S[3,1]"];
  N00 -> N01[label="R+"];
  N01 -> N02[label="S[1,1]"];
  N02 -> N03[label=any];
  N04 -> N05[label="R+"];
  N05 -> N06[label="S[1,1]"];
  N06 -> N07[label=any];
  N08 -> N09[label="R+"];
  N09 -> N10[label="S[1,1]"];
  N10 -> N11[label=any];
  N12 -> N13[label="R+"];
  N13 -> N14[label="S[1,1]"];
  N14 -> N15[label=any];
  N16 -> N17[label="MR+[2,1]"];
  N17 -> N18[label="S[2,1]"];
  N18 -> N19[label=any];
  N20 -> N21[label="MR+[2,1]"];
  N21 -> N22[label="S[2,1]"];
  N22 -> N23[label=any];
  N24 -> N25[label="R+"];
  N25 -> N26[label="S[1,1]"];
  N26 -> N27[label=any];
  N28 -> N29[label="MR+[3,1]"];
  N29 -> N30[label="S[3,1]"];
  N30 -> N31[label=any];
}
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_dirichlet_errors(self) -> None:
        self.maxDiff = None

        # If the constant tensor given is not supported at all by BMG because of
        # its dimensionality then that is the error we will report. If the tensor
        # is supported by BMG but not valid for a Dirichlet then that's what we say.

        # TODO: Error message is misleading in that it says that the requirement
        # is a 3x1 positive real matrix, when the real requirement is that it be
        # ANY 1-d positive real matrix.

        expected = (
            "The concentration of a Dirichlet is required to be"
            + " a 3 x 1 positive real matrix but is"
            + " a 3 x 2 positive real matrix.\n"
            + "The Dirichlet was created in function call d23()."
        )

        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([d23()], {}, 1)
        self.assertEqual(expected.strip(), str(ex.exception).strip())

    def test_dirichlet_fix_problems(self) -> None:

        # Can we take an input that is a valid tensor and deduce that we must
        # replace it with a positive real constant matrix node?

        self.maxDiff = None
        queries = [d2a()]
        observations = {d2a(): tensor([0.5, 0.5])}
        bmg = BMGRuntime().accumulate_graph(queries, observations)
        observed = to_dot(
            bmg,
            node_types=True,
            edge_requirements=True,
            after_transform=True,
            label_edges=False,
        )

        expected = """
digraph "graph" {
  N0[label="[2.5,3.0]:MR+[2,1]"];
  N1[label="Dirichlet:S[2,1]"];
  N2[label="Sample:S[2,1]"];
  N3[label="Observation tensor([0.5000, 0.5000]):S[2,1]"];
  N4[label="Query:S[2,1]"];
  N0 -> N1[label="MR+[2,1]"];
  N1 -> N2[label="S[2,1]"];
  N2 -> N3[label=any];
  N2 -> N4[label=any];
}
        """

        self.assertEqual(expected.strip(), observed.strip())

        # This is the tricky case: the degenerate case where we have only
        # one value, and we need to make sure that we generate a matrix
        # constant rather than a regular positive real constant:

        queries = [d1b()]
        bmg = BMGRuntime().accumulate_graph(queries, {})
        observed = to_dot(
            bmg,
            node_types=True,
            edge_requirements=True,
            after_transform=True,
            label_edges=False,
        )

        # This is subtle, but notice that we have a constant matrix here rather
        # than a constant; the value is [1.0], not 1.0.

        expected = """
digraph "graph" {
  N0[label="[1.0]:R+"];
  N1[label="Dirichlet:S[1,1]"];
  N2[label="Sample:S[1,1]"];
  N3[label="Query:S[1,1]"];
  N0 -> N1[label="R+"];
  N1 -> N2[label="S[1,1]"];
  N2 -> N3[label=any];
}"""

        self.assertEqual(expected.strip(), observed.strip())

    def test_dirichlet_bmg_inference(self) -> None:

        # Get Dirichlet samples; verify that the sample set
        # is in rows, not columns.

        self.maxDiff = None

        # 2-element Dirichlet

        queries = [d2a()]
        observations = {}
        num_samples = 10

        results = BMGInference().infer(queries, observations, num_samples, 1, nmc)
        samples = results[d2a()]
        self.assertEqual(Size([1, num_samples, 2]), samples.size())

        # Make sure we get the same thing when we use Bean Machine proper:

        results = SingleSiteNewtonianMonteCarlo().infer(
            queries, observations, num_samples, 1
        )
        samples = results[d2a()]
        self.assertEqual(Size([1, num_samples, 2]), samples.size())

        # 3-element Dirichlet

        queries = [d3()]
        observations = {}
        num_samples = 20

        results = BMGInference().infer(queries, observations, num_samples, 1, rejection)
        samples = results[d3()]
        self.assertEqual(Size([1, num_samples, 3]), samples.size())

        # If we observe a Dirichlet sample to be a value, we'd better get
        # that value when we query.

        queries = [d3()]
        observations = {d3(): tensor([0.5, 0.25, 0.25])}
        num_samples = 1

        results = BMGInference().infer(queries, observations, num_samples, 1, nmc)
        samples = results[d3()]
        expected = "tensor([[[0.5000, 0.2500, 0.2500]]], dtype=torch.float64)"
        self.assertEqual(expected, str(samples))

        # Make sure we get the same thing when we use Bean Machine proper:

        results = SingleSiteNewtonianMonteCarlo().infer(
            queries, observations, num_samples, 1
        )
        samples = results[d3()]
        expected = "tensor([[[0.5000, 0.2500, 0.2500]]])"
        self.assertEqual(expected, str(samples))

    def test_dirichlet_to_python(self) -> None:
        self.maxDiff = None

        queries = [d2a()]
        observations = {d2a(): tensor([0.5, 0.5])}

        observed = BMGInference().to_python(queries, observations)
        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_matrix(tensor([[2.5],[3.0]]))
n1 = g.add_distribution(
  graph.DistributionType.DIRICHLET,
  graph.ValueType(
    graph.VariableType.COL_SIMPLEX_MATRIX,
    graph.AtomicType.PROBABILITY,
    2,
    1,
  ),
  [n0],
)
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
g.observe(n2, tensor([0.5000, 0.5000]))
q0 = g.query(n2)"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_dirichlet_to_cpp(self) -> None:
        self.maxDiff = None

        queries = [d2a()]
        observations = {d2a(): tensor([0.5, 0.5])}

        observed = BMGInference().to_cpp(queries, observations)
        expected = """
graph::Graph g;
Eigen::MatrixXd m0(2, 1)
m0 << 2.5, 3.0;
uint n0 = g.add_constant_pos_matrix(m0);
uint n1 = g.add_distribution(
  graph::DistributionType::DIRICHLET,
  graph::ValueType(
    graph::VariableType::COL_SIMPLEX_MATRIX,
    graph::AtomicType::PROBABILITY,
    2,
    1
  ),
  std::vector<uint>({n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
Eigen::MatrixXd o0(2, 1)
o0 << 0.5, 0.5;
g.observe([n2], o0);
uint q0 = g.query(n2);"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_dirichlet_observation_errors(self) -> None:
        self.maxDiff = None

        queries = [d2a()]
        # Wrong size, wrong sum
        observations = {d2a(): tensor(2.0)}
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, observations, 1)
        expected = (
            "A Dirichlet distribution is observed to have value 2.0 "
            + "but only produces samples of type 2 x 1 simplex matrix."
        )
        self.assertEqual(expected, str(ex.exception))

        # Wrong size, right sum
        observations = {d2a(): tensor([0.25, 0.25, 0.25, 0.25])}
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, observations, 1)
        expected = (
            "A Dirichlet distribution is observed to have value "
            + "tensor([0.2500, 0.2500, 0.2500, 0.2500]) "
            + "but only produces samples of type 2 x 1 simplex matrix."
        )
        self.assertEqual(expected, str(ex.exception))

        # Right size, wrong sum
        observations = {d2a(): tensor([0.25, 0.25])}
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, observations, 1)
        expected = (
            "A Dirichlet distribution is observed to have value "
            + "tensor([0.2500, 0.2500]) "
            + "but only produces samples of type 2 x 1 simplex matrix."
        )
        self.assertEqual(expected, str(ex.exception))

    def test_dirichlet_index(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot([d3_index_0()], {})
        expected = """
digraph "graph" {
  N0[label="[1.0,1.0,1.0]"];
  N1[label=Dirichlet];
  N2[label=Sample];
  N3[label=0];
  N4[label=index];
  N5[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N4;
  N3 -> N4;
  N4 -> N5;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([d2a_index_flip()], {})
        expected = """
digraph "graph" {
  N00[label="[2.5,3.0]"];
  N01[label=Dirichlet];
  N02[label=Sample];
  N03[label=0.5];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label=1];
  N07[label=0];
  N08[label=if];
  N09[label=index];
  N10[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N09;
  N03 -> N04;
  N04 -> N05;
  N05 -> N08;
  N06 -> N08;
  N07 -> N08;
  N08 -> N09;
  N09 -> N10;
}
"""

        self.assertEqual(expected.strip(), observed.strip())

        queries = [d2a(), d2a_index_flip()]
        observations = {flip(): tensor(1.0)}

        results = BMGInference().infer(queries, observations, 1)
        d2a_sample = results[d2a()][0, 0]
        index_sample = results[d2a_index_flip()][0]
        # The sample and the indexed sample must be the same value
        self.assertEqual(d2a_sample[1], index_sample)
