#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Categorical compiler tests

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Categorical, Dirichlet, HalfCauchy


t = tensor([0.125, 0.125, 0.25, 0.5])


@bm.random_variable
def c_const_simplex():
    return Categorical(t)


@bm.random_variable
def c_const_unnormalized():
    # If we have samples of both the normalized and unnormalized distributions
    # the deduplicator should merge them into the same distribution, since
    # 2:2:4:8 :: 1/8:1/8:1/4:1/2
    return Categorical(t * 16.0)


@bm.random_variable
def c_const_logit_simplex():
    # Note that logits here means log probabilities, not log odds.
    # Since the argument is just a constant, the runtime should detect
    # that it can simply reuse the [0.125, 0.125, 0.25, 0.5] node
    # in the generated graph.
    return Categorical(logits=t.log())


@bm.random_variable
def c_trivial_simplex():
    # No sensible person would do this but we should ensure it works anyway.
    # Categorical(1.0) is already illegal in torch so we don't have to test that.
    # TODO: We could optimize this to the constant zero I suppose but it is
    # unlikely to help in realistic code. Better would be to detect this likely
    # bug and report it as a warning somehow.
    return Categorical(tensor([1.0]))


@bm.random_variable
def hc():
    return HalfCauchy(0.0)


@bm.random_variable
def c_random_logit():
    return Categorical(logits=tensor([0.0, 0.0, 0.0, -hc()]))


@bm.random_variable
def d4():
    return Dirichlet(tensor([1.0, 1.0, 1.0, 1.0]))


@bm.random_variable
def cd4():
    return Categorical(d4())


@bm.random_variable
def c_multi():
    return Categorical(tensor([[0.5, 0.5], [0.5, 0.5]]))


# NOTE: A random variable indexed by a categorical is tested in
# stochastic_control_flow_test.py.

# TODO: Once categorical inference is supported in BMG add a test
# here which demonstrates that.


class CategoricalTest(unittest.TestCase):
    def test_categorical_trivial(self) -> None:
        self.maxDiff = None

        queries = [c_trivial_simplex()]
        observations = {}
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N0[label="[1.0]"];
  N1[label=Categorical];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_categorical_dirichlet(self) -> None:
        self.maxDiff = None

        # It should be legal to use the output of a one-column
        # Dirichlet as the input to a categorical:

        queries = [cd4()]
        observations = {}
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N0[label="[1.0,1.0,1.0,1.0]"];
  N1[label=Dirichlet];
  N2[label=Sample];
  N3[label=Categorical];
  N4[label=Sample];
  N5[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_categorical_equivalent_consts(self) -> None:
        self.maxDiff = None

        # * If we have a categorical with a constant probability
        #   that does not sum to 1.0 then we automatically normalize it.
        # * If we have a categorical logits with constant probability
        #   then we automatically convert it to regular probs and
        #   normalize them.
        #
        # That means that we automatically deduplicate what looks
        # like three distinct distributions into three samples from
        # the same distribution:
        queries = [
            c_const_unnormalized(),
            c_const_simplex(),
            c_const_logit_simplex(),
        ]
        observations = {}
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N0[label="[0.125,0.125,0.25,0.5]"];
  N1[label=Categorical];
  N2[label=Sample];
  N3[label=Query];
  N4[label=Sample];
  N5[label=Query];
  N6[label=Sample];
  N7[label=Query];
  N0 -> N1;
  N1 -> N2;
  N1 -> N4;
  N1 -> N6;
  N2 -> N3;
  N4 -> N5;
  N6 -> N7;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

        # Note that we add a simplex-typed constant:

        observed = BMGInference().to_python(queries, observations)
        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_col_simplex_matrix(tensor([[0.125],[0.125],[0.25],[0.5]]))
n1 = g.add_distribution(
  graph.DistributionType.CATEGORICAL,
  graph.AtomicType.NATURAL,
  [n0],
)
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
q0 = g.query(n2)
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
q1 = g.query(n3)
n4 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
q2 = g.query(n4)
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_categorical_random_logit(self) -> None:
        self.maxDiff = None

        # We do not support Categorical(logits=something_random)
        # random variables.

        queries = [
            c_random_logit(),
        ]
        observations = {}
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, observations, 10)
        observed = str(ex.exception)
        expected = """
The model uses a categorical operation unsupported by Bean Machine Graph.
The unsupported node is the operand of a sample.
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_categorical_multi(self) -> None:
        self.maxDiff = None

        # We do not support Categorical with multiple dimensions.

        # TODO: This error message is not very well worded; what we want to communicate
        # is that ANY one-column simplex is the requirement.

        queries = [
            c_multi(),
        ]
        observations = {}
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, observations, 10)
        observed = str(ex.exception)
        expected = """
The probability of a categorical is required to be a 2 x 1 simplex matrix but is a 2 x 2 simplex matrix.
        """
        self.assertEqual(expected.strip(), observed.strip())
