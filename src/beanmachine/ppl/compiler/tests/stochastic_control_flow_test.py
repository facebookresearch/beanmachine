# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.compiler.runtime import BMGRuntime
from torch import tensor
from torch.distributions import Bernoulli, Beta, Normal


# Random variable that takes an argument
@bm.random_variable
def norm(n):
    return Normal(loc=0.0, scale=1.0)


# Random variable that takes no argument
@bm.random_variable
def coin():
    return Beta(2.0, 2.0)


# Call to random variable inside random variable
@bm.random_variable
def flip():
    return Bernoulli(coin())


@bm.random_variable
def flips(n):
    return Bernoulli(0.5)


@bm.random_variable
def spike_and_slab(n):
    if n:
        return Bernoulli(0.5)
    else:
        return Normal(0, 1)


# Try out a stochastic control flow where we choose
# a mean from one of two distributions depending on
# a coin flip.
@bm.random_variable
def choose_your_mean():
    return Normal(spike_and_slab(flip()), 1)


# Now let's try what looks like a stochastic workflow but is
# actually deterministic. We should detect this and avoid
# generating a stochastic workflow.


@bm.functional
def always_zero():
    return tensor(0)


@bm.random_variable
def any_index_you_want_as_long_as_it_is_zero():
    return Normal(spike_and_slab(always_zero()), 1)


# Now choose from one of three options; notice that we have
# computed a stochastic value inline here rather than putting
# it in a functional; that's fine.
@bm.random_variable
def three_possibilities():
    return Normal(spike_and_slab(flips(0) + flips(1)), 1)


@bm.random_variable
def choice_of_flips(n):
    if n:
        return Bernoulli(0.75)
    return Bernoulli(0.25)


# Demonstrate that composition is broken.
# TODO: When it is fixed, update this comment.
@bm.random_variable
def composition_is_broken():
    return Normal(spike_and_slab(choice_of_flips(flip())), 1)


# Make a choice of four possibilities based on two parameters.
@bm.random_variable
def multiple_choice(m, n):
    if n:
        if m:
            return Bernoulli(0.125)
        return Bernoulli(0.25)
    if m:
        return Bernoulli(0.75)
    return Bernoulli(0.875)


@bm.random_variable
def two_parameters():
    return Normal(multiple_choice(flips(0), flips(1)), 1)


class StochasticControlFlowTest(unittest.TestCase):
    def test_stochastic_control_flow_1(self) -> None:
        self.maxDiff = None

        queries = [any_index_you_want_as_long_as_it_is_zero()]
        observations = {}
        bmg = BMGRuntime().accumulate_graph(queries, observations)

        # Here we have what looks like a stochastic control flow but
        # in reality there is only one possibility. We should ensure
        # that we generate a graph with no choice points.

        observed = to_dot(bmg, after_transform=True, label_edges=False)
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Normal];
  N5[label=Sample];
  N6[label=Query];
  N0 -> N2;
  N1 -> N2;
  N1 -> N4;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
  N5 -> N6;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_stochastic_control_flow_2(self) -> None:
        self.maxDiff = None

        queries = [choose_your_mean()]
        observations = {}
        bmg = BMGRuntime().accumulate_graph(queries, observations)

        # Note that we generate an if-then-else node here to express the
        # flip that chooses between two alternatives, and therefore can
        # lower this to a form that BMG would accept.
        observed = to_dot(bmg, after_transform=True, label_edges=True)
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=Beta];
  N02[label=Sample];
  N03[label=Bernoulli];
  N04[label=Sample];
  N05[label=0.0];
  N06[label=1.0];
  N07[label=Normal];
  N08[label=Sample];
  N09[label=0.5];
  N10[label=Bernoulli];
  N11[label=Sample];
  N12[label=ToReal];
  N13[label=if];
  N14[label=Normal];
  N15[label=Sample];
  N16[label=Query];
  N00 -> N01[label=alpha];
  N00 -> N01[label=beta];
  N01 -> N02[label=operand];
  N02 -> N03[label=probability];
  N03 -> N04[label=operand];
  N04 -> N13[label=condition];
  N05 -> N07[label=mu];
  N06 -> N07[label=sigma];
  N06 -> N14[label=sigma];
  N07 -> N08[label=operand];
  N08 -> N13[label=alternative];
  N09 -> N10[label=probability];
  N10 -> N11[label=operand];
  N11 -> N12[label=operand];
  N12 -> N13[label=consequence];
  N13 -> N14[label=mu];
  N14 -> N15[label=operand];
  N15 -> N16[label=operator];
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_stochastic_control_flow_3(self) -> None:
        self.maxDiff = None

        queries = [three_possibilities()]
        observations = {}
        bmg = BMGRuntime().accumulate_graph(queries, observations)

        # Now we have three possibilities, and we cannot yet represent
        # that in BMG.
        # TODO: When we can represent this in BMG, update this test.
        observed = to_dot(bmg, after_transform=False, label_edges=True)
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=0.5];
  N05[label=Bernoulli];
  N06[label=Sample];
  N07[label=2.0];
  N08[label=Sample];
  N09[label=map];
  N10[label=Sample];
  N11[label=Sample];
  N12[label="+"];
  N13[label=index];
  N14[label=1];
  N15[label=Normal];
  N16[label=Sample];
  N17[label=Query];
  N00 -> N02[label=mu];
  N00 -> N09[label=0];
  N01 -> N02[label=sigma];
  N01 -> N09[label=2];
  N02 -> N03[label=operand];
  N03 -> N09[label=1];
  N04 -> N05[label=probability];
  N05 -> N06[label=operand];
  N05 -> N08[label=operand];
  N05 -> N10[label=operand];
  N05 -> N11[label=operand];
  N06 -> N09[label=3];
  N07 -> N09[label=4];
  N08 -> N09[label=5];
  N09 -> N13[label=left];
  N10 -> N12[label=left];
  N11 -> N12[label=right];
  N12 -> N13[label=right];
  N13 -> N15[label=mu];
  N14 -> N15[label=sigma];
  N15 -> N16[label=operand];
  N16 -> N17[label=operator];
}"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_stochastic_control_flow_composition_broken(self) -> None:
        self.maxDiff = None

        queries = [composition_is_broken()]
        observations = {}

        # Here we have a case where we have composed one stochastic control flow
        # as the input to another, and that is currently broken.
        #
        # TODO: Fix it.

        with self.assertRaises(ValueError):
            BMGRuntime().accumulate_graph(queries, observations)

    def test_stochastic_control_flow_4(self) -> None:
        self.maxDiff = None

        queries = [two_parameters()]
        observations = {}
        bmg = BMGRuntime().accumulate_graph(queries, observations)

        # Here we have four possibilities but since each is a Boolean choice
        # it turns out we can in fact represent it.
        observed = to_dot(bmg, after_transform=True, label_edges=True)
        expected = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=Sample];
  N04[label=0.875];
  N05[label=Bernoulli];
  N06[label=Sample];
  N07[label=0.25];
  N08[label=Bernoulli];
  N09[label=Sample];
  N10[label=0.75];
  N11[label=Bernoulli];
  N12[label=Sample];
  N13[label=0.125];
  N14[label=Bernoulli];
  N15[label=Sample];
  N16[label=if];
  N17[label=if];
  N18[label=if];
  N19[label=ToReal];
  N20[label=1.0];
  N21[label=Normal];
  N22[label=Sample];
  N23[label=Query];
  N00 -> N01[label=probability];
  N01 -> N02[label=operand];
  N01 -> N03[label=operand];
  N02 -> N18[label=condition];
  N03 -> N16[label=condition];
  N03 -> N17[label=condition];
  N04 -> N05[label=probability];
  N05 -> N06[label=operand];
  N06 -> N17[label=alternative];
  N07 -> N08[label=probability];
  N08 -> N09[label=operand];
  N09 -> N17[label=consequence];
  N10 -> N11[label=probability];
  N11 -> N12[label=operand];
  N12 -> N16[label=alternative];
  N13 -> N14[label=probability];
  N14 -> N15[label=operand];
  N15 -> N16[label=consequence];
  N16 -> N18[label=consequence];
  N17 -> N18[label=alternative];
  N18 -> N19[label=operand];
  N19 -> N21[label=mu];
  N20 -> N21[label=sigma];
  N21 -> N22[label=operand];
  N22 -> N23[label=operator];
}
"""
        self.assertEqual(expected.strip(), observed.strip())


# TODO: Test that shows that map support is broken
# TODO: Fix it!
# TODO: Test that shows what happens when multiple graph node
# arguments are not independent. Can get some false paths
# in the graph when this happens. Can we prune them?
