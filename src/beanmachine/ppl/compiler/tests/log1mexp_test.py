# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end test for log1mexp"""
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from beanmachine.ppl.utils.hint import log1mexp, math_log1mexp
from torch import tensor
from torch.distributions import Bernoulli, Beta, HalfCauchy


# New


@bm.random_variable
def hc():
    return HalfCauchy(42)  # positive real


@bm.functional
def right():
    return log1mexp(-hc())  # log1mexp takes a negative real


@bm.functional
def wrong():
    return log1mexp(hc())  # log1mexp takes a negative real!


@bm.functional
def math_right():
    return math_log1mexp(-hc())  # log1mexp takes a negative real


@bm.functional
def math_wrong():
    return math_log1mexp(hc())  # log1mexp takes a negative real!


# Old


@bm.random_variable
def beta():
    return Beta(2.0, -math_log1mexp(-2.0))


@bm.random_variable
def beta2():
    return Beta(2.0, -log1mexp(-beta()))


@bm.random_variable
def flip(n):
    return Bernoulli(beta())


class Log1mexpTest(unittest.TestCase):
    def test_log1mexp(self) -> None:
        """log1mexp"""

        # New
        #
        # First we look at the torch.tensor case
        #
        # Example of a model that is OK
        #
        queries = [right()]
        observations = {}
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N0[label=42.0];
  N1[label=HalfCauchy];
  N2[label=Sample];
  N3[label="-"];
  N4[label=Log1mexp];
  N5[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

        self.assertTrue(
            BMGInference().infer(queries, observations, 1),
            msg="Expected inference to complete successful on this example.",
        )

        #
        # Example of a model that is not OK, that is, should raise an error
        #
        queries = [wrong()]
        observations = {}
        with self.assertRaises(ValueError) as ex:
            observed = BMGInference().to_dot(queries, observations)
        expected = """The operand of a Log1mexp is required to be a negative real but is a positive real."""
        self.assertEqual(expected.strip(), str(ex.exception))

        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, observations, 1)
        self.assertEqual(expected.strip(), str(ex.exception))

        queries = [right()]
        observations = {hc(): tensor(1.0)}
        result = BMGInference().infer(queries, observations, 1)
        observed = result[right()]
        expected = log1mexp(tensor(-1.0))
        self.assertEqual(observed, expected)

        # Second we look at the math_ case
        #
        # Example of a model that is OK
        #
        queries = [math_right()]
        observations = {}
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N0[label=42.0];
  N1[label=HalfCauchy];
  N2[label=Sample];
  N3[label="-"];
  N4[label=Log1mexp];
  N5[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

        self.assertTrue(
            BMGInference().infer(queries, observations, 1),
            msg="Expected inference to complete successful on this example.",
        )

        #
        # Example of a model that is not OK, that is, should raise an error
        #
        queries = [math_wrong()]
        observations = {}
        with self.assertRaises(ValueError) as ex:
            observed = BMGInference().to_dot(queries, observations)
        expected = """The operand of a Log1mexp is required to be a negative real but is a positive real."""
        self.assertEqual(expected.strip(), str(ex.exception))

        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, observations, 1)
        self.assertEqual(expected.strip(), str(ex.exception))

        queries = [math_right()]
        observations = {hc(): tensor(1.0)}
        result = BMGInference().infer(queries, observations, 1)
        observed = result[math_right()]
        expected = math_log1mexp(-1.0)
        self.assertEqual(observed, expected)

        # ...

        # Old

    def test_log1mexp_coin_flip_inference(self) -> None:
        """Like a test in coin_flip_test.py but with log1mexp"""

        self.maxDiff = None
        queries = [beta()]
        observations = {
            flip(0): tensor(0.0),
            flip(1): tensor(0.0),
            flip(2): tensor(1.0),
            flip(3): tensor(0.0),
        }
        num_samples = 1000
        inference = BMGInference()
        mcsamples = inference.infer(queries, observations, num_samples)
        samples = mcsamples[beta()]
        observed = samples.mean()
        expected = tensor(0.4873)
        self.assertAlmostEqual(first=observed, second=expected, delta=0.05)

    def test_log1mexp_coin_flip_to_dot_cpp_python(self) -> None:
        """Like a test in coin_flip_test.py but with log1mexp"""
        self.maxDiff = None
        queries = [beta2()]
        observations = {
            flip(0): tensor(0.0),
            flip(1): tensor(0.0),
            flip(2): tensor(1.0),
            flip(3): tensor(0.0),
        }
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=0.14541345834732056];
  N02[label=Beta];
  N03[label=Sample];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label="Observation False"];
  N07[label=Sample];
  N08[label="Observation False"];
  N09[label=Sample];
  N10[label="Observation True"];
  N11[label=Sample];
  N12[label="Observation False"];
  N13[label=ToPosReal];
  N14[label="-"];
  N15[label=Log1mexp];
  N16[label="-"];
  N17[label=Beta];
  N18[label=Sample];
  N19[label=Query];
  N00 -> N02;
  N00 -> N17;
  N01 -> N02;
  N02 -> N03;
  N03 -> N04;
  N03 -> N13;
  N04 -> N05;
  N04 -> N07;
  N04 -> N09;
  N04 -> N11;
  N05 -> N06;
  N07 -> N08;
  N09 -> N10;
  N11 -> N12;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
  N16 -> N17;
  N17 -> N18;
  N18 -> N19;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

        observed = BMGInference().to_cpp(queries, observations)
        expected = """
graph::Graph g;
uint n0 = g.add_constant_pos_real(2.0);
uint n1 = g.add_constant_pos_real(0.14541345834732056);
uint n2 = g.add_distribution(
  graph::DistributionType::BETA,
  graph::AtomicType::PROBABILITY,
  std::vector<uint>({n0, n1}));
uint n3 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n2}));
uint n4 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n3}));
uint n5 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n4}));
g.observe([n5], false);
uint n7 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n4}));
g.observe([n7], false);
uint n9 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n4}));
g.observe([n9], true);
uint n11 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n4}));
g.observe([n11], false);
uint n13 = g.add_operator(
  graph::OperatorType::TO_POS_REAL, std::vector<uint>({n3}));
uint n14 = g.add_operator(
  graph::OperatorType::NEGATE, std::vector<uint>({n13}));
uint n15 = g.add_operator(
  graph::OperatorType::LOG1MEXP, std::vector<uint>({n14}));
uint n16 = g.add_operator(
  graph::OperatorType::NEGATE, std::vector<uint>({n15}));
uint n17 = g.add_distribution(
  graph::DistributionType::BETA,
  graph::AtomicType::PROBABILITY,
  std::vector<uint>({n0, n16}));
uint n18 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n17}));
g.query(n18);"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_python(queries, observations)
        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_real(2.0)
n1 = g.add_constant_pos_real(0.14541345834732056)
n2 = g.add_distribution(
  graph.DistributionType.BETA,
  graph.AtomicType.PROBABILITY,
  [n0, n1],
)
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n2])
n4 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n3],
)
n5 = g.add_operator(graph.OperatorType.SAMPLE, [n4])
g.observe(n5, False)
n6 = g.add_operator(graph.OperatorType.SAMPLE, [n4])
g.observe(n6, False)
n7 = g.add_operator(graph.OperatorType.SAMPLE, [n4])
g.observe(n7, True)
n8 = g.add_operator(graph.OperatorType.SAMPLE, [n4])
g.observe(n8, False)
n9 = g.add_operator(graph.OperatorType.TO_POS_REAL, [n3])
n10 = g.add_operator(graph.OperatorType.NEGATE, [n9])
n11 = g.add_operator(graph.OperatorType.LOG1MEXP, [n10])
n12 = g.add_operator(graph.OperatorType.NEGATE, [n11])
n13 = g.add_distribution(
  graph.DistributionType.BETA,
  graph.AtomicType.PROBABILITY,
  [n0, n12],
)
n14 = g.add_operator(graph.OperatorType.SAMPLE, [n13])
q0 = g.query(n14)
        """
        self.assertEqual(expected.strip(), observed.strip())
